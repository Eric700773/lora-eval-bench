import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.benchmarks.contracts import BenchmarkSpec, GenerationConfig, NormalizedRunResult
from app.benchmarks.registry import get_benchmark_runner, list_benchmark_runners
from app.benchmarks.lm_eval_runner import LmEvalRunner
from app.services.decontamination_service import check_sample_contamination
from app.services.flywheel_service import (
    _synthesis_leak_text,
    _validate_synthesized_sample,
    analyze_badcases_and_synthesize,
)
from app.services.llamafactory_config_service import generate_training_yaml
from app.services import lm_eval_import_service
from app.services.lm_eval_import_service import import_lm_eval_gate
from app.workers.qwen_wsl_worker import detect_repetition, sha256_text, sha256_token_ids
from app.workers import lm_eval_wsl_worker


class LmEvalRunnerTests(unittest.TestCase):
    def test_runner_builds_frozen_gate_command(self) -> None:
        runner = get_benchmark_runner("lm_eval")
        spec = BenchmarkSpec(
            framework="lm_eval",
            framework_version="0.4.13",
            framework_revision="ddd67220430a2470529f25fd5c05a576ca1057a0",
            task_name="gsm8k",
            task_version="3.0",
            dataset_revision="740312add88f781978c0658806c59bc2815b9866",
            split="test",
            few_shot=5,
            limit=10,
            mode="gate",
            model_id="local-qwen3.5-2b",
            generation=GenerationConfig(max_new_tokens=512, seed=42),
        )
        command = runner.build_command(spec, Path(r"C:\tmp\lm_eval_gate"))
        joined = " ".join(command)
        self.assertIn("local-chat-completions", command)
        self.assertIn("--apply_chat_template", command)
        self.assertIn("--log_samples", command)
        self.assertNotIn("--check_integrity", command)
        self.assertIn("--show_config", command)
        self.assertIn("tokenizer_backend=none", joined)
        self.assertIn("tokenized_requests=False", joined)
        self.assertIn("seed=42", joined)
        self.assertIn("--limit 10", joined)
        self.assertIn("--num_fewshot 5", joined)

    def test_runner_builds_frozen_full_command_without_limit(self) -> None:
        runner = get_benchmark_runner("lm_eval")
        spec = BenchmarkSpec(
            framework="lm_eval",
            framework_version="0.4.13",
            framework_revision="ddd67220430a2470529f25fd5c05a576ca1057a0",
            task_name="gsm8k",
            task_version="3.0",
            dataset_revision="740312add88f781978c0658806c59bc2815b9866",
            split="test",
            few_shot=5,
            limit=None,
            mode="full",
            model_id="local-qwen3.5-2b",
            generation=GenerationConfig(max_new_tokens=512, seed=42),
        )
        command = runner.build_command(spec, Path(r"C:\tmp\lm_eval_full"))
        self.assertNotIn("--limit", command)
        self.assertIn("--use_cache", command)
        self.assertIn("response_cache", " ".join(command))
        self.assertIn("seed=42", " ".join(command))

    def test_runner_normalizes_lm_eval_fixture(self) -> None:
        runner = LmEvalRunner()
        fixtures = BACKEND_DIR / "tests" / "fixtures" / "lm_eval"
        spec = BenchmarkSpec(
            framework="lm_eval",
            framework_version="0.4.13",
            framework_revision="ddd67220430a2470529f25fd5c05a576ca1057a0",
            task_name="gsm8k",
            task_version="3.0",
            dataset_revision="740312add88f781978c0658806c59bc2815b9866",
            split="test",
            few_shot=5,
            limit=2,
            mode="gate",
            model_id="local-qwen3.5-2b",
        )
        from app.benchmarks.contracts import RunManifest

        manifest = RunManifest(
            framework="lm_eval",
            framework_version="0.4.13",
            framework_revision=spec.framework_revision,
            task_name="gsm8k",
            benchmark_spec_hash=spec.spec_hash(),
        )
        normalized = runner.normalize_files(
            spec,
            manifest,
            fixtures / "results_fixture.json",
            fixtures / "samples_fixture.jsonl",
        )
        self.assertEqual(normalized.metrics["lm_eval_version"], "0.4.13")
        self.assertEqual(normalized.metrics["n_samples"]["gsm8k"]["effective"], 2)
        self.assertEqual(len(normalized.items), 2)
        self.assertTrue(normalized.items[0].passed)
        self.assertFalse(normalized.items[1].passed)
        self.assertEqual(normalized.items[0].parsed_output, "42")
        self.assertEqual(normalized.items[1].parsed_output, None)
        self.assertIn("strict-match", normalized.items[0].evidence["filters"])
        self.assertIn("flexible-extract", normalized.items[0].evidence["filters"])
        self.assertEqual(normalized.items[0].framework_request["request_type"], "generate_until")
        self.assertEqual(normalized.items[0].model_input.messages[0]["role"], "user")


    def test_runner_aligns_worker_evidence_and_passes_engineering_gate(self) -> None:
        runner = LmEvalRunner()
        fixtures = BACKEND_DIR / "tests" / "fixtures" / "lm_eval"
        spec = BenchmarkSpec(
            framework="lm_eval",
            framework_version="0.4.13",
            framework_revision="ddd67220430a2470529f25fd5c05a576ca1057a0",
            task_name="gsm8k",
            task_version="3.0",
            dataset_revision="740312add88f781978c0658806c59bc2815b9866",
            split="test",
            few_shot=5,
            limit=2,
            mode="gate",
            model_id="local-qwen3.5-2b",
        )
        from app.benchmarks.contracts import RunManifest

        manifest = RunManifest(
            framework="lm_eval",
            framework_version="0.4.13",
            framework_revision=spec.framework_revision,
            task_name="gsm8k",
            benchmark_spec_hash=spec.spec_hash(),
        )
        normalized = runner.normalize_files(
            spec,
            manifest,
            fixtures / "results_fixture.json",
            fixtures / "samples_fixture.jsonl",
        )
        with tempfile.TemporaryDirectory() as temporary:
            evidence_paths = []
            for item in normalized.items:
                evidence = {
                    "request_id": f"fixture-{item.sample_id}",
                    "api_request": {"messages": item.model_input.messages},
                    "model_input": {
                        "messages": item.model_input.messages,
                        "rendered_prompt": f"rendered-{item.sample_id}",
                        "rendered_prompt_sha256": f"render-hash-{item.sample_id}",
                        "chat_template_sha256": "template-hash",
                        "prompt_tokens": 3,
                        "input_ids": [1, 2, 3],
                        "input_ids_sha256": sha256_token_ids([1, 2, 3]),
                        "input_ids_head": [1, 2, 3],
                        "input_ids_tail": [1, 2, 3],
                        "truncated": False,
                        "model_context_length": 1024,
                        "context_exceeded": False,
                        "nested_chat_template_detected": False,
                        "render_token_roundtrip_matched": True,
                        "decoded_input_with_special_tokens": f"rendered-{item.sample_id}",
                    },
                    "model_output": {
                        "raw_generated_token_ids": [4, 5],
                        "raw_generated_text": item.raw_output["response"],
                        "content": item.raw_output["response"],
                        "content_sha256": sha256_text(item.raw_output["response"]),
                        "finish_reason": "stop",
                        "repetition_diagnostic": {"detected": False},
                    },
                }
                path = Path(temporary) / f"fixture-{item.sample_id}.json"
                path.write_text(json.dumps(evidence), encoding="utf-8")
                evidence_paths.append(path)
            alignment = runner.align_worker_evidence(normalized, evidence_paths)
            gate = runner.build_integrity_gate(normalized, alignment)
        self.assertEqual(alignment["aligned_items"], 2)
        self.assertEqual(gate.status, "passed")
        self.assertEqual(gate.engineering_failures, 0)
        self.assertEqual(gate.model_failures, 1)
        self.assertTrue(all(item.evidence["worker_evidence_status"] == "aligned" for item in normalized.items))
        self.assertTrue(all(item.model_input.input_ids == [1, 2, 3] for item in normalized.items))

    def test_offline_task_changes_only_dataset_source(self) -> None:
        runner = LmEvalRunner()
        with tempfile.TemporaryDirectory() as temporary:
            task_path, evidence = runner.prepare_execution_task(Path(temporary))
            execution = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        self.assertTrue(evidence["protocol_fields_unchanged"])
        self.assertEqual(execution["task"], "gsm8k_easytune_gate")
        self.assertEqual(execution["dataset_path"], "parquet")
        self.assertEqual(execution["num_fewshot"], 5)
        self.assertEqual(execution["output_type"], "generate_until")


class LmEvalImportTests(unittest.TestCase):
    def test_passed_gate_import_is_complete_and_idempotent(self) -> None:
        from app.benchmarks.contracts import IntegrityGateReport, RunManifest
        from app.database import Base
        from app.models import BenchmarkItemResult, BenchmarkRun, Dataset

        runner = LmEvalRunner()
        fixtures = BACKEND_DIR / "tests" / "fixtures" / "lm_eval"
        spec = BenchmarkSpec(
            framework="lm_eval",
            framework_version="0.4.13",
            framework_revision="fixture-commit",
            task_name="gsm8k",
            task_version="3.0",
            dataset_revision="fixture-revision",
            split="test",
            few_shot=5,
            limit=2,
            mode="gate",
            model_id="local-qwen3.5-2b",
            task_config={"test_fingerprint": "fixture-fingerprint"},
        )
        manifest = RunManifest(
            framework="lm_eval",
            framework_version="0.4.13",
            task_name="gsm8k",
            benchmark_spec_hash=spec.spec_hash(),
            environment={"test_rows": 1319},
        )
        normalized = runner.normalize_files(
            spec,
            manifest,
            fixtures / "results_fixture.json",
            fixtures / "samples_fixture.jsonl",
        )
        normalized.gate_report = IntegrityGateReport(
            status="passed",
            model_failures=1,
            engineering_failures=0,
        )

        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_root = root / "reports"
            run_dir = report_root / "fixture_gate"
            run_dir.mkdir(parents=True)
            (run_dir / "normalized_result.json").write_text(
                normalized.model_dump_json(indent=2), encoding="utf-8"
            )
            (run_dir / "integrity_gate.json").write_text(
                normalized.gate_report.model_dump_json(indent=2), encoding="utf-8"
            )
            snapshot = root / spec.dataset_revision
            parquet = snapshot / "main" / "test-00000-of-00001.parquet"
            parquet.parent.mkdir(parents=True)
            parquet.write_bytes(b"fixture")
            with (
                patch.object(lm_eval_import_service, "LM_EVAL_REPORT_ROOT", report_root),
                patch.object(lm_eval_import_service, "DEFAULT_DATASET_SNAPSHOT", snapshot),
                Session(engine) as db,
            ):
                first = import_lm_eval_gate(db, "fixture_gate", "Fixture Gate")
                second = import_lm_eval_gate(db, "fixture_gate", "Fixture Gate Again")
                self.assertEqual(first.id, second.id)
                self.assertEqual(db.query(BenchmarkRun).count(), 1)
                self.assertEqual(db.query(Dataset).count(), 1)
                self.assertEqual(db.query(BenchmarkItemResult).count(), 2)
                self.assertEqual(first.framework, "lm_eval")
                self.assertEqual(first.task_name, "gsm8k")
                self.assertEqual(first.gate_report_json["status"], "passed")
                self.assertEqual(
                    db.query(BenchmarkItemResult).filter(BenchmarkItemResult.is_passed == 0).count(),
                    1,
                )
                from fastapi import HTTPException

                with self.assertRaises(HTTPException) as blocked:
                    analyze_badcases_and_synthesize(db, first.id, "must-not-train")
                self.assertIn("analysis-only", str(blocked.exception.detail))


class LmEvalWorkerTests(unittest.TestCase):
    def test_worker_accepts_only_frozen_gate_command(self) -> None:
        runner = LmEvalRunner()
        spec = BenchmarkSpec(
            framework="lm_eval",
            framework_version="0.4.13",
            framework_revision="ddd67220430a2470529f25fd5c05a576ca1057a0",
            task_name="gsm8k",
            split="test",
            few_shot=5,
            limit=10,
            mode="gate",
            model_id="local-qwen3.5-2b",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary).resolve()
            command = runner.build_command(spec, output_dir)[4:]
            old_cli = getattr(lm_eval_wsl_worker, "LM_EVAL_CLI", None)
            old_root = getattr(lm_eval_wsl_worker, "ALLOWED_OUTPUT_ROOT", None)
            try:
                lm_eval_wsl_worker.LM_EVAL_CLI = command[command.index("run") - 1]
                lm_eval_wsl_worker.ALLOWED_OUTPUT_ROOT = Path(
                    next(value for value in command if value.startswith("/mnt/") and value.endswith(Path(temporary).name))
                ).resolve()
                lm_eval_wsl_worker.validate_command(command)
                full_spec = spec.model_copy(update={"mode": "full", "limit": None})
                full_command = runner.build_command(full_spec, output_dir)[4:]
                self.assertEqual(lm_eval_wsl_worker.validate_command(full_command), "full")
                unsafe = command + ["--confirm_run_unsafe_code"]
                with self.assertRaises(ValueError):
                    lm_eval_wsl_worker.validate_command(unsafe)
                wrong_limit = list(command)
                wrong_limit[wrong_limit.index("--limit") + 1] = "100"
                with self.assertRaises(ValueError):
                    lm_eval_wsl_worker.validate_command(wrong_limit)
            finally:
                if old_cli is not None:
                    lm_eval_wsl_worker.LM_EVAL_CLI = old_cli
                if old_root is not None:
                    lm_eval_wsl_worker.ALLOWED_OUTPUT_ROOT = old_root


class WorkerEvidenceTests(unittest.TestCase):
    def test_evidence_hashes_and_repetition_diagnostic_are_stable(self) -> None:
        self.assertEqual(
            sha256_text("Question: 2+2?"),
            "db6102aac7b4fd51194a237b3c25a3bb391bf9a7a9179fd1143ccdce72081c3a",
        )
        self.assertEqual(
            sha256_token_ids([1, 2, 3]),
            "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4",
        )
        self.assertFalse(detect_repetition("The answer is 42.")["detected"])
        repeated = detect_repetition("one two three four " * 6)
        self.assertTrue(repeated["detected"])
        self.assertGreaterEqual(repeated["max_repeated_ngram"], 4)


class LmEvalFreezeTests(unittest.TestCase):
    def test_frozen_gsm8k_protocol_and_environment(self) -> None:
        config_root = BACKEND_DIR / "configs" / "lm_eval"
        manifest = json.loads((config_root / "gsm8k-v0.4.13.manifest.json").read_text(encoding="utf-8"))
        task_path = config_root / "gsm8k-v0.4.13.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        task_hash = hashlib.sha256(task_path.read_bytes()).hexdigest()
        lock_text = (config_root / manifest["packages_lock"]).read_text(encoding="utf-8")

        self.assertEqual(manifest["framework_version"], "0.4.13")
        self.assertEqual(manifest["framework_commit"], "ddd67220430a2470529f25fd5c05a576ca1057a0")
        self.assertEqual(manifest["task_config_sha256"], task_hash)
        self.assertEqual(manifest["dataset_revision"], "740312add88f781978c0658806c59bc2815b9866")
        self.assertTrue(manifest["dataset_snapshot_verified"])
        self.assertEqual(manifest["splits"]["train"]["rows"], 7473)
        self.assertEqual(manifest["splits"]["test"]["rows"], 1319)
        self.assertEqual(task["task"], "gsm8k")
        self.assertEqual(task["output_type"], "generate_until")
        self.assertEqual(task["test_split"], "test")
        self.assertEqual(task["num_fewshot"], 5)
        self.assertTrue(
            "lm-eval==0.4.13" in lock_text or "lm_eval==0.4.13" in lock_text
        )


class BenchmarkAbstractionTests(unittest.TestCase):
    def test_benchmark_spec_hash_is_stable(self) -> None:
        spec = BenchmarkSpec(
            framework="lm_eval",
            task_name="gsm8k",
            framework_version="fixed-version",
            model_id="local-qwen3.5-2b",
            mode="gate",
            limit=10,
            generation=GenerationConfig(max_new_tokens=512, seed=42),
        )
        restored = BenchmarkSpec.model_validate_json(spec.model_dump_json())
        self.assertEqual(spec.spec_hash(), restored.spec_hash())
        self.assertEqual(len(spec.spec_hash()), 64)

    def test_custom_runner_is_registered_and_normalizes_existing_run(self) -> None:
        from app.database import SessionLocal, init_db

        init_db()
        self.assertIn("custom", list_benchmark_runners())
        runner = get_benchmark_runner("custom")
        db = SessionLocal()
        try:
            from app.models import BenchmarkRun

            existing = db.query(BenchmarkRun).filter(BenchmarkRun.status == "completed").order_by(BenchmarkRun.id.asc()).first()
            if existing is None:
                self.skipTest("No completed local benchmark run exists in this workspace.")
            existing.framework = existing.framework or "custom"
            existing.task_name = existing.task_name or "custom_30"
            db.commit()
            normalized = runner.normalize(db, existing.id)
            self.assertIsInstance(normalized, NormalizedRunResult)
            self.assertEqual(normalized.spec.framework, "custom")
            self.assertEqual(len(normalized.items), 30)
            self.assertEqual(normalized.manifest.benchmark_spec_hash, normalized.spec.spec_hash())
        finally:
            db.close()


class DecontaminationTests(unittest.TestCase):
    def test_detects_direct_leak_and_keeps_clean_sample(self) -> None:
        corpus = ["某仓库原有货物 240 箱，第一天运出总数的四分之一。"]
        leaked = check_sample_contamination(corpus[0], corpus)
        clean = check_sample_contamination("计算圆周长，半径为 7。", corpus)
        self.assertTrue(leaked["is_contaminated"])
        self.assertEqual(leaked["method"], "13-gram_exact_overlap")
        self.assertFalse(clean["is_contaminated"])

    def test_code_boilerplate_does_not_create_false_positive(self) -> None:
        benchmark = ["编写 Python 函数 first_unique_char(s) 返回第一个不重复字符索引。"]
        row = {
            "instruction": "编写 Python 函数 merge_intervals(intervals) 合并重叠区间。",
            "output": "def merge_intervals(intervals):\n    return intervals",
        }
        result = check_sample_contamination(
            _synthesis_leak_text(row, "code_sandbox"),
            benchmark,
            ngram_size=13,
            semantic_threshold=0.72,
        )
        self.assertFalse(result["is_contaminated"])


class SynthesisQualityTests(unittest.TestCase):
    def test_json_and_code_use_deterministic_gates(self) -> None:
        valid_json = {
            "instruction": "请提取订单信息并仅输出 JSON 对象。",
            "output": json.dumps({"order_id": "A-1"}, ensure_ascii=False),
        }
        invalid_json = {"instruction": "请提取订单信息并仅输出 JSON 对象。", "output": "{'order_id': 'A-1'}"}
        valid_code = {
            "instruction": "实现一个合并区间函数并返回新区间列表。",
            "output": "def merge_intervals(items):\n    return items",
        }
        invalid_code = {"instruction": "实现一个合并区间函数并返回新区间列表。", "output": "def broken("}
        self.assertTrue(_validate_synthesized_sample(valid_json, "json_format")[0])
        self.assertFalse(_validate_synthesized_sample(invalid_json, "json_format")[0])
        self.assertTrue(_validate_synthesized_sample(valid_code, "code_sandbox")[0])
        self.assertFalse(_validate_synthesized_sample(invalid_code, "code_sandbox")[0])


class QLoRAConfigTests(unittest.TestCase):
    def test_qlora_maps_to_lora_plus_four_bit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = SimpleNamespace(
                id=1,
                model_family="qwen",
                budget_level="low",
                base_model=r"D:\models\qwen3.5-2b",
                stage="sft",
                finetuning_type="qlora",
                template="qwen3_5_nothink",
                extra_config_json=None,
            )
            dataset = SimpleNamespace(
                id=12,
                status="converted",
                converted_file_path=str(root / "dataset.jsonl"),
            )
            paths = (root / "job.yaml", root / "output", root / "job.log")
            with patch("app.services.llamafactory_config_service._training_paths", return_value=paths):
                result = generate_training_yaml(job, dataset)
            config = yaml.safe_load(Path(result["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(config["finetuning_type"], "lora")
            self.assertEqual(config["quantization_bit"], 4)
            self.assertEqual(config["quantization_method"], "bitsandbytes")
            self.assertEqual(config["template"], "qwen3_5_nothink")
            self.assertEqual(config["cutoff_len"], 512)


if __name__ == "__main__":
    unittest.main()
