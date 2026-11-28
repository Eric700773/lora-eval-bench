from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.benchmarks.base import BenchmarkRunner
from app.benchmarks.contracts import (
    BenchmarkSpec,
    GateCheck,
    GenerationConfig,
    IntegrityGateReport,
    ModelInputEvidence,
    NormalizedItemResult,
    NormalizedRunResult,
    RunManifest,
)
from app.database import STORAGE_ROOT
from app.models import BenchmarkRun
from app.services.qwen_runtime_service import windows_to_wsl_path
from app.workers.qwen_wsl_worker import detect_repetition


BACKEND_DIR = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_DIR = BACKEND_DIR / "configs" / "lm_eval"
DEFAULT_MANIFEST_PATH = FROZEN_CONFIG_DIR / "gsm8k-v0.4.13.manifest.json"
DEFAULT_TASK_PATH = FROZEN_CONFIG_DIR / "gsm8k-v0.4.13.yaml"
DEFAULT_WSL_DISTRO = "Ubuntu"
DEFAULT_WSL_USER = os.getenv("EASYTUNE_WSL_USER", os.getenv("USERNAME", "user").lower())
DEFAULT_LM_EVAL_CLI = f"/home/{DEFAULT_WSL_USER}/.virtualenvs/easytune-lm-eval/bin/lm-eval"
DEFAULT_HF_HOME = windows_to_wsl_path(str(Path.home() / ".cache" / "huggingface"))
DEFAULT_MODEL_ENDPOINT = "http://127.0.0.1:8011/v1/chat/completions"
DEFAULT_LM_EVAL_WORKER_URL = "http://127.0.0.1:8012"
DEFAULT_MODEL_ID = "local-qwen3.5-2b"
DEFAULT_DATASET_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--openai--gsm8k"
    / "snapshots"
    / "740312add88f781978c0658806c59bc2815b9866"
)
RUNTIME_TASK_NAME = "gsm8k_easytune_gate"
WORKER_EVIDENCE_DIR = STORAGE_ROOT / "reports" / "worker_evidence"
PROTOCOL_KEYS = {
    "output_type",
    "training_split",
    "fewshot_split",
    "test_split",
    "doc_to_text",
    "doc_to_target",
    "metric_list",
    "generation_kwargs",
    "repeats",
    "num_fewshot",
    "filter_list",
    "metadata",
}


class LmEvalRunner(BenchmarkRunner):
    framework = "lm_eval"

    def __init__(self) -> None:
        self.frozen_manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))

    def build_spec(self, run: BenchmarkRun) -> BenchmarkSpec:
        if run.benchmark_spec_json:
            return BenchmarkSpec.model_validate(run.benchmark_spec_json)
        config = run.config_snapshot_json or {}
        requested_mode = str(config.get("mode", "gate")).lower()
        if requested_mode not in {"gate", "full"}:
            raise ValueError(f"Unsupported lm-eval mode: {requested_mode}")
        limit = 10 if requested_mode == "gate" else None
        mode = requested_mode
        return BenchmarkSpec(
            framework=self.framework,
            framework_version=self.frozen_manifest["framework_version"],
            framework_revision=self.frozen_manifest["framework_commit"],
            task_name=run.task_name or "gsm8k",
            task_version="3.0",
            dataset_revision=self.frozen_manifest["dataset_revision"],
            split="test",
            few_shot=5,
            limit=limit,
            mode=mode,
            model_id=run.base_model or DEFAULT_MODEL_ID,
            chat_template_sha256=config.get("chat_template_sha256"),
            generation=GenerationConfig(
                temperature=0.0,
                max_new_tokens=int(config.get("max_new_tokens", 512)),
                num_rollouts=1,
                enable_thinking=False,
                load_in_4bit=True,
                seed=42,
                stop_sequences=["Question:", "</s>", "<|im_end|>"],
            ),
            task_config={
                "task_config_sha256": self.frozen_manifest["task_config_sha256"],
                "dataset": self.frozen_manifest["dataset"],
                "dataset_name": self.frozen_manifest["dataset_name"],
                "train_fingerprint": self.frozen_manifest["splits"]["train"]["fingerprint"],
                "test_fingerprint": self.frozen_manifest["splits"]["test"]["fingerprint"],
                "output_type": "generate_until",
                "official_task_config": str(DEFAULT_TASK_PATH),
            },
            runner_config={
                "model_backend": "local-chat-completions",
                "model_endpoint": DEFAULT_MODEL_ENDPOINT,
                "batch_size": 1,
                "num_concurrent": 1,
                "tokenizer_backend": "none",
                "tokenized_requests": False,
                "apply_chat_template": True,
                "log_samples": True,
                "response_cache": True,
                "evidence_storage": "full" if mode == "gate" else "external_compact_index",
            },
        )

    def preflight(self, db: Session, run: BenchmarkRun, spec: BenchmarkSpec) -> IntegrityGateReport:
        checks: list[GateCheck] = []
        checks.append(self._path_check("frozen_manifest", DEFAULT_MANIFEST_PATH))
        checks.append(self._path_check("official_task_config", DEFAULT_TASK_PATH))
        task_hash = hashlib.sha256(DEFAULT_TASK_PATH.read_bytes()).hexdigest() if DEFAULT_TASK_PATH.exists() else None
        checks.append(
            GateCheck(
                name="task_config_hash",
                status="passed" if task_hash == self.frozen_manifest.get("task_config_sha256") else "failed",
                detail=task_hash,
            )
        )
        checks.append(
            GateCheck(
                name="frozen_protocol",
                status="passed"
                if spec.task_name == "gsm8k"
                and spec.few_shot == 5
                and spec.split == "test"
                and spec.framework_version == "0.4.13"
                else "failed",
                detail=f"task={spec.task_name}, few_shot={spec.few_shot}, split={spec.split}, version={spec.framework_version}",
            )
        )
        try:
            with urlopen(DEFAULT_MODEL_ENDPOINT.replace("/v1/chat/completions", "/v1/models"), timeout=10) as response:
                worker_ok = response.status == 200
        except Exception as exc:
            worker_ok = False
            worker_error = str(exc)
        else:
            worker_error = None
        checks.append(
            GateCheck(
                name="model_api_available",
                status="passed" if worker_ok else "failed",
                detail=DEFAULT_MODEL_ENDPOINT if worker_ok else worker_error,
            )
        )
        lm_eval_worker_url = os.getenv("EASYTUNE_LM_EVAL_WORKER_URL", DEFAULT_LM_EVAL_WORKER_URL)
        try:
            with urlopen(f"{lm_eval_worker_url.rstrip('/')}/health", timeout=10) as response:
                lm_eval_worker_ok = response.status == 200
        except Exception as exc:
            lm_eval_worker_ok = False
            lm_eval_worker_error = str(exc)
        else:
            lm_eval_worker_error = None
        checks.append(
            GateCheck(
                name="lm_eval_worker_available",
                status="passed" if lm_eval_worker_ok else "failed",
                detail=lm_eval_worker_url if lm_eval_worker_ok else lm_eval_worker_error,
            )
        )
        failed = [check for check in checks if check.status == "failed"]
        return IntegrityGateReport(
            status="failed" if failed else "passed",
            checks=checks,
            engineering_failures=len(failed),
        )

    def prepare_execution_task(self, output_dir: Path) -> tuple[Path, dict[str, Any]]:
        official = yaml.safe_load(DEFAULT_TASK_PATH.read_text(encoding="utf-8"))
        snapshot = Path(os.getenv("EASYTUNE_GSM8K_SNAPSHOT", str(DEFAULT_DATASET_SNAPSHOT))).resolve()
        train_path = snapshot / "main" / "train-00000-of-00001.parquet"
        test_path = snapshot / "main" / "test-00000-of-00001.parquet"
        if snapshot.name != self.frozen_manifest["dataset_revision"]:
            raise ValueError("GSM8K snapshot revision does not match frozen manifest.")
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Frozen GSM8K parquet files are missing under {snapshot}")
        execution = dict(official)
        execution["task"] = RUNTIME_TASK_NAME
        execution["dataset_path"] = "parquet"
        execution.pop("dataset_name", None)
        execution["dataset_kwargs"] = {
            "data_files": {
                "train": windows_to_wsl_path(str(train_path)),
                "test": windows_to_wsl_path(str(test_path)),
            }
        }
        protocol_diff = {
            key: {"official": official.get(key), "execution": execution.get(key)}
            for key in sorted(PROTOCOL_KEYS)
            if official.get(key) != execution.get(key)
        }
        if protocol_diff:
            raise ValueError(f"Offline task changed official protocol fields: {protocol_diff}")
        task_path = output_dir / f"{RUNTIME_TASK_NAME}.yaml"
        task_path.write_text(yaml.safe_dump(execution, allow_unicode=True, sort_keys=False), encoding="utf-8")
        evidence = {
            "official_task_sha256": hashlib.sha256(DEFAULT_TASK_PATH.read_bytes()).hexdigest(),
            "execution_task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
            "allowed_changes": {
                "task": {"official": official.get("task"), "execution": RUNTIME_TASK_NAME},
                "dataset_path": {"official": official.get("dataset_path"), "execution": "parquet"},
                "dataset_name": {"official": official.get("dataset_name"), "execution": None},
                "dataset_kwargs": execution["dataset_kwargs"],
            },
            "protocol_fields_unchanged": True,
            "dataset_revision": snapshot.name,
        }
        (output_dir / "offline_task_equivalence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return task_path, evidence

    def build_command(self, spec: BenchmarkSpec, output_dir: Path) -> list[str]:
        if spec.task_name != "gsm8k":
            raise ValueError("Stage-one LmEvalRunner only supports the frozen gsm8k task.")
        model_args = (
            f"model={DEFAULT_MODEL_ID},"
            f"base_url={DEFAULT_MODEL_ENDPOINT},"
            "tokenizer_backend=none,tokenized_requests=False,"
            "num_concurrent=1,timeout=900,max_length=262144,seed=42"
        )
        command = [
            "wsl.exe",
            "-d",
            os.getenv("EASYTUNE_WSL_DISTRO", DEFAULT_WSL_DISTRO),
            "--",
            "env",
            f"HF_HOME={os.getenv('EASYTUNE_LM_EVAL_HF_HOME', DEFAULT_HF_HOME)}",
            "HF_HUB_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
            "OPENAI_API_KEY=local-eval-key",
            os.getenv("EASYTUNE_LM_EVAL_CLI", DEFAULT_LM_EVAL_CLI),
            "run",
            "--model",
            "local-chat-completions",
            "--model_args",
            model_args,
            "--tasks",
            RUNTIME_TASK_NAME,
            "--include_path",
            windows_to_wsl_path(str(output_dir)),
            "--num_fewshot",
            str(spec.few_shot),
            "--apply_chat_template",
            "--batch_size",
            "1",
        ]
        if spec.mode == "gate":
            command.extend(["--limit", "10"])
        command.extend([
            "--gen_kwargs",
            f"max_gen_toks={spec.generation.max_new_tokens},temperature=0.0",
            "--seed",
            "42,42,42,42",
            "--log_samples",
            "--use_cache",
            windows_to_wsl_path(str(output_dir / "response_cache")),
            "--show_config",
            "--output_path",
            windows_to_wsl_path(str(output_dir)),
        ])
        return command

    def start_async(self, db: Session, run_id: int, resume: bool = False) -> BenchmarkRun:
        run = db.get(BenchmarkRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="BenchmarkRun not found.")
        spec = self.build_spec(run)
        if spec.mode != "full":
            raise HTTPException(status_code=400, detail="Async execution is reserved for full mode.")
        if run.status == "running" and not resume:
            return run
        output_dir = STORAGE_ROOT / "reports" / "lm_eval" / f"run_{run.id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        _, equivalence = self.prepare_execution_task(output_dir)
        gate = self.preflight(db, run, spec)
        gate.checks.append(
            GateCheck(
                name="offline_task_protocol_equivalence",
                status="passed" if equivalence["protocol_fields_unchanged"] else "failed",
                detail=str(output_dir / "offline_task_equivalence.json"),
                evidence=equivalence,
            )
        )
        if any(check.status == "failed" for check in gate.checks):
            gate.status = "failed"
            gate.engineering_failures = sum(check.status == "failed" for check in gate.checks)
            run.status = "failed"
            run.error_message = "lm-eval full preflight failed."
            run.gate_report_json = gate.model_dump(mode="json")
            db.commit()
            return run
        command = self.build_command(spec, output_dir)
        manifest = self._build_manifest(run, spec, command, output_dir)
        (output_dir / "run_manifest.before.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        previous_progress = run.progress_json or {}
        started_timestamp = float(previous_progress.get("evidence_started_timestamp") or time.time())
        attempt = int(previous_progress.get("attempt", 0)) + 1
        job_id = f"benchmark-{run.id}-full-{attempt}"
        payload = {
            "job_id": job_id,
            "command": command[4:],
            "log_path": windows_to_wsl_path(str(output_dir / "lm_eval.log")),
            "timeout": 21600,
        }
        worker_job = self._worker_json("POST", "/jobs", payload, timeout=30)
        run.framework = self.framework
        run.task_name = spec.task_name
        run.benchmark_spec_json = spec.model_dump(mode="json")
        run.config_hash = spec.spec_hash()
        run.run_manifest_json = manifest.model_dump(mode="json")
        run.gate_report_json = gate.model_dump(mode="json")
        run.progress_json = {
            "mode": "full",
            "job_id": worker_job["job_id"],
            "worker_status": worker_job["status"],
            "attempt": attempt,
            "expected_items": int(self.frozen_manifest["splits"]["test"]["rows"]),
            "completed_requests": 0,
            "percent": 0.0,
            "evidence_started_timestamp": started_timestamp,
            "started_at": datetime.now(UTC).isoformat(),
            "cache_path": str(output_dir / "response_cache"),
            "resumable": True,
        }
        run.status = "running"
        run.error_message = None
        db.commit()
        db.refresh(run)
        return run

    def refresh_async(self, db: Session, run_id: int) -> BenchmarkRun:
        run = db.get(BenchmarkRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="BenchmarkRun not found.")
        if run.status in {"completed", "gate_failed"} and run.normalized_result_path and Path(run.normalized_result_path).exists():
            return run
        progress = dict(run.progress_json or {})
        job_id = progress.get("job_id")
        if not job_id:
            raise HTTPException(status_code=400, detail="Run has no lm-eval worker job.")
        try:
            worker_job = self._worker_json("GET", f"/jobs/{job_id}", timeout=30)
        except RuntimeError as exc:
            progress["worker_status"] = "unavailable"
            progress["last_error"] = str(exc)
            progress["resumable"] = True
            run.progress_json = progress
            db.commit()
            return run
        expected = int(progress.get("expected_items") or self.frozen_manifest["splits"]["test"]["rows"])
        started = float(progress.get("evidence_started_timestamp") or 0)
        completed_requests = sum(
            path.stat().st_mtime >= started for path in WORKER_EVIDENCE_DIR.glob("*.json")
        )
        progress.update(
            {
                "worker_status": worker_job["status"],
                "completed_requests": min(completed_requests, expected),
                "percent": round(min(completed_requests, expected) / expected * 100, 2),
                "elapsed_seconds": worker_job.get("elapsed_seconds"),
                "returncode": worker_job.get("returncode"),
                "last_polled_at": datetime.now(UTC).isoformat(),
            }
        )
        run.progress_json = progress
        if worker_job["status"] == "completed":
            if run.status in {"completed", "gate_failed"} and run.normalized_result_path and Path(run.normalized_result_path).exists():
                return run
            self._finalize_async(db, run, started)
            return db.get(BenchmarkRun, run_id)
        if worker_job["status"] in {"failed", "timed_out"}:
            run.status = "failed"
            run.error_message = worker_job.get("error") or f"lm-eval worker exited with {worker_job.get('returncode')}"
            progress["resumable"] = True
        db.commit()
        db.refresh(run)
        return run

    def cancel_async(self, db: Session, run_id: int) -> BenchmarkRun:
        run = db.get(BenchmarkRun, run_id)
        if not run or not (run.progress_json or {}).get("job_id"):
            raise HTTPException(status_code=404, detail="Running lm-eval job not found.")
        job_id = run.progress_json["job_id"]
        result = self._worker_json("POST", f"/jobs/{job_id}/cancel", {}, timeout=30)
        progress = dict(run.progress_json or {})
        progress["worker_status"] = result["status"]
        progress["resumable"] = True
        run.progress_json = progress
        run.status = "cancelled"
        db.commit()
        db.refresh(run)
        return run

    def _finalize_async(self, db: Session, run: BenchmarkRun, evidence_started: float) -> None:
        spec = self.build_spec(run)
        output_dir = STORAGE_ROOT / "reports" / "lm_eval" / f"run_{run.id}"
        results_path, samples_path = self.discover_output_files(output_dir, spec.task_name)
        manifest = RunManifest.model_validate(run.run_manifest_json)
        preflight = IntegrityGateReport.model_validate(run.gate_report_json)
        normalized = self.normalize_files(spec, manifest, results_path, samples_path, gate_report=preflight)
        evidence_paths = [
            path for path in WORKER_EVIDENCE_DIR.glob("*.json") if path.stat().st_mtime >= evidence_started
        ]
        alignment = self.align_worker_evidence(normalized, evidence_paths)
        final_gate = self.build_integrity_gate(normalized, alignment)
        unique_checks: list[GateCheck] = []
        seen_checks: set[str] = set()
        for check in preflight.checks + final_gate.checks:
            if check.name not in seen_checks:
                unique_checks.append(check)
                seen_checks.add(check.name)
        final_gate.checks = unique_checks
        final_gate.engineering_failures = sum(check.status == "failed" for check in unique_checks)
        final_gate.status = "failed" if final_gate.engineering_failures else "passed"
        normalized.gate_report = final_gate
        normalized.manifest.completed_at = datetime.now(UTC)
        normalized.manifest.raw_result_path = str(results_path)
        gate_path = output_dir / "integrity_gate.json"
        normalized_path = output_dir / "normalized_result.json"
        gate_path.write_text(final_gate.model_dump_json(indent=2) + "\n", encoding="utf-8")
        normalized_path.write_text(normalized.model_dump_json(indent=2) + "\n", encoding="utf-8")
        from app.services.lm_eval_import_service import replace_run_items

        replace_run_items(db, run, normalized)
        progress = dict(run.progress_json or {})
        progress.update(
            {
                "worker_status": "completed",
                "completed_requests": len(normalized.items),
                "percent": 100.0,
                "completed_at": datetime.now(UTC).isoformat(),
                "resumable": False,
            }
        )
        run.progress_json = progress
        run.raw_result_path = str(results_path)
        run.normalized_result_path = str(normalized_path)
        run.metrics_json = normalized.metrics
        run.gate_report_json = final_gate.model_dump(mode="json")
        run.run_manifest_json = normalized.manifest.model_dump(mode="json")
        run.status = "completed" if final_gate.status == "passed" else "gate_failed"
        run.error_message = None if final_gate.status == "passed" else "Full integrity checks failed."
        db.commit()

    def _worker_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        worker_url = os.getenv("EASYTUNE_LM_EVAL_WORKER_URL", DEFAULT_LM_EVAL_WORKER_URL).rstrip("/")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = Request(
            f"{worker_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"lm-eval worker failed: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"lm-eval worker unavailable at {worker_url}: {exc}") from exc

    def run(self, db: Session, run_id: int) -> BenchmarkRun:
        run = db.get(BenchmarkRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="BenchmarkRun not found.")
        spec = self.build_spec(run)
        output_dir = STORAGE_ROOT / "reports" / "lm_eval" / f"run_{run.id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        _, offline_equivalence = self.prepare_execution_task(output_dir)
        gate = self.preflight(db, run, spec)
        gate.checks.append(
            GateCheck(
                name="offline_task_protocol_equivalence",
                status="passed" if offline_equivalence["protocol_fields_unchanged"] else "failed",
                detail=str(output_dir / "offline_task_equivalence.json"),
                evidence=offline_equivalence,
            )
        )
        command = self.build_command(spec, output_dir)
        manifest = self._build_manifest(run, spec, command, output_dir)
        run.framework = self.framework
        run.task_name = spec.task_name
        run.benchmark_spec_json = spec.model_dump(mode="json")
        run.gate_report_json = gate.model_dump(mode="json")
        run.run_manifest_json = manifest.model_dump(mode="json")
        if gate.status == "failed":
            run.status = "failed"
            run.error_message = "lm-eval preflight failed."
            db.commit()
            return run

        run.status = "running"
        db.commit()
        evidence_before = {path.name for path in WORKER_EVIDENCE_DIR.glob("*.json")}
        log_path = output_dir / "lm_eval.log"
        execution = self.execute_command(command, log_path, timeout=3600)
        if int(execution.get("returncode", -1)) != 0:
            run.status = "failed"
            run.error_message = f"lm-eval exited with code {execution.get('returncode')}; see {log_path}"
            db.commit()
            return run

        results_path, samples_path = self.discover_output_files(output_dir, spec.task_name)
        run.raw_result_path = str(results_path)
        normalized = self.normalize_files(spec, manifest, results_path, samples_path, gate_report=gate)
        evidence_paths = [
            path for path in WORKER_EVIDENCE_DIR.glob("*.json") if path.name not in evidence_before
        ]
        alignment = self.align_worker_evidence(normalized, evidence_paths)
        final_gate = self.build_integrity_gate(normalized, alignment)
        final_gate.checks = gate.checks + final_gate.checks
        final_gate.engineering_failures += gate.engineering_failures
        if any(check.status == "failed" for check in final_gate.checks):
            final_gate.status = "failed"
        normalized.gate_report = final_gate
        gate_path = output_dir / "integrity_gate.json"
        gate_path.write_text(
            json.dumps(final_gate.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        normalized_path = output_dir / "normalized_result.json"
        normalized_path.write_text(
            json.dumps(normalized.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        run.normalized_result_path = str(normalized_path)
        run.metrics_json = normalized.metrics
        run.gate_report_json = normalized.gate_report.model_dump(mode="json")
        run.status = "completed" if normalized.gate_report.status == "passed" else "gate_failed"
        run.error_message = None if normalized.gate_report.status == "passed" else "lm-eval integrity gate failed; see integrity_gate.json"
        run.run_manifest_json = normalized.manifest.model_dump(mode="json")
        db.commit()
        db.refresh(run)
        return run

    def normalize(self, db: Session, run_id: int) -> NormalizedRunResult:
        run = db.get(BenchmarkRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="BenchmarkRun not found.")
        if run.normalized_result_path and Path(run.normalized_result_path).exists():
            return NormalizedRunResult.model_validate_json(
                Path(run.normalized_result_path).read_text(encoding="utf-8")
            )
        if not run.raw_result_path:
            raise HTTPException(status_code=400, detail="lm-eval run has no raw result path.")
        spec = self.build_spec(run)
        result_path = Path(run.raw_result_path)
        _, samples_path = self.discover_output_files(result_path.parent, spec.task_name)
        manifest = RunManifest.model_validate(run.run_manifest_json or self._build_manifest(run, spec, [], result_path.parent).model_dump(mode="json"))
        return self.normalize_files(spec, manifest, result_path, samples_path)

    def discover_output_files(self, output_dir: Path, task_name: str) -> tuple[Path, Path]:
        results = sorted(output_dir.rglob("results_*.json"), key=lambda path: path.stat().st_mtime)
        samples = sorted(
            [
                path
                for path in output_dir.rglob("samples_*.jsonl")
                if task_name in path.name or RUNTIME_TASK_NAME in path.name
            ],
            key=lambda path: path.stat().st_mtime,
        )
        if not results or not samples:
            raise FileNotFoundError(f"lm-eval outputs missing under {output_dir}")
        return results[-1], samples[-1]

    def normalize_files(
        self,
        spec: BenchmarkSpec,
        manifest: RunManifest,
        results_path: Path,
        samples_path: Path,
        gate_report: IntegrityGateReport | None = None,
    ) -> NormalizedRunResult:
        raw_results = json.loads(results_path.read_text(encoding="utf-8"))
        sample_rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in sample_rows:
            grouped.setdefault(int(row["doc_id"]), []).append(row)
        items = [self._normalize_sample(doc_id, rows) for doc_id, rows in sorted(grouped.items())]
        metrics = {
            "framework_results": raw_results.get("results", {}),
            "n_samples": raw_results.get("n-samples", {}),
            "versions": raw_results.get("versions", {}),
            "n_shot": raw_results.get("n-shot", {}),
            "task_hashes": raw_results.get("task_hashes", {}),
            "lm_eval_version": raw_results.get("lm_eval_version"),
        }
        manifest.raw_result_path = str(results_path)
        manifest.completed_at = datetime.now(UTC)
        return NormalizedRunResult(
            spec=spec,
            manifest=manifest,
            gate_report=gate_report or IntegrityGateReport(status="not_run"),
            metrics=metrics,
            items=items,
            raw_result_path=str(results_path),
        )

    def _normalize_sample(self, doc_id: int, rows: list[dict[str, Any]]) -> NormalizedItemResult:
        preferred = next((row for row in rows if row.get("filter") == "strict-match"), rows[0])
        arguments = preferred.get("arguments") or {}
        first_argument = next(iter(arguments.values()), {})
        prompt = first_argument.get("arg_0", "")
        if isinstance(prompt, list):
            if len(prompt) == 1 and isinstance(prompt[0], str):
                try:
                    decoded_prompt = json.loads(prompt[0])
                except json.JSONDecodeError:
                    decoded_prompt = None
                if isinstance(decoded_prompt, list) and all(isinstance(message, dict) for message in decoded_prompt):
                    messages = decoded_prompt
                    prompt_text = json.dumps(decoded_prompt, ensure_ascii=False)
                else:
                    messages = [{"role": "user", "content": prompt[0]}]
                    prompt_text = prompt[0]
            else:
                messages = prompt
                prompt_text = json.dumps(prompt, ensure_ascii=False)
        elif isinstance(prompt, str):
            try:
                decoded_prompt = json.loads(prompt)
            except json.JSONDecodeError:
                decoded_prompt = None
            if isinstance(decoded_prompt, list) and all(isinstance(message, dict) for message in decoded_prompt):
                messages = decoded_prompt
                prompt_text = json.dumps(decoded_prompt, ensure_ascii=False)
            else:
                messages = [{"role": "user", "content": prompt}]
                prompt_text = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]
            prompt_text = str(prompt)
        resps = preferred.get("resps") or []
        raw_response = resps[0][0] if resps and resps[0] else ""
        filtered = preferred.get("filtered_resps") or []
        parsed = filtered[0] if filtered else None
        score = float(preferred.get("exact_match", 0.0))
        filters = {
            str(row.get("filter")): {
                "filtered_resps": row.get("filtered_resps"),
                "metrics": {key: value for key, value in row.items() if key not in self._fixed_sample_keys()},
            }
            for row in rows
        }
        return NormalizedItemResult(
            sample_id=str(doc_id),
            sample_index=doc_id,
            category="gsm8k",
            source_record=preferred.get("doc") or {},
            framework_request={
                "request_type": "generate_until",
                "arguments": arguments,
                "prompt_hash": preferred.get("prompt_hash"),
            },
            model_input=ModelInputEvidence(
                messages=messages,
                rendered_prompt=None,
                rendered_prompt_sha256=preferred.get("prompt_hash") or hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            ),
            raw_output={"response": raw_response, "resps": resps},
            parsed_output=parsed,
            reference=preferred.get("target"),
            score=score,
            passed=bool(score),
            error_type=None if score else "model_answer_error",
            error_reason=None if score else "lm-eval official exact_match failed",
            evidence={
                "filters": filters,
                "doc_hash": preferred.get("doc_hash"),
                "target_hash": preferred.get("target_hash"),
                "worker_evidence_status": "pending_alignment",
            },
        )

    def execute_command(self, command: list[str], log_path: Path, timeout: int) -> dict[str, Any]:
        if len(command) < 5 or command[0] != "wsl.exe" or command[3] != "--":
            raise ValueError("Expected a frozen WSL lm-eval command.")
        linux_command = command[4:]
        worker_url = os.getenv("EASYTUNE_LM_EVAL_WORKER_URL", DEFAULT_LM_EVAL_WORKER_URL).rstrip("/")
        payload = json.dumps(
            {
                "command": linux_command,
                "log_path": windows_to_wsl_path(str(log_path)),
                "timeout": timeout,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{worker_url}/run",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout + 30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"lm-eval worker failed: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"lm-eval worker unavailable at {worker_url}: {exc}") from exc

    def align_worker_evidence(
        self,
        normalized: NormalizedRunResult,
        evidence_paths: list[Path],
    ) -> dict[str, Any]:
        records: list[tuple[Path, dict[str, Any]]] = []
        records_by_key: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = {}
        for path in evidence_paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                pair = (path, record)
                records.append(pair)
                key = (
                    self._canonical_hash(record.get("api_request", {}).get("messages")),
                    str(record.get("model_output", {}).get("content_sha256") or ""),
                )
                records_by_key.setdefault(key, []).append(pair)
            except (OSError, json.JSONDecodeError):
                continue
        used: set[str] = set()
        aligned = 0
        for item in normalized.items:
            expected_messages = item.model_input.messages
            expected_response = str(item.raw_output.get("response") or "")
            key = (
                self._canonical_hash(expected_messages),
                hashlib.sha256(expected_response.encode("utf-8")).hexdigest(),
            )
            candidates = records_by_key.get(key, [])
            match: tuple[Path, dict[str, Any]] | None = None
            while candidates:
                candidate = candidates.pop(0)
                request_id = str(candidate[1].get("request_id") or candidate[0].stem)
                if request_id in used:
                    continue
                if candidate[1].get("api_request", {}).get("messages") != expected_messages:
                    continue
                if str(candidate[1].get("model_output", {}).get("content") or "") != expected_response:
                    continue
                match = candidate
                used.add(request_id)
                break
            if match is None:
                item.evidence["worker_evidence_status"] = "missing"
                continue
            path, record = match
            model_input = dict(record.get("model_input") or {})
            model_input["messages"] = record.get("api_request", {}).get("messages") or expected_messages
            item.model_input = ModelInputEvidence.model_validate(model_input)
            item.raw_output["worker_raw_generated_text"] = record.get("model_output", {}).get("raw_generated_text")
            if normalized.spec.mode == "gate":
                item.raw_output["worker_raw_generated_token_ids"] = record.get("model_output", {}).get("raw_generated_token_ids")
            item.evidence.update(
                {
                    "worker_evidence_status": "aligned",
                    "worker_request_id": record.get("request_id"),
                    "worker_evidence_path": str(path),
                    "framework_messages_sha256": self._canonical_hash(expected_messages),
                    "worker_messages_sha256": self._canonical_hash(record.get("api_request", {}).get("messages")),
                    "framework_raw_response_sha256": hashlib.sha256(expected_response.encode("utf-8")).hexdigest(),
                    "worker_content_sha256": record.get("model_output", {}).get("content_sha256"),
                    "finish_reason": record.get("model_output", {}).get("finish_reason"),
                    "repetition_diagnostic": detect_repetition(expected_response),
                    "worker_recorded_repetition_diagnostic": record.get("model_output", {}).get("repetition_diagnostic"),
                    "response_content_matched": True,
                    "model_input_complete": bool(model_input.get("rendered_prompt") and model_input.get("input_ids")),
                    "token_roundtrip_matched": model_input.get("render_token_roundtrip_matched") is True,
                    "truncated": model_input.get("truncated"),
                    "nested_chat_template_detected": model_input.get("nested_chat_template_detected"),
                    "input_ids_count": len(model_input.get("input_ids") or []),
                    "raw_generated_token_ids_count": len(record.get("model_output", {}).get("raw_generated_token_ids") or []),
                }
            )
            if normalized.spec.mode == "full":
                original_arguments = item.framework_request.get("arguments") or {}
                first_argument = next(iter(original_arguments.values()), {}) if isinstance(original_arguments, dict) else {}
                item.framework_request = {
                    "request_type": "generate_until",
                    "prompt_hash": item.framework_request.get("prompt_hash"),
                    "generation_kwargs": first_argument.get("arg_1") if isinstance(first_argument, dict) else None,
                    "messages_sha256": item.evidence.get("framework_messages_sha256"),
                }
                item.model_input.messages = []
                item.model_input.rendered_prompt = None
                item.model_input.decoded_input_with_special_tokens = None
                item.model_input.input_ids = None
                item.raw_output = {"response": expected_response}
            aligned += 1
        return {
            "evidence_files": len(records),
            "aligned_items": aligned,
            "unmatched_items": len(normalized.items) - aligned,
            "unused_evidence_files": len(records) - len(used),
        }

    def build_integrity_gate(
        self,
        normalized: NormalizedRunResult,
        alignment: dict[str, Any],
    ) -> IntegrityGateReport:
        items = normalized.items
        expected = normalized.spec.limit or int(self.frozen_manifest["splits"]["test"]["rows"])
        checks = [
            GateCheck(
                name="effective_sample_count",
                status="passed" if len(items) == expected else "failed",
                detail=f"expected={expected}, actual={len(items)}",
            ),
            GateCheck(
                name="worker_evidence_alignment",
                status="passed" if alignment["aligned_items"] == len(items) else "failed",
                detail=json.dumps(alignment, sort_keys=True),
                evidence=alignment,
            ),
            GateCheck(
                name="model_input_complete",
                status="passed"
                if all(item.evidence.get("model_input_complete") is True for item in items)
                else "failed",
                detail="Every sample must contain the rendered prompt and complete input token IDs.",
            ),
            GateCheck(
                name="token_roundtrip",
                status="passed"
                if all(item.evidence.get("token_roundtrip_matched") is True for item in items)
                else "failed",
                detail="Every rendered prompt must round-trip to the recorded token IDs.",
            ),
            GateCheck(
                name="no_truncation",
                status="passed" if all(item.evidence.get("truncated") is False for item in items) else "failed",
                detail="No evaluation sample may be silently truncated.",
            ),
            GateCheck(
                name="no_nested_chat_template",
                status="passed"
                if all(item.evidence.get("nested_chat_template_detected") is False for item in items)
                else "failed",
                detail="No request may contain pre-rendered Qwen chat markers.",
            ),
            GateCheck(
                name="response_content_alignment",
                status="passed"
                if all(item.evidence.get("response_content_matched") is True for item in items)
                else "failed",
                detail="Worker content must equal the raw response recorded by lm-eval.",
            ),
            GateCheck(
                name="official_filters_present",
                status="passed"
                if all({"strict-match", "flexible-extract"}.issubset(item.evidence.get("filters", {})) for item in items)
                else "failed",
                detail="Both official GSM8K filters must be preserved for every sample.",
            ),
        ]
        loop_count = sum(bool((item.evidence.get("repetition_diagnostic") or {}).get("detected")) for item in items)
        checks.append(
            GateCheck(
                name="repetition_diagnostics_recorded",
                status="passed"
                if all(item.evidence.get("repetition_diagnostic") is not None for item in items)
                else "failed",
                detail=f"loop_detected_samples={loop_count}; detected loops are model behavior, not infrastructure failure",
                evidence={"loop_detected_samples": loop_count},
            )
        )
        strict_metric = self._find_strict_metric(normalized.metrics.get("framework_results", {}))
        recomputed = sum(item.score for item in items) / len(items) if items else 0.0
        checks.append(
            GateCheck(
                name="metric_parity",
                status="passed" if strict_metric is not None and abs(strict_metric - recomputed) < 1e-12 else "failed",
                detail=f"framework={strict_metric}, recomputed={recomputed}",
            )
        )
        failed = [check for check in checks if check.status == "failed"]
        return IntegrityGateReport(
            status="failed" if failed else "passed",
            checks=checks,
            model_failures=sum(not item.passed for item in items),
            engineering_failures=len(failed),
            infrastructure_errors=[],
        )

    @staticmethod
    def _find_strict_metric(results: dict[str, Any]) -> float | None:
        for task_metrics in results.values():
            if "exact_match,strict-match" in task_metrics:
                return float(task_metrics["exact_match,strict-match"])
        return None

    @staticmethod
    def _canonical_hash(value: Any) -> str:
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_manifest(
        self,
        run: BenchmarkRun,
        spec: BenchmarkSpec,
        command: list[str],
        output_dir: Path,
    ) -> RunManifest:
        return RunManifest(
            framework=self.framework,
            framework_version=spec.framework_version,
            framework_revision=spec.framework_revision,
            task_name=spec.task_name,
            task_config_hash=str(spec.task_config.get("task_config_sha256")),
            benchmark_spec_hash=spec.spec_hash(),
            model_identity={"model_id": DEFAULT_MODEL_ID, "endpoint": DEFAULT_MODEL_ENDPOINT},
            environment={
                "python": "3.12.13",
                "platform": platform.platform(),
                "dataset_revision": spec.dataset_revision,
            },
            command=command,
            output_dir=str(output_dir),
            started_at=datetime.now(UTC),
        )

    @staticmethod
    def _path_check(name: str, path: Path) -> GateCheck:
        return GateCheck(
            name=name,
            status="passed" if path.exists() else "failed",
            detail=str(path),
        )

    @staticmethod
    def _fixed_sample_keys() -> set[str]:
        return {
            "doc_id",
            "doc",
            "target",
            "arguments",
            "resps",
            "filtered_resps",
            "filter",
            "metrics",
            "doc_hash",
            "prompt_hash",
            "target_hash",
        }
