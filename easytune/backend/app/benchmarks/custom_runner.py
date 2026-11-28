from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from app.models import BenchmarkItemResult, BenchmarkRun, Dataset
from app.services.eval_harness_service import execute_benchmark_run


class CustomBenchmarkRunner(BenchmarkRunner):
    framework = "custom"

    def build_spec(self, run: BenchmarkRun) -> BenchmarkSpec:
        config = run.config_snapshot_json or {}
        return BenchmarkSpec(
            framework=self.framework,
            task_name=run.task_name or "custom_dataset",
            task_version="1",
            split="uploaded",
            few_shot=0,
            mode="custom",
            model_id=run.base_model,
            generation=GenerationConfig(
                temperature=float(config.get("temperature", 0.0)),
                max_new_tokens=int(config.get("max_new_tokens", 1024)),
                num_rollouts=int(config.get("num_rollouts", 1)),
                enable_thinking=bool(config.get("enable_thinking", False)),
                load_in_4bit=bool(config.get("load_in_4bit", True)),
                seed=42,
            ),
            task_config={
                "benchmark_dataset_id": run.benchmark_dataset_id,
                "reference_train_dataset_id": run.reference_train_dataset_id,
                "decontamination_enabled": bool(run.decontamination_enabled),
                "dynamic_perturbation": bool(run.use_dynamic_perturbation),
                "judge_mode": run.judge_mode,
            },
            runner_config={
                "runner": "CustomBenchmarkRunner",
                "template": config.get("template", "qwen3_5_nothink"),
                "engine": config.get("engine"),
            },
        )

    def preflight(self, db: Session, run: BenchmarkRun, spec: BenchmarkSpec) -> IntegrityGateReport:
        checks: list[GateCheck] = []
        dataset = db.get(Dataset, run.benchmark_dataset_id)
        dataset_path = Path(dataset.converted_file_path or dataset.original_file_path) if dataset else None
        checks.append(
            GateCheck(
                name="benchmark_dataset_available",
                status="passed" if dataset_path and dataset_path.exists() else "failed",
                detail=str(dataset_path) if dataset_path else "Benchmark dataset is missing.",
            )
        )
        checks.append(
            GateCheck(
                name="model_identity_present",
                status="passed" if spec.model_id.strip() else "failed",
                detail=spec.model_id or "Model identity is empty.",
            )
        )
        checks.append(
            GateCheck(
                name="generation_config_valid",
                status="passed" if spec.generation.max_new_tokens > 0 and spec.generation.num_rollouts > 0 else "failed",
                detail=f"max_new_tokens={spec.generation.max_new_tokens}, num_rollouts={spec.generation.num_rollouts}",
            )
        )
        failed = [check for check in checks if check.status == "failed"]
        return IntegrityGateReport(
            status="failed" if failed else "passed",
            checks=checks,
            engineering_failures=len(failed),
        )

    def run(self, db: Session, run_id: int) -> BenchmarkRun:
        run = db.get(BenchmarkRun, run_id)
        if not run:
            raise ValueError(f"BenchmarkRun not found: {run_id}")
        spec = self.build_spec(run)
        gate = self.preflight(db, run, spec)
        run.framework = self.framework
        run.task_name = spec.task_name
        run.benchmark_spec_json = spec.model_dump(mode="json")
        run.gate_report_json = gate.model_dump(mode="json")
        run.run_manifest_json = self._build_manifest(run, spec).model_dump(mode="json")
        db.commit()
        if gate.status == "failed":
            run.status = "failed"
            run.error_message = "Custom benchmark preflight failed."
            db.commit()
            return run

        completed = execute_benchmark_run(db, run.id)
        normalized = self.normalize(db, run.id)
        result_path = STORAGE_ROOT / "reports" / f"benchmark_run_{run.id}_normalized.json"
        result_path.write_text(
            json.dumps(normalized.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        completed.normalized_result_path = str(result_path)
        completed.run_manifest_json = normalized.manifest.model_dump(mode="json")
        db.commit()
        db.refresh(completed)
        return completed

    def normalize(self, db: Session, run_id: int) -> NormalizedRunResult:
        run = db.get(BenchmarkRun, run_id)
        if not run:
            raise ValueError(f"BenchmarkRun not found: {run_id}")
        spec = BenchmarkSpec.model_validate(run.benchmark_spec_json or self.build_spec(run).model_dump(mode="json"))
        manifest = self._build_manifest(run, spec)
        gate = IntegrityGateReport.model_validate(run.gate_report_json or {})
        native_items = (
            db.query(BenchmarkItemResult)
            .filter(BenchmarkItemResult.benchmark_run_id == run.id)
            .order_by(BenchmarkItemResult.sample_index.asc())
            .all()
        )
        items = [self._normalize_item(item) for item in native_items]
        return NormalizedRunResult(
            spec=spec,
            manifest=manifest,
            gate_report=gate,
            metrics=run.metrics_json or {},
            items=items,
            raw_result_path=run.raw_result_path,
        )

    def _build_manifest(self, run: BenchmarkRun, spec: BenchmarkSpec) -> RunManifest:
        task_config = json.dumps(spec.task_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return RunManifest(
            framework=self.framework,
            framework_version="internal-v1",
            task_name=spec.task_name,
            task_config_hash=hashlib.sha256(task_config.encode("utf-8")).hexdigest(),
            benchmark_spec_hash=spec.spec_hash(),
            model_identity={
                "base_model": run.base_model,
                "adapter_path": run.adapter_path,
                "model_version_id": run.model_version_id,
            },
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "engine": (run.config_snapshot_json or {}).get("engine"),
            },
            command=[],
            output_dir=str(STORAGE_ROOT / "reports"),
            raw_result_path=run.raw_result_path,
            started_at=run.created_at,
            completed_at=datetime.now(UTC) if run.status == "completed" else None,
        )

    @staticmethod
    def _normalize_item(item: BenchmarkItemResult) -> NormalizedItemResult:
        actual_prompt = item.perturbed_prompt or item.prompt
        return NormalizedItemResult(
            sample_id=item.framework_sample_id or str(item.sample_index),
            sample_index=item.sample_index,
            category=item.category,
            source_record=item.source_record_json or {
                "prompt": item.prompt,
                "expected_answer": item.expected_answer,
            },
            framework_request=item.request_json or {
                "request_type": "custom_generate",
                "prompt": actual_prompt,
                "verifier_type": item.verifier_type,
            },
            model_input=ModelInputEvidence.model_validate(
                item.model_input_json
                or {
                    "messages": [{"role": "user", "content": actual_prompt}],
                }
            ),
            raw_output=item.raw_output_json or {"rollouts": item.rollouts_json},
            parsed_output=item.parsed_output_json or item.judge_detail_json,
            reference=item.expected_answer,
            score=item.score,
            passed=bool(item.is_passed),
            error_type=item.error_tag,
            error_reason=item.error_reason,
            evidence=item.evidence_json
            or {
                "contamination": item.contamination_detail_json,
                "capture_level": "application_prompt_and_decoded_output",
            },
        )
