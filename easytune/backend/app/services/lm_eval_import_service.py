from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.benchmarks.contracts import NormalizedItemResult, NormalizedRunResult
from app.benchmarks.lm_eval_runner import DEFAULT_DATASET_SNAPSHOT
from app.database import STORAGE_ROOT
from app.models import BenchmarkItemResult, BenchmarkRun, Dataset


LM_EVAL_REPORT_ROOT = (STORAGE_ROOT / "reports" / "lm_eval").resolve()


def replace_run_items(db: Session, run: BenchmarkRun, normalized: NormalizedRunResult) -> None:
    db.query(BenchmarkItemResult).filter(BenchmarkItemResult.benchmark_run_id == run.id).delete()
    for item in normalized.items:
        db.add(_to_item_record(run.id, item))
    db.flush()


def import_lm_eval_gate(
    db: Session,
    run_directory: str,
    name: str = "GSM8K lm-eval Gate (limit=10)",
) -> BenchmarkRun:
    run_dir = (LM_EVAL_REPORT_ROOT / run_directory).resolve()
    if LM_EVAL_REPORT_ROOT not in run_dir.parents:
        raise HTTPException(status_code=400, detail="Invalid lm-eval run directory.")
    normalized_path = run_dir / "normalized_result.json"
    gate_path = run_dir / "integrity_gate.json"
    if not normalized_path.exists() or not gate_path.exists():
        raise HTTPException(status_code=404, detail="Normalized result or integrity gate file is missing.")

    existing = (
        db.query(BenchmarkRun)
        .filter(BenchmarkRun.normalized_result_path == str(normalized_path))
        .first()
    )
    if existing:
        return existing

    normalized = NormalizedRunResult.model_validate_json(normalized_path.read_text(encoding="utf-8"))
    if normalized.spec.framework != "lm_eval" or normalized.spec.task_name != "gsm8k":
        raise HTTPException(status_code=400, detail="Only the frozen lm_eval/gsm8k Gate can be imported.")
    if normalized.gate_report.status != "passed":
        raise HTTPException(status_code=400, detail="Only a passed integrity Gate can be imported.")
    if len(normalized.items) != normalized.spec.limit:
        raise HTTPException(status_code=400, detail="Normalized item count does not match Gate limit.")

    dataset = _get_or_create_gsm8k_dataset(db, normalized)
    config_snapshot = {
        "framework": normalized.spec.framework,
        "framework_version": normalized.spec.framework_version,
        "framework_revision": normalized.spec.framework_revision,
        "task": normalized.spec.task_name,
        "task_version": normalized.spec.task_version,
        "dataset_revision": normalized.spec.dataset_revision,
        "split": normalized.spec.split,
        "few_shot": normalized.spec.few_shot,
        "limit": normalized.spec.limit,
        "mode": normalized.spec.mode,
        "generation": normalized.spec.generation.model_dump(mode="json"),
        "runner_config": normalized.spec.runner_config,
    }
    run = BenchmarkRun(
        name=name,
        framework="lm_eval",
        task_name="gsm8k",
        benchmark_spec_json=normalized.spec.model_dump(mode="json"),
        run_manifest_json=normalized.manifest.model_dump(mode="json"),
        gate_report_json=normalized.gate_report.model_dump(mode="json"),
        raw_result_path=normalized.raw_result_path,
        normalized_result_path=str(normalized_path),
        model_version_id=None,
        base_model=normalized.spec.model_id,
        adapter_path=None,
        benchmark_dataset_id=dataset.id,
        reference_train_dataset_id=None,
        decontamination_enabled=0,
        contaminated_count=0,
        use_dynamic_perturbation=0,
        config_snapshot_json=config_snapshot,
        config_hash=normalized.spec.spec_hash(),
        judge_mode="lm_eval_official",
        metrics_json=normalized.metrics,
        status="completed",
        error_message=None,
    )
    db.add(run)
    db.flush()
    replace_run_items(db, run, normalized)
    db.commit()
    db.refresh(run)
    return run


def _get_or_create_gsm8k_dataset(db: Session, normalized: NormalizedRunResult) -> Dataset:
    snapshot = Path(DEFAULT_DATASET_SNAPSHOT).resolve()
    test_path = snapshot / "main" / "test-00000-of-00001.parquet"
    if not test_path.exists():
        raise HTTPException(status_code=404, detail=f"Frozen GSM8K test parquet is missing: {test_path}")
    existing = db.query(Dataset).filter(Dataset.original_file_path == str(test_path)).first()
    if existing:
        return existing
    dataset = Dataset(
        name=f"[Official] openai/gsm8k@{normalized.spec.dataset_revision[:12]}",
        original_file_path=str(test_path),
        converted_file_path=None,
        file_type="parquet",
        file_size=test_path.stat().st_size,
        sample_count=int(normalized.manifest.environment.get("test_rows", 1319)),
        formatting="lm_eval",
        quality_score=100.0,
        report_json={
            "framework": "lm_eval",
            "task": "gsm8k",
            "dataset_revision": normalized.spec.dataset_revision,
            "split": normalized.spec.split,
            "fingerprint": normalized.spec.task_config.get("test_fingerprint"),
            "source": "openai/gsm8k",
            "role": "official_benchmark",
        },
        status="frozen",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def _to_item_record(run_id: int, item: NormalizedItemResult) -> BenchmarkItemResult:
    question = str(item.source_record.get("question") or item.framework_request.get("prompt") or "")
    raw_response = str(item.raw_output.get("response") or "")
    parsed = item.parsed_output
    return BenchmarkItemResult(
        benchmark_run_id=run_id,
        sample_index=item.sample_index,
        framework_sample_id=item.sample_id,
        source_record_json=item.source_record,
        request_json=item.framework_request,
        model_input_json=item.model_input.model_dump(mode="json"),
        raw_output_json=item.raw_output,
        parsed_output_json={"value": parsed, "filters": item.evidence.get("filters")},
        evidence_json=item.evidence,
        category=item.category,
        prompt=question,
        perturbed_prompt=None,
        expected_answer=json.dumps(item.reference, ensure_ascii=False) if not isinstance(item.reference, str) else item.reference,
        rollouts_json=[raw_response],
        is_contaminated=0,
        contamination_detail_json={"method": "not_applicable", "reason": "official lm-eval Gate"},
        verifier_type="lm_eval_exact_match",
        score=item.score,
        is_passed=1 if item.passed else 0,
        judge_detail_json={
            "framework": "lm_eval",
            "parsed_output": parsed,
            "reference": item.reference,
            "filters": item.evidence.get("filters"),
        },
        error_tag=item.error_type,
        error_reason=item.error_reason,
    )
