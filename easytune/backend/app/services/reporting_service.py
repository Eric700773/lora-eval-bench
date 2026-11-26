import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.database import STORAGE_ROOT
from app.models import BenchmarkItemResult, BenchmarkRun, FlywheelIteration, ModelVersion, TrainingJob


def generate_flywheel_report(
    db: Session,
    iteration_id: int,
    dynamic_baseline_run_id: int,
    dynamic_post_run_id: int,
) -> dict[str, Any]:
    iteration = db.get(FlywheelIteration, iteration_id)
    if not iteration or not iteration.post_eval_run_id or not iteration.training_job_id:
        raise HTTPException(status_code=400, detail="Completed Flywheel iteration is required.")
    original_base = db.get(BenchmarkRun, iteration.baseline_run_id)
    original_post = db.get(BenchmarkRun, iteration.post_eval_run_id)
    dynamic_base = db.get(BenchmarkRun, dynamic_baseline_run_id)
    dynamic_post = db.get(BenchmarkRun, dynamic_post_run_id)
    training_job = db.get(TrainingJob, iteration.training_job_id)
    model_version = (
        db.query(ModelVersion)
        .filter(ModelVersion.training_job_id == iteration.training_job_id)
        .order_by(ModelVersion.id.desc())
        .first()
    )
    runs = [original_base, original_post, dynamic_base, dynamic_post]
    if any(run is None or run.status != "completed" for run in runs):
        raise HTTPException(status_code=400, detail="All four Benchmark runs must be completed.")
    if not training_job or not model_version or model_version.status != "ready":
        raise HTTPException(status_code=400, detail="Validated training job and ModelVersion are required.")

    training_report_path = Path(training_job.output_dir or "") / "training-report.json"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8")) if training_report_path.exists() else {}
    original_comparison = _compare_runs(db, original_base, original_post)
    dynamic_comparison = _compare_runs(db, dynamic_base, dynamic_post)
    report = {
        "schema_version": 1,
        "generated_at": datetime.utcnow().isoformat(),
        "iteration_id": iteration.id,
        "model": {
            "base_model": original_base.base_model,
            "model_version_id": model_version.id,
            "adapter_path": model_version.adapter_path,
            "adapter_status": model_version.status,
        },
        "data": {
            "benchmark_dataset_id": original_base.benchmark_dataset_id,
            "reference_train_dataset_id": original_base.reference_train_dataset_id,
            "synthesized_dataset_id": iteration.synthesized_dataset_id,
            "attribution": iteration.attribution_summary_json,
        },
        "training": training_report,
        "original": original_comparison,
        "dynamic": dynamic_comparison,
        "decontamination": {
            "baseline_contaminated_count": original_base.contaminated_count,
            "post_contaminated_count": original_post.contaminated_count,
            "clean_subset_matched": original_base.reference_train_dataset_id == original_post.reference_train_dataset_id,
        },
        "configuration": {
            "baseline_config_hash": original_base.config_hash,
            "post_config_hash": original_post.config_hash,
            "evaluation_parameters_matched": _evaluation_parameters_match(original_base, original_post),
        },
        "claim_boundary": (
            "This is a six-sample, one-epoch end-to-end engineering validation. "
            "The score deltas are observed results, not evidence of statistically significant general capability improvement."
        ),
    }
    reports_dir = STORAGE_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"flywheel_iteration_{iteration.id}.json"
    markdown_path = reports_dir / f"flywheel_iteration_{iteration.id}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_to_markdown(report), encoding="utf-8")
    return {"report": report, "json_path": str(json_path), "markdown_path": str(markdown_path)}


def _compare_runs(db: Session, baseline: BenchmarkRun, candidate: BenchmarkRun) -> dict[str, Any]:
    base_metrics = baseline.metrics_json or {}
    cand_metrics = candidate.metrics_json or {}
    base_clean = base_metrics.get("clean_decontaminated", {})
    cand_clean = cand_metrics.get("clean_decontaminated", {})
    base_raw = base_metrics.get("raw_unfiltered", {})
    cand_raw = cand_metrics.get("raw_unfiltered", {})
    base_categories = base_metrics.get("by_category_clean", {})
    cand_categories = cand_metrics.get("by_category_clean", {})
    category_deltas = {
        category: round(
            float(cand_categories.get(category, {}).get("pass_at_1", 0.0))
            - float(base_categories.get(category, {}).get("pass_at_1", 0.0)),
            4,
        )
        for category in sorted(set(base_categories) | set(cand_categories))
    }
    base_items = {
        item.sample_index: item
        for item in db.query(BenchmarkItemResult).filter(BenchmarkItemResult.benchmark_run_id == baseline.id).all()
    }
    cand_items = {
        item.sample_index: item
        for item in db.query(BenchmarkItemResult).filter(BenchmarkItemResult.benchmark_run_id == candidate.id).all()
    }
    paired = {"improved": 0, "regressed": 0, "unchanged_pass": 0, "unchanged_fail": 0}
    for index in sorted(set(base_items) & set(cand_items)):
        before, after = base_items[index], cand_items[index]
        if before.is_contaminated or after.is_contaminated:
            continue
        if not before.is_passed and after.is_passed:
            paired["improved"] += 1
        elif before.is_passed and not after.is_passed:
            paired["regressed"] += 1
        elif before.is_passed:
            paired["unchanged_pass"] += 1
        else:
            paired["unchanged_fail"] += 1
    return {
        "baseline_run_id": baseline.id,
        "post_run_id": candidate.id,
        "variant_mode": cand_metrics.get("variant_mode", "original"),
        "clean": {
            "baseline_pass_at_1": base_clean.get("pass_at_1"),
            "post_pass_at_1": cand_clean.get("pass_at_1"),
            "delta": round(float(cand_clean.get("pass_at_1", 0.0)) - float(base_clean.get("pass_at_1", 0.0)), 4),
            "baseline_ci_95": base_clean.get("ci_95"),
            "post_ci_95": cand_clean.get("ci_95"),
        },
        "raw": {
            "baseline_pass_at_1": base_raw.get("pass_at_1"),
            "post_pass_at_1": cand_raw.get("pass_at_1"),
            "delta": round(float(cand_raw.get("pass_at_1", 0.0)) - float(base_raw.get("pass_at_1", 0.0)), 4),
        },
        "category_clean_deltas": category_deltas,
        "paired_clean_transitions": paired,
    }


def _evaluation_parameters_match(baseline: BenchmarkRun, candidate: BenchmarkRun) -> bool:
    keys = ("template", "temperature", "max_new_tokens", "num_rollouts", "enable_thinking", "load_in_4bit")
    return all((baseline.config_snapshot_json or {}).get(key) == (candidate.config_snapshot_json or {}).get(key) for key in keys)


def _to_markdown(report: dict[str, Any]) -> str:
    original = report["original"]
    dynamic = report["dynamic"]
    training = report["training"]
    rows = [
        "# EasyTune Flywheel Experiment Report",
        "",
        f"- Iteration: {report['iteration_id']}",
        f"- Base model: `{report['model']['base_model']}`",
        f"- ModelVersion: {report['model']['model_version_id']}",
        f"- Training samples: {training.get('samples')}",
        f"- Optimizer steps: {training.get('optimizer_steps')}",
        f"- Loss: {training.get('loss_first'):.4f} -> {training.get('loss_last'):.4f}",
        f"- Peak GPU memory: {training.get('peak_gpu_memory_bytes', 0) / (1024 ** 3):.2f} GiB",
        "",
        "| Evaluation | Baseline Clean | Post Clean | Delta | Baseline 95% CI | Post 95% CI |",
        "|---|---:|---:|---:|---|---|",
        f"| Original | {original['clean']['baseline_pass_at_1']:.4f} | {original['clean']['post_pass_at_1']:.4f} | {original['clean']['delta']:+.4f} | {original['clean']['baseline_ci_95']} | {original['clean']['post_ci_95']} |",
        f"| Dynamic | {dynamic['clean']['baseline_pass_at_1']:.4f} | {dynamic['clean']['post_pass_at_1']:.4f} | {dynamic['clean']['delta']:+.4f} | {dynamic['clean']['baseline_ci_95']} | {dynamic['clean']['post_ci_95']} |",
        "",
        "## Category Deltas",
        "",
        f"- Original: `{json.dumps(original['category_clean_deltas'], ensure_ascii=False)}`",
        f"- Dynamic: `{json.dumps(dynamic['category_clean_deltas'], ensure_ascii=False)}`",
        "",
        "## Paired Clean Transitions",
        "",
        f"- Original: `{json.dumps(original['paired_clean_transitions'], ensure_ascii=False)}`",
        f"- Dynamic: `{json.dumps(dynamic['paired_clean_transitions'], ensure_ascii=False)}`",
        "",
        "## Reproducibility",
        "",
        f"- Evaluation parameters matched: `{report['configuration']['evaluation_parameters_matched']}`",
        f"- Clean subset matched: `{report['decontamination']['clean_subset_matched']}`",
        f"- Contaminated samples: {report['decontamination']['baseline_contaminated_count']}",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
    ]
    return "\n".join(rows)
