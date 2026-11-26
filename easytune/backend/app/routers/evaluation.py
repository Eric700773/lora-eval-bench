from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import BenchmarkItemResult, BenchmarkRun, ModelVersion
from app.services.eval_harness_service import build_config_snapshot, execute_benchmark_run, recompute_benchmark_metrics

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


class BenchmarkRunCreate(BaseModel):
    name: str
    model_version_id: int | None = None
    base_model: str = r"D:\interview\local-llm-validation\models\qwen3.5-2b-bf16"
    benchmark_dataset_id: int
    reference_train_dataset_id: int | None = None
    decontamination_enabled: bool = True
    use_dynamic_perturbation: bool = False
    judge_mode: str = "hybrid"
    template: str = "qwen3_5_nothink"
    temperature: float = 0.0
    max_new_tokens: int = 1024
    num_rollouts: int = 1
    enable_thinking: bool = False
    load_in_4bit: bool = True


@router.post("/runs")
def create_and_run_benchmark(payload: BenchmarkRunCreate, db: Session = Depends(get_db)):
    base_model = payload.base_model
    adapter_path = None
    if payload.model_version_id:
        mv = db.get(ModelVersion, payload.model_version_id)
        if not mv:
            raise HTTPException(status_code=404, detail="ModelVersion not found.")
        base_model = mv.base_model
        adapter_path = mv.adapter_path

    snapshot, cfg_hash = build_config_snapshot(
        base_model=base_model,
        adapter_path=adapter_path,
        template=payload.template,
        temperature=payload.temperature,
        max_new_tokens=payload.max_new_tokens,
        num_rollouts=payload.num_rollouts,
        enable_thinking=payload.enable_thinking,
        load_in_4bit=payload.load_in_4bit,
    )
    run = BenchmarkRun(
        name=payload.name,
        model_version_id=payload.model_version_id,
        base_model=base_model,
        adapter_path=adapter_path,
        benchmark_dataset_id=payload.benchmark_dataset_id,
        reference_train_dataset_id=payload.reference_train_dataset_id,
        decontamination_enabled=1 if payload.decontamination_enabled else 0,
        use_dynamic_perturbation=1 if payload.use_dynamic_perturbation else 0,
        config_snapshot_json=snapshot,
        config_hash=cfg_hash,
        judge_mode=payload.judge_mode,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return execute_benchmark_run(db, run.id)


@router.get("/runs")
def list_benchmark_runs(db: Session = Depends(get_db)):
    return db.query(BenchmarkRun).order_by(BenchmarkRun.id.desc()).all()


@router.post("/runs/{run_id}/recompute-metrics")
def recompute_run_metrics(run_id: int, db: Session = Depends(get_db)):
    return recompute_benchmark_metrics(db, run_id)


@router.get("/runs/compare")
def compare_benchmark_runs(
    baseline_run_id: int,
    candidate_run_id: int,
    db: Session = Depends(get_db),
):
    baseline = db.get(BenchmarkRun, baseline_run_id)
    candidate = db.get(BenchmarkRun, candidate_run_id)
    if not baseline or not candidate:
        raise HTTPException(status_code=404, detail="BenchmarkRun not found.")
    if baseline.status != "completed" or candidate.status != "completed":
        raise HTTPException(status_code=400, detail="Both BenchmarkRuns must be completed.")

    baseline_metrics = baseline.metrics_json or {}
    candidate_metrics = candidate.metrics_json or {}
    baseline_clean = baseline_metrics.get("clean_decontaminated", {})
    candidate_clean = candidate_metrics.get("clean_decontaminated", {})
    baseline_raw = baseline_metrics.get("raw_unfiltered", {})
    candidate_raw = candidate_metrics.get("raw_unfiltered", {})
    baseline_categories = baseline_metrics.get("by_category_clean", {})
    candidate_categories = candidate_metrics.get("by_category_clean", {})

    category_deltas = {
        category: round(
            float(candidate_categories.get(category, {}).get("pass_at_1", 0.0))
            - float(baseline_categories.get(category, {}).get("pass_at_1", 0.0)),
            4,
        )
        for category in sorted(set(baseline_categories) | set(candidate_categories))
    }
    baseline_items = {
        item.sample_index: item
        for item in db.query(BenchmarkItemResult)
        .filter(BenchmarkItemResult.benchmark_run_id == baseline.id)
        .all()
    }
    candidate_items = {
        item.sample_index: item
        for item in db.query(BenchmarkItemResult)
        .filter(BenchmarkItemResult.benchmark_run_id == candidate.id)
        .all()
    }
    paired = {"improved": 0, "regressed": 0, "unchanged_pass": 0, "unchanged_fail": 0}
    for sample_index in sorted(set(baseline_items) & set(candidate_items)):
        base_item = baseline_items[sample_index]
        cand_item = candidate_items[sample_index]
        if base_item.is_contaminated or cand_item.is_contaminated:
            continue
        before, after = bool(base_item.is_passed), bool(cand_item.is_passed)
        if not before and after:
            paired["improved"] += 1
        elif before and not after:
            paired["regressed"] += 1
        elif before:
            paired["unchanged_pass"] += 1
        else:
            paired["unchanged_fail"] += 1

    return {
        "baseline_run_id": baseline.id,
        "candidate_run_id": candidate.id,
        "baseline_variant_mode": baseline_metrics.get("variant_mode", "original"),
        "candidate_variant_mode": candidate_metrics.get("variant_mode", "original"),
        "clean_pass_at_1": {
            "baseline": baseline_clean.get("pass_at_1", 0.0),
            "candidate": candidate_clean.get("pass_at_1", 0.0),
            "delta": round(float(candidate_clean.get("pass_at_1", 0.0)) - float(baseline_clean.get("pass_at_1", 0.0)), 4),
        },
        "raw_pass_at_1": {
            "baseline": baseline_raw.get("pass_at_1", 0.0),
            "candidate": candidate_raw.get("pass_at_1", 0.0),
            "delta": round(float(candidate_raw.get("pass_at_1", 0.0)) - float(baseline_raw.get("pass_at_1", 0.0)), 4),
        },
        "category_clean_deltas": category_deltas,
        "paired_clean_transitions": paired,
    }


@router.get("/runs/{run_id}/items")
def list_benchmark_items(run_id: int, only_badcases: bool = False, db: Session = Depends(get_db)):
    q = db.query(BenchmarkItemResult).filter(BenchmarkItemResult.benchmark_run_id == run_id)
    if only_badcases:
        q = q.filter(BenchmarkItemResult.is_passed == 0)
    return q.order_by(BenchmarkItemResult.sample_index.asc()).all()