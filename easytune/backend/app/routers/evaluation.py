import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.benchmarks.lm_eval_runner import LmEvalRunner, WORKER_EVIDENCE_DIR
from app.benchmarks.registry import get_benchmark_runner, list_benchmark_runners
from app.database import get_db
from app.models import BenchmarkItemResult, BenchmarkRun, Dataset, ModelVersion
from app.services.eval_harness_service import build_config_snapshot, recompute_benchmark_metrics
from app.services.lm_eval_import_service import import_lm_eval_gate

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


class BenchmarkRunCreate(BaseModel):
    name: str
    framework: str = "custom"
    task_name: str = "custom_30"
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


class LmEvalGateImport(BaseModel):
    run_directory: str = "gsm8k_gate_10_final"
    name: str = "GSM8K lm-eval Gate (limit=10)"


@router.post("/runs/import-lm-eval-gate")
def import_gate(payload: LmEvalGateImport, db: Session = Depends(get_db)):
    return import_lm_eval_gate(db, payload.run_directory, payload.name)


class LmEvalFullRunCreate(BaseModel):
    name: str = "GSM8K lm-eval Full (1319 samples)"
    base_model: str = "local-qwen3.5-2b"
    max_new_tokens: int = 512


@router.post("/runs/lm-eval/full")
def start_lm_eval_full(payload: LmEvalFullRunCreate, db: Session = Depends(get_db)):
    active = (
        db.query(BenchmarkRun)
        .filter(BenchmarkRun.framework == "lm_eval", BenchmarkRun.status == "running")
        .first()
    )
    if active:
        raise HTTPException(status_code=409, detail=f"lm-eval Run #{active.id} is already running.")
    dataset = next(
        (
            item
            for item in db.query(Dataset).order_by(Dataset.id.desc()).all()
            if (item.report_json or {}).get("role") == "official_benchmark"
            and (item.report_json or {}).get("task") == "gsm8k"
        ),
        None,
    )
    if not dataset:
        raise HTTPException(status_code=400, detail="Import the frozen GSM8K Gate dataset before starting Full mode.")
    run = BenchmarkRun(
        name=payload.name,
        framework="lm_eval",
        task_name="gsm8k",
        model_version_id=None,
        base_model=payload.base_model,
        adapter_path=None,
        benchmark_dataset_id=dataset.id,
        reference_train_dataset_id=None,
        decontamination_enabled=0,
        contaminated_count=0,
        use_dynamic_perturbation=0,
        config_snapshot_json={"mode": "full", "max_new_tokens": payload.max_new_tokens},
        config_hash="pending",
        judge_mode="lm_eval_official",
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return LmEvalRunner().start_async(db, run.id)


@router.post("/runs/{run_id}/lm-eval/resume")
def resume_lm_eval_full(run_id: int, db: Session = Depends(get_db)):
    return LmEvalRunner().start_async(db, run_id, resume=True)


@router.post("/runs/{run_id}/lm-eval/cancel")
def cancel_lm_eval_full(run_id: int, db: Session = Depends(get_db)):
    return LmEvalRunner().cancel_async(db, run_id)


@router.get("/runs/{run_id}/lm-eval/status")
def refresh_lm_eval_full(run_id: int, db: Session = Depends(get_db)):
    return LmEvalRunner().refresh_async(db, run_id)


@router.post("/runs")
def create_and_run_benchmark(payload: BenchmarkRunCreate, db: Session = Depends(get_db)):
    runner = get_benchmark_runner(payload.framework)
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
        framework=runner.framework,
        task_name=payload.task_name,
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
    return runner.run(db, run.id)


@router.get("/frameworks")
def list_frameworks():
    return {"available": list_benchmark_runners(), "planned": ["lm_eval", "opencompass", "evalplus", "interactive"]}


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


@router.get("/runs/{run_id}/normalized")
def get_normalized_run(run_id: int, db: Session = Depends(get_db)):
    run = db.get(BenchmarkRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="BenchmarkRun not found.")
    return get_benchmark_runner(run.framework or "custom").normalize(db, run.id)


@router.get("/runs/{run_id}/items/page")
def page_benchmark_items(
    run_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    only_badcases: bool = False,
    db: Session = Depends(get_db),
):
    query = db.query(BenchmarkItemResult).filter(BenchmarkItemResult.benchmark_run_id == run_id)
    if only_badcases:
        query = query.filter(BenchmarkItemResult.is_passed == 0)
    total = query.count()
    items = query.order_by(BenchmarkItemResult.sample_index.asc()).offset(offset).limit(limit).all()
    return {"total": total, "offset": offset, "limit": limit, "items": items}


@router.get("/runs/{run_id}/items/{item_id}/evidence")
def get_benchmark_item_evidence(run_id: int, item_id: int, db: Session = Depends(get_db)):
    item = db.get(BenchmarkItemResult, item_id)
    if not item or item.benchmark_run_id != run_id:
        raise HTTPException(status_code=404, detail="Benchmark item not found.")
    evidence_path_value = (item.evidence_json or {}).get("worker_evidence_path")
    if not evidence_path_value:
        raise HTTPException(status_code=404, detail="Worker evidence path is unavailable.")
    evidence_path = Path(evidence_path_value).resolve()
    evidence_root = WORKER_EVIDENCE_DIR.resolve()
    if evidence_root not in evidence_path.parents or not evidence_path.exists():
        raise HTTPException(status_code=404, detail="Worker evidence file is unavailable.")
    return json.loads(evidence_path.read_text(encoding="utf-8"))


@router.get("/runs/{run_id}/items")
def list_benchmark_items(run_id: int, only_badcases: bool = False, db: Session = Depends(get_db)):
    q = db.query(BenchmarkItemResult).filter(BenchmarkItemResult.benchmark_run_id == run_id)
    if only_badcases:
        q = q.filter(BenchmarkItemResult.is_passed == 0)
    return q.order_by(BenchmarkItemResult.sample_index.asc()).all()