from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import FlywheelIteration
from app.services.reporting_service import generate_flywheel_report
from app.services.flywheel_service import (
    analyze_badcases_and_synthesize,
    calibrate_decontamination,
    list_external_synthesis_models,
    run_post_training_regression_eval,
    synthesis_configuration,
    test_external_synthesis,
    trigger_flywheel_qlora_training,
)

router = APIRouter(prefix="/api/flywheel", tags=["flywheel"])


class FlywheelAnalyzeRequest(BaseModel):
    baseline_run_id: int
    iteration_name: str
    augment_per_badcase: int = 2
    general_replay_dataset_id: int | None = None
    replay_ratio: float = 0.3


class FlywheelTrainRequest(BaseModel):
    budget_level: str = "low"


class FlywheelRegressionRequest(BaseModel):
    model_version_id: int


@router.get("/decontamination/calibration")
def decontamination_calibration(
    baseline_run_id: int,
    candidate_dataset_id: int | None = None,
    db: Session = Depends(get_db),
):
    return calibrate_decontamination(db, baseline_run_id, candidate_dataset_id)


@router.get("/synthesis/config")
def synthesis_config():
    return synthesis_configuration()


@router.get("/synthesis/models")
def synthesis_models():
    return list_external_synthesis_models()


@router.post("/synthesis/test")
def synthesis_test():
    return test_external_synthesis()


@router.post("/analyze-and-synthesize")
def api_analyze_and_synthesize(payload: FlywheelAnalyzeRequest, db: Session = Depends(get_db)):
    return analyze_badcases_and_synthesize(
        db=db,
        baseline_run_id=payload.baseline_run_id,
        iteration_name=payload.iteration_name,
        augment_per_badcase=payload.augment_per_badcase,
        general_replay_dataset_id=payload.general_replay_dataset_id,
        replay_ratio=payload.replay_ratio,
    )


@router.post("/{iteration_id}/train")
def api_trigger_training(iteration_id: int, payload: FlywheelTrainRequest, db: Session = Depends(get_db)):
    return trigger_flywheel_qlora_training(db=db, iteration_id=iteration_id, budget_level=payload.budget_level)


@router.post("/{iteration_id}/regression-eval")
def api_regression_eval(iteration_id: int, payload: FlywheelRegressionRequest, db: Session = Depends(get_db)):
    return run_post_training_regression_eval(
        db=db,
        iteration_id=iteration_id,
        model_version_id=payload.model_version_id,
    )


@router.post("/{iteration_id}/report")
def create_flywheel_report(
    iteration_id: int,
    dynamic_baseline_run_id: int,
    dynamic_post_run_id: int,
    db: Session = Depends(get_db),
):
    return generate_flywheel_report(db, iteration_id, dynamic_baseline_run_id, dynamic_post_run_id)


@router.get("")
def list_flywheel_iterations(db: Session = Depends(get_db)):
    return db.query(FlywheelIteration).order_by(FlywheelIteration.id.desc()).all()