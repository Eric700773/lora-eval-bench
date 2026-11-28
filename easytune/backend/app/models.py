from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    converted_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    formatting: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    report_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    dataset_info_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    training_jobs: Mapped[list["TrainingJob"]] = relationship(back_populates="dataset")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    model_family: Mapped[str] = mapped_column(String(32), nullable=False, default="qwen")
    base_model: Mapped[str] = mapped_column(Text, nullable=False)
    template: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    stage: Mapped[str] = mapped_column(String(64), nullable=False, default="sft")
    finetuning_type: Mapped[str] = mapped_column(String(64), nullable=False, default="lora")
    budget_level: Mapped[str] = mapped_column(String(64), nullable=False, default="balanced")
    extra_config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    config_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    command: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    dataset: Mapped[Dataset] = relationship(back_populates="training_jobs")
    model_versions: Mapped[list["ModelVersion"]] = relationship(back_populates="training_job")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    training_job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_model: Mapped[str] = mapped_column(Text, nullable=False)
    adapter_path: Mapped[str] = mapped_column(Text, nullable=False)
    export_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ready")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    training_job: Mapped[TrainingJob] = relationship(back_populates="model_versions")
    validation_records: Mapped[list["ValidationRecord"]] = relationship(back_populates="model_version")


class ValidationRecord(Base):
    __tablename__ = "validation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    actual_answer: Mapped[str] = mapped_column(Text, nullable=False)
    human_score: Mapped[int] = mapped_column(Integer, nullable=False)
    human_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    model_version: Mapped[ModelVersion] = relationship(back_populates="validation_records")

class BenchmarkRun(Base):
    """一次完整的规范化 Benchmark 评测任务（含配置快照与统计置信指标）"""
    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    framework: Mapped[str] = mapped_column(String(32), nullable=False, default="custom")
    task_name: Mapped[str] = mapped_column(String(255), nullable=False, default="custom_dataset")
    benchmark_spec_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    run_manifest_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    gate_report_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    progress_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    raw_result_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_result_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 待测模型：可关联微调后的 model_version_id，也可直接填底座模型路径（如 Qwen/Qwen3.5-2B）做基线评测
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    base_model: Mapped[str] = mapped_column(Text, nullable=False)
    adapter_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 评测数据集与防污染控制
    benchmark_dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    reference_train_dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    decontamination_enabled: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    contaminated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    use_dynamic_perturbation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 参数快照（锁定模板、采样、Thinking开关、截断长度，生成唯一哈希防静默Bug）
    config_snapshot_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # 裁判与统计结果（Pass@1, Pass@k, Pass^k, Bootstrap 95% 置信区间, Kappa一致性）
    judge_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="hybrid")  # deterministic | llm_judge | hybrid
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    items: Mapped[list["BenchmarkItemResult"]] = relationship(
        back_populates="benchmark_run", cascade="all, delete-orphan"
    )


class BenchmarkItemResult(Base):
    """单道评测题的推理轨迹、污染标记、校准打分与错误归因"""
    __tablename__ = "benchmark_item_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    benchmark_run_id: Mapped[int] = mapped_column(ForeignKey("benchmark_runs.id"), nullable=False, index=True)
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False)
    framework_sample_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_record_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    request_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_input_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    raw_output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    parsed_output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    evidence_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="general")

    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    perturbed_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 多次采样轨迹列表（用于计算 Pass@k 与 Pass^k）
    rollouts_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # 防污染体检标记
    is_contaminated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    contamination_detail_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # 裁判结果（支持位置交换双盲打分校验）
    verifier_type: Mapped[str] = mapped_column(String(32), nullable=False, default="exact_match")
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_passed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    judge_detail_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Badcase 自动归因标签（如 instruction_following, logic_gap, format_error, hallucination）
    error_tag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    benchmark_run: Mapped[BenchmarkRun] = relationship(back_populates="items")


class FlywheelIteration(Base):
    """评测驱动优化飞轮：基线评测 -> Badcase 聚类 -> 定向数据合成 -> LoRA 微调 -> 回归复测"""
    __tablename__ = "flywheel_iterations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    baseline_run_id: Mapped[int] = mapped_column(ForeignKey("benchmark_runs.id"), nullable=False)
    attribution_summary_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    synthesized_dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    training_job_id: Mapped[int | None] = mapped_column(ForeignKey("training_jobs.id"), nullable=True)
    post_eval_run_id: Mapped[int | None] = mapped_column(ForeignKey("benchmark_runs.id"), nullable=True)
    gain_report_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="analyzed")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)