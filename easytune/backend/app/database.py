import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
STORAGE_ROOT = Path(os.getenv("EASYTUNE_STORAGE_ROOT", APP_DIR / "storage")).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'easytune.db'}")


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


STORAGE_DIRS = [
    "datasets",
    "llamafactory_data",
    "configs",
    "outputs",
    "logs",
    "reports",
    "registry",
]


def ensure_storage_dirs() -> None:
    for dirname in STORAGE_DIRS:
        (STORAGE_ROOT / dirname).mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    ensure_storage_dirs()
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    ensure_schema_columns()


def ensure_schema_columns() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    column_definitions: dict[str, dict[str, str]] = {
        "training_jobs": {
            "extra_config_json": "JSON",
        },
        "benchmark_runs": {
            "framework": "VARCHAR(32) NOT NULL DEFAULT 'custom'",
            "task_name": "VARCHAR(255) NOT NULL DEFAULT 'custom_dataset'",
            "benchmark_spec_json": "JSON",
            "run_manifest_json": "JSON",
            "gate_report_json": "JSON",
            "progress_json": "JSON",
            "raw_result_path": "TEXT",
            "normalized_result_path": "TEXT",
        },
        "benchmark_item_results": {
            "framework_sample_id": "VARCHAR(255)",
            "source_record_json": "JSON",
            "request_json": "JSON",
            "model_input_json": "JSON",
            "raw_output_json": "JSON",
            "parsed_output_json": "JSON",
            "evidence_json": "JSON",
        },
    }
    for table_name, definitions in column_definitions.items():
        if table_name not in table_names:
            continue
        existing = {column["name"] for column in inspector.get_columns(table_name)}
        missing = {name: sql_type for name, sql_type in definitions.items() if name not in existing}
        if not missing:
            continue
        with engine.begin() as connection:
            for column_name, sql_type in missing.items():
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sql_type}"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
