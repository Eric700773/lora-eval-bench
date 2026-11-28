from __future__ import annotations

from abc import ABC, abstractmethod

from sqlalchemy.orm import Session

from app.benchmarks.contracts import BenchmarkSpec, IntegrityGateReport, NormalizedRunResult
from app.models import BenchmarkRun


class BenchmarkRunner(ABC):
    framework: str

    @abstractmethod
    def build_spec(self, run: BenchmarkRun) -> BenchmarkSpec:
        """Build the frozen protocol and model specification for a run."""

    @abstractmethod
    def preflight(self, db: Session, run: BenchmarkRun, spec: BenchmarkSpec) -> IntegrityGateReport:
        """Validate runner-specific prerequisites without executing the full benchmark."""

    @abstractmethod
    def run(self, db: Session, run_id: int) -> BenchmarkRun:
        """Execute one benchmark run and persist its native records."""

    @abstractmethod
    def normalize(self, db: Session, run_id: int) -> NormalizedRunResult:
        """Convert native runner outputs into the common EasyTune result contract."""
