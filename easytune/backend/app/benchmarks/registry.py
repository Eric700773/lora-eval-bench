from __future__ import annotations

from fastapi import HTTPException

from app.benchmarks.base import BenchmarkRunner


_RUNNERS: dict[str, BenchmarkRunner] = {}


def register_benchmark_runner(runner: BenchmarkRunner) -> None:
    framework = runner.framework.strip().lower()
    if not framework:
        raise ValueError("Benchmark runner framework must be non-empty.")
    _RUNNERS[framework] = runner


def get_benchmark_runner(framework: str) -> BenchmarkRunner:
    normalized = framework.strip().lower()
    runner = _RUNNERS.get(normalized)
    if runner is None:
        available = ", ".join(sorted(_RUNNERS)) or "none"
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported benchmark framework: {framework}. Available runners: {available}.",
        )
    return runner


def list_benchmark_runners() -> list[str]:
    return sorted(_RUNNERS)


from app.benchmarks.custom_runner import CustomBenchmarkRunner
from app.benchmarks.lm_eval_runner import LmEvalRunner

register_benchmark_runner(CustomBenchmarkRunner())
register_benchmark_runner(LmEvalRunner())
