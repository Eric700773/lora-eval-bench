from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.benchmarks.lm_eval_runner import LmEvalRunner, WORKER_EVIDENCE_DIR
from app.database import STORAGE_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=STORAGE_ROOT / "reports" / "lm_eval" / "gsm8k_gate_10")
    parser.add_argument("--normalize-existing", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.normalize_existing:
        raise RuntimeError(f"Gate output directory must be empty: {output_dir}")

    runner = LmEvalRunner()
    run = SimpleNamespace(
        id="gsm8k-gate-10",
        benchmark_spec_json=None,
        config_snapshot_json={"limit": 10, "max_new_tokens": 512},
        task_name="gsm8k",
        base_model="local-qwen3.5-2b",
    )
    spec = runner.build_spec(run)
    manifest_before_path = output_dir / "run_manifest.before.json"
    if args.normalize_existing:
        from app.benchmarks.contracts import IntegrityGateReport, RunManifest

        preflight = IntegrityGateReport.model_validate_json(
            (output_dir / "preflight.json").read_text(encoding="utf-8")
        )
        manifest = RunManifest.model_validate_json(manifest_before_path.read_text(encoding="utf-8"))
        started = manifest.started_at or datetime.fromtimestamp(manifest_before_path.stat().st_mtime, tz=UTC)
        evidence_paths = [
            path
            for path in WORKER_EVIDENCE_DIR.glob("*.json")
            if path.stat().st_mtime >= manifest_before_path.stat().st_mtime - 1
        ]
    else:
        _, offline_equivalence = runner.prepare_execution_task(output_dir)
        preflight = runner.preflight(None, run, spec)
        preflight.checks.append(
            __import__("app.benchmarks.contracts", fromlist=["GateCheck"]).GateCheck(
                name="offline_task_protocol_equivalence",
                status="passed" if offline_equivalence["protocol_fields_unchanged"] else "failed",
                detail=str(output_dir / "offline_task_equivalence.json"),
                evidence=offline_equivalence,
            )
        )
        if any(check.status == "failed" for check in preflight.checks):
            preflight.status = "failed"
            (output_dir / "preflight.json").write_text(preflight.model_dump_json(indent=2) + "\n", encoding="utf-8")
            raise RuntimeError("lm-eval Gate preflight failed")
        (output_dir / "preflight.json").write_text(preflight.model_dump_json(indent=2) + "\n", encoding="utf-8")

        command = runner.build_command(spec, output_dir)
        manifest = runner._build_manifest(run, spec, command, output_dir)
        manifest_before_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        evidence_before = {path.name for path in WORKER_EVIDENCE_DIR.glob("*.json")}
        log_path = output_dir / "lm_eval.log"
        started = datetime.now(UTC)
        execution = runner.execute_command(command, log_path, timeout=3600)
        if int(execution.get("returncode", -1)) != 0:
            raise RuntimeError(f"lm-eval exited with code {execution.get('returncode')}; see {log_path}")
        evidence_paths = [path for path in WORKER_EVIDENCE_DIR.glob("*.json") if path.name not in evidence_before]

    results_path, samples_path = runner.discover_output_files(output_dir, spec.task_name)
    normalized = runner.normalize_files(spec, manifest, results_path, samples_path, gate_report=preflight)
    alignment = runner.align_worker_evidence(normalized, evidence_paths)
    final_gate = runner.build_integrity_gate(normalized, alignment)
    final_gate.checks = preflight.checks + final_gate.checks
    final_gate.engineering_failures += preflight.engineering_failures
    if any(check.status == "failed" for check in final_gate.checks):
        final_gate.status = "failed"
    normalized.gate_report = final_gate
    normalized.manifest.started_at = started
    normalized.manifest.completed_at = datetime.now(UTC)
    normalized.manifest.raw_result_path = str(results_path)

    normalized_path = output_dir / "normalized_result.json"
    gate_path = output_dir / "integrity_gate.json"
    manifest_path = output_dir / "run_manifest.json"
    normalized_path.write_text(normalized.model_dump_json(indent=2) + "\n", encoding="utf-8")
    gate_path.write_text(final_gate.model_dump_json(indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(normalized.manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": final_gate.status,
        "model_failures": final_gate.model_failures,
        "engineering_failures": final_gate.engineering_failures,
        "items": len(normalized.items),
        "results_path": str(results_path),
        "samples_path": str(samples_path),
        "normalized_path": str(normalized_path),
        "gate_path": str(gate_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if final_gate.status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
