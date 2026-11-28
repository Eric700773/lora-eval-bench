from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4


LM_EVAL_CLI: Path
ALLOWED_OUTPUT_ROOT: Path
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def validate_command(command: list[str]) -> str:
    if not command or command[0] != "env":
        raise ValueError("command must start with the frozen environment launcher")
    try:
        cli_index = command.index(str(LM_EVAL_CLI))
    except ValueError as exc:
        raise ValueError("command does not use the frozen lm-eval CLI") from exc
    arguments = command[cli_index + 1 :]
    if not arguments or arguments[0] != "run":
        raise ValueError("only lm-eval run is allowed")
    forbidden = {"--confirm_run_unsafe_code", "--trust_remote_code", "--config"}
    if forbidden.intersection(arguments):
        raise ValueError("unsafe or external lm-eval options are not allowed")
    required_pairs = {
        "--model": "local-chat-completions",
        "--tasks": "gsm8k_easytune_gate",
        "--num_fewshot": "5",
        "--batch_size": "1",
    }
    for option, expected in required_pairs.items():
        try:
            value = arguments[arguments.index(option) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"missing required option: {option}") from exc
        if value != expected:
            raise ValueError(f"unexpected {option}: {value}")
    if "--limit" in arguments:
        try:
            limit = arguments[arguments.index("--limit") + 1]
        except IndexError as exc:
            raise ValueError("missing --limit value") from exc
        if limit != "10":
            raise ValueError(f"only limit=10 Gate or no-limit Full is allowed, got {limit}")
        mode = "gate"
    else:
        mode = "full"
    for path_option in ("--include_path", "--output_path"):
        try:
            path_value = Path(arguments[arguments.index(path_option) + 1]).resolve()
        except (ValueError, IndexError) as exc:
            raise ValueError(f"missing required path option: {path_option}") from exc
        if ALLOWED_OUTPUT_ROOT not in path_value.parents and path_value != ALLOWED_OUTPUT_ROOT:
            raise ValueError(f"{path_option} is outside the allowed output root")
    if "--use_cache" not in arguments:
        raise ValueError("response cache is required for resumable evaluation")
    return mode


def _validated_payload(payload: dict[str, Any]) -> tuple[list[str], Path, int, str]:
    command = payload.get("command")
    log_path = Path(str(payload.get("log_path") or "")).resolve()
    timeout = int(payload.get("timeout", 21600))
    if not isinstance(command, list) or not command:
        raise ValueError("command must be a non-empty list")
    command = [str(value) for value in command]
    mode = validate_command(command)
    if ALLOWED_OUTPUT_ROOT not in log_path.parents:
        raise ValueError("log path is outside the allowed output root")
    if timeout < 60 or timeout > 43200:
        raise ValueError("timeout must be between 60 and 43200 seconds")
    return command, log_path, timeout, mode


def run_command(payload: dict[str, Any]) -> dict[str, Any]:
    command, log_path, timeout, mode = _validated_payload(payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.run(
            command,
            check=False,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    return {
        "returncode": process.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "log_path": str(log_path),
        "mode": mode,
    }


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    process = job.get("_process")
    if process is not None and job["status"] == "running" and process.poll() is not None:
        job["returncode"] = process.returncode
        job["status"] = "completed" if process.returncode == 0 else "failed"
        job["completed_at"] = time.time()
    elapsed = (job.get("completed_at") or time.time()) - job["started_at"]
    return {
        key: value
        for key, value in {
            "job_id": job["job_id"],
            "status": job["status"],
            "mode": job["mode"],
            "pid": process.pid if process is not None else None,
            "returncode": job.get("returncode"),
            "started_at": job["started_at"],
            "completed_at": job.get("completed_at"),
            "elapsed_seconds": round(elapsed, 3),
            "timeout": job["timeout"],
            "log_path": job["log_path"],
            "error": job.get("error"),
        }.items()
    }


def _monitor_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        process = job["_process"]
        timeout = job["timeout"]
    try:
        returncode = process.wait(timeout=timeout)
        status = "completed" if returncode == 0 else "failed"
        error = None
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = process.wait()
        status = "timed_out"
        error = f"lm-eval exceeded {timeout} seconds"
    finally:
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["_log_handle"].close()
            job["returncode"] = returncode
            job["status"] = status
            job["error"] = error
            job["completed_at"] = time.time()


def start_job(payload: dict[str, Any]) -> dict[str, Any]:
    command, log_path, timeout, mode = _validated_payload(payload)
    requested_id = str(payload.get("job_id") or f"lm-eval-{uuid4().hex}")
    if not requested_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in requested_id):
        raise ValueError("invalid job_id")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with JOBS_LOCK:
        if any(job["status"] == "running" for job in JOBS.values()):
            raise RuntimeError("another lm-eval job is already running")
        existing = JOBS.get(requested_id)
        if existing and existing["status"] == "running":
            return _public_job(existing)
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        JOBS[requested_id] = {
            "job_id": requested_id,
            "status": "running",
            "mode": mode,
            "command": command,
            "log_path": str(log_path),
            "timeout": timeout,
            "started_at": time.time(),
            "_process": process,
            "_log_handle": log_handle,
        }
        result = _public_job(JOBS[requested_id])
    threading.Thread(target=_monitor_job, args=(requested_id,), daemon=True).start()
    return result


def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise KeyError(job_id)
        return _public_job(job)


def cancel_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise KeyError(job_id)
        process = job.get("_process")
        if job["status"] == "running" and process is not None:
            process.terminate()
            job["status"] = "cancelling"
        return _public_job(job)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            with JOBS_LOCK:
                active = [_public_job(job) for job in JOBS.values() if job["status"] == "running"]
            self._send(
                200,
                {
                    "status": "ready",
                    "lm_eval_cli": str(LM_EVAL_CLI),
                    "allowed_output_root": str(ALLOWED_OUTPUT_ROOT),
                    "active_jobs": active,
                },
            )
            return
        if path.startswith("/jobs/"):
            job_id = path.removeprefix("/jobs/").strip()
            try:
                self._send(200, get_job(job_id))
            except KeyError:
                self._send(404, {"detail": "Job not found"})
            return
        self._send(404, {"detail": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            if path == "/run":
                self._send(200, run_command(payload))
                return
            if path == "/jobs":
                self._send(202, start_job(payload))
                return
            if path.startswith("/jobs/") and path.endswith("/cancel"):
                job_id = path.removeprefix("/jobs/").removesuffix("/cancel").strip("/")
                self._send(200, cancel_job(job_id))
                return
            self._send(404, {"detail": "Not found"})
        except Exception as exc:
            self._send(500, {"detail": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[lm-eval-worker] {self.address_string()} {format % args}", flush=True)

    def _send(self, status: int, body: dict[str, Any]) -> None:
        content = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm-eval-cli", type=Path, required=True)
    parser.add_argument("--allowed-output-root", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    global LM_EVAL_CLI, ALLOWED_OUTPUT_ROOT
    LM_EVAL_CLI = args.lm_eval_cli.resolve()
    ALLOWED_OUTPUT_ROOT = args.allowed_output_root.resolve()
    if not LM_EVAL_CLI.exists():
        raise FileNotFoundError(LM_EVAL_CLI)
    ALLOWED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "status": "ready",
                "host": args.host,
                "port": args.port,
                "lm_eval_cli": str(LM_EVAL_CLI),
                "allowed_output_root": str(ALLOWED_OUTPUT_ROOT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
