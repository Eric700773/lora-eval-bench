import json
import os
import urllib.error
import urllib.request
from typing import Any

from fastapi import HTTPException


DEFAULT_WORKER_URL = "http://127.0.0.1:8011"
DEFAULT_MODEL_PATH = r"/mnt/d/interview/local-llm-validation/models/qwen3.5-2b-bf16"


def runtime_config() -> dict[str, str]:
    return {
        "backend": "wsl_http_worker",
        "worker_url": os.getenv("EASYTUNE_QWEN_WORKER_URL", DEFAULT_WORKER_URL),
        "model_path": os.getenv("EASYTUNE_QWEN_BASE_MODEL", DEFAULT_MODEL_PATH),
    }


def qwen_runtime_preflight() -> dict[str, Any]:
    result = _request_json("GET", "/health")
    result["runtime_backend"] = "wsl_http_worker"
    result["worker_url"] = runtime_config()["worker_url"]
    return result


def run_single_inference_smoke(
    prompt: str,
    max_new_tokens: int = 64,
    load_in_4bit: bool = False,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    return _request_json(
        "POST",
        "/infer",
        {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "load_in_4bit": load_in_4bit,
            "temperature": temperature,
            "enable_thinking": enable_thinking,
            "adapter_path": windows_to_wsl_path(adapter_path) if adapter_path else None,
        },
        timeout=900,
    )


def release_qwen_runtime() -> dict[str, Any]:
    return _request_json("POST", "/release", {}, timeout=120)


def run_qlora_smoke(prompt: str, answer: str, rank: int = 8) -> dict[str, Any]:
    return _request_json(
        "POST",
        "/qlora-smoke",
        {"prompt": prompt, "answer": answer, "rank": rank},
        timeout=600,
    )


def windows_to_wsl_path(path: str) -> str:
    normalized = os.path.abspath(path).replace("\\", "/")
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return normalized


def run_qlora_training(dataset_path: str, output_dir: str, epochs: int = 1, rank: int = 8, max_length: int = 512, learning_rate: float = 2e-4) -> dict[str, Any]:
    return _request_json(
        "POST",
        "/qlora-train",
        {"dataset_path": windows_to_wsl_path(dataset_path), "output_dir": windows_to_wsl_path(output_dir), "epochs": epochs, "rank": rank, "max_length": max_length, "learning_rate": learning_rate},
        timeout=3600,
    )


def _request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    base_url = runtime_config()["worker_url"].rstrip("/")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Qwen worker error: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Qwen worker unavailable at {base_url}. Start the WSL worker first: {exc}",
        ) from exc
