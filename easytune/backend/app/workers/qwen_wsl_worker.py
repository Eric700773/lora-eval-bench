import argparse
import gc
import hashlib
import json
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4


MODEL_PATH: Path
EVIDENCE_DIR: Path
MODEL_ID = "local-qwen3.5-2b"
MODEL_CACHE: dict[tuple[bool, str], tuple[Any, Any]] = {}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_token_ids(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_evidence(record: dict[str, Any]) -> str:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / f"{record['request_id']}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return str(path)


def detect_repetition(text: str) -> dict[str, Any]:
    normalized = " ".join(text.split())
    tokens = normalized.split()
    if len(tokens) < 16:
        return {"detected": False, "max_repeated_ngram": 0, "repeated_line": False}
    counts: dict[tuple[str, ...], int] = {}
    ngram_size = 8
    for index in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[index : index + ngram_size])
        counts[ngram] = counts.get(ngram, 0) + 1
    max_repeat = max(counts.values(), default=0)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_line = len(lines) != len(set(lines)) if lines else False
    return {"detected": max_repeat >= 4 or repeated_line, "max_repeated_ngram": max_repeat, "repeated_line": repeated_line}


def get_inference_model(load_in_4bit: bool, adapter_path: str | None = None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    cache_key = (load_in_4bit, adapter_path or "")
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    release_models()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model_kwargs: dict[str, Any] = {"local_files_only": True, "low_cpu_mem_usage": True, "dtype": torch.bfloat16}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model_kwargs["device_map"] = {"": 0}
        model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, **model_kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, **model_kwargs).to("cuda")
    if adapter_path:
        from peft import PeftModel
        adapter = Path(adapter_path)
        if not adapter.exists():
            raise FileNotFoundError(f"adapter path not found: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def release_models() -> dict[str, Any]:
    import torch

    released = len(MODEL_CACHE)
    MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "released", "models_released": released}


def runtime_health() -> dict[str, Any]:
    import accelerate
    import bitsandbytes
    import peft
    import torch
    import transformers
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    return {
        "available": bool(torch.cuda.is_available()),
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "accelerate": accelerate.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_type": model_config.model_type,
        "architectures": model_config.architectures,
        "openai_api": {"models": "/v1/models", "chat_completions": "/v1/chat/completions"},
        "evidence_dir": str(EVIDENCE_DIR),
    }


def create_chat_completion(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if payload.get("stream"):
        raise ValueError("Streaming is not supported by this evaluation worker.")
    if int(payload.get("n", 1)) != 1:
        raise ValueError("Only n=1 is supported by this evaluation worker.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("Only string system/user/assistant messages are supported.")
        normalized_messages.append({"role": role, "content": content})

    request_id = str(payload.get("evidence_request_id") or f"chatcmpl-{uuid4().hex}")
    max_tokens = int(payload.get("max_tokens") or payload.get("max_new_tokens") or 512)
    temperature = float(payload.get("temperature", 0.0) or 0.0)
    top_p = float(payload.get("top_p", 1.0) or 1.0)
    seed = int(payload.get("seed", 42) or 42)
    load_in_4bit = bool(payload.get("load_in_4bit", True))
    adapter_path = str(payload.get("adapter_path") or "").strip() or None
    enable_thinking = bool(payload.get("enable_thinking", False) or (payload.get("chat_template_kwargs") or {}).get("enable_thinking", False))
    stop = payload.get("stop") or []
    stop_sequences = [stop] if isinstance(stop, str) else [str(item) for item in stop]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model = get_inference_model(load_in_4bit, adapter_path)
    rendered_prompt = tokenizer.apply_chat_template(normalized_messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    encoded = tokenizer(rendered_prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")
    input_ids_list = input_ids[0].tolist()
    roundtrip_text = tokenizer.decode(input_ids_list, skip_special_tokens=False)
    model_context = getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", None)
    if model_context and len(input_ids_list) + max_tokens > int(model_context):
        raise ValueError(f"Prompt plus generation exceeds model context: {len(input_ids_list)} + {max_tokens} > {model_context}")

    generation_kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask, "do_sample": temperature > 0, "max_new_tokens": max_tokens, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
    elapsed = time.perf_counter() - started
    generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
    raw_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    eos_values = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id]
    finish_reason = "stop" if generated_ids and generated_ids[-1] in set(eos_values) else "length"
    matched_stop: str | None = None
    for sequence in stop_sequences:
        position = content.find(sequence)
        if position >= 0:
            content = content[:position].rstrip()
            matched_stop = sequence
            finish_reason = "stop"
            break

    nested_markers = any("<|im_start|>" in message["content"] or "<|im_end|>" in message["content"] for message in normalized_messages)
    response = {"id": request_id, "object": "chat.completion", "created": int(time.time()), "model": str(payload.get("model") or MODEL_ID), "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}], "usage": {"prompt_tokens": len(input_ids_list), "completion_tokens": len(generated_ids), "total_tokens": len(input_ids_list) + len(generated_ids)}}
    evidence = {
        "schema_version": 1,
        "request_id": request_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": {"id": MODEL_ID, "path_local_only": str(MODEL_PATH), "load_in_4bit": load_in_4bit, "adapter_path_local_only": adapter_path},
        "api_request": {"messages": normalized_messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed, "stop": stop_sequences, "enable_thinking": enable_thinking},
        "model_input": {"rendered_prompt": rendered_prompt, "rendered_prompt_sha256": sha256_text(rendered_prompt), "chat_template_sha256": sha256_text(tokenizer.chat_template or ""), "prompt_tokens": len(input_ids_list), "input_ids": input_ids_list, "input_ids_sha256": sha256_token_ids(input_ids_list), "input_ids_head": input_ids_list[:16], "input_ids_tail": input_ids_list[-16:], "truncated": False, "model_context_length": model_context, "context_exceeded": False, "nested_chat_template_detected": nested_markers, "render_token_roundtrip_matched": tokenizer.encode(rendered_prompt, add_special_tokens=False) == input_ids_list, "decoded_input_with_special_tokens": roundtrip_text},
        "model_output": {"raw_generated_token_ids": generated_ids, "raw_generated_token_ids_sha256": sha256_token_ids(generated_ids), "raw_generated_text": raw_generated_text, "raw_generated_text_sha256": sha256_text(raw_generated_text), "content": content, "content_sha256": sha256_text(content), "finish_reason": finish_reason, "matched_stop_sequence": matched_stop, "repetition_diagnostic": detect_repetition(content)},
        "runtime": {"elapsed_seconds": round(elapsed, 3), "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "gpu": torch.cuda.get_device_name(0)},
        "response": response,
    }
    evidence_path = write_evidence(evidence)
    response["evidence"] = {"request_id": request_id, "path": evidence_path, "rendered_prompt_sha256": evidence["model_input"]["rendered_prompt_sha256"], "content_sha256": evidence["model_output"]["content_sha256"]}
    return response, evidence


def inference_smoke(payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    max_new_tokens = int(payload.get("max_new_tokens", 32))
    load_in_4bit = bool(payload.get("load_in_4bit", False))
    temperature = float(payload.get("temperature", 0.0))
    enable_thinking = bool(payload.get("enable_thinking", False))
    adapter_path = str(payload.get("adapter_path") or "").strip() or None

    torch.cuda.reset_peak_memory_stats()
    tokenizer, model = get_inference_model(load_in_4bit, adapter_path)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")
    started = time.perf_counter()
    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": temperature > 0,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = 0.9
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
    generated_ids = output_ids[0, input_ids.shape[1] :]
    result = {
        "status": "passed",
        "answer": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        "prompt_tokens": int(input_ids.shape[1]),
        "completion_tokens": int(generated_ids.shape[0]),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(0),
        "load_in_4bit": load_in_4bit,
        "model_path": str(MODEL_PATH),
    }
    return result


def qlora_train(payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dataset_path = Path(str(payload["dataset_path"])).resolve()
    output_dir = Path(str(payload["output_dir"])).resolve()
    epochs = int(payload.get("epochs", 1))
    rank = int(payload.get("rank", 8))
    max_length = int(payload.get("max_length", 512))
    learning_rate = float(payload.get("learning_rate", 2e-4))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"training output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with dataset_path.open(encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("training dataset is empty")

    release_models()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    encoded: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        instruction = str(row.get("instruction") or "").strip()
        input_text = str(row.get("input") or "").strip()
        answer = str(row.get("output") or "").strip()
        prompt = f"{instruction}\n{input_text}".strip()
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=True)["input_ids"]
        prefix = tokenizer.apply_chat_template([messages[0]], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True)["input_ids"]
        if full[:len(prefix)] != prefix:
            raise RuntimeError(f"assistant prefix mismatch at row {index}")
        if len(full) > max_length:
            raise RuntimeError(f"row {index} has {len(full)} tokens, exceeding max_length={max_length}")
        encoded.append({"input_ids": full, "prefix_length": len(prefix)})

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, local_files_only=True, quantization_config=quantization, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    adapted = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank * 2, lora_dropout=0.05, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    adapted.train()
    trainable = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    losses: list[float] = []
    optimizer_steps = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        for sample in encoded:
            optimizer.zero_grad(set_to_none=True)
            input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device="cuda")
            labels = input_ids.clone()
            labels[:, :sample["prefix_length"]] = -100
            output = adapted(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels, use_cache=False)
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(output.loss.detach().float().item()))

    adapter_dir = output_dir / "adapter"
    adapted.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.exists() or adapter_weights.stat().st_size == 0:
        raise RuntimeError("adapter_model.safetensors was not created")
    report = {"status": "completed", "mock": False, "dataset_path": str(dataset_path), "samples": len(encoded), "epochs": epochs, "optimizer_steps": optimizer_steps, "loss_first": losses[0], "loss_last": losses[-1], "loss_mean": sum(losses) / len(losses), "rank": rank, "max_length": max_length, "learning_rate": learning_rate, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "elapsed_seconds": round(time.perf_counter() - started, 3), "adapter_path": str(adapter_dir), "adapter_model_bytes": adapter_weights.stat().st_size, "gpu": torch.cuda.get_device_name(0), "model_path": str(MODEL_PATH)}
    (output_dir / "training-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del optimizer, adapted, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def qlora_smoke(payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    prompt = str(payload.get("prompt") or "Answer with only the number: What is 7 times 6?").strip()
    answer = str(payload.get("answer") or "42").strip()
    rank = int(payload.get("rank", 8))
    if not prompt or not answer:
        raise ValueError("prompt and answer are required")
    if rank <= 0:
        raise ValueError("rank must be positive")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    release_models()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )["input_ids"]
    prefix = tokenizer.apply_chat_template(
        [messages[0]],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
    )["input_ids"]
    if full[: len(prefix)] != prefix:
        raise RuntimeError("assistant prefix mismatch")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    adapted = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    adapted.train()
    input_ids = torch.tensor([full], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : len(prefix)] = -100
    optimizer = torch.optim.AdamW(
        (parameter for parameter in adapted.parameters() if parameter.requires_grad),
        lr=2e-4,
    )
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    output = adapted(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss.item()}")
    loss.backward()
    gradient_norms = [
        parameter.grad.detach().float().norm().item()
        for parameter in adapted.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradient_norms or not all(torch.isfinite(torch.tensor(value)) for value in gradient_norms):
        raise RuntimeError("missing or non-finite LoRA gradients")
    optimizer.step()

    output_root = Path(__file__).resolve().parents[1] / "storage" / "outputs" / f"qlora_smoke_{int(time.time())}"
    adapter_dir = output_root / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    adapted.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.exists() or adapter_weights.stat().st_size == 0:
        raise RuntimeError("adapter_model.safetensors was not created")
    trainable = sum(parameter.numel() for parameter in adapted.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in adapted.parameters())
    result = {
        "status": "passed",
        "smoke_only": True,
        "real_optimizer_step": True,
        "load_in_4bit": True,
        "loss": float(loss.detach().float().item()),
        "gradient_tensors": len(gradient_norms),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "rank": rank,
        "input_tokens": len(full),
        "assistant_loss_tokens": len(full) - len(prefix),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "adapter_path": str(adapter_dir),
        "adapter_model_bytes": adapter_weights.stat().st_size,
        "gpu": torch.cuda.get_device_name(0),
        "model_path": str(MODEL_PATH),
    }
    (output_root / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del optimizer, adapted, model, tokenizer, output, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._execute(runtime_health)
            return
        if path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": 0,
                            "owned_by": "easytune-local",
                        }
                    ],
                },
            )
            return
        if path.startswith("/evidence/"):
            request_id = path.removeprefix("/evidence/").strip()
            if not request_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in request_id):
                self._send(400, {"detail": "Invalid evidence request ID"})
                return
            evidence_path = EVIDENCE_DIR / f"{request_id}.json"
            if not evidence_path.exists():
                self._send(404, {"detail": "Evidence not found"})
                return
            self._send(200, json.loads(evidence_path.read_text(encoding="utf-8")))
            return
        self._send(404, {"detail": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if path == "/v1/chat/completions":
            self._execute(lambda: create_chat_completion(payload)[0])
            return
        if path == "/release":
            self._execute(release_models)
            return
        if path == "/infer":
            self._execute(lambda: inference_smoke(payload))
            return
        if path == "/qlora-train":
            self._execute(lambda: qlora_train(payload))
            return
        if path == "/qlora-smoke":
            self._execute(lambda: qlora_smoke(payload))
            return
        self._send(404, {"detail": "Not found"})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[qwen-worker] {self.address_string()} {format % args}", flush=True)

    def _execute(self, operation: Any) -> None:
        try:
            self._send(200, operation())
        except Exception as exc:
            self._send(500, {"detail": f"{type(exc).__name__}: {exc}"})

    def _send(self, status: int, body: dict[str, Any]) -> None:
        content = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--evidence-dir", type=Path, default=None)
    args = parser.parse_args()
    global MODEL_PATH, EVIDENCE_DIR
    MODEL_PATH = args.model_path.resolve()
    EVIDENCE_DIR = (args.evidence_dir or (Path(__file__).resolve().parents[1] / "storage" / "reports" / "worker_evidence")).resolve()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    server = HTTPServer((args.host, args.port), Handler)
    print(json.dumps({"status": "ready", "host": args.host, "port": args.port, "model_id": MODEL_ID, "model_path": str(MODEL_PATH), "evidence_dir": str(EVIDENCE_DIR)}, ensure_ascii=False), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
