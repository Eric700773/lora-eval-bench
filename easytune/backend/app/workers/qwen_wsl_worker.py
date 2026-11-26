import argparse
import gc
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


MODEL_PATH: Path
MODEL_CACHE: dict[tuple[bool, str], tuple[Any, Any]] = {}


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
    }


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
        if self.path != "/health":
            self._send(404, {"detail": "Not found"})
            return
        self._execute(runtime_health)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/release":
            self._execute(release_models)
            return
        if self.path == "/infer":
            self._execute(lambda: inference_smoke(payload))
            return
        if self.path == "/qlora-train":
            self._execute(lambda: qlora_train(payload))
            return
        if self.path == "/qlora-smoke":
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
    args = parser.parse_args()
    global MODEL_PATH
    MODEL_PATH = args.model_path.resolve()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    server = HTTPServer((args.host, args.port), Handler)
    print(json.dumps({"status": "ready", "host": args.host, "port": args.port, "model_path": str(MODEL_PATH)}, ensure_ascii=False), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
