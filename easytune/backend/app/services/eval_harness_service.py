import hashlib
import json
import os
import time
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import BenchmarkItemResult, BenchmarkRun, Dataset, ModelVersion
from app.services.calibrated_judge_service import compute_statistical_metrics, verify_single_rollout
from app.services.decontamination_service import check_sample_contamination
from app.services.perturbation_service import generate_dynamic_variant
from app.services.qwen_runtime_service import release_qwen_runtime, run_single_inference_smoke

_EVAL_MODEL_CACHE: dict[str, Any] = {}


def build_config_snapshot(
    base_model: str,
    adapter_path: str | None,
    template: str = "qwen",
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    num_rollouts: int = 1,
    enable_thinking: bool = False,
    load_in_4bit: bool = True,
) -> tuple[dict[str, Any], str]:
    """锁定所有影响评测分数的环境与采样参数，生成唯一 SHA256 哈希，杜绝静默 Bug"""
    snapshot = {
        "base_model": base_model,
        "adapter_path": adapter_path or "none",
        "template": template,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "num_rollouts": num_rollouts,
        "enable_thinking": enable_thinking,
        "load_in_4bit": load_in_4bit,
        "engine": "qwen_transformers_4bit" if load_in_4bit else "qwen_transformers_bf16",
    }
    canonical = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return snapshot, config_hash


def _load_dataset_records(dataset: Dataset) -> list[dict[str, Any]]:
    file_path = dataset.converted_file_path or dataset.original_file_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
        if isinstance(data, list):
            return data
        raise HTTPException(status_code=400, detail="Dataset JSON must be a list of records.")


def _extract_prompt_and_answer(row: dict[str, Any]) -> tuple[str, str | None, str, str, str | None]:
    """兼容 Alpaca / ShareGPT / 自定义 Benchmark 格式"""
    category = str(row.get("category") or row.get("tag") or "general")
    verifier_type = str(row.get("verifier_type") or "exact_match")
    unit_test = row.get("unit_test") or row.get("test")

    if "conversations" in row and isinstance(row["conversations"], list):
        user_msg = next((m["value"] for m in row["conversations"] if m.get("from") in {"human", "user"}), "")
        gpt_msg = next((m["value"] for m in row["conversations"] if m.get("from") in {"gpt", "assistant"}), None)
        return user_msg, gpt_msg, category, verifier_type, unit_test

    instruction = str(row.get("instruction") or row.get("prompt") or row.get("question") or "")
    inp = str(row.get("input") or "")
    prompt = f"{instruction}\n{inp}".strip() if inp else instruction
    answer = row.get("output") or row.get("answer") or row.get("expected_answer")
    return prompt, (str(answer) if answer is not None else None), category, verifier_type, unit_test


def _get_qwen_model_and_tokenizer(base_model: str, adapter_path: str | None, load_in_4bit: bool = True):
    """针对 RTX 4060 (8GB) 优化的 Qwen3.5 4-bit 推理加载器"""
    cache_key = f"{base_model}|{adapter_path or 'base'}|4bit={load_in_4bit}"
    if cache_key in _EVAL_MODEL_CACHE:
        return _EVAL_MODEL_CACHE[cache_key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    _EVAL_MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def _generate_responses(
    tokenizer: Any,
    model: Any,
    prompt: str,
    num_rollouts: int,
    temperature: float,
    max_new_tokens: int,
    enable_thinking: bool,
    load_in_4bit: bool = True,
    adapter_path: str | None = None,
) -> list[str]:
    if tokenizer is None or model is None:
        return [
            str(
                run_single_inference_smoke(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    load_in_4bit=load_in_4bit,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    adapter_path=adapter_path,
                )["answer"]
            )
            for _ in range(num_rollouts)
        ]

    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs: list[str] = []
    do_sample = temperature > 0 or num_rollouts > 1
    effective_temp = temperature if temperature > 0 else (0.6 if num_rollouts > 1 else 0.0)

    with torch.no_grad():
        for _ in range(num_rollouts):
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = effective_temp
                gen_kwargs["top_p"] = 0.9
            out_ids = model.generate(**inputs, **gen_kwargs)
            gen_ids = out_ids[:, inputs.input_ids.shape[-1] :]
            ans = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            outputs.append(ans)
    return outputs


def recompute_benchmark_metrics(db: Session, run_id: int) -> BenchmarkRun:
    run = db.get(BenchmarkRun, run_id)
    if not run or run.status != "completed":
        raise HTTPException(status_code=400, detail="Completed BenchmarkRun not found.")
    items = (
        db.query(BenchmarkItemResult)
        .filter(BenchmarkItemResult.benchmark_run_id == run.id)
        .order_by(BenchmarkItemResult.sample_index.asc())
        .all()
    )
    if not items:
        raise HTTPException(status_code=400, detail="BenchmarkRun has no stored item results.")

    raw_rollouts: list[list[bool]] = []
    clean_rollouts: list[list[bool]] = []
    raw_categories: dict[str, dict[str, int]] = {}
    clean_categories: dict[str, dict[str, int]] = {}
    for item in items:
        flags = [bool(item.is_passed)]
        raw_rollouts.append(flags)
        raw_bucket = raw_categories.setdefault(item.category, {"total": 0, "passed": 0})
        raw_bucket["total"] += 1
        raw_bucket["passed"] += int(bool(item.is_passed))
        if not item.is_contaminated:
            clean_rollouts.append(flags)
            clean_bucket = clean_categories.setdefault(item.category, {"total": 0, "passed": 0})
            clean_bucket["total"] += 1
            clean_bucket["passed"] += int(bool(item.is_passed))

    def summarize(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, int | float]]:
        return {
            category: {
                "total": values["total"],
                "passed": values["passed"],
                "pass_at_1": round(values["passed"] / values["total"], 4) if values["total"] else 0.0,
            }
            for category, values in stats.items()
        }

    raw_metrics = compute_statistical_metrics(raw_rollouts)
    clean_metrics = compute_statistical_metrics(clean_rollouts)
    by_category_raw = summarize(raw_categories)
    by_category_clean = summarize(clean_categories)
    previous = run.metrics_json or {}
    run.metrics_json = {
        "clean_decontaminated": clean_metrics,
        "raw_unfiltered": raw_metrics,
        "contamination_inflation": round(raw_metrics["pass_at_1"] - clean_metrics["pass_at_1"], 4),
        "by_category": {category: values["pass_at_1"] for category, values in by_category_raw.items()},
        "by_category_raw": by_category_raw,
        "by_category_clean": by_category_clean,
        "variant_mode": "dynamic_perturbed" if run.use_dynamic_perturbation else "original",
        "elapsed_seconds": previous.get("elapsed_seconds"),
    }
    run.contaminated_count = sum(int(bool(item.is_contaminated)) for item in items)
    db.commit()
    db.refresh(run)
    return run


def execute_benchmark_run(db: Session, run_id: int) -> BenchmarkRun:
    run = db.get(BenchmarkRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="BenchmarkRun not found.")

    run.status = "running"
    db.commit()

    start_ts = time.time()
    try:
        bench_ds = db.get(Dataset, run.benchmark_dataset_id)
        if not bench_ds:
            raise HTTPException(status_code=404, detail="Benchmark dataset not found.")

        eval_rows = _load_dataset_records(bench_ds)

        # 加载训练集文本用于去污染扫描
        train_corpus: list[str] = []
        if run.decontamination_enabled and run.reference_train_dataset_id:
            train_ds = db.get(Dataset, run.reference_train_dataset_id)
            if train_ds:
                train_rows = _load_dataset_records(train_ds)
                for tr in train_rows:
                    p, _, _, _, _ = _extract_prompt_and_answer(tr)
                    if p:
                        train_corpus.append(p)

        cfg = run.config_snapshot_json
        tokenizer, model = None, None

        # 清空旧记录（支持重跑）
        db.query(BenchmarkItemResult).filter(BenchmarkItemResult.benchmark_run_id == run.id).delete()

        contaminated_count = 0
        clean_rollout_passes: list[list[bool]] = []
        raw_rollout_passes: list[list[bool]] = []
        raw_category_stats: dict[str, dict[str, int]] = {}
        clean_category_stats: dict[str, dict[str, int]] = {}

        for idx, row in enumerate(eval_rows):
            prompt, expected, category, verifier_type, unit_test = _extract_prompt_and_answer(row)

            # 1. 双层去污染扫描
            contam_info = check_sample_contamination(prompt, train_corpus) if run.decontamination_enabled else {
                "is_contaminated": False, "method": "disabled", "score": 0.0
            }
            is_contam = bool(contam_info["is_contaminated"])
            if is_contam:
                contaminated_count += 1

            # 2. 动态扰动抗过拟合变体
            actual_prompt = prompt
            perturbed_prompt = None
            if run.use_dynamic_perturbation:
                variant = generate_dynamic_variant(prompt, expected, seed=42 + idx)
                perturbed_prompt = variant["perturbed_prompt"]
                actual_prompt = perturbed_prompt

            # 3. 多次 Rollout 推理
            rollouts = _generate_responses(
                tokenizer=tokenizer,
                model=model,
                prompt=actual_prompt,
                num_rollouts=int(cfg.get("num_rollouts", 1)),
                temperature=float(cfg.get("temperature", 0.0)),
                max_new_tokens=int(cfg.get("max_new_tokens", 1024)),
                enable_thinking=bool(cfg.get("enable_thinking", False)),
                load_in_4bit=bool(cfg.get("load_in_4bit", True)),
                adapter_path=run.adapter_path,
            )

            # 4. 逐条采样判分
            rollout_checks = [
                verify_single_rollout(
                    prompt=actual_prompt,
                    prediction=pred,
                    expected_answer=expected,
                    verifier_type=verifier_type,
                    unit_test_code=unit_test,
                )
                for pred in rollouts
            ]
            pass_flags = [bool(c["passed"]) for c in rollout_checks]
            first_check = rollout_checks[0]
            item_passed = pass_flags[0]

            raw_rollout_passes.append(pass_flags)
            if not is_contam:
                clean_rollout_passes.append(pass_flags)

            # 分类统计
            raw_bucket = raw_category_stats.setdefault(category, {"total": 0, "passed": 0})
            raw_bucket["total"] += 1
            if item_passed:
                raw_bucket["passed"] += 1
            if not is_contam:
                clean_bucket = clean_category_stats.setdefault(category, {"total": 0, "passed": 0})
                clean_bucket["total"] += 1
                if item_passed:
                    clean_bucket["passed"] += 1

            # 5. 初步错误标签归因
            error_tag = None
            error_reason = None
            if not item_passed:
                deterministic_tags = {
                    "math_numeric": "calculation_error",
                    "json_format": "format_error",
                    "code_sandbox": "code_error",
                }
                error_tag = first_check.get("error_category") or deterministic_tags.get(
                    verifier_type,
                    "logic_gap",
                )
                error_reason = first_check.get("reason")

            item_record = BenchmarkItemResult(
                benchmark_run_id=run.id,
                sample_index=idx,
                category=category,
                prompt=prompt,
                perturbed_prompt=perturbed_prompt,
                expected_answer=expected,
                rollouts_json=rollouts,
                is_contaminated=1 if is_contam else 0,
                contamination_detail_json=contam_info,
                verifier_type=verifier_type,
                score=float(first_check.get("score", 0.0)),
                is_passed=1 if item_passed else 0,
                judge_detail_json=first_check,
                error_tag=error_tag,
                error_reason=error_reason,
            )
            db.add(item_record)

        elapsed = round(time.time() - start_ts, 2)
        clean_metrics = compute_statistical_metrics(clean_rollout_passes)
        raw_metrics = compute_statistical_metrics(raw_rollout_passes)

        def category_metrics(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, int | float]]:
            return {
                cat: {
                    "total": stat["total"],
                    "passed": stat["passed"],
                    "pass_at_1": round(stat["passed"] / stat["total"], 4) if stat["total"] else 0.0,
                }
                for cat, stat in stats.items()
            }

        by_category_raw = category_metrics(raw_category_stats)
        by_category_clean = category_metrics(clean_category_stats)
        by_category_legacy = {cat: values["pass_at_1"] for cat, values in by_category_raw.items()}

        run.contaminated_count = contaminated_count
        run.metrics_json = {
            "clean_decontaminated": clean_metrics,
            "raw_unfiltered": raw_metrics,
            "contamination_inflation": round(raw_metrics["pass_at_1"] - clean_metrics["pass_at_1"], 4),
            "by_category": by_category_legacy,
            "by_category_raw": by_category_raw,
            "by_category_clean": by_category_clean,
            "variant_mode": "dynamic_perturbed" if run.use_dynamic_perturbation else "original",
            "elapsed_seconds": elapsed,
        }
        run.status = "completed"
        db.commit()
        db.refresh(run)
        try:
            release_qwen_runtime()
        except Exception:
            pass
        return run

    except Exception as exc:
        try:
            release_qwen_runtime()
        except Exception:
            pass
        run.status = "failed"
        run.error_message = str(exc)
        db.commit()
        raise