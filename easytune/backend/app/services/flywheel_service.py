import ast
import json
import os
import re
import urllib.request
from collections import Counter
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.database import STORAGE_ROOT
from app.models import BenchmarkItemResult, BenchmarkRun, Dataset, FlywheelIteration, ModelVersion, TrainingJob
from app.schemas import TrainingJobCreate
from app.services.dataset_service import convert_dataset
from app.services.decontamination_service import check_sample_contamination
from app.services.eval_harness_service import build_config_snapshot, execute_benchmark_run
from app.services.qwen_runtime_service import release_qwen_runtime, run_qlora_training, run_single_inference_smoke
from app.services.model_service import create_model_version_for_job
from app.services.training_service import create_training_job


def calibrate_decontamination(
    db: Session,
    baseline_run_id: int,
    candidate_dataset_id: int | None = None,
) -> dict[str, Any]:
    run = db.get(BenchmarkRun, baseline_run_id)
    if not run or run.status != "completed":
        raise HTTPException(status_code=400, detail="Completed baseline BenchmarkRun is required.")
    items = (
        db.query(BenchmarkItemResult)
        .filter(BenchmarkItemResult.benchmark_run_id == run.id)
        .order_by(BenchmarkItemResult.sample_index.asc())
        .all()
    )
    train_corpus: list[str] = []
    if run.reference_train_dataset_id:
        reference = db.get(Dataset, run.reference_train_dataset_id)
        if reference:
            path = reference.converted_file_path or reference.original_file_path
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    rows = json.load(handle) if path.endswith(".json") else [json.loads(line) for line in handle if line.strip()]
                train_corpus = [str(row.get("instruction") or row.get("prompt") or "") for row in rows]

    cases: list[dict[str, Any]] = []
    for item in items:
        result = check_sample_contamination(item.prompt, train_corpus)
        cases.append(
            {
                "source": "benchmark",
                "id": item.sample_index,
                "expected_contaminated": bool(item.is_contaminated),
                "actual_contaminated": bool(result["is_contaminated"]),
                "method": result["method"],
                "score": result["score"],
                "text": item.prompt,
            }
        )

    if candidate_dataset_id is not None:
        candidate = db.get(Dataset, candidate_dataset_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate Dataset not found.")
        path = candidate.converted_file_path or candidate.original_file_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Candidate Dataset file not found.")
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle) if path.endswith(".json") else [json.loads(line) for line in handle if line.strip()]
        eval_prompts = [item.prompt for item in items]
        for index, row in enumerate(rows):
            verifier_type = str(row.get("verifier_type") or _infer_synth_verifier(row))
            normalized = _synthesis_leak_text(
                {
                    "instruction": str(row.get("instruction") or row.get("prompt") or ""),
                    "output": str(row.get("output") or row.get("answer") or ""),
                },
                verifier_type,
            )
            result = check_sample_contamination(normalized, eval_prompts, ngram_size=13, semantic_threshold=0.72)
            cases.append(
                {
                    "source": f"candidate_dataset_{candidate_dataset_id}",
                    "id": index,
                    "expected_contaminated": False,
                    "actual_contaminated": bool(result["is_contaminated"]),
                    "method": result["method"],
                    "score": result["score"],
                    "text": str(row.get("instruction") or row.get("prompt") or ""),
                }
            )

    tp = sum(case["expected_contaminated"] and case["actual_contaminated"] for case in cases)
    fn = sum(case["expected_contaminated"] and not case["actual_contaminated"] for case in cases)
    fp = sum(not case["expected_contaminated"] and case["actual_contaminated"] for case in cases)
    tn = sum(not case["expected_contaminated"] and not case["actual_contaminated"] for case in cases)
    return {
        "baseline_run_id": run.id,
        "candidate_dataset_id": candidate_dataset_id,
        "summary": {
            "total": len(cases),
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        },
        "false_positives": [case for case in cases if not case["expected_contaminated"] and case["actual_contaminated"]],
        "false_negatives": [case for case in cases if case["expected_contaminated"] and not case["actual_contaminated"]],
        "cases": cases,
    }


def _infer_synth_verifier(row: dict[str, Any]) -> str:
    output = str(row.get("output") or row.get("answer") or "").strip()
    instruction = str(row.get("instruction") or row.get("prompt") or "").lower()
    if "def " in output:
        return "code_sandbox"
    if output.startswith("{") or "json" in instruction:
        return "json_format"
    if re.search(r"-?\d+(?:\.\d+)?", output):
        return "math_numeric"
    return "exact_match"


def synthesis_configuration() -> dict[str, Any]:
    base_url = (
        os.getenv("EASYTUNE_SYNTHESIS_BASE_URL")
        or os.getenv("EASYTUNE_JUDGE_BASE_URL")
        or ""
    ).rstrip("/")
    api_key = os.getenv("EASYTUNE_SYNTHESIS_API_KEY") or os.getenv("EASYTUNE_JUDGE_API_KEY") or ""
    model = os.getenv("EASYTUNE_SYNTHESIS_MODEL") or os.getenv("EASYTUNE_JUDGE_MODEL") or ""
    return {
        "configured": bool(base_url and api_key and model),
        "base_url": base_url or None,
        "model": model or None,
        "api_key_set": bool(api_key),
        "require_external": os.getenv("EASYTUNE_REQUIRE_EXTERNAL_SYNTHESIS", "true").lower() != "false",
    }


def list_external_synthesis_models() -> dict[str, Any]:
    config = synthesis_configuration()
    if not config["base_url"] or not config["api_key_set"]:
        raise HTTPException(status_code=400, detail="Synthesis BASE_URL and API_KEY must be configured first.")
    data = _external_json_request("GET", f"{config['base_url']}/models")
    rows = data.get("data") if isinstance(data, dict) else None
    model_ids = sorted(
        str(row.get("id"))
        for row in rows or []
        if isinstance(row, dict) and row.get("id")
    )
    return {"base_url": config["base_url"], "models": model_ids, "count": len(model_ids)}


def test_external_synthesis() -> dict[str, Any]:
    config = synthesis_configuration()
    if not config["configured"]:
        raise HTTPException(status_code=400, detail="Synthesis BASE_URL, API_KEY and MODEL must be configured first.")
    body = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return exactly one valid JSON object with string fields instruction and output. "
                    "Create a new arithmetic training example whose correct answer is 42. No Markdown."
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    data = _external_json_request("POST", f"{config['base_url']}/chat/completions", body)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Synthesis API response has no chat completion content.") from exc
    parsed = _parse_synthesized_rows(str(content))
    if not parsed:
        raise HTTPException(status_code=502, detail="Synthesis model did not return valid instruction/output JSON.")
    return {
        "status": "passed",
        "base_url": config["base_url"],
        "model": config["model"],
        "sample": parsed[0],
    }


def _external_json_request(method: str, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    api_key = os.getenv("EASYTUNE_SYNTHESIS_API_KEY") or os.getenv("EASYTUNE_JUDGE_API_KEY") or ""
    request = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Synthesis API request failed: {type(exc).__name__}: {exc}") from exc


def analyze_badcases_and_synthesize(
    db: Session,
    baseline_run_id: int,
    iteration_name: str,
    augment_per_badcase: int = 2,
    general_replay_dataset_id: int | None = None,
    replay_ratio: float = 0.3,
) -> FlywheelIteration:
    """
    评测驱动优化飞轮核心引擎（合规刷榜）：
    1. 自动聚合基线评测中的所有非污染 Badcase，按错误类型与能力维度切片归因；
    2. 针对每条 Badcase 调用强模型 API 合成同分布但不同实体的变体训练样本（严禁原题泄露）；
    3. 强制执行反向去污染校验，确保合成训练样本与原始评测集 13-gram 零重叠；
    4. 混入一定比例通用回放数据（Replay Buffer）防止灾难性遗忘，注册为 converted 数据集。
    """
    run = db.get(BenchmarkRun, baseline_run_id)
    if not run or run.status != "completed":
        raise HTTPException(status_code=400, detail="Baseline BenchmarkRun must be completed first.")

    items = (
        db.query(BenchmarkItemResult)
        .filter(BenchmarkItemResult.benchmark_run_id == baseline_run_id)
        .all()
    )
    badcases = [it for it in items if it.is_passed == 0 and it.is_contaminated == 0]
    if not badcases:
        raise HTTPException(status_code=400, detail="No clean badcases found in this baseline run.")

    tag_counter = Counter(_attribution_tag(it) for it in badcases)
    cat_counter = Counter(it.category for it in badcases)

    # 收集原始评测集所有题目文本，用于反向防污染自检
    eval_prompts = [it.prompt for it in items]

    synthesized_rows: list[dict[str, str]] = []
    blocked_by_decontam = 0
    blocked_by_quality = 0

    for bc in badcases:
        variants = _synthesize_remediation_samples(
            prompt=bc.prompt,
            expected_answer=bc.expected_answer or "",
            model_wrong_output=bc.rollouts_json[0] if bc.rollouts_json else "",
            error_reason=bc.error_reason or "",
            verifier_type=bc.verifier_type,
            count=augment_per_badcase,
        )
        for v in variants:
            quality_ok, _ = _validate_synthesized_sample(v, bc.verifier_type)
            if quality_ok and bc.verifier_type == "math_numeric" and _synthesis_provider() == "external_openai_compatible":
                quality_ok, _ = _external_quality_review(v, bc.verifier_type)
            if not quality_ok:
                blocked_by_quality += 1
                continue
            synth_text = _synthesis_leak_text(v, bc.verifier_type)
            leak = check_sample_contamination(synth_text, eval_prompts, ngram_size=13, semantic_threshold=0.72)
            duplicate = any(row["instruction"] == v["instruction"] for row in synthesized_rows)
            if leak["is_contaminated"] or duplicate:
                blocked_by_decontam += 1
                continue
            synthesized_rows.append(
                {
                    "instruction": v["instruction"],
                    "input": "",
                    "output": v["output"],
                }
            )

    try:
        release_qwen_runtime()
    except Exception:
        pass

    # 混入通用经验回放数据（防微调后通用能力遗忘）
    replay_added = 0
    if general_replay_dataset_id and synthesized_rows:
        replay_ds = db.get(Dataset, general_replay_dataset_id)
        if replay_ds:
            r_path = replay_ds.converted_file_path or replay_ds.original_file_path
            if r_path and os.path.exists(r_path):
                with open(r_path, "r", encoding="utf-8") as f:
                    r_data = json.load(f) if r_path.endswith(".json") else [json.loads(l) for l in f if l.strip()]
                take_n = max(1, int(len(synthesized_rows) * replay_ratio))
                for row in r_data[:take_n]:
                    synthesized_rows.append(
                        {
                            "instruction": str(row.get("instruction") or row.get("prompt") or ""),
                            "input": str(row.get("input") or ""),
                            "output": str(row.get("output") or row.get("answer") or ""),
                        }
                    )
                    replay_added += 1

    if not synthesized_rows:
        raise HTTPException(status_code=400, detail="No valid non-contaminated training samples were synthesized.")

    # 落盘为标准 Alpaca JSON 并注册为 converted 状态数据集，直接兼容原项目 LLaMA-Factory
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    ds_dir = STORAGE_ROOT / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    file_path = ds_dir / f"flywheel_{ts}_{uuid4().hex[:6]}.json"
    file_path.write_text(json.dumps(synthesized_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    new_ds = Dataset(
        name=f"[Flywheel] {iteration_name}",
        original_file_path=str(file_path),
        converted_file_path=None,
        file_type="json",
        file_size=file_path.stat().st_size,
        sample_count=len(synthesized_rows),
        formatting="alpaca",
        quality_score=95.0,
        status="checked",
    )
    db.add(new_ds)
    db.commit()
    db.refresh(new_ds)
    new_ds = convert_dataset(db, new_ds.id)

    attribution_summary = {
        "total_clean_eval_items": len([i for i in items if i.is_contaminated == 0]),
        "total_badcases": len(badcases),
        "by_error_tag": dict(tag_counter),
        "by_category": dict(cat_counter),
        "synthesized_count": len(synthesized_rows) - replay_added,
        "replay_buffer_count": replay_added,
        "blocked_contaminated_synth": blocked_by_decontam,
        "blocked_quality_synth": blocked_by_quality,
        "synthesis_provider": _synthesis_provider(),
    }

    iteration = FlywheelIteration(
        name=iteration_name,
        baseline_run_id=baseline_run_id,
        attribution_summary_json=attribution_summary,
        synthesized_dataset_id=new_ds.id,
        status="dataset_ready",
    )
    db.add(iteration)
    db.commit()
    db.refresh(iteration)
    return iteration


def trigger_flywheel_qlora_training(
    db: Session,
    iteration_id: int,
    budget_level: str = "low",
) -> FlywheelIteration:
    """调用原项目的 LLaMA-Factory QLoRA 训练流水线（默认 low 预算适配 RTX 4060 8GB）"""
    it = db.get(FlywheelIteration, iteration_id)
    if not it or not it.synthesized_dataset_id:
        raise HTTPException(status_code=404, detail="FlywheelIteration or synthesized dataset not found.")

    baseline_run = db.get(BenchmarkRun, it.baseline_run_id)
    if not baseline_run:
        raise HTTPException(status_code=404, detail="Baseline run missing.")

    job_payload = TrainingJobCreate(
        name=f"Flywheel-QLoRA-{it.name}",
        dataset_id=it.synthesized_dataset_id,
        model_family="qwen",
        base_model=baseline_run.base_model,
        template="qwen3_5_nothink",
        stage="sft",
        finetuning_type="qlora",
        budget_level=budget_level,
    )
    job = create_training_job(db, job_payload)
    it.training_job_id = job.id
    it.status = "training"
    db.commit()
    db.refresh(it)

    dataset = db.get(Dataset, it.synthesized_dataset_id)
    if not dataset or not dataset.converted_file_path:
        it.status = "training_failed"
        db.commit()
        raise HTTPException(status_code=400, detail="Synthesized dataset has no converted JSONL path.")
    try:
        from app.services.log_service import append_log

        append_log(job.log_path, "[real-qlora] Starting WSL NF4 QLoRA training.")
        report = run_qlora_training(
            dataset_path=dataset.converted_file_path,
            output_dir=job.output_dir or "",
            epochs=1,
            rank=8,
            max_length=512 if budget_level == "low" else 1024,
            learning_rate=2e-4,
        )
        if report.get("mock") or not report.get("adapter_model_bytes"):
            raise RuntimeError("QLoRA worker did not produce a real adapter artifact.")
        job.status = "completed"
        job.progress = 100
        job.error_message = None
        db.commit()
        model_version = create_model_version_for_job(db, job)
        if model_version.status != "ready":
            raise RuntimeError(f"ModelVersion artifact validation failed: {model_version.status}")
        append_log(job.log_path, f"[real-qlora] ModelVersion {model_version.id} created.")
        it.status = "training_completed"
        db.commit()
        db.refresh(it)
        return it
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
        it.status = "training_failed"
        db.commit()
        raise HTTPException(status_code=502, detail=f"Real QLoRA training failed: {exc}") from exc


def run_post_training_regression_eval(
    db: Session,
    iteration_id: int,
    model_version_id: int,
) -> FlywheelIteration:
    """
    训练完成后，使用与基线评测【完全一致的配置快照】重跑回归测试，输出严谨的前后对比与防过拟合报告
    """
    it = db.get(FlywheelIteration, iteration_id)
    if not it:
        raise HTTPException(status_code=404, detail="FlywheelIteration not found.")
    base_run = db.get(BenchmarkRun, it.baseline_run_id)
    mv = db.get(ModelVersion, model_version_id)
    if not base_run or not mv:
        raise HTTPException(status_code=404, detail="Baseline run or ModelVersion not found.")
    training_job = db.get(TrainingJob, it.training_job_id) if it.training_job_id else None
    adapter_weights = os.path.join(mv.adapter_path, "adapter_model.safetensors")
    if not training_job or training_job.status != "completed":
        raise HTTPException(status_code=400, detail="Flywheel training job is not completed.")
    if mv.training_job_id != training_job.id:
        raise HTTPException(status_code=400, detail="ModelVersion does not belong to this Flywheel training job.")
    if mv.status != "ready" or not os.path.isfile(adapter_weights):
        raise HTTPException(status_code=400, detail="ModelVersion has no validated real adapter weights.")

    base_cfg = base_run.config_snapshot_json
    snapshot, cfg_hash = build_config_snapshot(
        base_model=mv.base_model,
        adapter_path=mv.adapter_path,
        template=base_cfg.get("template", "qwen3_5_nothink"),
        temperature=float(base_cfg.get("temperature", 0.0)),
        max_new_tokens=int(base_cfg.get("max_new_tokens", 1024)),
        num_rollouts=int(base_cfg.get("num_rollouts", 1)),
        enable_thinking=bool(base_cfg.get("enable_thinking", False)),
        load_in_4bit=bool(base_cfg.get("load_in_4bit", True)),
    )

    post_run = BenchmarkRun(
        name=f"[Regression] {base_run.name} -> {mv.name}",
        model_version_id=mv.id,
        base_model=mv.base_model,
        adapter_path=mv.adapter_path,
        benchmark_dataset_id=base_run.benchmark_dataset_id,
        reference_train_dataset_id=base_run.reference_train_dataset_id,
        decontamination_enabled=1,
        use_dynamic_perturbation=base_run.use_dynamic_perturbation,
        config_snapshot_json=snapshot,
        config_hash=cfg_hash,
        judge_mode=base_run.judge_mode,
        status="pending",
    )
    db.add(post_run)
    db.commit()
    db.refresh(post_run)

    completed_post = execute_benchmark_run(db, post_run.id)

    b_clean = (base_run.metrics_json or {}).get("clean_decontaminated", {})
    p_clean = (completed_post.metrics_json or {}).get("clean_decontaminated", {})
    b_cat = (base_run.metrics_json or {}).get("by_category_clean", {})
    p_cat = (completed_post.metrics_json or {}).get("by_category_clean", {})

    category_deltas = {
        cat: round(
            float(p_cat.get(cat, {}).get("pass_at_1", 0.0))
            - float(b_cat.get(cat, {}).get("pass_at_1", 0.0)),
            4,
        )
        for cat in set(b_cat.keys()) | set(p_cat.keys())
    }

    it.post_eval_run_id = completed_post.id
    it.gain_report_json = {
        "baseline_pass_at_1": b_clean.get("pass_at_1", 0.0),
        "post_pass_at_1": p_clean.get("pass_at_1", 0.0),
        "net_gain_pass_at_1": round(p_clean.get("pass_at_1", 0.0) - b_clean.get("pass_at_1", 0.0), 4),
        "baseline_ci_95": b_clean.get("ci_95"),
        "post_ci_95": p_clean.get("ci_95"),
        "post_contaminated_count": completed_post.contaminated_count,
        "category_deltas": category_deltas,
        "model_version_id": mv.id,
        "adapter_path": mv.adapter_path,
        "clean_subset_matched": base_run.reference_train_dataset_id == completed_post.reference_train_dataset_id,
        "config_matched": all(
            base_cfg.get(key) == snapshot.get(key)
            for key in ("template", "temperature", "max_new_tokens", "num_rollouts", "enable_thinking", "load_in_4bit")
        ),
    }
    it.status = "completed"
    db.commit()
    db.refresh(it)
    return it


def _synthesis_leak_text(row: dict[str, str], verifier_type: str) -> str:
    instruction = str(row.get("instruction") or "")
    output = str(row.get("output") or "")
    if verifier_type == "code_sandbox":
        boilerplate = [
            r"编写\s*Python\s*函数",
            r"请编写一个\s*Python\s*函数",
            r"Write\s+(?:a|the)\s+(?:Python\s+)?function",
            r"Implement\s+(?:a|the)\s+(?:Python\s+)?function",
            r"返回",
            r"returns?",
        ]
        for pattern in boilerplate:
            instruction = re.sub(pattern, " ", instruction, flags=re.IGNORECASE)
    return f"{instruction}\n{output}"


def _synthesis_provider() -> str:
    if os.getenv("EASYTUNE_SYNTHESIS_BASE_URL") or os.getenv("EASYTUNE_JUDGE_BASE_URL"):
        return "external_openai_compatible"
    return "local_qwen_fallback"


def _external_quality_review(row: dict[str, str], verifier_type: str) -> tuple[bool, str]:
    config = synthesis_configuration()
    review_prompt = (
        "You are a strict training-data reviewer. Check whether the instruction is self-contained, "
        "has a valid solution, and whether the proposed output correctly answers it. "
        "For JSON tasks, require valid JSON. For code tasks, require executable Python solving the instruction. "
        "Return exactly one JSON object: {\"valid\":true|false,\"reason\":\"short reason\"}.\n"
        f"Verifier type: {verifier_type}\n"
        f"Instruction: {row['instruction']}\n"
        f"Proposed output: {row['output']}"
    )
    data = _external_json_request(
        "POST",
        f"{config['base_url']}/chat/completions",
        {
            "model": config["model"],
            "messages": [{"role": "user", "content": review_prompt}],
            "temperature": 0.0,
            "max_tokens": 256,
        },
    )
    try:
        content = str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return False, "review_response_missing"
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    try:
        review = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return False, "review_json_invalid"
        try:
            review = json.loads(match.group(0))
        except json.JSONDecodeError:
            return False, "review_json_invalid"
    return bool(review.get("valid")), str(review.get("reason") or "")


def _validate_synthesized_sample(row: dict[str, str], verifier_type: str) -> tuple[bool, str]:
    instruction = str(row.get("instruction") or "").strip()
    output = str(row.get("output") or "").strip()
    if len(instruction) < 12 or not output:
        return False, "empty_or_too_short"
    if verifier_type == "json_format":
        try:
            json.loads(output)
        except json.JSONDecodeError:
            return False, "invalid_json_answer"
    if verifier_type == "math_numeric":
        instruction_numbers = re.findall(r"-?\d+(?:\.\d+)?", instruction)
        output_numbers = re.findall(r"-?\d+(?:\.\d+)?", output)
        if not instruction_numbers or not output_numbers:
            return False, "missing_numeric_problem_or_answer"
    if verifier_type == "code_sandbox":
        if "def " not in output:
            return False, "missing_code_answer"
        try:
            tree = ast.parse(output)
        except SyntaxError:
            return False, "python_syntax_error"
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)):
            return False, "missing_function_definition"
    return True, "passed"


def _attribution_tag(item: BenchmarkItemResult) -> str:
    deterministic_tags = {
        "math_numeric": "calculation_error",
        "json_format": "format_error",
        "code_sandbox": "code_error",
    }
    return deterministic_tags.get(item.verifier_type, item.error_tag or "unknown")


def _build_synthesis_prompt(
    prompt: str,
    expected_answer: str,
    error_reason: str,
    verifier_type: str,
    variant_index: int,
) -> str:
    common = (
        "Return exactly one valid JSON object with two string fields: instruction and output. "
        "Do not use Markdown and do not add explanation. Change entities, scenario, numbers, and wording; "
        "do not copy any sentence or long phrase from the source. The answer must be correct.\n"
        f"Variant index: {variant_index}\nSource problem: {prompt}\n"
        f"Reference answer: {expected_answer or 'derive a correct answer'}\nFailure reason: {error_reason}\n"
    )
    if verifier_type == "json_format":
        return common + (
            "This is a JSON extraction task. Create a new Chinese extraction scenario with different names, "
            "values, or entities. The instruction must require JSON-only output with named keys. "
            "The output field itself must be a valid JSON object using double quotes and correct values."
        )
    if verifier_type == "code_sandbox":
        return common + (
            "This is a Python coding task. Do NOT create any uniqueness, frequency-counting, first-occurrence, "
            "or non-repeating-element problem. Instead create a clearly different algorithmic task such as "
            "merging intervals, validating parentheses, rotating a matrix, or processing numeric ranges. "
            "Use a different input domain and function name. The output field must contain executable Python code "
            "with a function definition, no prose, and it must solve the instruction."
        )
    return common + (
        "This is a deterministic arithmetic reasoning task. Create a new solvable problem with changed numbers and wording. "
        "The output field must be the correct final numeric answer."
    )


def _synthesize_remediation_samples(
    prompt: str,
    expected_answer: str,
    model_wrong_output: str,
    error_reason: str,
    verifier_type: str = "exact_match",
    count: int = 2,
) -> list[dict[str, str]]:
    """
    优先调用外部强模型 API；未配置时调用本地 Qwen3.5 worker 真实生成。
    任一模型未返回合法 JSON 训练样本时明确失败，不生成模板伪数据。
    """
    base_url = (
        os.getenv("EASYTUNE_SYNTHESIS_BASE_URL")
        or os.getenv("EASYTUNE_JUDGE_BASE_URL")
        or os.getenv("EASYTUNE_INFERENCE_BASE_URL", "")
    )
    api_key = (
        os.getenv("EASYTUNE_SYNTHESIS_API_KEY")
        or os.getenv("EASYTUNE_JUDGE_API_KEY")
        or os.getenv("EASYTUNE_INFERENCE_API_KEY", "")
    )
    synth_model = os.getenv("EASYTUNE_SYNTHESIS_MODEL") or os.getenv("EASYTUNE_JUDGE_MODEL", "qwen-max")
    require_external = os.getenv("EASYTUNE_REQUIRE_EXTERNAL_SYNTHESIS", "true").lower() != "false"

    if base_url:
        external_rows: list[dict[str, str]] = []
        for variant_index in range(count):
            single_prompt = _build_synthesis_prompt(
                prompt=prompt,
                expected_answer=expected_answer,
                error_reason=error_reason,
                verifier_type=verifier_type,
                variant_index=variant_index + 1,
            )
            for attempt in range(2):
                body = {
                    "model": synth_model,
                    "messages": [{"role": "user", "content": single_prompt}],
                    "temperature": 0.2 + attempt * 0.2,
                    "max_tokens": 1200,
                }
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                req = urllib.request.Request(
                    url=f"{base_url.rstrip('/')}/chat/completions",
                    data=json.dumps(body).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=90) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    parsed = _parse_synthesized_rows(data["choices"][0]["message"]["content"])
                    if parsed:
                        external_rows.append(parsed[0])
                        break
                except Exception as exc:
                    if attempt == 1 and require_external:
                        raise HTTPException(
                            status_code=502,
                            detail=f"External synthesis request failed: {type(exc).__name__}: {exc}",
                        ) from exc
        if external_rows:
            return external_rows
        if require_external:
            raise HTTPException(status_code=502, detail="External synthesis model returned no valid instruction/output JSON.")

    if require_external:
        raise HTTPException(
            status_code=503,
            detail=(
                "External synthesis model is required. Configure EASYTUNE_SYNTHESIS_BASE_URL, "
                "EASYTUNE_SYNTHESIS_API_KEY and EASYTUNE_SYNTHESIS_MODEL. "
                "Set EASYTUNE_REQUIRE_EXTERNAL_SYNTHESIS=false only for local fallback experiments."
            ),
        )

    local_rows: list[dict[str, str]] = []
    for variant_index in range(count):
        single_prompt = _build_synthesis_prompt(
            prompt=prompt,
            expected_answer=expected_answer,
            error_reason=error_reason,
            verifier_type=verifier_type,
            variant_index=variant_index + 1,
        )
        for attempt in range(2):
            local_result = run_single_inference_smoke(
                prompt=single_prompt,
                max_new_tokens=768,
                load_in_4bit=True,
                temperature=0.4 + attempt * 0.2,
                enable_thinking=False,
            )
            parsed = _parse_synthesized_rows(str(local_result.get("answer") or ""))
            if parsed:
                local_rows.append(parsed[0])
                break
    if not local_rows:
        raise HTTPException(status_code=502, detail="Local synthesis model returned no valid JSON training samples.")
    return local_rows


def _parse_synthesized_rows(content: str) -> list[dict[str, str]]:
    candidates: list[Any] = []
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    try:
        candidates.append(json.loads(stripped))
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
                candidates.append(value)
                break
            except json.JSONDecodeError:
                continue
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            rows.append(candidate)
        elif isinstance(candidate, list):
            rows.extend(row for row in candidate if isinstance(row, dict))
    return [
        {"instruction": str(row["instruction"]).strip(), "output": str(row["output"]).strip()}
        for row in rows
        if str(row.get("instruction") or "").strip() and str(row.get("output") or "").strip()
    ]
