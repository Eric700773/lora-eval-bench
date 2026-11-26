import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import urllib.request
from typing import Any


def verify_single_rollout(
    prompt: str,
    prediction: str,
    expected_answer: str | None,
    verifier_type: str = "exact_match",
    unit_test_code: str | None = None,
) -> dict[str, Any]:
    """
    优先级：确定性程序校验（代码沙盒/数学数值/JSON格式/选项匹配） > 校准 LLM-as-a-Judge
    """
    # 剥离 Qwen3.5 可能输出的 <think>...</think> 思维链过程，仅提取最终作答
    clean_pred = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL).strip()

    if verifier_type == "code_sandbox" and unit_test_code:
        return _run_python_sandbox(clean_pred, unit_test_code)

    if verifier_type == "json_format":
        return _verify_json_output(clean_pred, expected_answer)

    if verifier_type == "math_numeric" and expected_answer is not None:
        return _verify_math_numeric(clean_pred, expected_answer)

    if verifier_type == "llm_judge":
        return _run_calibrated_llm_judge(prompt, clean_pred, expected_answer)

    # 默认 exact_match / 选项匹配
    return _verify_exact_or_choice(clean_pred, expected_answer or "")


def _run_python_sandbox(prediction: str, unit_test_code: str, timeout_sec: int = 5) -> dict[str, Any]:
    """子进程隔离沙盒：提取代码块并运行断言单元测试"""
    code_match = re.search(r"```(?:python)?\s*(.*?)```", prediction, flags=re.DOTALL | re.IGNORECASE)
    extracted_code = code_match.group(1).strip() if code_match else prediction
    full_script = f"{extracted_code}\n\n# --- Unit Tests ---\n{unit_test_code}\n"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(full_script)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, "-I", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        passed = proc.returncode == 0
        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "reason": "Unit tests passed" if passed else f"Assertion/Runtime Error: {proc.stderr[:300]}",
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "score": 0.0, "reason": f"Sandbox execution timed out ({timeout_sec}s)"}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _verify_json_output(prediction: str, expected_answer: str | None) -> dict[str, Any]:
    json_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", prediction, flags=re.DOTALL)
    raw = json_match.group(1) if json_match else prediction
    try:
        parsed = json.loads(raw)
        if expected_answer:
            exp_obj = json.loads(expected_answer)
            if isinstance(exp_obj, dict) and isinstance(parsed, dict):
                missing_keys = [k for k in exp_obj.keys() if k not in parsed]
                if missing_keys:
                    return {"passed": False, "score": 0.5, "reason": f"Missing required JSON keys: {missing_keys}"}
        return {"passed": True, "score": 1.0, "reason": "Valid JSON structure"}
    except Exception as exc:
        return {"passed": False, "score": 0.0, "reason": f"Invalid JSON: {exc}"}


def _verify_math_numeric(prediction: str, expected_answer: str) -> dict[str, Any]:
    exp_nums = re.findall(r"-?\d+(?:\.\d+)?", expected_answer)
    pred_nums = re.findall(r"-?\d+(?:\.\d+)?", prediction)
    if not exp_nums:
        return _verify_exact_or_choice(prediction, expected_answer)
    if not pred_nums:
        return {"passed": False, "score": 0.0, "reason": "No numeric answer found in prediction"}
    target = float(exp_nums[-1])
    actual = float(pred_nums[-1])
    passed = math.isclose(target, actual, rel_tol=1e-4, abs_tol=1e-4)
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": f"Matched numeric {actual}" if passed else f"Expected {target}, got {actual}",
    }


def _verify_exact_or_choice(prediction: str, expected_answer: str) -> dict[str, Any]:
    exp = expected_answer.strip().lower()
    pred = prediction.strip().lower()
    if not exp:
        return {"passed": bool(pred), "score": 1.0 if pred else 0.0, "reason": "Non-empty output"}
    # 若参考答案是单字母选项 (A/B/C/D)
    if re.fullmatch(r"[a-d]", exp):
        m = re.search(r"(?:答案[是为：:]\s*|answer\s*[:is]*\s*|\b)([a-d])\b", pred)
        extracted = m.group(1) if m else (pred[0] if pred and pred[0] in "abcd" else "")
        passed = extracted == exp
        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "reason": f"Choice matched: {extracted.upper()}" if passed else f"Expected {exp.upper()}, got '{extracted}'",
        }
    passed = exp in pred or pred == exp
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Exact/substring match" if passed else f"Mismatch with expected: {expected_answer[:80]}",
    }


def _run_calibrated_llm_judge(prompt: str, prediction: str, expected_answer: str | None) -> dict[str, Any]:
    """
    使用强模型 API 作为 Judge，引入严格 Rubric 与 Unknown 拒答逃生舱，防止裁判幻觉
    """
    base_url = os.getenv("EASYTUNE_JUDGE_BASE_URL") or os.getenv("EASYTUNE_INFERENCE_BASE_URL", "")
    api_key = os.getenv("EASYTUNE_JUDGE_API_KEY") or os.getenv("EASYTUNE_INFERENCE_API_KEY", "")
    judge_model = os.getenv("EASYTUNE_JUDGE_MODEL", "qwen-max")

    if not base_url:
        # 若未配置外部裁判 API，退化为关键词覆盖率校验
        return _verify_exact_or_choice(prediction, expected_answer or "")

    judge_prompt = (
        "你是一位严格、客观的大模型评测裁判。请根据【问题】和【参考标准答案】，评估【待测模型回答】是否准确、完整且无幻觉。\n"
        "请严格返回合法 JSON，包含字段：\n"
        '- "verdict": 只能是 "PASS"、"FAIL" 或 "UNKNOWN"（当参考信息不足以判定时选 UNKNOWN）\n'
        '- "score": 0.0 到 1.0 之间的浮点数\n'
        '- "error_category": 若未通过，从 ["instruction_following", "logic_gap", "format_error", "hallucination", "other"] 中选一个；若通过则为 "none"\n'
        '- "reason": 一句话中文归因\n\n'
        f"【问题】：{prompt}\n"
        f"【参考标准答案】：{expected_answer or '无固定标准答案，请依据事实与指令约束判断'}\n"
        f"【待测模型回答】：{prediction}\n"
    )
    body = {
        "model": judge_model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        parsed = json.loads(m.group(0)) if m else {}
        verdict = str(parsed.get("verdict", "FAIL")).upper()
        score = float(parsed.get("score", 1.0 if verdict == "PASS" else 0.0))
        return {
            "passed": verdict == "PASS",
            "score": score,
            "verdict": verdict,
            "error_category": parsed.get("error_category", "other"),
            "reason": parsed.get("reason", content[:150]),
        }
    except Exception as exc:
        return {"passed": False, "score": 0.0, "verdict": "UNKNOWN", "reason": f"Judge API error: {exc}"}


def compute_statistical_metrics(item_rollout_passes: list[list[bool]], bootstrap_rounds: int = 500) -> dict[str, Any]:
    """
    计算工业级评测统计指标：
    - Pass@1: 首次采样平均准确率
    - Pass@k: k 次采样中至少成功 1 次的比例（衡量模型能力上限）
    - Pass^k: k 次采样全部成功的比例（衡量输出一致性与稳定性）
    - 95% Bootstrap 置信区间（确认提分是否具有统计显著性）
    """
    n = len(item_rollout_passes)
    if n == 0:
        return {"total_samples": 0, "pass_at_1": 0.0, "pass_at_k": 0.0, "pass_power_k": 0.0, "ci_95": [0.0, 0.0]}

    k = max(len(r) for r in item_rollout_passes)
    pass1_scores = [1.0 if (r and r[0]) else 0.0 for r in item_rollout_passes]
    pass_at_k_scores = [1.0 if any(r) else 0.0 for r in item_rollout_passes]
    pass_power_k_scores = [1.0 if (r and all(r)) else 0.0 for r in item_rollout_passes]

    mean_p1 = sum(pass1_scores) / n
    mean_pk = sum(pass_at_k_scores) / n
    mean_ppk = sum(pass_power_k_scores) / n

    # Bootstrap 95% 置信区间
    rng = random.Random(42)
    boot_means = []
    for _ in range(bootstrap_rounds):
        sample = [pass1_scores[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    ci_low = boot_means[int(0.025 * bootstrap_rounds)]
    ci_high = boot_means[int(0.975 * bootstrap_rounds)]

    return {
        "total_samples": n,
        "rollouts_per_item": k,
        "pass_at_1": round(mean_p1, 4),
        "pass_at_k": round(mean_pk, 4),
        "pass_power_k": round(mean_ppk, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
    }