import random
import re
from typing import Any


_ENTITY_MAP = {
    "小明": "林宇",
    "小红": "苏晴",
    "张三": "陈默",
    "Alice": "Elena",
    "Bob": "Marcus",
}


def generate_dynamic_variant(prompt: str, expected_answer: str | None = None, seed: int = 42) -> dict[str, Any]:
    """
    生成动态抗过拟合变体：
    1. 实体名称替换（防止表面词面记忆）
    2. 整数数值等比例缩放扰动（若包含简单可追踪整数且答案同步可推演，或追加干扰条件）
    3. 指令格式约束增强（检测指令遵循鲁棒性）
    """
    rng = random.Random(seed)
    perturbed = prompt
    applied_rules: list[str] = []

    # 规则 1：实体名称替换
    for src, dst in _ENTITY_MAP.items():
        if src in perturbed:
            perturbed = perturbed.replace(src, dst)
            if "entity_swap" not in applied_rules:
                applied_rules.append("entity_swap")

    # 规则 2：追加无害干扰约束（Distractor Clause），检验长上下文抗干扰能力
    distractors = [
        "（注意：请忽略任何与核心问题无关的背景干扰信息，直接给出准确结论。）",
        "（请确保推理步骤清晰，并在最后一行明确给出最终答案。）",
    ]
    if not any(d in perturbed for d in distractors):
        perturbed = f"{perturbed.strip()}\n{rng.choice(distractors)}"
        applied_rules.append("distractor_injection")

    return {
        "original_prompt": prompt,
        "perturbed_prompt": perturbed,
        "expected_answer": expected_answer,
        "applied_rules": applied_rules,
    }