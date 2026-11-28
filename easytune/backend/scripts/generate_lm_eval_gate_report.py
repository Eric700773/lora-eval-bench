from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(value: str, limit: int = 140) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    normalized = json.loads(args.normalized.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    task_results = next(iter(normalized["metrics"]["framework_results"].values()))
    items = normalized["items"]
    spec = normalized["spec"]
    mode = spec["mode"]
    is_full = mode == "full"
    title = "GSM8K lm-eval Full Evaluation Report" if is_full else "GSM8K lm-eval Gate Report"
    scope_note = (
        "This is a full 1,319-sample GSM8K evaluation under the frozen EasyTune protocol."
        if is_full
        else "This is a 10-sample engineering integrity gate, not a publishable GSM8K benchmark score."
    )
    strict_label = "Strict match" if is_full else "Strict match (sample only)"
    flexible_label = "Flexible extraction" if is_full else "Flexible extraction (sample only)"
    rows = [
        f"# {title}",
        "",
        f"> {scope_note}",
        "",
        "## Frozen Protocol",
        "",
        f"- Framework: lm-eval {spec['framework_version']}",
        f"- Framework commit: `{spec['framework_revision']}`",
        f"- Task: `{spec['task_name']}` (metadata version {spec['task_version']})",
        f"- Dataset revision: `{spec['dataset_revision']}`",
        f"- Split / few-shot / mode: `{spec['split']}` / `{spec['few_shot']}` / `{mode}`",
        f"- Effective samples: `{len(items)}`",
        f"- Task config SHA256: `{spec['task_config']['task_config_sha256']}`",
        "",
        "## Outcome",
        "",
        f"- Integrity status: **{gate['status']}**",
        f"- Engineering failures: **{gate['engineering_failures']}**",
        f"- Model-answer failures under strict match: **{gate['model_failures']}**",
        f"- {strict_label}: **{task_results['exact_match,strict-match']:.10f}**",
        f"- Strict-match standard error: **{task_results['exact_match_stderr,strict-match']:.10f}**",
        f"- {flexible_label}: **{task_results['exact_match,flexible-extract']:.10f}**",
        f"- Flexible-extract standard error: **{task_results['exact_match_stderr,flexible-extract']:.10f}**",
        "",
        "All engineering checks:",
        "",
    ]
    rows.extend(f"- `{check['name']}`: **{check['status']}** - {check.get('detail') or ''}" for check in gate["checks"])
    rows.extend(
        [
            "",
            "## Three-Sample Evidence Review",
            "",
            "| ID | Question | Prompt tokens | Parsed | Passed | Finish | Input/output aligned |",
            "|---:|---|---:|---|---|---|---|",
        ]
    )
    for item in items[:3]:
        rows.append(
            "| {id} | {question} | {tokens} | {parsed} | {passed} | {finish} | {aligned} |".format(
                id=item["sample_id"],
                question=compact(item["source_record"]["question"]).replace("|", "\\|"),
                tokens=item["model_input"]["prompt_tokens"],
                parsed=str(item["parsed_output"]).replace("|", "\\|"),
                passed=item["passed"],
                finish=item["evidence"].get("finish_reason"),
                aligned=item["evidence"].get("response_content_matched"),
            )
        )
    evidence_text = (
        "For every Full sample, the compact normalized index retains the official document, raw response, parser output, score, token counts, hashes, and evidence path. Complete messages, rendered prompt, and input/output token IDs remain in the per-request evidence JSON and are loaded on demand."
        if is_full
        else "For every Gate sample, the normalized result retains the official document, lm-eval request, API messages, rendered Qwen prompt, complete input token IDs, raw generated tokens/text, filtered responses, reference, score, and evidence hashes."
    )
    rows.extend(
        [
            "",
            evidence_text,
            "",
            "## Boundary",
            "",
            "Official GSM8K test badcases are analysis-only and are blocked from the synthesis and QLoRA flywheel. The score is specific to the frozen model, quantization, chat-template, generation, framework, Task, and dataset revisions recorded above.",
            "",
        ]
    )
    args.output.write_text("\n".join(rows), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
