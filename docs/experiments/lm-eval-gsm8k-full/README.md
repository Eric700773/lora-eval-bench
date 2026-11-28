# GSM8K Full Evaluation

EasyTune completed the full 1,319-sample GSM8K test split through a frozen `lm-evaluation-harness` protocol and the local Qwen3.5-2B OpenAI-compatible worker.

## Result

| Metric | Value | Standard error |
|---|---:|---:|
| Strict match | 0.4897649735 | 0.0137695989 |
| Flexible extraction | 0.6307808946 | 0.0132930195 |

This is a real full-split result, not the earlier 10-sample engineering Gate.

## Frozen protocol

- Framework: `lm-eval==0.4.13`
- Framework commit: `ddd67220430a2470529f25fd5c05a576ca1057a0`
- Task: `gsm8k`, metadata version `3.0`
- Task YAML SHA256: `82eb1780b263bc040729032b3e076990c7933f42163c91e359f38820dd1d8833`
- Dataset revision: `740312add88f781978c0658806c59bc2815b9866`
- Test fingerprint: `978d5a0bffccb5c2`
- Split: `test`
- Few-shot: `5`
- Limit: none
- Sampling: greedy, temperature `0`, seeds `42,42,42,42`
- Max generation tokens: `512`
- Local runtime: Qwen3.5-2B, NF4 4-bit, non-thinking template

WSL could not access Hugging Face during the run, so the execution Task used Parquet files from the frozen dataset revision. EasyTune compared all protocol-bearing fields with the official YAML. Only the Task ID and dataset loading source changed; Prompt construction, split selection, few-shot count, stop sequences, filters, metrics, aggregation, and Task metadata remained unchanged.

## Integrity outcome

- Effective samples: `1319/1319`
- Worker evidence aligned: `1319/1319`
- Unmatched samples: `0`
- Unused evidence files: `0`
- Silent truncations: `0`
- Nested chat templates: `0`
- Engineering failures: `0`
- Strict metric parity: framework `0.489764973464746`, recomputed `0.489764973464746`
- Repetition diagnostics flagged: `28` model outputs

Repetition flags are retained as model-behavior diagnostics. They do not invalidate the request, transport, parser, or scorer chain.

## Evidence storage

Each request retains a complete local evidence JSON:

```text
lm-eval document and 5-shot messages
-> Qwen rendered prompt
-> complete input token IDs
-> raw generated token IDs and text
-> API content
-> lm-eval raw response
-> strict/flexible filtered responses
-> reference and sample score
```

To keep API and frontend responses bounded, the 1,319-item normalized index stores hashes, token counts, output, parser result, score, and evidence path. Full token evidence is loaded only when a specific item is opened.

The full local prompts and outputs stay under ignored runtime storage. This public package contains sanitized protocol, aggregate metrics, and integrity counts.

## Runtime

- Wall time: `9973.344` seconds, about 2 hours 46 minutes
- Batch size: `1`
- Response cache: enabled
- Backend execution: asynchronous WSL worker
- Recovery: same output directory and response cache

## Test-set boundary

Official GSM8K test badcases are analysis-only. EasyTune blocks this Run from synthesis and QLoRA. Any evaluation-driven optimization must use a separate Discovery/Dev dataset.
