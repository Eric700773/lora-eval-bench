# lm-eval GSM8K Integrity Gate

This evidence package records a 10-sample engineering gate for the EasyTune `lm-evaluation-harness` integration. It is not a full GSM8K score and must not be compared with leaderboard results.

## Frozen protocol

| Field | Value |
|---|---|
| Framework | `lm-eval==0.4.13` |
| Framework commit | `ddd67220430a2470529f25fd5c05a576ca1057a0` |
| Task | `gsm8k`, metadata version `3.0` |
| Task YAML SHA256 | `82eb1780b263bc040729032b3e076990c7933f42163c91e359f38820dd1d8833` |
| Dataset | `openai/gsm8k`, config `main` |
| Dataset revision | `740312add88f781978c0658806c59bc2815b9866` |
| Split | `test` |
| Few-shot | `5` |
| Limit | `10` |
| Generation | greedy, temperature `0`, max generation tokens `512` |

The host could not reach Hugging Face from WSL during the run. EasyTune therefore loaded Parquet files from the frozen dataset revision. A generated execution Task changed only the dataset source and runtime Task ID; Prompt construction, splits, few-shot count, stop sequences, filters, metric, aggregation, and Task metadata were checked field-by-field against the official YAML.

## Outcome

- Integrity gate: **passed**
- Engineering failures: **0**
- Worker evidence aligned: **10/10**
- Strict-match model failures: **6/10**
- Strict match on this sample: **0.4**
- Flexible extraction on this sample: **0.6**
- Framework strict metric and item-level recomputation: **0.4 / 0.4**

The sample scores are included only to demonstrate that official extraction and aggregation executed. They do not estimate full-dataset performance.

## Evidence chain

Every Gate item retained and verified:

```text
frozen GSM8K document
-> lm-eval 5-shot chat messages
-> Qwen chat-template rendered prompt
-> complete input token IDs and hashes
-> raw generated token IDs and text
-> OpenAI-compatible response content
-> lm-eval raw response
-> strict/flexible filtered response
-> reference and per-item exact-match score
```

All ten items passed these engineering checks:

- model input and complete token evidence present;
- rendered Prompt token roundtrip matched;
- no silent input truncation;
- no nested Qwen chat template;
- Worker content exactly matched lm-eval raw response;
- official strict and flexible filters were retained;
- no unexplained repetition loop;
- framework aggregate metric matched item-level recomputation.

## Boundary

Official GSM8K test badcases are analysis-only in EasyTune. The API rejects sending this Run into synthesis or QLoRA. Evaluation-driven optimization must use a separate Discovery/Dev dataset and reserve the official test set for final evaluation.

The full local evidence contains prompts and model responses and remains under ignored runtime storage. This public package contains only sanitized protocol and aggregate evidence.
