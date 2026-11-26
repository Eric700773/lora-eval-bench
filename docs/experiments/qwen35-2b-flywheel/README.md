# Qwen3.5-2B Evaluation Flywheel Experiment

This directory contains sanitized evidence from an end-to-end, no-mock engineering validation of the EasyTune evaluation flywheel.

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- Backend: Python 3.12
- GPU runtime: PyTorch 2.10.0 + CUDA 12.6
- Transformers: 5.9.0
- PEFT: 0.21.3
- Quantization: bitsandbytes NF4 4-bit
- Base model: local Qwen3.5-2B BF16 weights
- Synthesis model: DeepSeek official `deepseek-v4-pro`

No model weights, adapter weights, API keys, databases, user paths, or runtime logs are included in this evidence package.

## Evaluation Protocol

The static benchmark contains 30 items:

- 10 numerical reasoning tasks
- 10 JSON instruction-following tasks
- 10 Python function tasks with executable checks

A five-item reference training corpus contains two independently contaminated evaluation questions, one additional paraphrase of an already contaminated question, and two clean examples. Contamination is counted per evaluation item, so the expected contaminated count is two.

The experiment compares four runs:

| Run | Model | Prompt mode |
|---|---|---|
| Baseline original | Base Qwen3.5-2B | Original benchmark |
| Baseline dynamic | Base Qwen3.5-2B | Entity/distractor perturbation |
| Post original | Base + QLoRA adapter | Original benchmark |
| Post dynamic | Base + QLoRA adapter | Entity/distractor perturbation |

All four runs use the same benchmark, clean subset definition, chat template, generation parameters, and deterministic verifiers. The only intended model difference is the trained adapter.

## Data Flywheel

Three clean baseline failures were attributed as:

- calculation error: 1
- JSON format/instruction error: 1
- code error: 1

DeepSeek generated candidate remediation samples. Candidates passed:

1. deterministic JSON or Python AST validation;
2. mathematical answer review where applicable;
3. reverse contamination checks against the full evaluation set;
4. standard Alpaca JSONL conversion and dataset registration.

Six accepted examples were used for the training validation: two numerical, two JSON, and two Python examples.

## Real QLoRA Training

- Mock training: disabled
- Training samples: 6
- Epochs: 1
- Optimizer steps: 6
- LoRA rank: 8
- Maximum length: 512
- Learning rate: 2e-4
- First loss: 5.3008
- Last loss: 0.3331
- Peak allocated GPU memory: 3.69 GiB
- Adapter artifact: real `adapter_model.safetensors`, 2,956,152 bytes

## Results

| Evaluation | Baseline clean Pass@1 | Post clean Pass@1 | Delta | Baseline 95% CI | Post 95% CI |
|---|---:|---:|---:|---|---|
| Original | 0.8929 | 0.9286 | +0.0357 | [0.7857, 1.0000] | [0.8214, 1.0000] |
| Dynamic | 0.8214 | 0.8571 | +0.0357 | [0.6786, 0.9643] | [0.7500, 0.9643] |

Original clean category deltas:

- numerical reasoning: +0.1111
- JSON instruction: 0.0000
- code: 0.0000

Dynamic clean category deltas:

- numerical reasoning: -0.1111
- JSON instruction: 0.0000
- code: +0.2000

The paired clean transitions show both improvements and regressions. This is important evidence that a single aggregate gain does not imply uniform capability improvement.

## Decontamination Calibration

On the available 36-case calibration set:

- true positive: 2
- false negative: 0
- false positive: 0
- true negative: 34
- precision: 1.0000
- recall: 1.0000

This only establishes behavior on the included small calibration set. It is not a claim of universal perfect contamination detection.

## Claim Boundary

This is a 30-item benchmark and a six-sample, one-epoch engineering validation. The confidence intervals overlap substantially. The experiment demonstrates a reproducible no-mock system and an observed small-sample score increase; it does not establish statistically significant general capability improvement.

See `metrics.json` and `training-report.json` for machine-readable evidence.
