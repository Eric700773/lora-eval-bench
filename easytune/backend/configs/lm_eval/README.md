# Frozen lm-eval Environment

This directory freezes the first official-framework benchmark supported by EasyTune.

## Framework

- Package: `lm-eval`
- Version: `0.4.13`
- Git tag: `v0.4.13`
- Git commit: `ddd67220430a2470529f25fd5c05a576ca1057a0`
- Python: `3.12.13`
- Environment location used during validation: `$HOME/.virtualenvs/easytune-lm-eval`

The environment is intentionally separate from the FastAPI backend and the Qwen GPU runtime.

## Task

- Task ID: `gsm8k`
- Task source: `lm_eval/tasks/gsm8k/gsm8k.yaml`
- Output type: `generate_until`
- Dataset: `openai/gsm8k`, config `main`
- Training/few-shot split: `train`
- Evaluation split: `test`
- Few-shot examples: `5`
- Official strict and flexible extraction filters are retained unchanged.

## Frozen files

- `gsm8k-v0.4.13.yaml`: exact installed Task YAML snapshot
- `gsm8k-v0.4.13.manifest.json`: framework, dataset revision and split fingerprints
- `requirements-lm-eval-0.4.13.lock.txt`: exact environment package versions

## Recreate the environment

```bash
mkdir -p ~/.virtualenvs
uv venv --python 3.12 ~/.virtualenvs/easytune-lm-eval
uv pip install \
  --python ~/.virtualenvs/easytune-lm-eval/bin/python \
  "lm-eval[api]==0.4.13"
```

Validate the installed official Task:

```bash
~/.virtualenvs/easytune-lm-eval/bin/lm-eval validate --tasks gsm8k
~/.virtualenvs/easytune-lm-eval/bin/lm-eval ls tasks | grep gsm8k
```

The PyPI wheel does not ship the repository test root required by the runtime `--check_integrity` option. EasyTune therefore validates the installed Task explicitly, freezes its SHA256, verifies the offline Task protocol field-by-field, and checks sample/metric parity after execution instead of using that unavailable flag.

The dataset revision is frozen in the manifest. A shared Hugging Face cache may be pre-populated from the Windows host when WSL networking is unavailable.
