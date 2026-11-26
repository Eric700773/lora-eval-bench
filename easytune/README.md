# EasyTune 微调傻瓜包

EasyTune 是围绕现有微调框架的产品化控制台。推荐部署方式是前后端分离：前端可以部署到阿里云等公网服务器，后端部署在本机 GPU 机器上，负责数据、训练、推理和模型文件访问。

当前已在 RTX 4060 8GB、Python 3.12 和本地 Qwen3.5-2B 上跑通无 Mock 链路：

```text
Dataset upload -> decontaminated benchmark -> dynamic perturbation
-> clean badcase attribution -> DeepSeek synthesis and review
-> NF4 QLoRA -> adapter regression -> JSON/Markdown report
```

实验性平台能力包括原始/去污染分数、Bootstrap 95% CI、能力切片、配对动态变体、合成数据反污染、真实 adapter 注册与同口径回归。当前仓库中的 30 题、6 条合成样本和 1 epoch 训练用于端到端工程验证，不代表统计显著的通用能力提升。

## 模型方向

第一版只预留两个模型方向：

- `qwen`：按 LLaMA-Factory 训练配置生成 YAML。优先检测系统 `llamafactory-cli`，没有时使用内置 `backend/app/vendor/llamafactory`。
- `rwkv`：按 RWKV-PEFT 生成训练数据和配置。优先检测系统 `rwkv-finetune`，没有时使用内置 `backend/app/vendor/rwkv_peft/train.py`。当前支持 LoRA、QLoRA、State Tuning、Miss/DiSHA、AdaLoRA、Prefix 和全量训练入口。默认 `data_type=sft`；离线 smoke 或不想走 HuggingFace tokenizer 时可选 `data_type=jsonl`，平台会生成 RWKV-PEFT 原生 `{"text": ...}` 数据。
- `rwkv_lightning`：内置 `backend/app/vendor/rwkv_lightning`，用于启动本机 RWKV 推理服务。

当前已经有基础前置检查、runner 检测、GPU 提示和失败摘要。后续可以继续增强：

- Qwen 实习生：继续扩展不同 qwen 尺寸的模板、显存估算和失败解析规则。
- RWKV 实习生：继续完善不同模型尺寸的默认参数、显存估算、GPU 真训 smoke 和训练后推理合并闭环。

## 目录结构

```text
easytune/
  backend/
    app/
      routers/
      services/
      storage/
      vendor/
        llamafactory/
        rwkv_lightning/
        rwkv_peft/
  frontend/
  docker-compose.yml
  README.md
```

`storage/` 会自动创建并保存上传文件、转换数据、配置、输出、日志、报告和模型 registry。仓库只保留 `.gitkeep`，运行生成物不会提交。

## 本地后端启动

```bash
cd easytune/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

真实训练/推理机器需要额外安装可选依赖：

```bash
pip install -r requirements-ml.txt
```

进入「Deploy / Inference」可以查看 `RWKV deps`。如果缺 `torch`、`lightning`、`peft`、`rwkv`、`jsonlines` 等模块，RWKV-PEFT 真实训练会在启动前失败并提示缺失依赖；这是为了避免把 mock 或半启动状态误认为正式训练。

RWKV-PEFT 默认仍可使用 DeepSpeed fused optimizer；小模型 smoke 或不希望编译 DeepSpeed optimizer 时，在创建训练任务里把 `optimizer` 设为 `adamw`。

健康检查：

```bash
curl http://localhost:8000/api/health
```

## Qwen3.5 评测飞轮

当前验证配置：

```text
Windows backend: Python 3.12
WSL GPU worker: PyTorch 2.10.0+cu126 / Transformers 5.9.0 / PEFT 0.21.3
GPU: NVIDIA GeForce RTX 4060 Laptop GPU 8GB
Base model: D:\interview\local-llm-validation\models\qwen3.5-2b-bf16
Synthesis model: DeepSeek official deepseek-v4-pro
```

先启动 WSL GPU worker：

```powershell
wsl.exe -d Ubuntu -- /mnt/d/interview/local-llm-validation/.venv-transformers/bin/python `
  /mnt/c/Users/Eric/lora-eval-bench/easytune/backend/app/workers/qwen_wsl_worker.py `
  --model-path /mnt/d/interview/local-llm-validation/models/qwen3.5-2b-bf16 `
  --host 0.0.0.0 --port 8011
```

正式数据合成需要在本机 User 环境变量中配置：

```text
EASYTUNE_SYNTHESIS_BASE_URL=https://api.deepseek.com
EASYTUNE_SYNTHESIS_API_KEY=<DeepSeek API Key>
EASYTUNE_SYNTHESIS_MODEL=deepseek-v4-pro
EASYTUNE_REQUIRE_EXTERNAL_SYNTHESIS=true
```

API Key 不应写入仓库。后端提供以下安全预检：

```text
GET  /api/flywheel/synthesis/config
GET  /api/flywheel/synthesis/models
POST /api/flywheel/synthesis/test
```

验证流程：

1. 上传 Benchmark 和参考训练集。
2. `POST /api/benchmarks/runs` 运行原题和动态变体基线。
3. `POST /api/flywheel/analyze-and-synthesize` 进行归因、DeepSeek 合成、质量门禁和反污染。
4. `POST /api/flywheel/{iteration_id}/train` 执行真实 NF4 QLoRA。
5. `POST /api/flywheel/{iteration_id}/regression-eval` 使用真实 adapter 重跑同口径回归。
6. `POST /api/flywheel/{iteration_id}/report` 输出 JSON 和 Markdown 报告。

本次验证产物：

```text
Baseline original: Run 2
Baseline dynamic: Run 3
Adapter original regression: Run 4
Adapter dynamic regression: Run 5
Flywheel iteration: 8
Training job: 1
ModelVersion: 1
Synthesized dataset: 12
Report: app/storage/reports/flywheel_iteration_8.{json,md}
```

结果摘要：原题 clean Pass@1 `0.8929 -> 0.9286`，动态变体 clean Pass@1 `0.8214 -> 0.8571`。两组 95% CI 均有明显重叠，因此只能声明观察到小样本提升，不能声明统计显著。


可选安全配置：

```bash
export EASYTUNE_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
export EASYTUNE_API_TOKEN=change-this-token
```

设置 `EASYTUNE_API_TOKEN` 后，上传数据、创建/启动训练、保存验收记录等写接口需要 `Authorization: Bearer <token>`。前端可以在「Deploy / Inference」页面保存该 token。

可选推理验收环境变量：

```bash
# RWKV Lightning, from built-in vendor
cd easytune/backend/app/vendor/rwkv_lightning
python app.py --model-path /path/to/rwkv-model.pth --port 8001 --password optional-token

export EASYTUNE_INFERENCE_PROVIDER=rwkv_lightning
export EASYTUNE_INFERENCE_BASE_URL=http://127.0.0.1:8001
export EASYTUNE_INFERENCE_API_KEY=optional-token

# Qwen local Transformers
export EASYTUNE_INFERENCE_PROVIDER=qwen_transformers
export EASYTUNE_QWEN_BASE_MODEL=/path/to/qwen-base

# OpenAI-compatible/vLLM/Open WebUI
export EASYTUNE_INFERENCE_PROVIDER=openai_compatible
export EASYTUNE_INFERENCE_BASE_URL=http://localhost:8001/openai/v1
export EASYTUNE_INFERENCE_MODEL=easytune-served-model
export EASYTUNE_INFERENCE_API_KEY=optional-token

# 或 Ollama
export EASYTUNE_INFERENCE_PROVIDER=ollama
export EASYTUNE_INFERENCE_BASE_URL=http://localhost:11434
export EASYTUNE_INFERENCE_MODEL=qwen2.5:7b
```

## 本地前端启动

```bash
cd easytune/frontend
npm install
npm run dev
```

打开：

```text
http://localhost:3000
```

前端部署到阿里云时：

- 构建时可设置 `NEXT_PUBLIC_API_BASE_URL=http://你的本机后端地址:8000`。
- 也可以进入前端页面「Deploy / Inference」，把后端 API 地址保存到当前浏览器。
- 后端默认开放 CORS，实际公网部署建议在反向代理层限制来源和访问令牌。

## Docker Compose 启动

```bash
cd easytune
docker compose up --build
```

访问：

```text
前端：http://localhost:3000
后端：http://localhost:8000
```

## 页面设计

| 页面 | 内容 |
| --- | --- |
| Dashboard | GPU 状态、最近训练任务、失败任务、模型数量、数据集数量 |
| Dataset | 上传 JSONL/CSV、格式检查、样本预览、train/valid 切分、错误样本提示 |
| Create Fine-tune | base model、训练方式、RWKV-PEFT 参数、模板选择、epoch/lr/batch/output |
| Training Monitor | 任务进度、日志、命令、配置、输出目录，后续接 loss 曲线和停止/恢复 |
| Evaluation | 去污染 benchmark、动态变体、Badcase、飞轮状态、base vs adapter 回归和报告 |
| Chat Playground | 调用本机推理后端，做问答对比 |
| Model Registry | checkpoint、adapter、export、版本说明 |
| Deploy / Inference | 前端 API 地址、后端健康、vLLM/Ollama/Transformers/RWKV Lightning 配置 |

## 使用流程

1. 进入「数据集管理」，上传 `.json`、`.jsonl` 或 `.csv`。
2. 查看质检报告，包括空文件、行数、非空行数、空行、行长度、格式识别和质量分。
3. 点击「转换」，生成 `storage/llamafactory_data/dataset_{id}.jsonl` 和 `dataset_info.json`。
4. 进入「创建训练任务」，选择已转换数据集，选择 `qwen` 或 `rwkv`。
5. 提交后查看任务详情页，可见 `command`、`config_path`、`log_path`、`output_dir`。
6. 点击启动。正式模式找不到真实 runner 时会直接失败，不会静默进入 Mock；仅显式设置 `EASYTUNE_ALLOW_MOCK_TRAINING=true` 才允许测试用 Mock。
7. 训练完成后自动生成 ModelVersion、export manifest 和 `storage/registry/model_versions.json`。RWKV-PEFT 的 `rwkv-*.pth` 和同级 `job_x-adapter/` 目录会被识别为模型产物。
8. 进入「人工验收」，选择模型版本；可手填实际答案，也可在配置推理服务后生成实际答案并保存人工评测记录。

## API

```text
POST /api/datasets/upload
GET  /api/datasets
GET  /api/datasets/{id}
POST /api/datasets/{id}/convert
GET  /api/datasets/{id}/dataset-info

POST /api/training-jobs
GET  /api/training-jobs
GET  /api/training-jobs/{id}
POST /api/training-jobs/{id}/start
GET  /api/training-jobs/{id}/logs

GET  /api/model-versions
GET  /api/model-versions/{id}

POST /api/validation-records
POST /api/validation-records/generate
GET  /api/validation-records

GET  /api/health
```

## 后续接入点

- `backend/app/services/dataset_service.py`：继续增强数据清洗、字段映射、采样预览和质检规则。
- `backend/app/services/llamafactory_config_service.py`：继续扩展 qwen/RWKV 不同训练配置模板和默认参数。
- `backend/app/services/training_service.py`：继续增强中断、重试和更细的失败解析。
- `backend/app/services/model_service.py`：后续可接 MLflow 或远端模型仓库。
- `backend/app/services/inference_service.py`：已经支持 OpenAI-compatible/vLLM/Open WebUI/Ollama，后续可加更多推理后端。

## LLaMA-Factory 参考目录

仓库根目录的 `LlamaFactory/` 只作为本地代码参考，不上传 GitHub；根 `.gitignore` 已忽略该目录。
EasyTune 不 import 或读取这个本地目录；qwen 真实训练只检测系统 PATH 里的 `llamafactory-cli`，没有 CLI 时自动走 mock train。
