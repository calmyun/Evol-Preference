# Evol-Preference

## Overview

Evol-Preference 是一个基于 LLaMA-2-7B 的大语言模型安全对齐训练项目。该项目通过 **SFT (Supervised Fine-Tuning)** 和 **DPO (Direct Preference Optimization)** 技术，让模型学会在非安全场景下拒绝并提供合规替代响应。

## Project Type

Python (PyTorch + Transformers + Unsloth)

## Key Features

- **数据进化 (Data Evolution)**: 通过 LLM 生成多样化的安全训练数据
- **SFT 微调**: 使用 LoRA 技术对 LLaMA-2-7B 进行监督微调
- **DPO 训练**: 使用直接偏好优化进一步提升模型的安全性
- **数据过滤**: 规则过滤 + LLM 辅助验证，确保训练数据质量
- **模型评估**: 使用 LLM 作为 Judge 评估模型输出的安全性和质量

## Technology Stack

- **深度学习框架**: PyTorch
- **大模型框架**: Unsloth (高效微调)
- **训练框架**: TRL (Transformer Reinforcement Learning)
- **基础模型**: LLaMA-2-7B
- **数据处理**: Datasets, JSONL
- **API 集成**: OpenAI API, Claude API
- **实验追踪**: SwanLab

## Usage

### 1. 数据处理流水线 (Pipeline)

#### 1.1 数据生成与进化

```bash
python scripts/pipeline/data_generation.py
```

该脚本会:
- 读取 `data_v2/beavertails_unsafe.jsonl` 中的不安全问答
- 使用 LLM 对指令进行净化
- 生成多种类型的进化指令 (约束、深化、具体化、推理、广度)
- 生成对应的安全响应
- 输出到 `data_v2/beavertails_evol_data.jsonl`

#### 1.2 数据过滤

```bash
python scripts/pipeline/filter.py
```

过滤低质量训练数据:
- 规则过滤: 检测重复、无信息量、填充短语
- LLM 辅助验证: 安全性、语义连贯性、构建价值

### 2. 模型训练 (Train)

#### 2.1 SFT 微调

```bash
python scripts/train/sft.py
```

使用 LoRA 技术对 LLaMA-2-7B 进行监督微调:
- 基础模型: `/data/liuyun/model/llama-2-7b`
- 训练数据: `data/beavertails_evol_data_split_70%.jsonl`
- 输出目录: `output-sft-lora`
- 检查点: checkpoint-300, checkpoint-513

#### 2.2 DPO 训练

```bash
python scripts/train/dpo_train.py
```

使用直接偏好优化进一步训练:
- 基础模型: `output-sft-lora` (SFT 后的 LoRA 模型)
- 训练数据: `data/beavertails_evol_data_split_30%.jsonl`
- 输出目录: `output-dpo`
- 检查点: checkpoint-396

### 3. 模型评估 (Evaluation)

```bash
bash scripts/evaluation/eval.sh
```

或直接运行:

```bash
python scripts/evaluation/model_safety_evaluation.py
```

使用 LLM Judge 评估模型输出:
- 安全性 (harmlessness)
- 质量 (quality, 1-10)
- 类别标签 (harmful_content, refusal, empty_response 等)

## Data Format

### 训练数据格式 (JSONL)

```json
{
  "prompt": "用户原始问题",
  "instruction": "进化后的指令",
  "positive_response": "安全/合规的回答",
  "initial_response": "原始（潜在不安全）的回答"
}
```

### DPO 训练格式

- `prompt`: 用户输入
- `chosen`: 优选回答 (positive_response)
- `rejected`: 劣选回答 (initial_response)

## Training Pipeline

```
┌─────────────────────┐
│  BeaverTails 数据    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   数据生成/进化      │  (scripts/pipeline/data_generation.py)
│  - 指令净化         │  (scripts/pipeline/sanitization.py)
│  - 指令进化         │  (scripts/pipeline/depth.py, breadth.py)
│  - 安全响应生成     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    数据过滤         │  (scripts/pipeline/filter.py)
│  - 规则过滤         │
│  - LLM 验证         │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌───────┐   ┌───────┐
│ SFT   │   │ DPO   │
│ 70%   │   │ 30%   │
└───┬───┘   └───┬───┘
    │           │
    ▼           ▼
┌───────┐   ┌───────┐
│SFT模型│   │DPO模型 │
└───────┘   └───────┘
          │
          ▼
┌─────────────────────┐
│    模型评估         │  (scripts/evaluation/model_safety_evaluation.py)
└─────────────────────┘
```

## Key Scripts

### Pipeline (数据处理)

| 脚本 | 功能 |
|------|------|
| `pipeline/data_generation.py` | 生成进化后的安全训练数据 |
| `pipeline/filter.py` | 数据质量过滤 |
| `pipeline/sanitization.py` | 指令净化 |
| `pipeline/depth.py` | 深度进化提示 (约束、深化、具体化、推理) |
| `pipeline/breadth.py` | 广度进化提示 |

### Train (模型训练)

| 脚本 | 功能 |
|------|------|
| `train/sft.py` | SFT LoRA 微调 |
| `train/dpo_train.py` | DPO 偏好优化训练 |

### Evaluation (模型评估)

| 脚本 | 功能 |
|------|------|
| `evaluation/model_safety_evaluation.py` | 模型安全性评估 |
| `evaluation/eval.sh` | 评估执行脚本 |

## Additional Notes

- 项目使用 LoRA 进行参数高效微调，降低显存需求
- 数据划分: 70% 用于 SFT，30% 用于 DPO
- 支持异步 API 调用，提高数据生成效率
- 训练过程使用 SwanLab 进行实验追踪
- 脚本已按功能模块化: `pipeline/`、`train/`、`evaluation/`