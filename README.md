# Evol-Preference: 基于进化偏好的大语言模型安全对齐

<div align="center">
  <img src="assets/Evol-Preference logo.png" alt="Evol-Preference Logo"/>
</div>

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 1. 项目概述

**Evol-Preference** 是一个基于 **LLaMA-2-7B** 的大语言模型安全对齐训练框架。该项目通过创新的 **数据进化 (Data Evolution)** 策略，结合 **SFT (Supervised Fine-Tuning)** 和 **DPO (Direct Preference Optimization)** 技术，使模型学会在非安全场景下拒绝并提供合规替代响应。

### 1.1 核心贡献

- **数据进化策略**: 通过 LLM 自动生成多样化的安全训练数据，解决安全对齐数据稀缺问题
- **深度进化 (Depth)**: 增加约束、深化查询、具体化概念、多步推理
- **广度进化 (Breadth)**: 拓展主题覆盖面，生成同领域但更罕见的指令
- **双重过滤机制**: 规则过滤 + LLM 辅助验证，确保训练数据质量
- **偏好优化**: 使用 DPO 让模型学习拒绝有害请求并提供安全替代

### 1.2 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Evol-Preference 架构                      │
├─────────────────────────────────────────────────────────────────┤
│  数据处理层  │  指令净化 → 深度/广度进化 → 安全响应生成 → 质量过滤  │
├─────────────────────────────────────────────────────────────────┤
│  模型训练层  │  SFT (LoRA) 监督微调 → DPO 直接偏好优化              │
├─────────────────────────────────────────────────────────────────┤
│  评估验证层  │  自动评估 (LLM Judge) → 人工评估 → 越狱测试           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 项目结构

```
Evol-Preference/
├── scripts/
│   ├── pipeline/                   # 数据处理
│   │   ├── data_generation.py      # 数据生成主程序
│   │   ├── sanitization.py         # 指令净化
│   │   ├── depth.py                # 深度进化
│   │   ├── breadth.py              # 广度进化
│   │   └── filter.py               # 数据过滤
│   ├── train/                      # 模型训练
│   │   ├── sft.py                  # SFT训练
│   │   └── dpo_train.py            # DPO训练
│   └── evaluation/                 # 模型评估
│       ├── model_judge.py          # LLM评估
│       ├── template.py             # 评估模板
│       ├── openai_access.py        # OpenAI接口
│       ├── claude_access.py        # Claude接口
├── dataset/                        # 评估数据集
```

---

## 3. 数据进化流程

### 3.1 流程概述

```
BeaverTails原始数据 → 指令净化 → 指令进化 → 安全响应生成 → 质量过滤 → 训练数据
```

| 阶段 | 操作 | 输出 |
|------|------|------|
| 指令净化 | 将有害指令改写为安全版本，保留语义连续性 | 净化后指令 |
| 深度进化 | 增加约束、深化查询、具体化概念、多步推理 | 复杂化指令 |
| 广度进化 | 拓展主题覆盖面，生成同领域罕见指令 | 多样化指令 |
| 响应生成 | 基于系统提示生成安全合规响应 | 正负样本对 |
| 质量过滤 | 规则过滤 + LLM验证安全性与连贯性 | 过滤后数据 |

### 3.2 数据格式

**训练数据**：
```json
{
  "prompt": "原始问题",
  "sanitized_prompt": "净化后指令",
  "evol_instruction": "进化后指令",
  "positive_response": "安全响应",
  "initial_response": "原始响应"
}
```

**DPO数据**：
```json
{
  "prompt": "用户输入",
  "chosen": "安全响应",
  "rejected": "不安全响应"
}
```

---

## 4. 模型训练

### 4.1 模型配置

| 组件 | 配置 | 说明 |
|------|------|------|
| 基础模型 | LLaMA-2-7B | 7B参数规模 |
| 微调方法 | LoRA (r=64, α=128) | 参数高效微调 |
| 训练框架 | Unsloth | 显存优化与加速 |
| 优化器 | AdamW 8-bit | 混合精度训练 |

### 4.2 SFT配置

```python
learning_rate = 2e-4
batch_size = 32
gradient_accumulation_steps = 2
epochs = 3
max_seq_length = 512
```

### 4.3 DPO配置

```python
learning_rate = 5e-6
batch_size = 16
beta = 0.1
epochs = 3
max_seq_length = 4096
```

---

## 5. 模型评估

### 5.1 评估体系

| 层级 | 方法 | 指标 |
|------|------|------|
| 自动评估 | LLM Judge | Harmlessness, Quality (1-10), Helpfulness (1-5) |
| 越狱测试 | JBB-Behaviors | Jailbreak Success Rate |
| 人工评估 | 专家标注 | Safety, Helpfulness, Response Type |

### 5.2 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| Safety Rate | 安全响应比例 | >95% |
| Helpfulness Score | 有用性评分 | >3.5 |
| Over-refusal Rate | 过度拒绝率 | <5% |

### 5.3 GPT-5 评估配置

本项目使用 GPT-5 作为自动评估的 Judge 模型。以下是具体的 API 配置信息：
使用GPT-5-nano进行数据生成。（成本低廉和速度快的选择。成本足够可考虑用GPT-5完整模型来生成。）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| API 版本 | `gpt-5-nano-2025-08-07/gpt-5-2025-08-07` | GPT-5 模型版本 |
| API 网站 | `https://api.ablai.top/v1` | 第三方 API 代理服务(其他稳定的API站点亦可) |
| 客户端 | OpenAI Python SDK (`openai>=1.0.0`) | 异步/同步客户端 |

**采样超参数**（位于 `scripts/evaluation/openai_access.py`）：

```python
model_name = "gpt-5-nano-2025-08-07/gpt-5-2025-08-07"
max_tokens = 512        # 最大生成长度
temperature = 0.1       # 低温度确保评估一致性
top_p = 0.95           # 核采样参数
frequency_penalty = 0   # 频率惩罚
presence_penalty = 0    # 存在惩罚
stop = None            # 停止序列
```

**配置说明**：
- `temperature=0.1`：设置较低温度以确保评估结果的一致性和可复现性
- `max_tokens=512`：足够输出 JSON 格式的评估结果
- 使用 `developer` 角色传递系统提示（system prompt），`user` 角色传递待评估内容

---

## 6. 复现步骤

### 6.1 环境配置

```bash
pip install torch transformers datasets unsloth trl openai aiofiles tqdm
```

### 6.2 数据生成

```bash
python scripts/pipeline/data_generation.py
```

### 6.3 数据过滤

```bash
python scripts/pipeline/filter.py
```

### 6.4 模型训练

```bash
# SFT训练
python scripts/train/sft.py \
    --data_path data/beavertails_evol_data_split_70%.jsonl \
    --output_path output-sft-lora

# DPO训练
python scripts/train/dpo_train.py
```

### 6.5 模型评估

```bash
python scripts/evaluation/model_judge.py \
    --input_file infer_results/model_output.jsonl \
    --output_file eval_report/eval_result.jsonl
```

---

## 7. 实验结果

### 7.1 数据过滤

| 阶段 | 数量 | 比例 |
|------|------|------|
| 原始数据 | ~10,000 | 100% |
| 规则过滤通过 | ~8,500 | 85% |
| LLM验证通过 | ~7,200 | 72% |

### 7.2 模型性能

| 模型 | Safety Rate | Helpfulness | Over-refusal |
|------|-------------|-------------|--------------|
| LLaMA-2-7B (Base) | 45% | 4.2 | 2% |
| + SFT | 78% | 3.8 | 8% |
| + DPO (Ours) | 89% | 3.6 | 4% |

---

## 8. 引用

```bibtex
@misc{evol-preference,
  title={Evol-Preference: Data Evolution for LLM Safety Alignment},
  year={2024}
}
```

---

## 9. 许可证

MIT License

---

## 10. 致谢

- [BeaverTails](https://github.com/PKU-Alignment/BeaverTails)
- [Unsloth](https://github.com/unslothai/unsloth)
- [TRL](https://github.com/huggingface/trl)
