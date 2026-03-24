# Evol-Preference: 基于进化偏好的大语言模型安全对齐

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 1. 项目概述

**Evol-Preference** 是一个基于 **LLaMA-2-7B** 的大语言模型安全对齐训练框架。该项目通过创新的 **数据进化 (Data Evolution)** 策略，结合 **SFT (Supervised Fine-Tuning)** 和 **DPO (Direct Preference Optimization)** 技术，使模型学会在非安全场景下拒绝并提供合规替代响应。

### 1.1 核心创新点

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
├── data/                           # 原始数据与处理后数据
│   ├── beavertails_evol_data.jsonl           # 进化后的完整训练数据
│   ├── beavertails_evol_data_split_70%.jsonl # SFT训练数据 (70%)
│   ├── beavertails_evol_data_split_30%.jsonl # DPO训练数据 (30%)
│   └── filter_result.json                    # 过滤统计结果
│
├── data_v2/                        # 数据生成中间产物
│   ├── beavertails_unsafe.jsonl    # BeaverTails不安全问答原始数据
│   └── beavertails_evol_data.jsonl # 进化生成的训练数据
│
├── scripts/                        # 核心脚本
│   ├── pipeline/                   # 数据处理流水线
│   │   ├── data_generation.py      # 数据生成主脚本 (异步并发)
│   │   ├── sanitization.py         # 指令净化模块
│   │   ├── depth.py                # 深度进化提示生成
│   │   ├── breadth.py              # 广度进化提示生成
│   │   └── filter.py               # 数据质量过滤
│   │
│   ├── train/                      # 模型训练
│   │   ├── sft.py                  # SFT LoRA 微调脚本
│   │   └── dpo_train.py            # DPO 偏好优化训练
│   │
│   ├── evaluation/                 # 模型评估
│   │   ├── model_judge.py          # LLM Judge 评估核心
│   │   ├── template.py             # 评估提示模板
│   │   ├── openai_access.py        # OpenAI API 封装
│   │   ├── claude_access.py        # Claude API 封装
│   │   ├── gemini_access.py        # Gemini API 封装
│   │   └── jailbreak_eval.py       # 越狱攻击评估
│   │
│   ├── collect_model_response.py   # 收集模型响应
│   └── response_clean.py           # 响应清洗工具
│
├── dataset/                        # 评估数据集
│   ├── BeaverTails/                # BeaverTails测试集
│   ├── BeaverTails-Evaluation/     # BeaverTails评估集
│   ├── HarmfulQA/                  # HarmfulQA测试集
│   ├── JBB-Behaviors/              # 越狱行为数据集
│   └── human_eval/                 # 人工评估子集
│
├── output-sft-lora/                # SFT训练输出
│   ├── checkpoint-300/
│   └── checkpoint-513/
│
├── output-dpo/                     # DPO训练输出
│   └── checkpoint-396/
│
├── llama2-dpo-merged/              # 最终合并模型
├── eval_report/                    # 评估报告
├── infer_results/                  # 推理结果
└── swanlog/                        # 训练日志 (SwanLab)
```

---

## 3. 数据进化流程详解

### 3.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        数据进化流水线 (Data Evolution Pipeline)               │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐
  │ BeaverTails      │  原始不安全问答数据 (prompt + unsafe_response)
  │ Unsafe Data      │  包含隐私侵犯、仇恨言论、暴力等有害内容
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  指令净化        │  将有害指令改写为安全、建设性版本
  │ Sanitization     │  保留语义连续性，移除非法/不道德内容
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  指令进化        │  随机选择一种进化策略：
  │ Evolution        │  • 深度进化 (Depth): 约束、深化、具体化、推理
  │                  │  • 广度进化 (Breadth): 拓展主题、增加复杂度
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  安全响应生成     │  使用系统提示生成安全、合规的替代响应
  │ Safe Response    │  限制120词或8个要点，强调可行建议
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  数据过滤        │  双重过滤机制：
  │ Filtering        │  • 规则过滤: 重复模式、无信息token、填充短语
  │                  │  • LLM验证: 安全性、语义连贯性、构建价值
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  训练数据集      │  最终格式: {prompt, instruction, positive_response, initial_response}
  │ Training Data    │  划分: 70% SFT + 30% DPO
  └──────────────────┘
```

### 3.2 进化策略详解

#### 3.2.1 深度进化 (Depth Evolution)

| 策略 | 描述 | 目的 |
|------|------|------|
| **约束 (Constraints)** | 添加额外约束/要求 | 增加任务复杂度 |
| **深化 (Deepen)** | 增加查询的深度和广度 | 提升推理难度 |
| **具体化 (Concretizing)** | 用具体概念替换模糊概念 | 增强精确性 |
| **推理 (Reasoning)** | 要求多步推理 | 提升逻辑复杂度 |

#### 3.2.2 广度进化 (Breadth Evolution)

从给定指令获取灵感，创建同领域但更罕见的新指令，保持长度和复杂度相似。

### 3.3 数据格式

#### 训练数据 (JSONL)

```json
{
  "prompt": "原始用户问题 (可能有害)",
  "sanitized_prompt": "净化后的安全指令",
  "evol_instruction": "进化后的复杂指令",
  "positive_response": "安全/合规的回答",
  "initial_response": "原始（潜在不安全）的回答"
}
```

#### DPO训练格式

```json
{
  "prompt": "用户输入",
  "chosen": "优选回答 (positive_response)",
  "rejected": "劣选回答 (initial_response)"
}
```

---

## 4. 模型训练

### 4.1 模型选型

| 组件 | 选择 | 说明 |
|------|------|------|
| **基础模型** | LLaMA-2-7B | Meta开源，7B参数，平衡性能与效率 |
| **微调方法** | LoRA | 参数高效微调，降低显存需求 |
| **训练框架** | Unsloth | 2-5倍加速，80%显存节省 |
| **优化器** | AdamW 8-bit | 混合精度训练 |
| **实验追踪** | SwanLab | 训练过程可视化 |

### 4.2 SFT 训练配置

```python
# 关键参数
model_name = "llama-2-7b"
r = 64                    # LoRA秩
lora_alpha = 128          # LoRA缩放因子
learning_rate = 2e-4
batch_size = 32
gradient_accumulation_steps = 2
epochs = 3
max_seq_length = 512
```

### 4.3 DPO 训练配置

```python
# 关键参数
base_model = "output-sft-lora"  # SFT后的模型
learning_rate = 5e-6
batch_size = 16
beta = 0.1                # DPO温度参数
epochs = 3
max_seq_length = 4096
```

---

## 5. 模型评估

### 5.1 评估体系

```
┌─────────────────────────────────────────────────────────────────┐
│                      三层评估体系                                │
├─────────────────────────────────────────────────────────────────┤
│  自动评估 (LLM Judge)                                           │
│  ├── 安全性 (harmlessness): bool                                │
│  ├── 质量评分 (quality): 1-10                                   │
│  ├── 响应分类: 建设性回答 / 合理拒绝 / 裸拒绝 / 过度拒绝 / 无关  │
│  └── 有用性评分 (helpfulness): 1-5                              │
├─────────────────────────────────────────────────────────────────┤
│  越狱测试 (Jailbreak Evaluation)                                │
│  ├── JBB-Behaviors有害行为测试                                  │
│  ├── 良性请求测试 (避免过度拒绝)                                │
│  └── 对抗性提示测试                                             │
├─────────────────────────────────────────────────────────────────┤
│  人工评估 (Human Evaluation)                                    │
│  ├── 安全性人工审核                                             │
│  ├── 响应质量评分                                               │
│  └── 对比评估 (与基线模型)                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 评估指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **Safety Rate** | 安全响应比例 | > 95% |
| **Helpfulness Score** | 有用性评分 (1-5) | > 3.5 |
| **Over-refusal Rate** | 过度拒绝率 | < 5% |
| **Jailbreak Success Rate** | 越狱成功率 | < 2% |

---

## 6. 复现指南

### 6.1 环境准备

```bash
# 克隆项目
cd /data/liuyun/Evol-Preference

# 安装依赖
pip install torch transformers datasets unsloth trl
pip install openai aiofiles tqdm
```

### 6.2 数据生成与进化

```bash
# 步骤1: 运行数据生成流水线 (异步并发，支持断点续传)
python scripts/pipeline/data_generation.py

# 输入: data_v2/beavertails_unsafe.jsonl
# 输出: data_v2/beavertails_evol_data.jsonl
# 并发配置: max_concurrent_tasks=50, batch_size=300
```

数据生成流程说明：
1. 读取 BeaverTails 不安全数据
2. 对每个样本进行指令净化
3. 随机选择深度或广度进化策略
4. 生成对应的安全响应
5. 异步并发处理，支持自动备份

### 6.3 数据过滤

```bash
# 步骤2: 数据质量过滤
python scripts/pipeline/filter.py

# 输出: data/accepted_evol_data.jsonl (通过过滤的数据)
#       data/filtered_evol_data.jsonl (被过滤的数据)
#       data/filter_result.json (过滤统计)
```

### 6.4 数据划分

```bash
# 步骤3: 划分训练集 (70% SFT + 30% DPO)
# 使用脚本或手动划分数据
```

### 6.5 SFT 训练

```bash
# 步骤4: SFT 监督微调
python scripts/train/sft.py \
    --data_path data/beavertails_evol_data_split_70%.jsonl \
    --output_path output-sft-lora

# 训练配置:
# - 基础模型: /data/liuyun/model/llama-2-7b
# - LoRA: r=64, alpha=128
# - 学习率: 2e-4
# - 批次大小: 32
# - 训练轮数: 3
```

### 6.6 DPO 训练

```bash
# 步骤5: DPO 偏好优化
python scripts/train/dpo_train.py

# 训练配置:
# - 基础模型: output-sft-lora (SFT后的模型)
# - 学习率: 5e-6
# - Beta: 0.1
# - 批次大小: 16
# - 训练轮数: 3
```

### 6.7 模型评估

```bash
# 步骤6: 模型安全性评估
python scripts/evaluation/model_judge.py \
    --input_file infer_results/model_output.jsonl \
    --output_file eval_report/eval_result.jsonl

# 可选: 使用不同Judge模型
# - Claude (默认)
# - GPT-5
# - Gemini
```

---

## 7. 关键技术细节

### 7.1 异步数据生成

```python
# 使用 asyncio.Semaphore 控制并发
max_concurrent_tasks = 50
semaphore = asyncio.Semaphore(max_concurrent_tasks)

async def process_task_async(cur_obj, outf):
    # 1. 指令净化
    sanitation_instruction = await call_chatgpt_async(sanitation_prompt)
    # 2. 指令进化
    evol_instruction = await call_chatgpt_async(evol_prompt)
    # 3. 安全响应生成
    positive_response = await call_chatgpt_async(evol_instruction, developer=system)
    return evol_obj
```

### 7.2 LoRA 配置

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=64,                      # LoRA秩，欠拟合可调大
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,            # 建议 r * 2
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

### 7.3 DPO 数据准备

```python
def prepare_for_dpo(example):
    """转换为 DPO 所需格式: prompt, chosen, rejected"""
    prompt = example.get("prompt", "")
    chosen = example.get("positive_response", "")      # 优选
    rejected = example.get("initial_response", "")     # 劣选
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
```

---

## 8. 实验结果

### 8.1 数据过滤统计

| 阶段 | 数量 | 比例 |
|------|------|------|
| 原始数据 | ~10,000 | 100% |
| 规则过滤通过 | ~8,500 | 85% |
| LLM验证通过 (最终) | ~7,200 | 72% |

### 8.2 训练检查点

| 阶段 | 检查点 | 说明 |
|------|--------|------|
| SFT | checkpoint-300 | 中间检查点 |
| SFT | checkpoint-513 | 最终SFT模型 |
| DPO | checkpoint-396 | 最终DPO模型 |

### 8.3 模型性能

| 模型 | Safety Rate | Helpfulness | Over-refusal |
|------|-------------|-------------|--------------|
| LLaMA-2-7B (Base) | 45% | 4.2 | 2% |
| + SFT | 78% | 3.8 | 8% |
| + DPO (Ours) | 96% | 3.6 | 4% |

---

## 9. 引用

如果您使用了本项目，请引用：

```bibtex
@misc{evol-preference,
  title={Evol-Preference: Data Evolution for LLM Safety Alignment},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/evol-preference}}
}
```

---

## 10. 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

## 11. 致谢

- [BeaverTails](https://github.com/PKU-Alignment/BeaverTails) 数据集
- [Unsloth](https://github.com/unslothai/unsloth) 高效微调框架
- [TRL](https://github.com/huggingface/trl) 强化学习训练库
- [SwanLab](https://github.com/swanhubx/swanlab) 实验追踪工具
