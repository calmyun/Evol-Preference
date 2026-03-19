#!/usr/bin/env python3
# train_dpo_safe.py
"""
DPO 训练脚本 —— 基于用户提供的偏好数据（jsonl），目标：让模型学会
在非安全场景下拒绝并提供合规替代（训练依赖于正样本 positive_response）。
"""

import os
import json
from typing import Dict

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig

# ----------------------------
# Config (按需修改)
# ----------------------------
DATA_FILE = "/data/liuyun/Evol-Preference/data/beavertails_evol_data_split_30%.jsonl"
OUTPUT_DIR = "/data/liuyun/Evol-Preference/output-dpo"
MODEL_NAME_OR_PATH = "/data/liuyun/Evol-Preference/output-sft-lora" 
MAX_SEQ_LENGTH = 4096
MAX_PROMPT_LENGTH = 512
MAX_GEN_LENGTH = 1024
TRAIN_SAMPLE_FRACTION = 1.0  # 若数据大，可设置小一点
BATCH_SIZE = 16
GRAD_ACC_STEPS = 2
NUM_EPOCHS = 3
LEARNING_RATE = 5e-6
BETA = 0.1  # DPO beta 超参，按需调整
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 小工具：加载 jsonl 并转换为 HF Dataset
# ----------------------------
def load_jsonl_as_dataset(path: str, sample_frac: float = 1.0) -> Dataset:
    # 每行是一个 JSON 对象,sample_frac是采样比例
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"warning: skip line {i} in {path}: {e}")
                continue
            records.append(obj)
    if sample_frac < 1.0:
        cutoff = max(1, int(len(records) * sample_frac))
        records = records[:cutoff]
    return Dataset.from_list(records)


# ----------------------------
# 转换为 DPO 所需字段： prompt, chosen, rejected
# ----------------------------
def prepare_for_dpo(example: Dict) -> Dict:
    """
    输入 example 预期包含:
      - prompt: 用户原始问句（可选）
      - instruction: 重写或指令文本（可选）
      - positive_response: 安全 / 合规的改写（必需）
      - initial_response: 原始（潜在不安全）的回答（必需）
    输出：
      - prompt (str): 文本提示给模型
      - chosen (str): 优选回答（positive_response）
      - rejected (str): 劣选回答（initial_response）
    """

    # prompt = example.get("instruction") or example.get("prompt") or ""
    prompt = example.get("prompt") or ""
    chosen = example.get("positive_response") or ""
    rejected = example.get("initial_response") or ""

    # 基本清洗：去除多余的空行
    def clean(s: str) -> str:
        return "\n".join([ln.rstrip() for ln in s.strip().splitlines() if ln.strip()])

    prompt = clean(prompt)
    chosen = clean(chosen)
    rejected = clean(rejected)

    # 若缺失必要字段，则抛弃（DPO 需要 prompt/chosen/rejected）
    if not prompt or not chosen or not rejected:
        return {}

    # 截断超长（按 token 近似用字符长度）
    max_prompt_chars = MAX_PROMPT_LENGTH * 4
    max_gen_chars = MAX_GEN_LENGTH * 4
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars]
    if len(chosen) > max_gen_chars:
        chosen = chosen[:max_gen_chars]
    if len(rejected) > max_gen_chars:
        rejected = rejected[:max_gen_chars]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

# ----------------------------
# 主流程
# ----------------------------
def main():
    torch.manual_seed(SEED)

    # 1) 载入原始 jsonl 数据为 Dataset
    print("Loading jsonl dataset from:", DATA_FILE)
    raw_ds = load_jsonl_as_dataset(DATA_FILE, sample_frac=TRAIN_SAMPLE_FRACTION)
    print(f"Loaded {len(raw_ds)} raw records.")

    # 2) map -> 转换为 DPO 所需字段并过滤无效样本
    print("Preparing dataset for DPO (mapping to prompt/chosen/rejected)...")
    mapped = raw_ds.map(
        lambda ex: prepare_for_dpo(ex),
        remove_columns=raw_ds.column_names,
    )
    # 过滤掉 prepare_for_dpo 返回为空 dict 的样本
    mapped = mapped.filter(lambda ex: bool(ex.get("prompt") and ex.get("chosen") and ex.get("rejected")))
    print(f"After filtering, {len(mapped)} usable samples.")

    # 3) 划分 train/test（简单按 90/10）
    split = mapped.train_test_split(test_size=0.1, seed=SEED)
    datasets = DatasetDict({"train": split["train"], "test": split["test"]})
    print("Train size:", len(datasets["train"]), "Test size:", len(datasets["test"]))

    # 4) 初始化 tokenizer 和模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME_OR_PATH,
        load_in_4bit = False,   # 单卡80G，可以直接fp16/bf16跑
        dtype = torch.bfloat16,
        max_seq_length = MAX_SEQ_LENGTH
    )

    # Lora配置，若已有lora则直接使用现有配置
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                     "gate_proj", "up_proj", "down_proj",],
    #     lora_alpha = 64,
    #     lora_dropout = 0, # Currently only supports dropout = 0
    #     bias = "none",    # Currently only supports bias = "none"
    #     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    #     random_state = 3407,
    #     use_rslora = False,  # We support rank stabilized LoRA
    #     loftq_config = None, # And LoftQ
    # )

    # print("Loading tokenizer and model:", MODEL_NAME_OR_PATH)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    # # Ensure tokenizer has eos token
    # if tokenizer.eos_token is None:
    #     tokenizer.add_special_tokens({"eos_token": ""})

    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME_OR_PATH,
    #     device_map="auto" if torch.cuda.is_available() else None,
    #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    #     low_cpu_mem_usage=True,
    # )

    # # 如果添加了新 token（如 eos），resize
    # model.resize_token_embeddings(len(tokenizer))

    # 5) 为 DPOTrainer 准备数据：DPOTrainer 接受普通的 HF Dataset 包含字段 prompt/chosen/rejected（文本）
    #    （trl 的 DPOTrainer 在内部会对 prompt/chosen/rejected 进行 tokenization）
    # 我们不再在此手动 tokenize，直接传 dataset

    # 6) 初始化 DPOTrainer
    print("初始化 DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # None 表示 TRL 会在内部复制一份作为 reference（或你可以显式提供）
        args=DPOConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            warmup_ratio=0.1,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.0,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir=OUTPUT_DIR,
            report_to="swanlab",
        ),
        beta=BETA,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        max_length=MAX_GEN_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )

    # 8) 训练
    print("开始DPO 训练！")
    dpo_trainer.train()

    # 9) 保存模型和 tokenizer
    print("保存模型至:", OUTPUT_DIR)
    dpo_trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()