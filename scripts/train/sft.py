from unsloth import FastLanguageModel
from trl import SFTTrainer,SFTConfig
from datasets import load_dataset
import torch
import argparse

argparser = argparse.ArgumentParser("sft")
argparser.add_argument("--data_path", type=str, required=True)
argparser.add_argument("--output_path", type=str, required=True)
args = argparser.parse_args()


# 1. 参数配置
model_name = "/data/liuyun/model/llama-2-7b"
# data_path = "/data/liuyun/Evol-Preference/data/beavertails_evol_data_split_70%.jsonl"
data_path = args.data_path
output_dir = args.output_path

max_length = 512
batch_size = 32
gradient_accumulation_steps = 2
learning_rate = 2e-4
epochs = 3
weight_decay = 0.0

# 2. 加载 LLaMA2-7B
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = False,   # 单卡80G，可以直接fp16/bf16跑
    dtype = torch.bfloat16,
    max_seq_length = max_length
)

# Lora配置
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128,欠拟合可以稍微大一点
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128,            # suggested r * 2
    lora_dropout = 0,           # Supports any, but = 0 is optimized
    bias = "none",              # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # 节省显存
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

PROMPT_TEMPLATES = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}
"""

# 3. 数据预处理,examples必须包含instruction,input,output
def format_example(examples):
    train_texts = []
    instructions = examples["instruction"]
    responses = examples["positive_response"]
    for instruction,response in zip(instructions,responses):
        text = f"{PROMPT_TEMPLATES.format(instruction=instruction,response=response)}{tokenizer.eos_token}"
        train_texts.append(text)
    return {
        "text": train_texts
    }

dataset = load_dataset("json", data_files=data_path,split="train")
dataset = dataset.map(format_example,batched=True)

# 4. 定义 Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_length,
    packing = False,  # 每条样本单独训练，避免打包混乱

    args = SFTConfig(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        num_train_epochs = epochs,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        optim = "adamw_8bit",
        warmup_steps = 500,
        logging_steps = 20,
        save_steps = 300,
        save_total_limit = 2,
        lr_scheduler_type = "linear",
        completion_only_loss = True,
        bf16 = True,
        tf32 = True,
        output_dir = output_dir,
        report_to = "swanlab"
    )
)

# 5. 开始训练
trainer.train()

# 6. 保存模型
trainer.save_model(output_dir)