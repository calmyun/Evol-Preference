import torch
import argparse
import json
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


argparser = argparse.ArgumentParser("Collect model response for different models")
argparser.add_argument("--dataset_path", type=str, required=True)
argparser.add_argument("--output_path", type=str, required=True)
argparser.add_argument("--model_path", type=str, required=True)
argparser.add_argument("--model_type", type=str, required=True, choices=["qwen3", "llama2", "beaver7b"])
argparser.add_argument("--device", type=int, default=0)
args = argparser.parse_args()

MODEL_PATH = args.model_path
MODEL_TYPE = args.model_type
DATASET_PATH = args.dataset_path
OUTPUT_PATH = args.output_path
DEVICE = args.device
BATCH_SIZE = 8
MAX_NEW_TOKENS = 1024


def detect_model_type(model_path: str, tokenizer: AutoTokenizer) -> str:
    if MODEL_TYPE != "auto":
        return MODEL_TYPE

    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            model_name = config.get("model_type", "").lower()
            if "qwen" in model_name:
                return "qwen3"

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        if "Qwen" in str(tokenizer.chat_template):
            return "qwen3"

    if "llama2" in model_path.lower() or "llama" in model_path.lower():
        return "llama2"

    return "llama2"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map=f"cuda:{DEVICE}",
    )
    model.eval()
    return model, tokenizer


def generate_qwen3(model, tokenizer, prompts, max_new_tokens):
    messages_list = [[{"role": "user", "content": p.strip()}] for p in prompts]
    texts = [
        tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True, enable_thinking=True
        ) for m in messages_list
    ]
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(f"cuda:{DEVICE}")

    with torch.inference_mode():
        outputs = model.generate(**encodings, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    responses = []
    for i, output in enumerate(outputs):
        output_ids = output[len(encodings.input_ids[i]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        responses.append(content)
    return responses


ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def generate_llama2(model, tokenizer, prompts, max_new_tokens):
    texts = [ALPACA_PROMPT_TEMPLATE.format(instruction=p.strip()) for p in prompts]
        
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(f"cuda:{DEVICE}")

    with torch.inference_mode():
        outputs = model.generate(
            **encodings, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    responses = []
    for i, output in enumerate(outputs):
        full_text = tokenizer.decode(output[len(encodings.input_ids[i]):], skip_special_tokens=True).strip("\n")
        match = re.search(r'###\s*(Response|Analysis)\s*:?\s*\n?(.*?)(?=\n\s*###\s*[A-Za-z]+|$)', full_text, re.IGNORECASE | re.DOTALL)
        if match:
            response = re.sub(r'\n{3,}', '\n\n', match.group(2).strip())
        else:
            response = full_text
        responses.append(response)
    return responses


def generate_beaver7b(model, tokenizer, prompts, max_new_tokens):
    prompt = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:'
    formatted_prompts = [prompt.format(input=p) for p in prompts]

    # 接着encode，然后decode，得到输出的推理结果
    encodings = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{DEVICE}")

    with torch.inference_mode():
        outputs = model.generate(**encodings, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    responses = []
    for i, output in enumerate(outputs):
        full_text = tokenizer.decode(output[len(encodings.input_ids[i]):], skip_special_tokens=True).strip("\n")
        responses.append(full_text)
    return responses


GENERATORS = {
    "qwen3": generate_qwen3,
    "llama2": generate_llama2,
    "beaver7b": generate_beaver7b,
}


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    model_type = detect_model_type(MODEL_PATH, tokenizer)
    print(f"Detected model type: {model_type}")
    generator = GENERATORS[model_type]

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    # with open(OUTPUT_PATH, "w", encoding="utf-8") as wf:
    with open(OUTPUT_PATH, "a+", encoding="utf-8") as wf:
        for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
            batch = test_data[i:i + BATCH_SIZE]
            prompts = [d["prompt"] for d in batch]

            try:
                responses = generator(model, tokenizer, prompts, MAX_NEW_TOKENS)
            except Exception as e:
                print(f"[WARN] Batch {i} error: {e}")
                responses = [f"生成错误: {e}"] * len(batch)

            for data, resp in zip(batch, responses):
                wf.write(json.dumps({"prompt": data["prompt"], "model_response": resp}, ensure_ascii=False) + "\n")
            wf.flush()
            os.fsync(wf.fileno())

    print(f"✅ 模型推理结束! 结果保存至: {OUTPUT_PATH}")