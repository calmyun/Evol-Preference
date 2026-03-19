import torch
import argparse
import json
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ==== 配置 ====
argparser = argparse.ArgumentParser("Collect model response according safety benchmark dataset (Transformers version)")
argparser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/Evol-Preference/dataset/HarmfulQA/test.jsonl")
argparser.add_argument("--output_path", type=str, default="/root/autodl-tmp/Evol-Preference/infer_harmfulqa/Qwen3-4B-SafeRL.jsonl")
argparser.add_argument("--model_path", type=str, default="/root/autodl-tmp/Evol-Preference/Qwen3-4B-SafeRL")
argparser.add_argument("--device", type=int, default=0, help="GPU device id to use")
args = argparser.parse_args()

MODEL_PATH = args.model_path
DATASET_PATH = args.dataset_path
OUTPUT_PATH = args.output_path
DEVICE = args.device
BATCH_SIZE = 8
MAX_NEW_TOKENS = 1024

# ==== 1. 模型加载 ====
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side='left'
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map=f"cuda:{DEVICE}",
    )
    model.eval()

    return model, tokenizer

# ==== 3. 提取响应(针对llama2-dpo-merged) ====
def extract_response_text(full_text: str) -> str:
    pattern = re.compile(
        r'###\s*(Response|Analysis)\s*:?\s*\n?(.*?)'
        r'(?=\n\s*###\s*[A-Za-z]+|$)',
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(full_text)
    if match:
        response = match.group(2).strip()
        response = re.sub(r'\n{3,}', '\n\n', response).strip()
        return response
    return full_text.strip()

# ==== 4. 批量生成响应 ====
def parse_qwen3_response(output_ids, input_ids, tokenizer):
    output_ids = output_ids[0][len(input_ids[0]):].tolist()
    
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    return thinking_content, content

def generate_responses_batch(model, tokenizer, prompts, max_new_tokens=1024):
    messages_list = [[{"role": "user", "content": p.strip()}] for p in prompts]
    
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        ) for messages in messages_list
    ]
    
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(f"cuda:{DEVICE}")

    with torch.inference_mode():
        outputs = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    responses = []
    for i, output in enumerate(outputs):
        thinking_content, content = parse_qwen3_response(
            output.unsqueeze(0), 
            encodings.input_ids[i:i+1], 
            tokenizer
        )
        responses.append(content)
    
    return responses


# ==== 5. 主程序 ====
if __name__ == "__main__":
    model, tokenizer = load_model()

    # 读取测试集
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as wf:
        for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
            batch = test_data[i : i + BATCH_SIZE]
            prompts = [d["prompt"] for d in batch]

            try:
                responses = generate_responses_batch(model, tokenizer, prompts, MAX_NEW_TOKENS)
            except Exception as e:
                print(f"[WARN] Batch {i} 生成错误: {e}")
                responses = [f"生成错误: {e}"] * len(batch)

            results = []
            for data, resp in zip(batch, responses):
                results.append({
                    "prompt": data["prompt"],
                    "model_response": resp,
                })

            wf.write("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
            wf.flush()
            os.fsync(wf.fileno())

    print(f"✅ 收集完成，结果保存至：{OUTPUT_PATH}")