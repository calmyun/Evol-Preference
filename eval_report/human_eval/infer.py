import torch
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


MODEL_PATH = "/data/liuyun/Evol-Preference/llama2-dpo-merged"
INPUT_PATH = "/data/liuyun/Evol-Preference/eval_report/human_eval/mistake_response.jsonl"
OUTPUT_PATH = "/data/liuyun/Evol-Preference/eval_report/human_eval/correct_response.jsonl"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 512

ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompts, max_new_tokens):
    responses = []

    for prompt in prompts:
        text = ALPACA_PROMPT_TEMPLATE.format(instruction=prompt.strip())

        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

        with torch.inference_mode():
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        full_text = tokenizer.decode(outputs[0][encodings.input_ids.shape[1]:], skip_special_tokens=True).strip("\n")
        
        match = re.search(r'###\s*Response\s*:?\s*\n?(.*?)(?=\n\s*###\s*[A-Za-z]+|$)', full_text, re.IGNORECASE | re.DOTALL)
        if match:
            response = re.sub(r'\n{3,}', '\n\n', match.group(1).strip())
        else:
            response = full_text
        
        responses.append(response)

    return responses


if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model loaded from: {MODEL_PATH}")

    print(f"Reading prompts from: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    print(f"Total prompts: {len(test_data)}")

    prompts = [d["prompt"] for d in test_data]

    print(f"Starting inference... Output to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as wf:
        for i, prompt in enumerate(tqdm(prompts)):
            try:
                response = generate_response(model, tokenizer, [prompt], MAX_NEW_TOKENS)[0]
            except Exception as e:
                print(f"\n[ERROR] Prompt {i}: {e}")
                response = f"生成错误: {e}"

            wf.write(json.dumps({"prompt": prompt, "model_response": response}, ensure_ascii=False) + "\n")
            wf.flush()
            os.fsync(wf.fileno())

    print(f"✅ Inference complete! Results saved to: {OUTPUT_PATH}")
