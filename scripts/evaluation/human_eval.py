# filename: beavertails_evaluator.py
import json
import re
import time
import argparse
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

# Import the provided API wrapper functions
# from openai_access import get_completion, get_completion_async
# from gemini_access import get_completion, get_completion_async
from claude_access import get_completion, get_completion_async

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file, returning list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} in {path}: {e}")
    return items

def save_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    """Save list of dicts to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

import template
JUDGE_DEVELOPER_INSTRUCTIONS = template.HUMAN_EVALUATOR_PROMPT

def build_judge_user_message(user_prompt: str, model_response: str) -> str:
    content = (
        f'prompt: """{user_prompt}"""\n'
        f'model_response: """{model_response}"""\n'
    )
    return content

def extract_json_block(text: str) -> Optional[str]:
    """
    Extract first valid JSON object from text.
    Tries direct parse first, then bracket matching, finally regex fallback.
    """
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 括号匹配：查找第一个完整的 {...} 块
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += 1 if text[i] == "{" else -1 if text[i] == "}" else 0
            if depth == 0:
                try:
                    json.loads(text[start:i+1])
                    return text[start:i+1]
                except json.JSONDecodeError:
                    pass

    # 正则兜底：匹配最后一个 {...} 块
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            json.loads(match.group(0))
            return match.group(0)
        except json.JSONDecodeError:
            pass

    return None

async def judge_single_async(prompt: str, model_response: str, developer_instructions: str = JUDGE_DEVELOPER_INSTRUCTIONS, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
    """
    Async version using call_chatgpt_async.
    Returns parsed JSON with fields: safety, response_type, helpfulness_score, reasoning
    """
    user_msg = build_judge_user_message(prompt, model_response)
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            raw = await get_completion_async(user_msg, system=developer_instructions)
            if raw is None:
                raise ValueError("API returned None")
            parsed_str = extract_json_block(raw)
            if parsed_str:
                parsed = json.loads(parsed_str)
                return parsed
            raise ValueError("Failed to extract JSON block")
        except Exception as e:
            print(f"[judge_single_async] attempt {attempt} failed: {e}")
            if attempt >= max_retries:
                return {
                    "safety": "Unsafe",
                    "response_type": None,
                    "helpfulness_score": 1,
                    "reasoning": "failed to get valid evaluation"
                }
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
    return {
        "safety": "Unsafe",
        "response_type": None,
        "helpfulness_score": 1,
        "reasoning": "max retries exceeded"
    }

# ---------------------------
# Main evaluation orchestration
# ---------------------------
async def evaluate_model_file_async(model_label: str, responses_path: str, output_path: str, dataset_name: str, concurrency: int = 8, verbose: bool = True) -> Dict[str, Any]:
    """
    Async evaluation: launches judge calls concurrently (bounded by 'concurrency').
    
    Evaluates based on HUMAN_EVALUATOR_PROMPT output fields:
    - safety: "Safe" | "Unsafe"
    - response_type: classification of response
    - helpfulness_score: 1-5
    - reasoning: explanation
    """
    items = load_jsonl(responses_path)
    results = []
    counts = {
        "total": 0,
        "safe_count": 0,
        "unsafe_count": 0,
        "helpfulness_sum": 0,
        "response_types": {}
    }
    start_time = time.time()

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int, it: Dict[str, Any]):
        async with semaphore:
            prompt = it.get("prompt", "")
            model_resp = it.get("model_response", "")
            judge = await judge_single_async(prompt, model_resp)
            out = {
                "model_label": model_label,
                "index": index,
                "prompt": prompt,
                "model_response": model_resp,
                "judge": judge
            }
            return out

    tasks = [worker(i+1, it) for i, it in enumerate(items)]
    BATCH = 32
    pbar = tqdm(total=len(items), desc=f"Evaluating {model_label}")
    for i in range(0, len(tasks), BATCH):
        batch = tasks[i:i+BATCH]
        finished = await asyncio.gather(*batch)
        for out in finished:
            results.append(out)
            judge = out["judge"]
            counts["total"] += 1
            
            safety = judge.get("safety", "Unsafe")
            if safety == "Safe":
                counts["safe_count"] += 1
            else:
                counts["unsafe_count"] += 1
            
            helpfulness = judge.get("helpfulness_score", 1)
            if isinstance(helpfulness, (int, float)):
                counts["helpfulness_sum"] += int(helpfulness)
            else:
                counts["helpfulness_sum"] += 1
            
            response_type = judge.get("response_type")
            if response_type:
                counts["response_types"][response_type] = counts["response_types"].get(response_type, 0) + 1
            
        pbar.update(len(finished))
    pbar.close()

    save_jsonl(results, output_path)
    duration = time.time() - start_time
    
    total = counts["total"]
    safety_rate = counts["safe_count"] / total if total else 0.0
    avg_helpfulness = counts["helpfulness_sum"] / total if total else 0.0
    
    stats = {
        "model_label": model_label,
        "eval_dataset": dataset_name,
        "total": total,
        "safe_count": counts["safe_count"],
        "unsafe_count": counts["unsafe_count"],
        "safety_rate": safety_rate,
        "avg_helpfulness": avg_helpfulness,
        "response_type_distribution": counts["response_types"],
        "duration_seconds": duration,
        "output_path": output_path
    }
    return stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate model responses using GPT-5 judge.")
    parser.add_argument("--data_path", required=True, help="Path to JSONL file of model responses (fields: prompt, model_response)")
    parser.add_argument("--output_path", required=True, help="Output JSONL path for judge results")
    parser.add_argument("--model", required=True, help="Label for the model under test (e.g., alpaca-7b)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., beavertails)")
    parser.add_argument("--concurrency", type=int, default=16, help="(async) concurrency level")
    args = parser.parse_args()

    print("Starting asynchronous evaluation...")
    stats = asyncio.run(evaluate_model_file_async(args.model, args.data_path, args.output_path, args.dataset, concurrency=args.concurrency, verbose=True))
    print("Done. Stats:")
    stats_json_str = json.dumps(stats, indent=2, ensure_ascii=False)
    print(stats_json_str)
    
    # 保存结果
    with open(f"/data/liuyun/Evol-Preference/eval_report/human_eval/{args.model}.jsonl", "a+" ,encoding='utf-8') as f:
        f.write(stats_json_str + "\n")

if __name__ == "__main__":
    main()
