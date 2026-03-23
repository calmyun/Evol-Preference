# filename: beavertails_evaluator.py
import json
import re
import time
import argparse
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from claude_access import get_completion_async

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
begign_eval_prompt = template.JAILBREAK_BENIGN_EVALUATOR_PROMPT
harmful_eval_prompt = template.JAILBREAK_HARMFUL_EVALUATOR_PROMPT

async def judge_single_async(prompt: str, model_response: str, eval_prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
    """
    Async version using call_chatgpt_async.
    Returns parsed evaluation result based on dataset type:
    - harmful: "safe" or "unsafe"
    - benign: "yes" or "no"
    """
    user_msg = eval_prompt.format(goal=prompt, response=model_response)
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            raw = await get_completion_async(user_msg)
            if raw is None:
                raise ValueError("API returned None")
            
            # For benign prompt, expect "Yes"/"No" answer
            if "Answer either \"Yes\" or \"No\"" in eval_prompt:
                result = raw.strip().lower()
                if result == "yes":
                    return {"result": "yes"}
                elif result == "no":
                    return {"result": "no"}
                else:
                    raise ValueError(f"Unexpected response: {raw}")
            
            # For harmful prompt, expect "safe"/"unsafe" answer
            if "Answer with either 'safe' or 'unsafe'" in eval_prompt:
                result = raw.strip().lower()
                if result == "safe":
                    return {"result": "safe"}
                elif result == "unsafe":
                    return {"result": "unsafe"}
                else:
                    raise ValueError(f"Unexpected response: {raw}")
            
            raise ValueError("Failed to parse response")
        except Exception as e:
            print(f"[judge_single_async] attempt {attempt} failed: {e}")
            if attempt >= max_retries:
                return {"result": "error", "error": str(e)}
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
    return {"result": "error", "error": "max retries exceeded"}

# ---------------------------
# Main evaluation orchestration
# ---------------------------
async def evaluate_model_file_async(model_label: str, responses_path: str, output_path: str, dataset_name: str, concurrency: int = 8, verbose: bool = True) -> Dict[str, Any]:
    """
    Async evaluation: launches judge calls concurrently (bounded by 'concurrency').
    
    Evaluates based on dataset type:
    - harmful: uses JAILBREAK_HARMFUL_EVALUATOR_PROMPT, counts "safe" and "unsafe"
    - benign: uses JAILBREAK_BENIGN_EVALUATOR_PROMPT, counts "yes" and "no"
    """
    items = load_jsonl(responses_path)
    results = []
    
    # Select evaluation prompt based on dataset type
    if dataset_name.lower() == "harmful":
        eval_prompt = harmful_eval_prompt
        counts = {"total": 0, "safe": 0, "unsafe": 0}
    elif dataset_name.lower() == "benign":
        eval_prompt = begign_eval_prompt
        counts = {"total": 0, "yes": 0, "no": 0}
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}. Must be 'harmful' or 'benign'")
    
    start_time = time.time()

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int, it: Dict[str, Any]):
        async with semaphore:
            prompt = it.get("prompt", "")
            model_resp = it.get("model_response", "")
            judge = await judge_single_async(prompt, model_resp, eval_prompt)
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
    pbar = tqdm(total=len(items), desc=f"Evaluating {model_label} on {dataset_name}")
    for i in range(0, len(tasks), BATCH):
        batch = tasks[i:i+BATCH]
        finished = await asyncio.gather(*batch)
        for out in finished:
            results.append(out)
            judge = out["judge"]
            counts["total"] += 1
            
            result = judge.get("result", "error")
            if result in counts:
                counts[result] += 1
            
        pbar.update(len(finished))
    pbar.close()

    save_jsonl(results, output_path)
    duration = time.time() - start_time
    
    stats = {
        "model_label": model_label,
        "eval_dataset": dataset_name,
        "counts": counts,
        "duration_seconds": duration,
        "output_path": output_path
    }
    return stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate model responses using LLM judge.")
    parser.add_argument("--data_path", required=False, help="Path to JSONL file of model responses")
    parser.add_argument("--output_path", required=False, help="Output JSONL path for judge results")
    parser.add_argument("--model", required=True, help="Label for the model under test")
    parser.add_argument("--dataset", required=True, help="Dataset name: 'harmful' or 'benign'")
    parser.add_argument("--concurrency", type=int, default=16, help="(async) concurrency level")
    args = parser.parse_args()

    # Validate dataset type
    dataset_type = args.dataset.lower()
    if dataset_type not in ["harmful", "benign"]:
        raise ValueError(f"Dataset must be 'harmful' or 'benign', got: {args.dataset}")

    # Set default paths based on dataset type
    base_dir = "/data/liuyun/Evol-Preference"
    
    if dataset_type == "harmful":
        # Use harmful dataset and responses
        data_path = args.data_path or f"{base_dir}/infer_jailbreak/{args.model}_harmful.jsonl"
        output_dir = f"{base_dir}/eval_report/jailbreak"
        output_path = args.output_path or f"{output_dir}/{args.model}_harmful_eval.jsonl"
    else:  # benign
        # Use benign dataset and responses
        data_path = args.data_path or f"{base_dir}/infer_jailbreak/{args.model}_benign.jsonl"
        output_dir = f"{base_dir}/eval_report/jailbreak"
        output_path = args.output_path or f"{output_dir}/{args.model}_benign_eval.jsonl"

    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting evaluation for {args.model} on {dataset_type} dataset...")
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    
    stats = asyncio.run(evaluate_model_file_async(args.model, data_path, output_path, dataset_type, concurrency=args.concurrency, verbose=True))
    
    print("Done. Stats:")
    stats_json_str = json.dumps(stats, indent=2, ensure_ascii=False)
    print(stats_json_str)
    
    # Save summary results
    summary_path = f"{output_dir}/{args.model}_{dataset_type}_summary.json"
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(stats_json_str)
    
    print(f"✅ Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
