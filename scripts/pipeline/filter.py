import re
import json
import asyncio
from typing import List, Dict
from tqdm import tqdm
from evaluation.openai_access import call_chatgpt_async  # ✅ 改为异步版本
import aiofiles

# ---------- Step 1: Rule-Based Filtering ---------- #
def rule_based_filter(prompt: str) -> Dict:
    """Apply deterministic lexical & structural rules."""
    text = prompt.strip()
    reasons = []

    # 1. Content duplication patterns
    if re.search(r'#Rewritten Prompt#|given prompt|original instruction', text, re.I):
        reasons.append("duplication")

    # 2. Uninformative tokens / artifacts
    if len(re.sub(r"[a-zA-Z]", "", text)) / max(len(text), 1) > 0.4:
        reasons.append("uninformative")
    if re.search(r"(please refer|as an ai model|thank you for understanding)", text, re.I):
        reasons.append("filler_phrase")

    passed = len(reasons) == 0
    return {"text": text, "passed_rule": passed, "reasons": reasons}


# ---------- Step 2: LLM-Assisted Validation (Async version) ---------- #
async def llm_assisted_validation_async(prompt: str) -> Dict:
    developer = """You are a critical safety and quality validator for AI-generated instructions.
    Your task is to assess each prompt according to the following criteria, and respond in strict JSON format."""

    input_prompt = f"""
    [Instruction to validate begins below]
    ----
    {prompt}
    ----
    Evaluate this prompt on the following three criteria:

    1. Safety_Compliance:
    - The instruction must not encourage, or describe unsafe, illegal, or unethical behaviors.
    - If such content appears, mark this criterion as "FAIL".

    2. Semantic_Coherence:
    - The prompt must be syntactically complete, logically clear, and semantically understandable.

    3. Constructive_Value:
    - The prompt should encourage a meaningful, instructive, or task-oriented response.

    Respond **only** in this JSON format:
    {{
    "Safety_Compliance": "PASS" or "FAIL",
    "Semantic_Coherence": "PASS" or "FAIL",
    "Constructive_Value": "PASS" or "FAIL",
    "Overall": "ACCEPT" if all PASS else "REJECT",
    "Short_Reason": "One concise sentence summarizing the decision."
    }}
    """
    try:
        response_text = await call_chatgpt_async(input_prompt, developer=developer)
        return json.loads(response_text)
    except Exception as e:
        print(f"❌ LLM验证失败: {e}")
        return {
            "Safety_Compliance": "FAIL",
            "Semantic_Coherence": "FAIL",
            "Constructive_Value": "FAIL",
            "Overall": "REJECT",
            "Short_Reason": f"Exception during validation: {e}"
        }


# ---------- Step 3: Async Filtering Pipeline ---------- #
async def evol_filter_pipeline_async(data: List[Dict], save_dir="/data/liuyun/Evol-Preference/data"):
    filtered, accepted = [], []

    print(f"开始异步过滤，共 {len(data)} 条数据...")
    tasks = []
    rule_results = []

    # Step 1: Rule-based filtering (同步执行)
    for item in data:
        prompt = item["instruction"]
        rule_result = rule_based_filter(prompt)
        rule_results.append((item, prompt, rule_result))
        if rule_result["passed_rule"]:
            tasks.append(llm_assisted_validation_async(prompt))
        else:
            # 不通过rule直接拒绝
            filtered.append({
                "prompt": prompt,
                "status": "rejected_rule",
                "rule_result": rule_result,
                "llm_result": None
            })

    # Step 2: Run LLM validation asynchronously
    llm_results = []
    for i in tqdm(range(0, len(tasks), 10), desc="LLM并发验证中..."):  # 每次10个并发
        batch = tasks[i:i+10]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        llm_results.extend(batch_results)
        await asyncio.sleep(0.2)  # 稍作间隔以防触发RateLimit

    # Step 3: Combine results
    llm_index = 0
    for item, prompt, rule_result in rule_results:
        if rule_result["passed_rule"]:
            llm_result = llm_results[llm_index]
            llm_index += 1
            if llm_result.get("Overall") == "ACCEPT":
                accepted.append(item)
            else:
                filtered.append({
                    "prompt": prompt,
                    "status": "rejected_llm",
                    "rule_result": rule_result,
                    "llm_result": llm_result
                })

    # Step 4: Save asynchronously
    accepted_path = f"{save_dir}/accepted_evol_data.jsonl"
    filtered_path = f"{save_dir}/filtered_evol_data.jsonl"
    async with aiofiles.open(accepted_path, 'w', encoding='utf-8') as fa:
        for item in accepted:
            await fa.write(json.dumps(item, ensure_ascii=False) + '\n')
    async with aiofiles.open(filtered_path, 'w', encoding='utf-8') as ff:
        for item in filtered:
            await ff.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Step 5: Compute statistics
    total = len(accepted) + len(filtered)
    accept_ratio = len(accepted) / total if total > 0 else 0
    filter_ratio = len(filtered) / total if total > 0 else 0

    print(f"筛选完成！✅ 已接受 {len(accepted)} 条，已拒绝 {len(filtered)} 条")
    print(f"通过率（accepted ratio）: {accept_ratio:.3f}")
    print(f"过滤率（filtered ratio）: {filter_ratio:.3f}")

    final_result = {
        "accepted_count": len(accepted),
        "filtered_count": len(filtered),
        "accepted_ratio": accept_ratio,
        "filtered_ratio": filter_ratio
    }
    with open("/data/liuyun/Evol-Preference/data/filter_result.json",'w',encoding='utf-8') as wf:
        json.dump(final_result, wf, ensure_ascii=False, indent=2)
        
    return final_result


# ---------- Step 4: Entry ---------- #
if __name__ == "__main__":
    data_path = "/data/liuyun/Evol-Preference/data/beavertails_evol_data.jsonl"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]


    asyncio.run(evol_filter_pipeline_async(data))
