import re
import json
import asyncio
from typing import List, Dict, Tuple
from tqdm import tqdm
from evaluation.openai_access import call_chatgpt_async
import aiofiles

# ---------- Step 1: Enhanced Rule-Based Filtering ---------- #
def rule_based_filter_enhanced(item: Dict) -> Tuple[Dict, bool]:
    """Apply enhanced deterministic lexical & structural rules."""
    instruction = item.get("instruction", "").strip()
    positive_response = item.get("positive_response", "").strip()
    reasons = []

    # 1. Content duplication patterns (原有规则)
    if re.search(r'#Rewritten Prompt#|given prompt|original instruction', instruction, re.I):
        reasons.append("duplication")

    # 2. Uninformative tokens / artifacts (原有规则)
    if len(re.sub(r"[a-zA-Z]", "", instruction)) / max(len(instruction), 1) > 0.4:
        reasons.append("uninformative")
    if re.search(r"(please refer|as an ai model|thank you for understanding)", instruction, re.I):
        reasons.append("filler_phrase")

    # ========== 新增规则 ========== #

    # 3. 指令长度检查
    if len(instruction) < 30:
        reasons.append("instruction_too_short")
    if len(instruction) > 500:
        reasons.append("instruction_too_long")

    # 4. 响应长度检查
    if len(positive_response) < 80:
        reasons.append("response_too_short")
    if len(positive_response) > 2000:
        reasons.append("response_too_long")

    # 5. 指令-响应长度比检查
    if len(positive_response) > 0:
        ratio = len(instruction) / len(positive_response)
        if ratio > 0.8:
            reasons.append("instruction_response_ratio_imbalanced")

    # 6. 响应质量检查 - 检测过短或无意义的响应
    if len(positive_response) > 0:
        # 检测只有标题或列表项的响应
        lines = positive_response.split('\n')
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        if len(non_empty_lines) <= 2 and len(positive_response) < 150:
            reasons.append("response_insufficient")

        # 检测重复的短语
        words = positive_response.lower().split()
        if len(words) > 0:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:
                reasons.append("response_repetitive")

        # 检测只有占位符的响应
        placeholder_patterns = [r'^i can(\'t| not)?$', r'^i don\'t know$', r'^i can\'t help with',
                               r'^i cannot help$', r'^i\'m unable to$']
        for pattern in placeholder_patterns:
            if re.match(pattern, positive_response.strip(), re.I):
                reasons.append("response_placeholder")
                break

    # 7. 指令质量检查 - 检测过度复杂的指令
    constraint_count = len(re.findall(r'\b(please|ensure|include|additionally|furthermore|moreover|while|also|must|should|need to)\b', instruction, re.I))
    if constraint_count > 8:
        reasons.append("instruction_overconstrained")

    # 8. 检测指令中的有害模式（即使 LLM 会检查，规则层先过滤）
    harmful_patterns = [
        r'\bhow to (kill|murder|poison|bomb|hack|steal|cheat)\b',
        r'\bstep\s+by\s+step.*(?:kill|murder|poison)',
        r'\binstruction.*(?:illegal|criminal)\b'
    ]
    for pattern in harmful_patterns:
        if re.search(pattern, instruction, re.I):
            reasons.append("instruction_harmful")
            break

    passed = len(reasons) == 0
    return {"passed_rule": passed, "reasons": reasons}


# ---------- Step 2: Enhanced LLM-Assisted Validation ---------- #
async def llm_assisted_validation_enhanced(instruction: str, positive_response: str) -> Dict:
    """Enhanced LLM validation with more strict criteria."""
    developer = """You are a critical safety and quality validator for AI-generated instruction-response pairs.
    Your task is to assess each pair according to following criteria, and respond in strict JSON format."""

    input_prompt = f"""
    [Instruction-Response Pair to validate begins below]
    ----
    Instruction: {instruction}
    
    Response: {positive_response}
    ----

    Evaluate this pair on the following four criteria:

    1. Safety_Compliance:
    - The instruction must not encourage, or describe unsafe, illegal, or unethical behaviors.
    - The response must be safe, helpful, and not enable harm.
    - If either instruction or response fails safety, mark this criterion as "FAIL".

    2. Semantic_Coherence:
    - The instruction must be syntactically complete, logically clear, and semantically understandable.
    - The response must directly address the instruction.
    - If either fails coherence, mark this criterion as "FAIL".

    3. Constructive_Value:
    - The instruction should encourage a meaningful, instructive, or task-oriented response.
    - The response should be substantive, informative, and provide value.
    - If either fails constructive value, mark this criterion as "FAIL".

    4. Response_Quality:
    - The response must be sufficiently detailed (at least 100 characters).
    - The response should not be a simple refusal or placeholder.
    - The response should provide actionable information or guidance.
    - If response fails quality, mark this criterion as "FAIL".

    Respond **only** in this JSON format:
    {{
    "Safety_Compliance": "PASS" or "FAIL",
    "Semantic_Coherence": "PASS" or "FAIL",
    "Constructive_Value": "PASS" or "FAIL",
    "Response_Quality": "PASS" or "FAIL",
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
            "Response_Quality": "FAIL",
            "Overall": "REJECT",
            "Short_Reason": f"Exception during validation: {e}"
        }


# ---------- Step 3: Enhanced Async Filtering Pipeline ---------- #
async def evol_filter_pipeline_enhanced(data: List[Dict], save_dir="/data/liuyun/Evol-Preference/data"):
    """Enhanced filtering pipeline with better quality control."""
    filtered, accepted = [], []

    print(f"开始增强异步过滤，共 {len(data)} 条数据...")
    tasks = []
    rule_results = []

    # Step 1: Enhanced rule-based filtering
    for item in data:
        rule_result = rule_based_filter_enhanced(item)
        rule_results.append((item, rule_result))
        if rule_result["passed_rule"]:
            instruction = item.get("instruction", "")
            positive_response = item.get("positive_response", "")
            tasks.append(llm_assisted_validation_enhanced(instruction, positive_response))
        else:
            filtered.append({
                "instruction": item.get("instruction", ""),
                "positive_response": item.get("positive_response", ""),
                "status": "rejected_rule",
                "rule_result": rule_result,
                "llm_result": None
            })

    # Step 2: Run LLM validation asynchronously
    llm_results = []
    batch_size = 10
    for i in tqdm(range(0, len(tasks), batch_size), desc="LLM并发验证中..."):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        llm_results.extend(batch_results)
        await asyncio.sleep(0.3)  # 稍微增加间隔以防触发RateLimit

    # Step 3: Combine results
    llm_index = 0
    for item, rule_result in rule_results:
        if rule_result["passed_rule"]:
            llm_result = llm_results[llm_index]
            llm_index += 1
            if llm_result.get("Overall") == "ACCEPT":
                accepted.append(item)
            else:
                filtered.append({
                    "instruction": item.get("instruction", ""),
                    "positive_response": item.get("positive_response", ""),
                    "status": "rejected_llm",
                    "rule_result": rule_result,
                    "llm_result": llm_result
                })

    # Step 4: Save asynchronously
    accepted_path = f"{save_dir}/accepted_evol_data_enhanced.jsonl"
    filtered_path = f"{save_dir}/filtered_evol_data_enhanced.jsonl"
    
    async with aiofiles.open(accepted_path, 'w', encoding='utf-8') as fa:
        for item in accepted:
            await fa.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    async with aiofiles.open(filtered_path, 'w', encoding='utf-8') as ff:
        for item in filtered:
            await ff.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Step 5: Compute detailed statistics
    total = len(accepted) + len(filtered)
    accept_ratio = len(accepted) / total if total > 0 else 0
    filter_ratio = len(filtered) / total if total > 0 else 0

    # 统计拒绝原因
    rule_rejection_reasons = {}
    for item in filtered:
        if item["status"] == "rejected_rule":
            for reason in item["rule_result"]["reasons"]:
                rule_rejection_reasons[reason] = rule_rejection_reasons.get(reason, 0) + 1

    llm_rejection_reasons = {}
    for item in filtered:
        if item["status"] == "rejected_llm" and item["llm_result"]:
            reason = item["llm_result"].get("Short_Reason", "Unknown")
            llm_rejection_reasons[reason] = llm_rejection_reasons.get(reason, 0) + 1

    print(f"\n{'='*80}")
    print(f"筛选完成！✅ 已接受 {len(accepted)} 条，已拒绝 {len(filtered)} 条")
    print(f"{'='*80}")
    print(f"\n总体统计:")
    print(f"  总样本数: {total}")
    print(f"  通过率（accepted ratio）: {accept_ratio:.3f} ({accept_ratio*100:.1f}%)")
    print(f"  过滤率（filtered ratio）: {filter_ratio:.3f} ({filter_ratio*100:.1f}%)")

    print(f"\n规则拒绝原因统计:")
    for reason, count in sorted(rule_rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} ({count/total*100:.2f}%)")

    print(f"\nLLM拒绝原因统计 (Top 10):")
    sorted_llm_reasons = sorted(llm_rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
    for reason, count in sorted_llm_reasons:
        print(f"  {reason}: {count}")

    final_result = {
        "accepted_count": len(accepted),
        "filtered_count": len(filtered),
        "accepted_ratio": accept_ratio,
        "filtered_ratio": filter_ratio,
        "rule_rejection_reasons": rule_rejection_reasons,
        "llm_rejection_reasons": llm_rejection_reasons
    }
    
    with open(f"{save_dir}/filter_result_enhanced.json", 'w', encoding='utf-8') as wf:
        json.dump(final_result, wf, ensure_ascii=False, indent=2)
        
    return final_result


# ---------- Step 4: Entry ---------- #
if __name__ == "__main__":
    data_path = "/data/liuyun/Evol-Preference/data/beavertails_evol_data.jsonl"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    asyncio.run(evol_filter_pipeline_enhanced(data))
