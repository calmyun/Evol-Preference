import json
import random
import time
import os
import shutil
import logging
import asyncio
from evaluation.openai_access import call_chatgpt, call_chatgpt_async
from pipeline.depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from pipeline.breadth import createBreadthPrompt
from pipeline.sanitization import createSanitizationPrompt

# ============ 日志系统配置 ============
log_file = "/data/liuyun/Evol-Preference/main.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============ 系统配置 ============
system = """You are a safety-aligned assistant.
Respond to instructions in a safe, factual, and concise manner, adhering to the following constraints:
- Limit your response to a maximum of 120 words or 8 bullet points.
- Only provide step-by-step tutorials if the user explicitly requests them.
- Emphasize actionable advice and avoid lengthy justifications.
"""

input_data_path = "/data/liuyun/Evol-Preference/data_v2/beavertails_unsafe.jsonl"
output_file_path = "/data/liuyun/Evol-Preference/data_v2/beavertails_evol_data.jsonl"


# ============ 工具函数 ============
def safe_write(outf, data_obj):
    """写入并立即落盘"""
    outf.write(json.dumps(data_obj, ensure_ascii=False) + '\n')
    outf.flush()
    os.fsync(outf.fileno())


def backup_file(file_path, backup_index):
    """每1000条数据备份一次"""
    base, ext = os.path.splitext(file_path)
    backup_path = f"{base}_backup_{backup_index}{ext}"
    shutil.copy(file_path, backup_path)
    logger.info(f"📦 备份完成: {os.path.basename(backup_path)}")


def process_task(cur_obj, outf):
    """处理单条任务逻辑"""
    prompt = cur_obj["prompt"]
    initial_response = cur_obj["response"]

    sanitation_prompt = createSanitizationPrompt(prompt)
    sanitation_instruction = call_chatgpt(sanitation_prompt)
    if not sanitation_instruction:
        raise Exception("指令净化失败")

    evol_prompts = [
        createConstraintsPrompt(sanitation_instruction),
        createDeepenPrompt(sanitation_instruction),
        createConcretizingPrompt(sanitation_instruction),
        createReasoningPrompt(sanitation_instruction),
        createBreadthPrompt(sanitation_instruction)
    ]

    selected_evol_prompt = random.choice(evol_prompts)
    evol_instruction = call_chatgpt(selected_evol_prompt)
    if not evol_instruction:
        raise Exception("生成进化指令失败")

    positive_response = call_chatgpt(evol_instruction, developer=system)
    if not positive_response:
        raise Exception("生成进化响应失败")

    evol_obj = {
        "prompt": prompt,
        "instruction": evol_instruction,
        "positive_response": positive_response,
        "initial_response": initial_response
    }

    safe_write(outf, evol_obj)
    return evol_instruction, positive_response

async def process_task_async(cur_obj, outf):
    """异步处理单条任务逻辑"""
    prompt = cur_obj["prompt"]
    initial_response = cur_obj["response"]

    sanitation_prompt = createSanitizationPrompt(prompt)
    sanitation_instruction = await call_chatgpt_async(sanitation_prompt)
    if not sanitation_instruction:
        raise Exception("指令净化失败")

    evol_prompts = [
        createConstraintsPrompt(sanitation_instruction),
        createDeepenPrompt(sanitation_instruction),
        createConcretizingPrompt(sanitation_instruction),
        createReasoningPrompt(sanitation_instruction),
        createBreadthPrompt(sanitation_instruction)
    ]

    selected_evol_prompt = random.choice(evol_prompts)
    evol_instruction = await call_chatgpt_async(selected_evol_prompt)
    if not evol_instruction:
        raise Exception("生成进化指令失败")

    positive_response = await call_chatgpt_async(evol_instruction, developer=system)
    if not positive_response:
        raise Exception("生成进化响应失败")

    evol_obj = {
        "prompt": prompt,
        "sanitized_prompt": sanitation_instruction,
        "evol_instruction": evol_instruction,
        "positive_response": positive_response,
        "initial_response": initial_response
    }

    # 文件写入保持同步，避免并发写入冲突
    safe_write(outf, evol_obj)
    return evol_instruction, positive_response


# ============ 主流程 ============
async def process_with_semaphore(semaphore, cur_obj, outf, task_id, total_tasks):
    """使用信号量限制并发的任务处理函数"""
    async with semaphore:
        try:
            evol_instruction, positive_response = await process_task_async(cur_obj, outf)
            logger.info(f"[{task_id/total_tasks*100:5.1f}%] #{task_id}/{total_tasks} 🧬 成功 | 当前生成指令片段: {evol_instruction[:40]}...")
            logger.info(f"[{task_id/total_tasks*100:5.1f}%] #{task_id}/{total_tasks} 💡 成功 | 当前进化响应片段: {positive_response[:40]}...")
            return True
        except Exception as e:
            logger.warning(f"[{task_id/total_tasks*100:5.1f}%] #{task_id}/{total_tasks} ❌ 失败: {str(e)}")
            return False

async def main_async():
    """异步主函数，支持并发处理多个任务"""
    with open(input_data_path, 'r', encoding='utf-8') as inpf, \
         open(output_file_path, 'a', newline='', encoding='utf-8') as outf:

        all_objs = [json.loads(line) for line in inpf]
        total_tasks = len(all_objs)
        logger.info(f"开始异步处理 {total_tasks} 个任务")
        logger.info("=" * 40)

        start_time = time.time()
        success_count, error_count = 0, 0
        
        # 配置并发参数
        max_concurrent_tasks = 50  # 最大并发数，可根据API配额调整
        batch_size = 300  # 批次大小
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # 分批次处理任务
        for batch_start in range(0, total_tasks, batch_size):
            batch_end = min(batch_start + batch_size, total_tasks)
            batch_objs = all_objs[batch_start:batch_end]
            batch_tasks = []
            
            # 为当前批次的每个任务创建协程
            for i, cur_obj in enumerate(batch_objs):
                task_id = batch_start + i + 1
                task = process_with_semaphore(semaphore, cur_obj, outf, task_id, total_tasks)
                batch_tasks.append(task)
            
            # 并发执行当前批次的所有任务
            results = await asyncio.gather(*batch_tasks, return_exceptions=False)
            
            # 统计结果
            batch_success = sum(1 for result in results if result)
            batch_error = len(results) - batch_success
            
            success_count += batch_success
            error_count += batch_error
            
            # 每1000条创建一次备份
            if success_count > 0 and success_count % 1000 == 0:
                backup_file(output_file_path, success_count)
            
            logger.info(f"批次 {batch_start//batch_size + 1} 完成: 成功 {batch_success}, 失败 {batch_error}")
            
            # 批次之间添加短暂延迟，避免API过载
            if batch_end < total_tasks:
                await asyncio.sleep(2)  # 批次间延迟
        
        # 计算耗时
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes = remainder // 60

        logger.info("=" * 40)
        logger.info(f"处理完成! 总耗时: {int(hours)}小时{int(minutes)}分钟")
        logger.info(f"成功: {success_count}/{total_tasks}  |  失败: {error_count}/{total_tasks}")
        logger.info("=" * 40)
        
        return success_count, error_count, total_tasks

def main():
    # 使用异步模式执行
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
