#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict


DIRECTORIES = [
    "/data/liuyun/Evol-Preference/infer_harmfulqa",
    "/data/liuyun/Evol-Preference/infer_results_beavertails_evaluation",
    "/data/liuyun/Evol-Preference/infer_results_beavertails_test",
]

OUTPUT_DIR = "/data/liuyun/Evol-Preference/eval_report"

QUALITY_THRESHOLD = 3


@dataclass
class DisagreementRecord:
    prompt: str
    model_name: str
    directory: str
    gpt_judge: Dict[str, Any]
    claude_judge: Dict[str, Any]
    disagreement_type: str
    quality_diff: int


def load_judge_data(file_path: str) -> Dict[str, Dict]:
    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    prompt = record.get('prompt', '')
                    if prompt:
                        data[prompt] = record
                except json.JSONDecodeError as e:
                    print(f"  [警告] 文件 {os.path.basename(file_path)} 第 {line_num} 行 JSON 解析错误: {e}")
                except Exception as e:
                    print(f"  [警告] 文件 {os.path.basename(file_path)} 第 {line_num} 行处理错误: {e}")
    except FileNotFoundError:
        print(f"  [警告] 文件不存在: {file_path}")
    except Exception as e:
        print(f"  [错误] 读取文件 {file_path} 失败: {e}")
    return data


def compare_judgments(
    gpt_data: Dict[str, Dict],
    claude_data: Dict[str, Dict],
    model_name: str,
    directory: str ) -> List[DisagreementRecord]:
    disagreements = []
    
    common_prompts = set(gpt_data.keys()) & set(claude_data.keys())
    
    for prompt in common_prompts:
        gpt_record = gpt_data[prompt]
        claude_record = claude_data[prompt]
        
        try:
            gpt_judge = gpt_record.get('judge', {})
            claude_judge = claude_record.get('judge', {})
            
            gpt_harmlessness = gpt_judge.get('harmlessness')
            claude_harmlessness = claude_judge.get('harmlessness')
            
            gpt_quality = gpt_judge.get('quality')
            claude_quality = claude_judge.get('quality')
            
            if gpt_harmlessness is None or claude_harmlessness is None:
                continue
            if gpt_quality is None or claude_quality is None:
                continue
            
            harmlessness_diff = gpt_harmlessness != claude_harmlessness
            quality_diff = abs(gpt_quality - claude_quality)
            quality_diff_significant = quality_diff >= QUALITY_THRESHOLD
            
            if harmlessness_diff or quality_diff_significant:
                if harmlessness_diff and quality_diff_significant:
                    disagreement_type = "both"
                elif harmlessness_diff:
                    disagreement_type = "harmlessness_only"
                else:
                    disagreement_type = "quality_only"
                
                record = DisagreementRecord(
                    prompt=prompt,
                    model_name=model_name,
                    directory=directory,
                    gpt_judge={
                        "harmlessness": gpt_harmlessness,
                        "quality": gpt_quality
                    },
                    claude_judge={
                        "harmlessness": claude_harmlessness,
                        "quality": claude_quality
                    },
                    disagreement_type=disagreement_type,
                    quality_diff=quality_diff
                )
                disagreements.append(record)
                
        except Exception as e:
            print(f"  [警告] 处理 prompt 时出错: {str(e)[:50]}...")
            continue
    
    return disagreements


def process_directory(directory: str) -> List[DisagreementRecord]:
    all_disagreements = []
    
    if not os.path.exists(directory):
        print(f"[警告] 目录不存在: {directory}")
        return all_disagreements
    
    gpt_files = {}
    claude_files = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('_judge_gpt.jsonl'):
            model_name = filename.replace('_judge_gpt.jsonl', '')
            gpt_files[model_name] = os.path.join(directory, filename)
        elif filename.endswith('_judge_claude.jsonl'):
            model_name = filename.replace('_judge_claude.jsonl', '')
            claude_files[model_name] = os.path.join(directory, filename)
    
    common_models = set(gpt_files.keys()) & set(claude_files.keys())
    
    print(f"\n处理目录: {os.path.basename(directory)}")
    print(f"  发现 {len(common_models)} 对可比较的模型文件")
    
    for model_name in common_models:
        gpt_path = gpt_files[model_name]
        claude_path = claude_files[model_name]
        
        print(f"  处理模型: {model_name}")
        
        gpt_data = load_judge_data(gpt_path)
        claude_data = load_judge_data(claude_path)
        
        if not gpt_data or not claude_data:
            print(f"    [跳过] 数据为空")
            continue
        
        disagreements = compare_judgments(
            gpt_data, claude_data, model_name, os.path.basename(directory)
        )
        
        all_disagreements.extend(disagreements)
        print(f"    发现 {len(disagreements)} 条分歧记录")
    
    return all_disagreements


def generate_report(all_disagreements: List[DisagreementRecord]) -> Dict:
    summary = {
        "total_disagreements": len(all_disagreements),
        "harmlessness_only": 0,
        "quality_only": 0,
        "both": 0,
    }
    
    by_directory = defaultdict(lambda: {
        "total": 0,
        "harmlessness_only": 0,
        "quality_only": 0,
        "both": 0
    })
    
    by_model = defaultdict(lambda: {
        "total": 0,
        "harmlessness_only": 0,
        "quality_only": 0,
        "both": 0
    })
    
    for record in all_disagreements:
        summary[record.disagreement_type] += 1
        by_directory[record.directory][record.disagreement_type] += 1
        by_directory[record.directory]["total"] += 1
        by_model[record.model_name][record.disagreement_type] += 1
        by_model[record.model_name]["total"] += 1
    
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "directories_analyzed": len(DIRECTORIES),
            "quality_threshold": QUALITY_THRESHOLD,
        },
        "summary": summary,
        "by_directory": dict(by_directory),
        "by_model": dict(by_model),
        "detailed_disagreements": [asdict(r) for r in all_disagreements]
    }
    
    return report


def print_console_report(report: Dict):
    print("\n" + "=" * 60)
    print("GPT vs Claude 判断分歧分析报告")
    print(f"生成时间: {report['metadata']['generated_at']}")
    print(f"Quality 阈值: ≥ {report['metadata']['quality_threshold']}")
    print("=" * 60)
    
    summary = report['summary']
    total = summary['total_disagreements']
    
    print("\n一、总体统计")
    print("-" * 40)
    print(f"总分歧记录数: {total}")
    print(f"  - harmlessness 不一致: {summary['harmlessness_only']}")
    print(f"  - quality 不一致 (差值≥3): {summary['quality_only']}")
    print(f"  - 两者都不一致: {summary['both']}")
    
    print("\n二、分目录统计")
    print("-" * 40)
    for dir_name, stats in report['by_directory'].items():
        print(f"\n{dir_name}:")
        print(f"  总分歧数: {stats['total']}")
        print(f"  - harmlessness 不一致: {stats['harmlessness_only']}")
        print(f"  - quality 不一致: {stats['quality_only']}")
        print(f"  - 两者都不一致: {stats['both']}")
    
    print("\n三、分模型统计")
    print("-" * 40)
    for model_name, stats in sorted(report['by_model'].items()):
        print(f"\n{model_name}:")
        print(f"  总分歧数: {stats['total']}")
        print(f"  - harmlessness 不一致: {stats['harmlessness_only']}")
        print(f"  - quality 不一致: {stats['quality_only']}")
        print(f"  - 两者都不一致: {stats['both']}")
    
    print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("GPT vs Claude 判断分歧分析")
    print("=" * 60)
    
    all_disagreements = []
    
    for directory in DIRECTORIES:
        disagreements = process_directory(directory)
        all_disagreements.extend(disagreements)
    
    report = generate_report(all_disagreements)
    
    print_console_report(report)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "judge_disagreement_report.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细报告已保存至: {output_path}")
    
    return report


if __name__ == "__main__":
    main()
