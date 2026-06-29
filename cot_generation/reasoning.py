"""
主 CoT 推理生成脚本
====================

本脚本是确定性推理管道的核心入口，负责：
1. 加载训练数据
2. 对每道题目进行分类
3. 调用对应的推理器生成确定性解题过程
4. 将结果保存为 JSONL 格式

运行方式：
    python reasoning.py [--input INPUT_CSV] [--output OUTPUT_DIR]

设计理念：
- 确定性推理：对于每种已知题型，通过规则求解而非LLM生成
- 保证答案正确性：通过示例验证推断出的规则
- 生成高质量 CoT：推理过程详细且逻辑严密，用于 SFT 训练
"""

import os
import sys
import json
import math
import re
import argparse
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict

# 确保能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_CSV, REASONING_OUTPUT_DIR, OUTPUT_DIR,
    CATEGORY_PATTERNS, CORPUS_CONFIG,
)
from store_types import Problem, ReasoningResult, load_problems, classify_problem
from reasoners import get_reasoner, REASONER_REGISTRY


def compare_answer(stored_answer: str, predicted: str) -> bool:
    """
    验证答案是否匹配（移植自原始方案的比较逻辑）。
    
    对于数值答案，允许 1% 相对误差；
    对于二进制位串，严格字符串比较；
    否则忽略大小写进行字符串比较。
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    
    # 如果答案是纯二进制位串，严格比较
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()
    
    try:
        # 尝试将答案转为浮点数比较
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # 回退为忽略大小写的字符串比较
        return predicted.lower() == stored_answer.lower()


# ============================================================
# 日志配置
# ============================================================
def setup_logging(log_dir: str = None):
    """配置日志系统"""
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"reasoning_{datetime.now():%Y%m%d_%H%M%S}.log")
        handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]
    else:
        handlers = [logging.StreamHandler()]
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


# ============================================================
# 主推理流程
# ============================================================

def run_reasoning(
    input_csv: str,
    output_dir: str,
    categories: List[str] = None,
    verify: bool = True,
) -> Dict[str, dict]:
    """
    运行确定性推理流程。
    
    Args:
        input_csv: 输入 CSV 文件路径
        output_dir: 输出目录
        categories: 要处理的题型列表（None 表示全部）
        verify: 是否验证预测答案与标准答案的一致性
        
    Returns:
        统计信息字典
    """
    logger = logging.getLogger(__name__)
    
    # 加载数据
    logger.info(f"加载数据: {input_csv}")
    problems = load_problems(input_csv)
    logger.info(f"总题目数: {len(problems)}")
    
    # 按类别分组
    by_category = defaultdict(list)
    for p in problems:
        by_category[p.category].append(p)
    
    logger.info("题目分类统计:")
    for cat, probs in sorted(by_category.items()):
        logger.info(f"  {cat}: {len(probs)} 题")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    stats = {}
    all_results = []
    
    # 对每个类别运行推理
    target_categories = categories if categories else list(REASONER_REGISTRY.keys())
    
    for category in target_categories:
        if category not in by_category:
            logger.warning(f"类别 '{category}' 没有对应题目，跳过")
            continue
        
        probs = by_category[category]
        reasoner = get_reasoner(category)
        
        if reasoner is None:
            logger.warning(f"类别 '{category}' 没有对应推理器，跳过")
            stats[category] = {"total": len(probs), "success": 0, "correct": 0, "rate": 0.0}
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"处理类别: {category} ({len(probs)} 题)")
        logger.info(f"{'='*60}")
        
        category_results = []
        success_count = 0
        correct_count = 0
        
        for i, problem in enumerate(probs):
            result = reasoner.solve(problem)
            category_results.append(result)
            
            if result.success:
                success_count += 1
                # 验证答案（使用容差比较）
                if verify and compare_answer(problem.answer, result.predicted_answer):
                    correct_count += 1
                elif verify:
                    # 答案不匹配时记录日志（仅前几个）
                    if success_count - correct_count <= 5:
                        logger.debug(
                            f"  答案不匹配 [{problem.id}]: "
                            f"预测='{result.predicted_answer}' vs 标准='{problem.answer}'"
                        )
            
            # 进度报告
            if (i + 1) % 200 == 0:
                logger.info(f"  进度: {i+1}/{len(probs)}, 成功率: {success_count}/{i+1}")
        
        # 保存该类别的结果
        output_file = os.path.join(output_dir, f"{category}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in category_results:
                if result.success:
                    record = {
                        "problem_id": result.problem_id,
                        "category": result.category,
                        "predicted_answer": result.predicted_answer,
                        "reasoning_text": result.reasoning_text,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # 统计
        rate = correct_count / len(probs) * 100 if probs else 0
        stats[category] = {
            "total": len(probs),
            "success": success_count,
            "correct": correct_count,
            "rate": rate,
        }
        
        logger.info(f"  类别 {category} 完成:")
        logger.info(f"    总题数: {len(probs)}")
        logger.info(f"    推理成功: {success_count} ({success_count/len(probs)*100:.1f}%)")
        logger.info(f"    答案正确: {correct_count} ({rate:.1f}%)")
        
        all_results.extend(category_results)
    
    # 汇总统计
    total = sum(s["total"] for s in stats.values())
    total_success = sum(s["success"] for s in stats.values())
    total_correct = sum(s["correct"] for s in stats.values())
    
    logger.info(f"\n{'='*60}")
    logger.info("总体统计:")
    logger.info(f"  总题数: {total}")
    logger.info(f"  推理成功: {total_success} ({total_success/total*100:.1f}%)" if total > 0 else "  无数据")
    logger.info(f"  答案正确: {total_correct} ({total_correct/total*100:.1f}%)" if total > 0 else "  无数据")
    logger.info(f"{'='*60}")
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, "stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计信息已保存: {stats_file}")
    
    return stats


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="CoT 确定性推理生成")
    parser.add_argument(
        "--input", type=str, default=TRAIN_CSV,
        help="输入 CSV 文件路径"
    )
    parser.add_argument(
        "--output", type=str, default=REASONING_OUTPUT_DIR,
        help="输出目录"
    )
    parser.add_argument(
        "--categories", type=str, nargs='+', default=None,
        help="指定要处理的题型（默认全部）"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="跳过答案验证"
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="日志输出目录"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_dir)
    
    run_reasoning(
        input_csv=args.input,
        output_dir=args.output,
        categories=args.categories,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
