"""
本地离线评测脚本（Task 1）
===========================
加载 base + LoRA adapter，对 data/holdout_500.csv 做 vLLM 推理，
按竞赛评测标准（精确字符串匹配 或 1% 相对数值容忍）打分，
并按 category 分桶输出准确率与错题清单。

预期用法（AutoDL）:
    python eval_local.py \
        --model_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --lora_path ./output/nemotron_sft/checkpoint-xxx \
        --holdout_csv data/holdout_500.csv \
        --output_dir ./eval_reports

产出:
    eval_reports/eval_report_{timestamp}.json  # 总分 + 分桶分 + 错题
    eval_reports/eval_detail_{timestamp}.csv   # 每条样本的推理明细
"""

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ---------- 评测核心（严格对齐官方 nvidia-nemotron-metric.ipynb） ----------


def extract_final_answer(text: Optional[str]) -> str:
    r"""官方 metric extract_final_answer 的逐行复刻（v15 版本）。

    优先级：\\boxed{} → 4 种 'final answer' 启发式 → 末尾数字 → 末尾行。
    找不到任何东西返回 'NOT_FOUND'。

    v15 改动：boxed 提取改为"定位所有 \\boxed{ 起点 + 取段内最后一个 }"，
    可正确处理嵌套 LaTeX（如 \\frac{1}{2}）和答案本身含 } 的情况。
    """
    if text is None:
        return "NOT_FOUND"

    # 1) \boxed{...}（v15: 多阶段定位 + 段内末尾 } 匹配）
    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # 2) 4 种 final-answer 启发式
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        m = re.findall(pattern, text, re.IGNORECASE)
        if m:
            return m[-1].strip()

    # 3) 末尾数字
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    if m:
        return m[-1]

    # 4) 末尾非空行
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """官方 metric verify 函数的逐行复刻。

    - 二进制位串：严格字符串比对（lower）
    - 数值：math.isclose(rel_tol=1e-2, abs_tol=1e-5)
    - 其它：lower 后字符串比对
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


# ---------- Prompt 构造（与官方 generate_predictions 完全一致） ----------


def build_user_content(prompt: str) -> str:
    """官方:
    user_content = item.prompt
        + '\\nPlease put your final answer inside `\\\\boxed{}`. '
          'For example: `\\\\boxed{your answer}`'
    """
    return (
        prompt
        + "\nPlease put your final answer inside `\\boxed{}`. "
          "For example: `\\boxed{your answer}`"
    )


def format_prompts_with_chat_template(
    tokenizer, user_prompts: List[str], enable_thinking: bool = True
) -> List[str]:
    formatted = []
    for user_prompt in user_prompts:
        messages = [{"role": "user", "content": user_prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        formatted.append(text)
    return formatted


# ---------- vLLM 推理 ----------


def run_vllm_inference(
    model_path: str,
    lora_path: Optional[str],
    prompts: List[str],
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    top_p: float,
    gpu_memory_utilization: float,
    max_lora_rank: int,
    max_num_seqs: int,
) -> List[str]:
    """与官方 generate_predictions 中 LLM/SamplingParams 初始化完全对齐。"""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank

    print(f"[vLLM] init model={model_path} lora={lora_path}")
    llm = LLM(**llm_kwargs)

    if lora_path:
        lora_request = LoRARequest("eval_adapter", 1, lora_path)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    return [o.outputs[0].text for o in outputs]


# ---------- 主流程 ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="base model path")
    parser.add_argument("--lora_path", default=None, help="LoRA adapter path (可选)")
    parser.add_argument(
        "--holdout_csv",
        default="data/holdout_500.csv",
    )
    parser.add_argument("--output_dir", default="./eval_reports")
    parser.add_argument("--tag", default="", help="报告文件名附加标签，区分不同实验")

    # 推理参数（严格对齐 Kaggle metric）
    parser.add_argument("--max_tokens", type=int, default=7680)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_lora_rank", type=int, default=32)
    parser.add_argument("--max_num_seqs", type=int, default=64)

    parser.add_argument(
        "--num_samples", type=int, default=-1, help="评测前 N 条（-1 = 全部）"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""

    # ---- 1. 加载 holdout ----
    df = pd.read_csv(args.holdout_csv)
    required_cols = {"id", "category", "prompt", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"holdout csv missing columns {required_cols - set(df.columns)}"
        )
    df["answer"] = df["answer"].astype(str)

    if args.num_samples > 0 and args.num_samples < len(df):
        df = df.head(args.num_samples).reset_index(drop=True)

    print(f"[data] loaded {len(df)} samples from {args.holdout_csv}")
    print(f"[data] category dist:\n{df['category'].value_counts().to_string()}")

    # ---- 2. 构造 prompt ----
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    user_prompts = [build_user_content(p) for p in df["prompt"].tolist()]
    formatted = format_prompts_with_chat_template(
        tokenizer, user_prompts, enable_thinking=True
    )
    print(f"[prompt sample]\n{formatted[0][:400]}\n...")

    # ---- 3. 推理 ----
    raw_outputs = run_vllm_inference(
        model_path=args.model_path,
        lora_path=args.lora_path,
        prompts=formatted,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_lora_rank=args.max_lora_rank,
        max_num_seqs=args.max_num_seqs,
    )

    # ---- 4. 评估（严格对齐 Kaggle verify） ----
    records = []
    per_cat = defaultdict(lambda: {"total": 0, "correct": 0, "extract_fail": 0})
    wrong_cases = []

    for i, row in df.iterrows():
        output = raw_outputs[i]
        pred = extract_final_answer(output)
        extract_fail = pred == "NOT_FOUND"
        correct = verify(str(row["answer"]), pred)
        cat = row["category"]

        per_cat[cat]["total"] += 1
        if correct:
            per_cat[cat]["correct"] += 1
        if extract_fail:
            per_cat[cat]["extract_fail"] += 1

        records.append(
            dict(
                id=row["id"],
                category=cat,
                answer=row["answer"],
                prediction=pred,
                correct=bool(correct),
                extract_fail=bool(extract_fail),
                output_len=len(output),
                output=output,
            )
        )

        if not correct:
            wrong_cases.append(
                dict(
                    id=row["id"],
                    category=cat,
                    gt=row["answer"],
                    pred=pred,
                    output_tail=output[-300:],
                )
            )

    total = len(df)
    total_correct = sum(1 for r in records if r["correct"])
    total_extract_fail = sum(1 for r in records if r["extract_fail"])
    overall_acc = total_correct / total if total else 0.0

    per_cat_report = {}
    for cat, st in per_cat.items():
        per_cat_report[cat] = {
            "total": st["total"],
            "correct": st["correct"],
            "accuracy": st["correct"] / st["total"] if st["total"] else 0.0,
            "extract_fail": st["extract_fail"],
        }

    # ---- 5. 产出报告 ----
    report = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "holdout_csv": args.holdout_csv,
        "tag": args.tag,
        "total": total,
        "correct": total_correct,
        "accuracy": overall_acc,
        "extract_fail": total_extract_fail,
        "per_category": per_cat_report,
        "wrong_cases": wrong_cases,
    }

    report_path = Path(args.output_dir) / f"eval_report_{timestamp}{suffix}.json"
    detail_path = Path(args.output_dir) / f"eval_detail_{timestamp}{suffix}.csv"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # detail csv 只保留核心列，output 另存副本（可能很大）
    detail_df = pd.DataFrame(records)
    detail_df.to_csv(detail_path, index=False)

    # 控制台摘要
    print("=" * 60)
    print(f"[eval summary] total={total} correct={total_correct} acc={overall_acc:.4f}")
    print(f"extract_fail={total_extract_fail}")
    print("-- per category --")
    for cat, st in sorted(per_cat_report.items()):
        print(
            f"  {cat:<22} total={st['total']:>3} correct={st['correct']:>3} "
            f"acc={st['accuracy']:.4f} extract_fail={st['extract_fail']}"
        )
    print("=" * 60)
    print(f"report -> {report_path}")
    print(f"detail -> {detail_path}")


if __name__ == "__main__":
    main()
