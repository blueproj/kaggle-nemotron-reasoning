"""
Nemotron LoRA SFT 推理脚本 (vLLM)
====================================
使用 vLLM 加载基础模型 + LoRA 适配器进行高效推理。
在本地 train.csv 上验证模型效果并计算准确率。

使用方法:
    python infer_vllm.py \
        --model_path nvidia/Nemotron-Mini-4B-Instruct \
        --lora_path ./output/nemotron_lora \
        --data_path data/train.csv \
        --num_samples 500
"""

import argparse
import re
import pandas as pd
import numpy as np
import torch
import random
from typing import List, Optional, Tuple


def setup_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    从模型输出中提取 \\boxed{} 中的答案。

    支持嵌套大括号的情况，使用栈匹配最外层 \\boxed{} 的内容。
    例如:
        "The answer is \\boxed{42}" -> "42"
        "\\boxed{x^2 + 1}" -> "x^2 + 1"
    """
    # 方法1: 正则匹配（适用于简单情况）
    # 查找最后一个 \boxed{...}（模型可能在思考过程中也输出 boxed）
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # 方法2: 手动栈匹配（处理复杂嵌套）
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i-1].strip()

    return None


def build_inference_prompt(prompt: str) -> str:
    """
    构造推理时的用户提示。
    与训练时格式保持一致。
    """
    user_content = (
        f"{prompt}\n"
        f"Please put your final answer inside \\boxed{{}}. "
        f"For example: \\boxed{{your answer}}"
    )
    return user_content


def run_vllm_inference(
    model_path: str,
    lora_path: Optional[str],
    prompts: List[str],
    max_tokens: int = 7680,
    max_model_len: int = 8192,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.9,
    max_lora_rank: int = 32,
) -> List[str]:
    """
    使用 vLLM 进行批量推理。

    参数:
        model_path: 基础模型路径
        lora_path: LoRA 适配器路径（None 则不加载 LoRA）
        prompts: 格式化后的 prompt 列表
        max_tokens: 最大生成 token 数（竞赛要求 7680）
        max_model_len: 模型最大上下文长度（竞赛要求 8192）
        temperature: 采样温度（竞赛要求 0.0，即 greedy）
        gpu_memory_utilization: GPU 显存利用率
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # 配置采样参数（竞赛评测参数）
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        # temperature=0 时不需要 top_p
    )

    # 初始化 vLLM 引擎
    llm_kwargs = {
        "model": model_path,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }

    # 如果需要加载 LoRA
    # LoRA adapter 由 peft 的 save_pretrained() 保存，包含:
    #   - adapter_config.json: LoRA 配置（rank, alpha, target_modules 等）
    #   - adapter_model.safetensors: LoRA 权重
    # vLLM 可直接加载此格式，无需额外转换
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank  # 需 >= 训练时的 lora_r

    print(f"正在初始化 vLLM 引擎...")
    print(f"  模型: {model_path}")
    if lora_path:
        print(f"  LoRA: {lora_path}")
    print(f"  最大序列长度: {max_model_len}")
    print(f"  最大生成 token: {max_tokens}")

    llm = LLM(**llm_kwargs)

    # 执行推理
    print(f"开始推理 {len(prompts)} 条数据...")
    if lora_path:
        lora_request = LoRARequest("nemotron_lora", 1, lora_path)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    # 提取生成结果
    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(generated_text)

    return results


def evaluate_accuracy(
    predictions: List[Optional[str]],
    ground_truths: List[str],
) -> Tuple[float, int, int]:
    """
    计算预测准确率。

    返回: (准确率, 正确数, 总数)
    """
    correct = 0
    total = len(ground_truths)

    for pred, gt in zip(predictions, ground_truths):
        if pred is not None and pred.strip() == gt.strip():
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser(description="Nemotron vLLM Inference")

    parser.add_argument(
        "--model_path", type=str, required=True,
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora_path", type=str, default=None,
        help="LoRA 适配器路径（不指定则只用基础模型）"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="data/train.csv",
        help="验证数据 CSV 路径"
    )
    parser.add_argument(
        "--num_samples", type=int, default=500,
        help="验证样本数（-1 表示全部）"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=7680,
        help="最大生成 token 数（竞赛要求 7680）"
    )
    parser.add_argument(
        "--max_model_len", type=int, default=8192,
        help="模型最大上下文长度（竞赛要求 8192）"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="采样温度（竞赛要求 0.0）"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="GPU 显存利用率"
    )
    parser.add_argument(
        "--max_lora_rank", type=int, default=32,
        help="LoRA 最大 rank（需 >= 训练时的 lora_r，默认 32）"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="推理结果保存路径（CSV）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    setup_seed(args.seed)

    # ---- 1. 加载数据 ----
    print(f"加载数据: {args.data_path}")
    df = pd.read_csv(args.data_path)
    df["answer"] = df["answer"].astype(str)

    # 抽样
    if args.num_samples > 0 and args.num_samples < len(df):
        df = df.sample(n=args.num_samples, random_state=args.seed).reset_index(drop=True)
        print(f"随机抽样 {args.num_samples} 条数据进行验证")
    else:
        print(f"使用全部 {len(df)} 条数据进行验证")

    # ---- 2. 构造推理 prompt ----
    # 使用 tokenizer 的 chat_template 格式化
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    formatted_prompts = []
    for _, row in df.iterrows():
        user_content = build_inference_prompt(row["prompt"])
        messages = [{"role": "user", "content": user_content}]
        # 使用 chat_template 格式化，添加 generation prompt
        # enable_thinking=True: 某些模型支持在 apply_chat_template 中设置
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # 启用思考模式
            )
        except TypeError:
            # 如果 tokenizer 不支持 enable_thinking 参数
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        formatted_prompts.append(text)

    print(f"Prompt 示例:\n{formatted_prompts[0][:300]}...\n")

    # ---- 3. 执行推理 ----
    results = run_vllm_inference(
        model_path=args.model_path,
        lora_path=args.lora_path,
        prompts=formatted_prompts,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_lora_rank=args.max_lora_rank,
    )

    # ---- 4. 提取答案并评估 ----
    print("\n正在提取答案并计算准确率...")
    predictions = []
    for i, result in enumerate(results):
        answer = extract_boxed_answer(result)
        predictions.append(answer)

        # 打印前几个样本的结果
        if i < 5:
            print(f"\n--- 样本 {i+1} ---")
            print(f"真实答案: {df.iloc[i]['answer']}")
            print(f"提取答案: {answer}")
            print(f"模型输出 (前200字): {result[:200]}")

    # 计算准确率
    accuracy, correct, total = evaluate_accuracy(predictions, df["answer"].tolist())
    print(f"\n{'='*60}")
    print(f"验证结果:")
    print(f"  总样本数: {total}")
    print(f"  正确数: {correct}")
    print(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  答案提取失败: {predictions.count(None)}")
    print(f"{'='*60}")

    # ---- 5. 保存结果 ----
    if args.output_file:
        result_df = df[["id", "prompt", "answer"]].copy()
        result_df["prediction"] = predictions
        result_df["model_output"] = results
        result_df["correct"] = [
            pred is not None and pred.strip() == gt.strip()
            for pred, gt in zip(predictions, df["answer"].tolist())
        ]
        result_df.to_csv(args.output_file, index=False)
        print(f"\n结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
