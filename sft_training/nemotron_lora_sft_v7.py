"""
Nemotron-3-Nano-30B-A3B LoRA SFT v7 训练脚本 (严格 quota 分层 sampler)
=====================================================================
v7 核心改动 (相对 v6):
  1. 采样从 WeightedRandomSampler (概率期望保证) 换为 StratifiedSampler
     (严格 quota 保证):
     - v6: cryptarithm 期望 2/batch, 实际服从 B(32, 0.0625) 分布,
           仍有01-8 条波动, 约 15.5% 的 batch 完全无 cryptarithm
     - v7: cryptarithm 严格保底 2 条/batch, 其他 30 条按自然分布
           在其他 6 类中随机分配
  2. CategoryBalancedTrainer → StratifiedTrainer
  3. dataloader_num_workers 4 → 0 (避免多 worker 复制 sampler 状态导致样本重复)
  4. 其余超参 (lr/alpha=32/min_lr=1e-5/epochs=1/warmup=0.05) 与 v6 完全一致

设计要点:
  - StratifiedSampler 是样本级 sampler (yield 单个 index),
    在每 eff_batch (per_device_batch_size × grad_accum) 个连续 yield 中
    严格包含: cryptarithm 2 条 + 其他类 30 条(按自然分布分配)。
  - HuggingFace Trainer 用 batch_size=1 + grad_accum=32, sampler 顺序 yield 后,
    每 32 个 micro-batch 形成一个梯度累积单元 = 一个 effective batch,
    则该梯度单元严格含 2 条 cryptarithm。
  - 多 epoch 时, sampler.set_epoch 使用不同 seed, 避免顺序重复。

基础模型: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (MoE 架构)
框架: transformers + peft + bitsandbytes (QLoRA 模式)
"""

import copy
import json
import logging
import os
import argparse
import torch
import transformers
import numpy as np
import random
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Sequence
from functools import partial

from datasets import Dataset
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ============================================================
# 常量定义
# ============================================================
IGNORE_INDEX = -100  # 用于标记不计算 loss 的 token (即 prompt 部分)

# 与 Kaggle 评测、推理端一致的 user prompt 后缀
PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

# v3: 权威 7 类 reasoner_key 顺序
# (与 code/cot/reasoners/__init__.py 注册表 + build_train_corpus_v2.py type 列一致)
FINE_CATEGORIES: List[str] = [
    "bit_manipulation",
    "cipher",
    "gravity",
    "numeral",
    "unit_conversion",
    "cryptarithm",
    "equation_numeric",
]


# ============================================================
# 工具函数
# ============================================================

def setup_seed(seed: int = 42):
    """设置全局随机种子，确保实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: str) -> logging.Logger:
    """创建带时间戳的日志系统，同时输出到文件和控制台。"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    os.makedirs(log_dir, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_v7_{time_str}.log")

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# ============================================================
# 数据处理 — CSV-CoT 格式专用 (9 类版)
# ============================================================

def load_csv_cot(data_path: str) -> pd.DataFrame:
    """
    加载 CSV-CoT 训练数据 (v3 7 类 reasoner_key 版)。
    必须包含 5 列: id, prompt, answer, type, generated_cot
    - type: 7 类 reasoner_key 之一 (FINE_CATEGORIES)
    - generated_cot: 完整推理链（不含 <think> 前缀，脚本会自动加）

    若 type 列出现未知类别直接报错退出, 避免采样桶错位。
    """
    df = pd.read_csv(data_path)
    required_cols = {"id", "prompt", "answer", "type", "generated_cot"}
    missing = required_cols - set(df.columns)
    assert not missing, f"CSV 缺少必需列: {missing}，当前列: {df.columns.tolist()}"

    # 基础清洗
    df = df.dropna(subset=["prompt", "answer", "generated_cot", "type"])
    df["answer"] = df["answer"].astype(str)
    df["generated_cot"] = df["generated_cot"].astype(str)

    unknown = set(df["type"].unique()) - set(FINE_CATEGORIES)
    if unknown:
        raise ValueError(
            f"v4 type 列出现未知类别 {unknown}, "
            f"合法类别: {FINE_CATEGORIES}"
        )

    return df


def build_cot_messages(prompt: str, generated_cot: str) -> List[Dict[str, str]]:
    """
    构造 chat 格式训练数据（与 v2 一致）。

    user: prompt + PROMPT_SUFFIX（与 Kaggle 评测一致）
    assistant: <think>\\n + generated_cot（generated_cot 尾部已含 </think>\\n\\n\\boxed{GT}）
    """
    user_content = prompt + PROMPT_SUFFIX

    # generated_cot 不含 <think> 前缀，训练时补上
    # chat_template + enable_thinking=True 的 generation_prompt 会加 <think>\n
    # 为保证 prompt_len 对齐，assistant content 必须以 <think>\n 开头
    assistant_content = "<think>\n" + generated_cot

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def tokenize_chat_sample(
    messages: List[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
    max_response_length: int = 7680,
) -> Optional[Dict[str, torch.Tensor]]:
    """将 chat 消息 tokenize, 只对 assistant 部分计算 loss。
    过滤策略 (与 vLLM 评测端对齐):
      1. 总 token 数 > max_length (8192) → 丢弃 (max_model_len)
      2. response token 数 > max_response_length (7680) → 丢弃 (max_tokens)
    """
    try:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True
        )
    except TypeError:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    prompt_messages = [messages[0]]
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

    full_tokens = tokenizer(
        full_text, max_length=max_length, truncation=True,
        return_tensors="pt", add_special_tokens=False,
    )
    prompt_tokens = tokenizer(
        prompt_text, max_length=max_length, truncation=True,
        return_tensors="pt", add_special_tokens=False,
    )

    input_ids = full_tokens["input_ids"][0]
    labels = input_ids.clone()
    prompt_len = prompt_tokens["input_ids"].shape[1]
    labels[:prompt_len] = IGNORE_INDEX

    # 过滤1: 总长度超过 max_model_len (8192)
    if len(input_ids) > max_length:
        return None

    # 过滤2: response 长度超过 max_tokens (7680)
    response_len = len(input_ids) - prompt_len
    if response_len > max_response_length:
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": full_tokens["attention_mask"][0],
    }


def prepare_dataset(
    df: pd.DataFrame,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
    max_response_length: int = 7680,
    num_proc: int = 8,
) -> Dataset:
    """
    将 CSV-CoT DataFrame 转换为 tokenized HuggingFace Dataset (v3 7 类版)。

    与 v2 差异: cat2id 固定使用 FINE_CATEGORIES 顺序 (7 类), 不再 sort(unique)。
    保证不同数据集间 category_id 编码一致, 与采样桶绑定稳定。
    """
    logger = logging.getLogger(__name__)

    # ★ v3: 类别编码强制按 FINE_CATEGORIES 顺序
    cat2id = {cat: i for i, cat in enumerate(FINE_CATEGORIES)}
    actual_cats = sorted(df["type"].unique().tolist())
    logger.info(f"类别编码 (固定 7 类): {cat2id}")
    logger.info(f"数据中实际出现的类别: {actual_cats}")

    dataset = Dataset.from_dict({
        "prompt": df["prompt"].tolist(),
        "generated_cot": df["generated_cot"].tolist(),
        "category_id": [cat2id[t] for t in df["type"]],
    })
    logger.info(f"CSV-CoT 数据集大小: {len(dataset)} 条")

    def tokenize_function(examples):
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_category_id = []
        all_valid = []

        for prompt, cot, cat_id in zip(
            examples["prompt"], examples["generated_cot"], examples["category_id"]
        ):
            messages = build_cot_messages(prompt, cot)
            result = tokenize_chat_sample(messages, tokenizer, max_length, max_response_length)

            if result is None:
                all_input_ids.append([])
                all_labels.append([])
                all_attention_mask.append([])
                all_category_id.append(cat_id)
                all_valid.append(False)
            else:
                all_input_ids.append(result["input_ids"].tolist())
                all_labels.append(result["labels"].tolist())
                all_attention_mask.append(result["attention_mask"].tolist())
                all_category_id.append(cat_id)
                all_valid.append(True)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
            "category_id": all_category_id,
            "valid": all_valid,
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=["prompt", "generated_cot"],
        desc="Tokenizing CSV-CoT dataset (v7)",
    )

    total_before = len(tokenized_dataset)

    # v6: per-type 过滤统计 + 长度分布
    id2cat = {i: c for c, i in cat2id.items()}
    filter_by_cat: Dict[str, int] = {cat: 0 for cat in cat2id}
    keep_lengths_by_cat: Dict[str, List[int]] = {cat: [] for cat in cat2id}
    for sample in tokenized_dataset:
        cat_name = id2cat[sample["category_id"]]
        if not sample["valid"]:
            filter_by_cat[cat_name] += 1
        else:
            keep_lengths_by_cat[cat_name].append(len(sample["input_ids"]))

    tokenized_dataset = tokenized_dataset.filter(
        lambda x: x["valid"], desc="Filtering truncated samples"
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["valid"])
    num_filtered = total_before - len(tokenized_dataset)
    if num_filtered > 0:
        logger.info(
            f"过滤了 {num_filtered} 条样本 "
            f"(总长>{max_length} 或 response>{max_response_length} tokens)"
        )
        logger.info("各 type 过滤数:")
        for cat, n in sorted(filter_by_cat.items()):
            if n > 0:
                logger.info(f"  {cat:<22} {n:>5}")

    # v6: per-type token 长度分布
    logger.info("各 type token 长度分布 (有效样本):")
    logger.info(f"  {'cat':<22} {'n':>6} {'p50':>6} {'p95':>6} {'max':>6}")
    for cat in sorted(keep_lengths_by_cat):
        lens = keep_lengths_by_cat[cat]
        if not lens:
            continue
        p50 = int(np.percentile(lens, 50))
        p95 = int(np.percentile(lens, 95))
        mx = max(lens)
        logger.info(
            f"  {cat:<22} {len(lens):>6} {p50:>6} {p95:>6} {mx:>6}"
        )

    logger.info(f"Tokenization 完成，有效样本数: {len(tokenized_dataset)}")
    return tokenized_dataset, cat2id


# ============================================================
# v7 严格 quota 分层 Sampler
# ============================================================

class StratifiedSampler(Sampler):
    """严格 quota 分层采样器 (v7)。

    在每个 effective batch (per_device_batch_size × grad_accum) 内,
    按指定 quota 严格分配各类样本数量。

    例: eff_batch=32, target_per_batch={"cryptarithm": 2}
      - 每 32 个连续 yield 严格包含 2 条 cryptarithm + 30 条其他类
      - 其他 30 条按其余 6 类的样本数量比例(自然分布)随机分配

    抽样策略 (无放回顺序消费 + 用完 reshuffle):
      - boost 类: 池子打乱后顺序取, 用完重新 shuffle, 保证均匀轮到每条
      - 其他类: 一次性扁平 6 个池 → 打乱 → 顺序消费, 保证每条在 epoch 内被使用 ≥1 次
      这与 v6 (有放回 WeightedRandomSampler) 的区别: v6 下 ~38% 样本在 1 epoch 内
      完全没被见到, v7 采样使满覆盖率达 ~96% (仅最后不足一轮的部分未被见)。

    与 v6 WeightedRandomSampler 的区别:
      - v6: 概率期望保证 (单 batch 可能 0~8 条 cryptarithm, 均值 2)
      - v7: 严格 quota (每 batch 固定 2 条 cryptarithm)

    需要 batch_size=1 + grad_accum=eff_batch 的训练设置:
      Trainer 顺序消费 sampler yield 的 index, 每 grad_accum 个 micro-batch
      形成一个优化器 step 的梯度累积单元, 该单元严格含指定 quota。
    """

    def __init__(
        self,
        cat_ids_per_sample: Sequence[int],
        cat2id: Dict[str, int],
        target_per_batch: Dict[str, int],
        eff_batch: int,
        num_eff_batches: int,
        seed: int = 42,
    ):
        self.cat_ids = list(cat_ids_per_sample)
        self.cat2id = cat2id
        self.id2cat = {v: k for k, v in cat2id.items()}
        self.target_per_batch = dict(target_per_batch)
        self.eff_batch = eff_batch
        self.num_eff_batches = num_eff_batches
        self.base_seed = seed
        self._epoch = 0

        # 按 cat_id 分组 sample indices
        self.indices_by_cid: Dict[int, List[int]] = {cid: [] for cid in cat2id.values()}
        for i, cid in enumerate(self.cat_ids):
            if cid in self.indices_by_cid:
                self.indices_by_cid[cid].append(i)

        # 校验 boost 配置
        for cat in target_per_batch:
            if cat not in cat2id:
                raise ValueError(f"target_per_batch 中未知类别 {cat!r}")
            n = target_per_batch[cat]
            if not isinstance(n, int) or n <= 0:
                raise ValueError(f"target_per_batch[{cat!r}] = {n!r}, 必须为正整数")
            if not self.indices_by_cid[cat2id[cat]]:
                raise ValueError(f"类 {cat!r} 在数据集中无样本, 无法 boost")

        boost_total = sum(target_per_batch.values())
        if boost_total >= eff_batch:
            raise ValueError(
                f"boost 类总配额 {boost_total} >= eff_batch {eff_batch}, 无剩余配额给其他类"
            )

        self.boost_cids: Dict[int, int] = {cat2id[c]: n for c, n in target_per_batch.items()}
        self.other_cids: List[int] = [
            cid for cid in cat2id.values() if cid not in self.boost_cids
        ]
        self.other_quota = eff_batch - boost_total

        # 其他类的自然分布概率 (按样本数量加权)
        other_counts = np.array(
            [len(self.indices_by_cid[cid]) for cid in self.other_cids],
            dtype=np.float64,
        )
        if other_counts.sum() == 0:
            raise ValueError("其他类全部无样本")
        self.other_probs = other_counts / other_counts.sum()

    def set_epoch(self, epoch: int) -> None:
        """与 DistributedSampler 类似, 允许外部控制 epoch 偏移。"""
        self._epoch = epoch

    def __len__(self) -> int:
        return self.num_eff_batches * self.eff_batch

    def __iter__(self):
        rng = np.random.default_rng(self.base_seed + self._epoch)
        self._epoch += 1  # 每次 __iter__ 自动不同 seed (多 epoch 顺序不同)

        # boost 类: 打乱池子顺序消费, 用完 reshuffle 重启
        # 保证每条 boost 样本都会被均匀轮到 (cryptarithm 1136 次 / 407 池 → 平均 ~2.8 次/条)
        boost_state: Dict[int, Dict] = {}
        for cid in self.boost_cids:
            pool = list(self.indices_by_cid[cid])
            rng.shuffle(pool)
            boost_state[cid] = {"pool": pool, "pos": 0}

        # 其他类: 一次性扁平打乱, 顺序消费, 用完 reshuffle
        # 保证每条其他类样本在 epoch 内被使用 ≥1 次 (仅最后不足一轮的身位未被使用)
        other_pool: List[int] = []
        for cid in self.other_cids:
            other_pool.extend(self.indices_by_cid[cid])
        rng.shuffle(other_pool)
        other_pos = 0

        def _draw_boost_cid(cid: int, n: int) -> List[int]:
            st = boost_state[cid]
            out = []
            for _ in range(n):
                if st["pos"] >= len(st["pool"]):
                    rng.shuffle(st["pool"])
                    st["pos"] = 0
                out.append(int(st["pool"][st["pos"]]))
                st["pos"] += 1
            return out

        def _draw_other_one() -> int:
            nonlocal other_pos
            if other_pos >= len(other_pool):
                rng.shuffle(other_pool)
                other_pos = 0
            v = int(other_pool[other_pos])
            other_pos += 1
            return v

        # 日志辅助: 统计每个 batch 内各类别样本数 (移植自 v12 CurriculumSampler)
        _log = logging.getLogger(__name__)
        cids_ordered = sorted(self.cat2id.values())

        def _log_batch_dist(batch_idx: int, indices: List[int]) -> None:
            from collections import Counter as _Counter
            cnt = _Counter(self.cat_ids[i] for i in indices)
            parts = [f"{self.id2cat[cid]}={cnt.get(cid, 0)}" for cid in cids_ordered]
            _log.info(f"  [eff_batch {batch_idx:>4d}] {' | '.join(parts)}")

        for b in range(self.num_eff_batches):
            batch_indices: List[int] = []

            # 1) boost 类: 严格 quota, 无放回顺序消费
            for cid, n in self.boost_cids.items():
                batch_indices.extend(_draw_boost_cid(cid, n))

            # 2) 其他类: 无放回顺序消费扯平池, 按自然分布分配
            for _ in range(self.other_quota):
                batch_indices.append(_draw_other_one())

            # 3) batch 内 shuffle (避免 cryptarithm 总聚集在 batch 头部)
            rng.shuffle(batch_indices)

            # 4) 日志: 每个 eff_batch 都打印类别分布
            _log_batch_dist(b, batch_indices)

            for idx in batch_indices:
                yield idx


# ============================================================
# v7 Trainer (调用严格 quota sampler)
# ============================================================

class StratifiedTrainer(Trainer):
    """继承 HuggingFace Trainer, 用 StratifiedSampler 实现严格 quota 采样。

    与 v6 CategoryBalancedTrainer (WeightedRandomSampler / 概率期望) 不同,
    本 Trainer 让每个 effective batch 严格包含指定数量的 boost 类样本。
    """

    def __init__(self, *args, stratified_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratified_sampler = stratified_sampler

    def _get_train_sampler(self, dataset=None):
        if self.stratified_sampler is None:
            return super()._get_train_sampler(dataset)
        return self.stratified_sampler


# ============================================================
# 模型加载 (与 v2 完全一致)
# ============================================================

def load_model_and_tokenizer(
    model_name_or_path: str,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    use_gradient_checkpointing: bool = True,
    use_4bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16",
    use_double_quant: bool = True,
) -> tuple:
    """加载基础模型并应用 LoRA。支持 BF16 和 QLoRA 4-bit 两种模式。"""
    logger = logging.getLogger(__name__)

    logger.info(f"加载 tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map.get(bnb_4bit_compute_dtype, torch.bfloat16)

    if use_4bit:
        logger.info(f"QLoRA 模式: 4-bit {bnb_4bit_quant_type}, compute={bnb_4bit_compute_dtype}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing,
        )
    else:
        logger.info(f"BF16 模式加载: {model_name_or_path}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                use_cache=False,
                attn_implementation="flash_attention_2",
            )
            logger.info("使用 flash_attention_2 加速")
        except (ValueError, ImportError) as e:
            logger.warning(f"flash_attention_2 不可用 ({e})，回退到 eager")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                use_cache=False,
                attn_implementation="eager",
            )
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            logger.info("已启用 Gradient Checkpointing")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj", "out_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {lora_config.target_modules}")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================
# Callback
# ============================================================

class TrainingLogCallback(transformers.TrainerCallback):
    """自定义训练日志回调：格式化输出 loss/lr/grad_norm"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logger = logging.getLogger(__name__)
        if "loss" in logs:
            msg_parts = [f"step={state.global_step}"]
            for key in ["loss", "grad_norm", "learning_rate", "epoch"]:
                if key in logs:
                    val = logs[key]
                    msg_parts.append(f"{key}={val:.6f}" if isinstance(val, float) else f"{key}={val}")
            logger.info(" | ".join(msg_parts))


# ============================================================
# 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    """基于 v3, 新增 max_response_length 过滤。"""
    parser = argparse.ArgumentParser(description="Nemotron LoRA SFT v7 (严格 quota 分层 sampler)")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/nemotron_lora_v7")

    parser.add_argument(
        "--data_path", type=str, required=True,
        help="训练 CSV 路径 (必须含 id/prompt/answer/type/generated_cot 5 列, "
             "type ∈ FINE_CATEGORIES 7 类)"
    )
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument(
        "--max_response_length", type=int, default=7680,
        help="assistant response 最大 token 数 (与 vLLM max_tokens 对齐, 默认 7680)"
    )

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--use_4bit", action="store_true", default=False)
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16")
    parser.add_argument("--use_double_quant", action="store_true", default=True)
    parser.add_argument("--no_double_quant", action="store_true", default=False)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--no_balanced_sampling", action="store_true", default=False)
    parser.add_argument(
        "--class_target_per_batch", type=str, default="cryptarithm:2",
        help="指定类别的每 batch 严格 quota, 格式 'class1:n1,class2:n2' (n 必须为正整数)。"
             "未指定的类按自然分布在剩余 quota 内随机分配。"
             "v7 默认: cryptarithm:2 (小类严格保底, 其他类自然分布)。"
             "传空字符串则退化到默认 RandomSampler (无 boost)。"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_total_limit", type=int, default=6)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="./logs")

    args = parser.parse_args()
    if args.no_double_quant:
        args.use_double_quant = False
    if args.use_4bit and args.deepspeed:
        raise ValueError("QLoRA 4-bit 模式与 DeepSpeed 不兼容")
    return args


# ============================================================
# 主训练流程
# ============================================================

def main():
    args = parse_args()
    setup_seed(args.seed)
    logger = setup_logger(args.log_dir)

    logger.info("=" * 60)
    logger.info("Nemotron LoRA SFT v7 — Strict Quota Stratified Sampler")
    logger.info("=" * 60)
    logger.info(f"训练参数: {vars(args)}")

    # ---- 1. 加载模型 ----
    logger.info("正在加载模型...")
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        use_double_quant=args.use_double_quant,
    )

    # ---- 2. 加载 CSV-CoT 数据 ----
    logger.info(f"加载 CSV-CoT 数据: {args.data_path}")
    df = load_csv_cot(args.data_path)
    logger.info(f"原始数据量: {len(df)} 条")
    logger.info(f"类别分布:\n{df['type'].value_counts().to_string()}")

    train_dataset, cat2id = prepare_dataset(
        df=df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_response_length=args.max_response_length,
        num_proc=args.num_proc,
    )

    # ---- 3. 类别采样 (v7: 严格 quota 分层 sampler) ----
    # v7: 每 eff_batch 严格包含指定 quota 的 boost 类样本 (不再是概率期望)
    stratified_sampler = None
    if not args.no_balanced_sampling:
        cat_ids = train_dataset["category_id"]
        from collections import Counter
        cat_counts = Counter(cat_ids)
        total_n = sum(cat_counts.values())
        eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps

        # 解析 --class_target_per_batch (默认 cryptarithm:2)
        target_spec = (args.class_target_per_batch or "").strip()
        target_per_batch: Dict[str, int] = {}
        if target_spec:
            for item in target_spec.split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" not in item:
                    raise ValueError(
                        f"--class_target_per_batch 项 {item!r} 格式错误, "
                        f"应为 'cat:n' 且 n 为正整数"
                    )
                cat_name, n_str = item.split(":", 1)
                cat_name = cat_name.strip()
                try:
                    n_target = int(n_str.strip())  # v7: 严格 quota 必须为正整数
                except ValueError:
                    raise ValueError(
                        f"v7 --class_target_per_batch 中 {n_str!r} 不是整数, "
                        f"严格 quota 必须为正整数"
                    )
                if cat_name not in cat2id:
                    raise ValueError(
                        f"--class_target_per_batch 中未知类别 {cat_name!r}, "
                        f"合法类别: {list(cat2id)}"
                    )
                target_per_batch[cat_name] = n_target

        # 计算总 effective batch 数 (= total optimizer steps)
        num_eff_batches = (len(train_dataset) // eff_batch) * args.num_train_epochs

        if target_per_batch:
            stratified_sampler = StratifiedSampler(
                cat_ids_per_sample=cat_ids,
                cat2id=cat2id,
                target_per_batch=target_per_batch,
                eff_batch=eff_batch,
                num_eff_batches=num_eff_batches,
                seed=args.seed,
            )
            logger.info(f"v7 严格 quota sampler: target_per_batch={target_per_batch}")
            logger.info(
                f"  eff_batch={eff_batch}, num_eff_batches={num_eff_batches}, "
                f"total samples seen = {num_eff_batches * eff_batch}"
            )
            other_n_total = sum(
                cat_counts[cid] for cid in stratified_sampler.other_cids
            )
            logger.info(
                f"  {'cat':<22} {'n':>6} {'CSV占比':>9} {'每eff_batch':>12} {'每step占比':>10}  tag"
            )
            for cat_name, cat_id in sorted(cat2id.items(), key=lambda x: x[1]):
                cnt = cat_counts.get(cat_id, 0)
                ratio = cnt / total_n if total_n > 0 else 0.0
                if cat_name in target_per_batch:
                    expect = float(target_per_batch[cat_name])
                    tag = "*boost (严格)"
                else:
                    expect = stratified_sampler.other_quota * (cnt / other_n_total)
                    tag = "(自然分布)"
                logger.info(
                    f"  {cat_name:<22} {cnt:>6} {ratio:>8.2%} {expect:>11.2f} "
                    f"{expect/eff_batch:>9.2%}  {tag}"
                )
        else:
            logger.warning(
                "未指定 --class_target_per_batch, v7 退化到默认 RandomSampler (无 boost)"
            )

    train_dataset = train_dataset.remove_columns(["category_id"])

    # ---- 4. 首样本预览 + Loss Mask 验证 ----
    logger.info("=" * 60)
    logger.info("首 5 条训练样本预览:")
    logger.info("=" * 60)
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        total_tokens = len(input_ids)

        mask_boundary = 0
        for idx, l in enumerate(labels):
            if l != IGNORE_INDEX:
                mask_boundary = idx
                break

        num_label = sum(1 for l in labels if l != IGNORE_INDEX)
        prompt_text = tokenizer.decode(input_ids[:mask_boundary], skip_special_tokens=False)
        response_text = tokenizer.decode(input_ids[mask_boundary:], skip_special_tokens=False)

        logger.info(f"--- 样本 {i} ---")
        logger.info(f"  总tokens={total_tokens} | mask={total_tokens - num_label} | loss={num_label}")
        logger.info(f"  [prompt尾50]: ...{prompt_text[-50:]}")
        logger.info(f"  [response头100]: {response_text[:100]}")
        logger.info(f"  [response尾80]: ...{response_text[-80:]}")
    logger.info("=" * 60)

    # ---- 5. 训练配置 ----
    total_steps = (
        len(train_dataset)
        // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    ) * args.num_train_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    logger.info(f"total_steps={total_steps}, warmup_steps={warmup_steps}")

    optim = "paged_adamw_32bit" if args.use_4bit else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs={"min_lr": args.min_lr},
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataloader_num_workers=0,  # v7: 自定义 sampler 在多 worker 下会重复抽样, 强制 0
        remove_unused_columns=True,
        report_to="none",
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=optim,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        padding=True,
    )

    trainer = StratifiedTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[TrainingLogCallback()],
        stratified_sampler=stratified_sampler,
    )

    logger.info("开始训练...")
    logger.info(f"  模式 = {'QLoRA 4-bit' if args.use_4bit else 'BF16 LoRA'}")
    logger.info(f"  总样本数 = {len(train_dataset)}")
    logger.info(f"  有效batch = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Epochs = {args.num_train_epochs}")
    logger.info(f"  LR = {args.learning_rate}, Scheduler = {args.lr_scheduler_type}")
    logger.info(f"  Warmup = {warmup_steps} steps ({args.warmup_ratio*100:.1f}%)")
    logger.info(f"  Grad Clip = {args.max_grad_norm}")
    logger.info(f"  Weight Decay = {args.weight_decay}")
    logger.info(f"  类别采样 (v7 严格 quota) = {'ON' if stratified_sampler else 'OFF'}")

    trainer.train()

    logger.info(f"保存 LoRA adapter: {args.output_dir}")
    trainer.save_state()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("=" * 60)
    logger.info("训练完毕！")
    logger.info(f"LoRA adapter: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
