"""
训练数据 v2 采样脚本
====================
从 data/v1/train_corpus_aug_full_v1.csv 中按新配比抽样，产出 data/v2/ 训练集。

v2 配比设计（相对 v1 的变化）:
  - bit_manipulation: 3000 → 5000 (全部 1364 原 + 3636 扩充, 每 source_id 最多 3 条扩充)
  - gravity/unit_conversion/cipher/numeral: 只用原样本 (不用扩充)
  - cryptarithm: 407 (不变, 全量)
  - equation_numeric: 2500 训练 + 280 验证 (从全量 2780 中随机抽)

产出:
  - data/v2/train_corpus_v2.csv (14250 行, 5 列: id/prompt/answer/type/generated_cot)
  - data/v2/val_280_equation_numeric.csv (280 行, 4 列: id/category/prompt/answer)
"""

import os
import sys
import pandas as pd
import numpy as np

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FULL_CSV = os.path.join(PROJECT_ROOT, "data/v1/train_corpus_aug_full_v1.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/v2")

SEED = 42
np.random.seed(SEED)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"读取: {FULL_CSV}")
    full = pd.read_csv(FULL_CSV)
    full["answer"] = full["answer"].astype(str)
    print(f"  总行数: {len(full)}")
    print(f"  列: {full.columns.tolist()}")
    print()

    parts = []

    # --- 1. gravity / unit_conversion / cipher / numeral: 只用原样本 ---
    for cat in ["gravity", "unit_conversion", "cipher", "numeral"]:
        cat_orig = full[(full["type"] == cat) & (full["is_augmented"] == 0)]
        print(f"{cat}: 原样本 {len(cat_orig)} 条 (全部入选)")
        parts.append(cat_orig)

    # --- 2. bit_manipulation: 全部原样本 + 扩充补到 5000 ---
    bit_all = full[full["type"] == "bit_manipulation"]
    bit_orig = bit_all[bit_all["is_augmented"] == 0]
    bit_aug = bit_all[bit_all["is_augmented"] == 1]
    print(f"\nbit_manipulation: 原 {len(bit_orig)}, 扩充可用 {len(bit_aug)}")

    # 全部原样本入选
    parts.append(bit_orig)
    need_aug = 5000 - len(bit_orig)  # 3636

    # 每 source_id 最多 3 条扩充
    bit_aug_grouped = bit_aug.groupby("source_id")
    aug_selected = []
    for src_id, grp in bit_aug_grouped:
        n = min(3, len(grp))
        aug_selected.append(grp.sample(n=n, random_state=SEED))
    bit_aug_pool = pd.concat(aug_selected)

    if len(bit_aug_pool) >= need_aug:
        bit_aug_final = bit_aug_pool.sample(n=need_aug, random_state=SEED)
    else:
        # 如果限制 max_per_source=3 后不够，放宽
        bit_aug_final = bit_aug_pool
        print(f"  WARNING: max_per_source=3 只有 {len(bit_aug_pool)} 条, 不够 {need_aug}")

    parts.append(bit_aug_final)
    print(f"  bit_manipulation 最终: {len(bit_orig) + len(bit_aug_final)} 条 "
          f"(原 {len(bit_orig)} + 扩充 {len(bit_aug_final)})")

    # --- 3. cryptarithm: 全量 ---
    crypt = full[full["type"] == "cryptarithm"]
    print(f"\ncryptarithm: 全量 {len(crypt)} 条")
    parts.append(crypt)

    # --- 4. equation_numeric: 2500 训练 + 280 验证 ---
    eq = full[full["type"] == "equation_numeric"]
    print(f"\nequation_numeric: 全量 {len(eq)} 条")

    eq_shuffled = eq.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    eq_train = eq_shuffled.iloc[:2500]
    eq_val = eq_shuffled.iloc[2500:]
    print(f"  训练: {len(eq_train)}, 验证: {len(eq_val)}")
    parts.append(eq_train)

    # --- 合并训练集 ---
    train_df = pd.concat(parts, ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # 只保留 SFT 需要的 5 列
    train_out = train_df[["id", "prompt", "answer", "type", "generated_cot"]]

    print(f"\n{'='*60}")
    print(f"v2 训练集总行数: {len(train_out)}")
    print(f"类别分布:")
    print(train_out["type"].value_counts().to_string())
    print(f"{'='*60}")

    # 保存训练集
    train_path = os.path.join(OUTPUT_DIR, "train_corpus_v2.csv")
    train_out.to_csv(train_path, index=False)
    print(f"\n训练集已保存: {train_path}")

    # 保存 equation_numeric 验证集 (格式与 holdout/val_1000 一致)
    val_out = eq_val[["id", "type", "prompt", "answer"]].copy()
    val_out = val_out.rename(columns={"type": "category"})
    val_path = os.path.join(OUTPUT_DIR, "val_280_equation_numeric.csv")
    val_out.to_csv(val_path, index=False)
    print(f"验证集已保存: {val_path} ({len(val_out)} 条)")


if __name__ == "__main__":
    main()
