# 训练数据 v2 - 配比调整版

> 调整数据配比：bit_manipulation 加量到 5000，4 大类去掉扩充只用原样本，equation_numeric 分出 280 条验证集。
> 生成时间: 2026-05-27
> 流程代码: `code/new_cot/sample_train_corpus_v2.py`
> 数据来源: `data/v1/train_corpus_aug_full_v1.csv`（47792 条 V1 reasoner 命中全集）

---

## 一、设计动机

v1 数据 (18187 条) 训练的 v7 模型在 val_1000_learned 上发现：
- gravity/unit_conversion/cipher/numeral: 99~100% ✓（扩充没增益，只用原样本即可）
- bit_manipulation: 92.2%（错了 26/335）— 规则选择阶段出错，需要更多训练样本覆盖
- equation_numeric: 需要验证集评估泛化

v2 核心变化：
1. bit_manipulation 3000 → 5000（+2000 扩充样本，增加规则组合覆盖度）
2. 4 大类去掉扩充，只用原样本（已证明不需要扩充）
3. equation_numeric 留出 280 条验证集

---

## 二、文件清单

### 1. `train_corpus_aug_sampled_train_v2.csv`

| 项 | 值 |
|----|----|
| 行数 | 14250 |
| 列 | `id, prompt, answer, type, generated_cot` |
| 来源 | `data/v1/train_corpus_aug_full_v1.csv` |
| 生成脚本 | `code/new_cot/sample_train_corpus_v2.py` |

**类别分布**:

| type | 行数 | 策略 |
|------|---:|------|
| bit_manipulation | 5000 | 全部 1364 原 + 3636 扩充（每 source_id ≤ 3 条） |
| equation_numeric | 2500 | 从 2780 全量中随机取（seed=42） |
| gravity | 1597 | 只用原样本 |
| unit_conversion | 1594 | 只用原样本 |
| numeral | 1576 | 只用原样本 |
| cipher | 1576 | 只用原样本 |
| cryptarithm | 407 | 全量 |
| **合计** | **14250** | |

---

### 2. `val_1000_learned.csv`

| 项 | 值 |
|----|----|
| 行数 | 1280 |
| 列 | `id, category, prompt, answer` |
| 用途 | 验证模型对 reasoner 能解的题的泛化能力 |

**来源**: v1 val_1000_learned (5 大类 1000 条) + equation_numeric 280 条

| category | 行数 |
|------|---:|
| bit_manipulation | 335 |
| equation_numeric | 280 |
| gravity | 171 |
| unit_conversion | 167 |
| cipher | 165 |
| numeral | 162 |

---

### 3. `val_1000_unseen.csv`

| 项 | 值 |
|----|----|
| 行数 | 1000 |
| 列 | `id, category, prompt, answer` |
| 用途 | 验证模型对 reasoner 无法解的难题的泛化能力 |
| 说明 | 与 v1 完全相同（直接复制） |

---

### 4. `val_280_equation_numeric.csv`

| 项 | 值 |
|----|----|
| 行数 | 280 |
| 列 | `id, category, prompt, answer` |
| 用途 | equation_numeric 专项验证集（已并入 val_1000_learned） |

---

## 三、与 v1 对比

| 维度 | v1 | v2 |
|------|---:|---:|
| 总训练量 | 18187 | 14250 |
| bit_manipulation | 3000 (1364原+1636扩) | **5000** (1364原+3636扩) |
| gravity | 3000 (1597原+1403扩) | 1597 (只原) |
| unit_conversion | 3000 (1594原+1406扩) | 1594 (只原) |
| cipher | 3000 (1576原+1424扩) | 1576 (只原) |
| numeral | 3000 (1576原+1424扩) | 1576 (只原) |
| cryptarithm | 407 | 407 |
| equation_numeric | 2780 | 2500 (训练) + 280 (验证) |

---

## 四、训练命令

```bash
cd sft_training
bash train_v7.sh --mode bf16 \
    --data_path ../data/v2/train_corpus_aug_sampled_train_v2.csv
```

训练参数与 v7 一致（1 epoch, lr=2e-4, alpha=32, min_lr=1e-5, cryptarithm boost 2/batch）。
total_steps ≈ 14250 / 32 = 445 步。

---

## 五、验证命令

```bash
# val_learned (含 equation_numeric)
python eval_local.py \
    --model_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --lora_path <ckpt_path> \
    --holdout_csv ../data/v2/val_1000_learned.csv \
    --output_dir ./eval_reports --tag v2_learned

# val_unseen (难题)
python eval_local.py \
    --model_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --lora_path <ckpt_path> \
    --holdout_csv ../data/v2/val_1000_unseen.csv \
    --output_dir ./eval_reports --tag v2_unseen
```
