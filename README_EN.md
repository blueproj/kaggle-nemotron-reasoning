# NVIDIA Nemotron Reasoning Challenge — LoRA SFT Silver Medal Solution

> Kaggle **NVIDIA Nemotron Model Reasoning Challenge** Silver Medal Solution
>
> Enhancing reasoning capabilities of Nemotron-3-Nano-30B-A3B through deterministic CoT generation and class-balanced LoRA fine-tuning

---

## Table of Contents

- [Competition Overview](#competition-overview)
- [Final Results](#final-results)
- [Technical Approach](#technical-approach)
- [Key Innovations](#key-innovations)
- [Repository Structure](#repository-structure)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Quick Start](#quick-start)
- [Training Configuration](#training-configuration)
- [Dataset Details](#dataset-details)
- [Tech Stack](#tech-stack)

---

## Competition Overview

**NVIDIA Nemotron Model Reasoning Challenge** is a Kaggle competition focused on natural language reasoning:

- **Task**: Enhance reasoning capabilities of the Nemotron-3-Nano-30B-A3B baseline model via LoRA fine-tuning
- **Submission**: LoRA adapter (weight files)
- **Evaluation**: Exact string match or 1% relative numerical tolerance
- **Model**: NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (MoE + Mamba hybrid architecture, 30B params, 3B active)

### Task Categories

The competition covers 7 reasoning task types, each requiring a different solving strategy:

| Category | Description | Solving Method |
|----------|-------------|----------------|
| `bit_manipulation` | 8-bit binary transformation rule inference | XOR/AND/OR brute-force search + rule verification |
| `cryptarithm` | Symbolic equation transformation | CSP constraint propagation + algebraic deduction |
| `gravity` | Gravitational constant physics calculation | Median fitting + long division step-by-step |
| `numeral` | Numeral system conversion (Wonderland base) | Rule inference + base conversion |
| `unit_conversion` | Unit conversion | Linear fitting + exact string arithmetic |
| `cipher` | Text encryption/decryption | Letter substitution rule inference |
| `equation_numeric` | Equation numerical solving | Symbolic search + numerical computation |

---

## Final Results

🏆 **Kaggle Silver Medal**

### Key Metrics

| Metric | Value |
|--------|-------|
| Training data size | 14,250 samples (7 categories) |
| Validation accuracy (learned) | ~97% |
| Validation accuracy (unseen) | ~85% |
| Training steps | 445 steps (1 epoch) |
| Effective Batch Size | 32 |
| Training hardware | RTX PRO 6000 (96GB) |

### Per-Category Performance

| Category | Samples | Accuracy |
|----------|---------|----------|
| gravity | 171 | ~100% |
| unit_conversion | 167 | ~99% |
| cipher | 165 | ~99% |
| numeral | 162 | ~99% |
| bit_manipulation | 335 | ~92% |
| equation_numeric | 280 | ~90% |
| cryptarithm | 1280 (all) | ~85% |

---

## Technical Approach

```
Raw Data (9,500 problems)
    │
    ├─ 1. Deterministic CoT Generation (reasoning.py + 7 reasoners)
    │      └─ Rule-based solving → GT consistency verification → high-quality CoT
    │
    ├─ 2. Data Augmentation (build_augmented_dataset.py)
    │      └─ N-1 replacement: swap query with examples to generate new samples
    │
    ├─ 3. Data Sampling (sample_train_corpus_v2.py)
    │      └─ bit_manipulation boosted to 5000 + original-only for others + held-out val
    │
    ├─ 4. LoRA SFT Training (nemotron_lora_sft_v7.py)
    │      └─ StratifiedSampler strict quota + Loss Mask + Gradient Checkpointing
    │
    └─ 5. Evaluation (eval_local.py)
           └─ vLLM inference + official metric replication
```

---

## Key Innovations

### 1. Deterministic CoT Generation Pipeline

**Core idea**: Instead of using LLMs to generate CoT, we build deterministic solvers for each task type, guaranteeing answer correctness.

Each reasoner is responsible for:
1. Parsing examples and queries from the problem prompt
2. Inferring hidden rules from examples (e.g., XOR constants, substitution tables, gravitational constants)
3. Applying rules to compute the answer
4. Generating detailed step-by-step reasoning text (including long multiplication/division)

**Advantages**:
- ✅ 100% correct answers (algorithmic solving, not probabilistic generation)
- ✅ Rigorous reasoning chains suitable for SFT training
- ✅ String-based arithmetic throughout, avoiding floating-point precision loss

See `cot_generation/reasoners/` directory.

### 2. StratifiedSampler — Strict Quota Stratified Sampling

**Problem**: The cryptarithm category has only 407 samples (2.8% of data). With traditional WeightedRandomSampler, ~15.5% of batches contain zero cryptarithm samples.

**Solution**: A custom PyTorch Sampler that guarantees exactly 2 cryptarithm samples + 30 other-class samples (naturally distributed) in every effective batch of 32.

**Results**:
- Minority class coverage increased from ~62% to ~96%
- Each cryptarithm sample is trained ~2.8 times on average

See the `StratifiedSampler` class in `sft_training/nemotron_lora_sft_v7.py`.

### 3. N-1 Replacement Data Augmentation

**Strategy**: For each original sample (with N examples), replace the k-th example with (query, answer), making the original example the new query, generating N augmented samples.

- Preserves the problem's semantic structure
- Answers derived from original GT, ensuring correctness
- Max 3 augmented samples per source_id to prevent overfitting

See `cot_generation/build_augmented_dataset.py`.

### 4. Token-Level Filtering

Training data is strictly filtered according to competition evaluation constraints:
- Total token count ≤ 8192 (max_model_len)
- Response token count ≤ 7680 (max_tokens)

Ensures training data aligns with the inference-time generation space.

### 5. Loss Mask Mechanism

Loss is computed only on assistant response tokens (`IGNORE_INDEX = -100` for prompt tokens), fully aligned with the chat template format.

---

## Repository Structure

```
kaggle-nemotron-reasoning/
│
├── cot_generation/                 # Deterministic CoT Generation Pipeline
│   ├── config.py                   # Configuration (paths, patterns, params)
│   ├── reasoning.py                # Main entry: load → classify → reason → output
│   ├── store_types.py              # Data structures + parsing utils + long arithmetic
│   ├── sample_train_corpus_v2.py   # v2 training data sampling script
│   ├── build_augmented_dataset.py  # N-1 replacement augmentation
│   └── reasoners/                  # 7 deterministic reasoners
│       ├── __init__.py             # Reasoner registry
│       ├── _utils.py               # \boxed{} answer extraction
│       ├── gravity.py              # Gravitational constant inference
│       ├── unit_conversion.py      # Unit conversion + linear fitting
│       ├── cipher.py               # Substitution cipher cracking
│       ├── bit_manipulation.py     # 8-bit binary rule search
│       ├── numeral.py              # Wonderland numeral conversion
│       ├── cryptarithm.py          # Symbolic equation solving
│       └── equation_numeric.py     # Equation numerical solving
│
├── sft_training/                   # LoRA SFT Training
│   ├── nemotron_lora_sft_v7.py     # ★ Silver medal training script
│   ├── train_v7.sh                 # Training launch script (BF16 / QLoRA)
│   ├── eval_local.py               # Local evaluation (vLLM + official metric)
│   └── infer_vllm.py              # vLLM batch inference script
│
├── data/                           # Data documentation and samples
│   ├── README.md                   # v2 dataset detailed documentation
│   └── samples/                    # Data samples (first 20 rows each)
│       ├── train_sample.csv
│       ├── val_learned_sample.csv
│       └── val_unseen_sample.csv
│
├── requirements.txt                # Python dependencies
└── .gitignore
```

---

## End-to-End Pipeline

### Step 1: Generate Deterministic CoT

```bash
cd cot_generation

# Run deterministic reasoning on all problems
python reasoning.py --input ../data/train.csv --output ./output/reasoning

# Output: one JSONL per category + stats.json
```

### Step 2: Data Augmentation

```bash
cd cot_generation

# N-1 replacement augmentation
python build_augmented_dataset.py
```

### Step 3: Build Training Set (v2 ratios)

```bash
cd cot_generation

# Sample with silver medal ratios
python sample_train_corpus_v2.py
```

### Step 4: LoRA SFT Training

```bash
cd sft_training

# BF16 mode (recommended, requires ~80GB VRAM)
bash train_v7.sh --mode bf16 \
    --data_path ../data/v2/train_corpus_aug_sampled_train_v2.csv

# QLoRA mode (24GB VRAM)
bash train_v7.sh --mode qlora \
    --data_path ../data/v2/train_corpus_aug_sampled_train_v2.csv
```

### Step 5: Local Evaluation

```bash
cd sft_training

# Evaluate on validation set
python eval_local.py \
    --model_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --lora_path ./output/nemotron_v7_bf16_xxx/checkpoint-445 \
    --holdout_csv ../data/v2/val_1000_learned.csv \
    --output_dir ./eval_reports --tag silver_v7
```

---

## Training Configuration

### Model Information

| Item | Value |
|------|-------|
| Base model | NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |
| Architecture | MoE (Mixture of Experts) + Mamba |
| Parameters | 30B (3B active) |
| Precision | BF16 |

### LoRA Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| LoRA rank | 32 | Low-rank matrix dimension |
| LoRA alpha | 32 | Scaling factor (alpha/rank = 1.0) |
| LoRA dropout | 0.05 | Regularization |
| Target modules | q/k/v/o_proj, in/out/up/down_proj | Attention + MLP |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Learning Rate | 2e-4 |
| Scheduler | cosine_with_min_lr |
| Min LR | 2e-5 |
| Warmup Ratio | 0.05 |
| Effective Batch Size | 32 (1 × 32 grad_accum) |
| Max Length | 8192 |
| Max Response Length | 7680 |
| Weight Decay | 0.01 |
| Max Grad Norm | 1.0 |
| Gradient Checkpointing | ON |
| Optimizer | AdamW (Torch) |
| Seed | 42 |

### Sampling Strategy

| Category | Training Count | Strategy |
|----------|---------------|----------|
| bit_manipulation | 5000 | 1364 original + 3636 augmented |
| equation_numeric | 2500 | Random sample (280 held out for validation) |
| gravity | 1597 | Original only |
| unit_conversion | 1594 | Original only |
| numeral | 1576 | Original only |
| cipher | 1576 | Original only |
| cryptarithm | 407 | All + strict 2/batch quota |
| **Total** | **14250** | |

---

## Dataset Details

### v2 Training Set Design

The v2 dataset was optimized based on v1 evaluation results:

1. **bit_manipulation boosted**: 3000 → 5000 (insufficient rule combination coverage, needs more augmented samples)
2. **4 major categories de-augmented**: gravity/unit_conversion/cipher/numeral already at 99%+ accuracy, augmentation provides no benefit
3. **equation_numeric held-out**: 280 samples split from 2780 as a dedicated validation set

See `data/README.md` for full details.

### Data Format

Training CSV contains 5 columns:

| Column | Description |
|--------|-------------|
| `id` | Unique sample identifier |
| `prompt` | Original problem text |
| `answer` | Ground truth answer |
| `type` | Category (one of 7 types) |
| `generated_cot` | Deterministic reasoning chain (without `\:thinking` prefix) |

---

## Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| PyTorch | 2.6.0 + CUDA 12.4 | Deep learning framework |
| Transformers | 5.8.0 | Model loading & tokenizer |
| PEFT | 0.19.1 | LoRA fine-tuning |
| BitsAndBytes | 0.42.0 | QLoRA 4-bit quantization |
| Datasets | 4.0.0 | Data processing |
| vLLM | 0.8.0+ | Efficient inference |
| DeepSpeed | latest | Distributed training support |
| Mamba SSM | 2.3.2 | Nemotron Mamba architecture |
| Flash Attention | 2.8.0+ | Training acceleration |

---

## Hardware Requirements

| Use Case | GPU | VRAM | Mode |
|----------|-----|------|------|
| Full training | RTX PRO 6000 / A100 | 80GB+ | BF16 LoRA |
| Consumer GPU | RTX 3090 / 4090 | 24GB | QLoRA 4-bit |
| Inference eval | RTX PRO 6000 | 80GB+ | vLLM + LoRA |

---

## License

This project code is for educational reference only. Competition data is copyrighted by Kaggle/NVIDIA.
