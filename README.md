# NVIDIA Nemotron 推理挑战赛 — LoRA SFT 银牌方案

> Kaggle **NVIDIA Nemotron Model Reasoning Challenge** 银牌解决方案
>
> 通过确定性 CoT 生成 + 类别均衡 LoRA 微调，在 Nemotron-3-Nano-30B-A3B 模型上显著提升推理能力

---

## 目录

- [比赛简介](#比赛简介)
- [最终成绩](#最终成绩)
- [技术方案概览](#技术方案概览)
- [核心创新点](#核心创新点)
- [项目结构](#项目结构)
- [端到端流程](#端到端流程)
- [快速开始](#快速开始)
- [训练配置](#训练配置)
- [数据集详情](#数据集详情)
- [技术栈](#技术栈)

---

## 比赛简介

**NVIDIA Nemotron Model Reasoning Challenge** 是 Kaggle 上的自然语言推理竞赛：

- **任务**: 在 NVIDIA Nemotron-3-Nano-30B-A3B 基线模型上通过 LoRA 微调提升推理能力
- **提交**: LoRA adapter（权重文件）
- **评测**: 精确字符串匹配 或 1% 相对数值容忍度
- **模型**: NVIDIA-Nemotron-3-Nano-30B-A3B-BF16（MoE + Mamba 混合架构，30B 参数，激活 3B）

### 题型分布

竞赛涵盖 7 类推理任务，每类需要不同的求解策略：

| 题型 | 描述 | 求解方法 |
|------|------|----------|
| `bit_manipulation` | 8 位二进制数变换规则推断 | XOR/AND/OR 暴力搜索 + 规则验证 |
| `cryptarithm` | 符号方程变换求解 | CSP 约束传播 + 代数推导 |
| `gravity` | 重力常数物理公式计算 | 中位数拟合 + 长除法逐步展示 |
| `numeral` | 数字系统转换（Wonderland 进制） | 规则推断 + 进制转换 |
| `unit_conversion` | 单位转换 | 线性拟合 + 精确字符串算术 |
| `cipher` | 文本加密/解密 | 字母替换规则推断 |
| `equation_numeric` | 方程数值求解 | 符号搜索 + 数值计算 |

---

## 最终成绩

🏆 **Kaggle 银牌 (Silver Medal)**

### 关键指标

| 指标 | 值 |
|------|-----|
| 训练数据量 | 14,250 条（7 类） |
| 验证集准确率 (learned) | ~97% |
| 验证集准确率 (unseen) | ~85% |
| 训练步数 | 445 steps (1 epoch) |
| 有效 Batch Size | 32 |
| 训练硬件 | RTX PRO 6000 (96GB) |

### 验证集分桶表现

| 类别 | 样本数 | 准确率 |
|------|--------|--------|
| gravity | 171 | ~100% |
| unit_conversion | 167 | ~99% |
| cipher | 165 | ~99% |
| numeral | 162 | ~99% |
| bit_manipulation | 335 | ~92% |
| equation_numeric | 280 | ~90% |
| cryptarithm | 1280(含全部) | ~85% |

---

## 技术方案概览

```
原始数据 (9,500题)
    │
    ├─ 1. 确定性 CoT 生成 (reasoning.py + 7 reasoners)
    │      └─ 规则求解 → GT一致性验证 → 高质量推理链
    │
    ├─ 2. 数据增强 (build_augmented_dataset.py)
    │      └─ N-1 替换式扩充: 用(query,answer)替换示例，生成新样本
    │
    ├─ 3. 采样配比 (sample_train_corpus_v2.py)
    │      └─ bit_manipulation 加量到5000 + 其他类用原样本 + 留出验证集
    │
    ├─ 4. LoRA SFT 训练 (nemotron_lora_sft_v7.py)
    │      └─ StratifiedSampler 严格quota + Loss Mask + Gradient Checkpointing
    │
    └─ 5. 评测提交 (eval_local.py)
           └─ vLLM 推理 + 官方 metric 复刻验证
```

---

## 核心创新点

### 1. 确定性 CoT 生成管道

**核心思路**: 不依赖 LLM 生成 CoT，而是为每类题目构建确定性求解器，保证答案正确性。

每个 reasoner 负责：
1. 从题目 prompt 中解析示例和查询
2. 根据示例推断隐藏规则（如 XOR 常数、替换表、重力常数等）
3. 应用规则计算答案
4. 生成详细的逐步推理文本（含长乘法/长除法展示）

**优势**:
- ✅ 答案 100% 正确（算法求解，非概率生成）
- ✅ 推理链逻辑严密，适合 SFT 教学
- ✅ 全程使用字符串算术，避免浮点精度损失

详见 `cot_generation/reasoners/` 目录。

### 2. StratifiedSampler — 严格配额分层采样

**问题**: cryptarithm 类仅 407 条（占 2.8%），传统 WeightedRandomSampler 下约 15.5% 的 batch 完全没有 cryptarithm 样本。

**方案**: 自定义 PyTorch Sampler，在每个 effective batch（32 条）中严格保证 2 条 cryptarithm + 30 条其他类（按自然分布分配）。

**效果**: 
- 小类覆盖率从 ~62% 提升到 ~96%
- 每条 cryptarithm 样本平均被训练 ~2.8 次

详见 `sft_training/nemotron_lora_sft_v7.py` 中的 `StratifiedSampler` 类。

### 3. N-1 替换式数据增强

**策略**: 对每条原样本（含 N 个示例），用 (query, answer) 替换第 k 个示例，将原示例变为新 query，生成 N 条增强样本。

- 保持题目语义结构不变
- 答案来自原始 GT，保证正确性
- 每 source_id 限制最多 3 条增强，避免过拟合

详见 `cot_generation/build_augmented_dataset.py`。

### 4. Token 级别过滤

训练数据严格按竞赛评测约束过滤：
- 总 token 数 ≤ 8192（max_model_len）
- Response token 数 ≤ 7680（max_tokens）

确保训练数据与推理时生成空间一致。

### 5. Loss Mask 机制

仅对 assistant 回复部分计算 loss（`IGNORE_INDEX = -100` 标记 prompt 部分），与 chat template 格式完全对齐。

---

## 项目结构

```
kaggle-nemotron-reasoning/
│
├── cot_generation/                 # 确定性 CoT 生成管道
│   ├── config.py                   # 配置文件（路径、题型模式、参数）
│   ├── reasoning.py                # 主推理入口：加载题目 → 分类 → 调用reasoner → 输出
│   ├── store_types.py              # 数据结构定义 + 解析工具 + 长乘法/长除法
│   ├── sample_train_corpus_v2.py   # v2 训练数据采样脚本
│   ├── build_augmented_dataset.py  # N-1 替换式数据增强
│   └── reasoners/                  # 7 类确定性推理器
│       ├── __init__.py             # 推理器注册表
│       ├── _utils.py               # \boxed{} 答案提取工具
│       ├── gravity.py              # 重力常数推断 + 长除法展示
│       ├── unit_conversion.py      # 单位转换 + 线性拟合
│       ├── cipher.py               # 字母替换密码破解
│       ├── bit_manipulation.py     # 8位二进制变换规则搜索
│       ├── numeral.py              # Wonderland 数字系统转换
│       ├── cryptarithm.py          # 符号方程变换求解
│       └── equation_numeric.py     # 方程数值求解
│
├── sft_training/                   # LoRA SFT 训练
│   ├── nemotron_lora_sft_v7.py     # ★ 银牌训练脚本（StratifiedSampler + Loss Mask）
│   ├── train_v7.sh                 # 训练启动脚本（BF16 / QLoRA 双模式）
│   ├── eval_local.py               # 本地评测（vLLM + 官方 metric 复刻）
│   └── infer_vllm.py              # vLLM 批量推理脚本
│
├── data/                           # 数据说明与样本
│   ├── README.md                   # v2 数据集详细说明
│   └── samples/                    # 数据样本（各取前20行）
│       ├── train_sample.csv
│       ├── val_learned_sample.csv
│       └── val_unseen_sample.csv
│
├── requirements.txt                # Python 依赖
└── .gitignore
```

---

## 端到端流程

### Step 1: 生成确定性 CoT

```bash
cd cot_generation

# 对所有题目运行确定性推理
python reasoning.py --input ../data/train.csv --output ./output/reasoning

# 输出: 每类一个 JSONL 文件 + stats.json 统计
```

### Step 2: 数据增强

```bash
cd cot_generation

# N-1 替换式扩充
python build_augmented_dataset.py
```

### Step 3: 构建训练集 (v2 配比)

```bash
cd cot_generation

# 按银牌配比采样
python sample_train_corpus_v2.py
```

### Step 4: LoRA SFT 训练

```bash
cd sft_training

# BF16 模式（推荐，需要 ~80GB 显存）
bash train_v7.sh --mode bf16 \
    --data_path ../data/v2/train_corpus_aug_sampled_train_v2.csv

# QLoRA 模式（24GB 显卡可用）
bash train_v7.sh --mode qlora \
    --data_path ../data/v2/train_corpus_aug_sampled_train_v2.csv
```

### Step 5: 本地评测

```bash
cd sft_training

# 在验证集上评测
python eval_local.py \
    --model_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --lora_path ./output/nemotron_v7_bf16_xxx/checkpoint-445 \
    --holdout_csv ../data/v2/val_1000_learned.csv \
    --output_dir ./eval_reports --tag silver_v7
```

---

## 训练配置

### 模型信息

| 项 | 值 |
|----|-----|
| 基础模型 | NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |
| 架构 | MoE (Mixture of Experts) + Mamba |
| 参数量 | 30B（激活 3B） |
| 精度 | BF16 |

### LoRA 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA rank | 32 | 低秩矩阵维度 |
| LoRA alpha | 32 | 缩放系数 (alpha/rank = 1.0) |
| LoRA dropout | 0.05 | 正则化 |
| Target modules | q/k/v/o_proj, in/out/up/down_proj | 覆盖注意力和 MLP |

### 训练超参数

| 参数 | 值 |
|------|-----|
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

### 采样策略

| 类别 | 训练量 | 策略 |
|------|--------|------|
| bit_manipulation | 5000 | 原样本 1364 + 增强 3636 |
| equation_numeric | 2500 | 随机采样（留出 280 验证） |
| gravity | 1597 | 仅原样本 |
| unit_conversion | 1594 | 仅原样本 |
| numeral | 1576 | 仅原样本 |
| cipher | 1576 | 仅原样本 |
| cryptarithm | 407 | 全量 + 每batch严格2条 |
| **合计** | **14250** | |

---

## 数据集详情

### v2 训练集设计

v2 数据集基于 v1 的评测结果进行了配比优化：

1. **bit_manipulation 加量**: 3000 → 5000（规则组合覆盖度不足，需要更多增强样本）
2. **4 大类去扩充**: gravity/unit_conversion/cipher/numeral 已达 99%+，扩充无增益
3. **equation_numeric 留验证集**: 2780 条中分出 280 条作为专项验证集

详见 `data/README.md`。

### 数据格式

训练 CSV 包含 5 列：

| 列名 | 说明 |
|------|------|
| `id` | 样本唯一标识 |
| `prompt` | 题目原文 |
| `answer` | 标准答案 |
| `type` | 题型（7 类之一） |
| `generated_cot` | 确定性推理链（不含 `\:thinking` 前缀） |

---

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| PyTorch | 2.6.0 + CUDA 12.4 | 深度学习框架 |
| Transformers | 5.8.0 | 模型加载与 Tokenizer |
| PEFT | 0.19.1 | LoRA 微调 |
| BitsAndBytes | 0.42.0 | QLoRA 4-bit 量化 |
| Datasets | 4.0.0 | 数据处理 |
| vLLM | 0.8.0+ | 高效推理 |
| DeepSpeed | latest | 分布式训练支持 |
| Mamba SSM | 2.3.2 | Nemotron Mamba 架构 |
| Flash Attention | 2.8.0+ | 训练加速 |

---

## 硬件需求

| 用途 | GPU | 显存 | 模式 |
|------|-----|------|------|
| 正式训练 | RTX PRO 6000 / A100 | 80GB+ | BF16 LoRA |
| 消费级显卡 | RTX 3090 / 4090 | 24GB | QLoRA 4-bit |
| 推理评测 | RTX PRO 6000 | 80GB+ | vLLM + LoRA |

---

## License

本项目代码仅供学习参考，比赛数据版权归 Kaggle/NVIDIA 所有。
