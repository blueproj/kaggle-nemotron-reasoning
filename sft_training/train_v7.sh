#!/bin/bash
# ============================================================
# Nemotron LoRA SFT v7 训练启动脚本 (严格 quota 分层 sampler)
# ============================================================
# v7 特性 (相对 v6):
#   - 采样从 WeightedRandomSampler (概率期望) 换为 StratifiedSampler
#     (严格 quota): 每 eff_batch 严格包含 cryptarithm 2 条 + 其他 30 条
#   - dataloader_num_workers 强制 0 (避免多 worker 重复抽样)
#   - 其余超参 (lr/alpha=32/min_lr=1e-5/epochs=1) 与 v6 完全一致
#
# 使用方法:
#   BF16 单卡:  bash train_v7.sh --mode bf16
#   QLoRA 3090: bash train_v7.sh --mode qlora
#
# 可覆盖参数:
#   --data_path /path/to/csv    (训练 CSV 路径, 必须 type=7 类)
#   --model_path /path/to/model
#   --epochs N
#   --lr 2e-4
#   --batch_size N
#   --grad_accum N
#   --boost "cryptarithm:2"     (覆盖默认 boost 配置, n 必须为正整数)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="bf16"

# ---- 解析命令行参数 ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)       MODE=$2; shift 2;;
        --model_path) MODEL_PATH_OVERRIDE=$2; shift 2;;
        --data_path)  DATA_PATH_OVERRIDE=$2; shift 2;;
        --max_length) MAX_LENGTH_OVERRIDE=$2; shift 2;;
        --batch_size) BATCH_SIZE_OVERRIDE=$2; shift 2;;
        --grad_accum) GRAD_ACCUM_OVERRIDE=$2; shift 2;;
        --epochs)     NUM_EPOCHS_OVERRIDE=$2; shift 2;;
        --lr)         LEARNING_RATE_OVERRIDE=$2; shift 2;;
        --scheduler)  SCHEDULER_OVERRIDE=$2; shift 2;;
        --warmup)     WARMUP_OVERRIDE=$2; shift 2;;
        --no_balanced) NO_BALANCED="--no_balanced_sampling"; shift;;
        --boost)      BOOST_OVERRIDE=$2; shift 2;;
        *)
            echo "未知参数: $1"
            exit 1;;
    esac
done

# ============================================================
# BF16 模式 (96GB 显卡如 RTX PRO 6000 / A100)
# ============================================================
if [ "${MODE}" = "bf16" ]; then
    echo "============================================================"
    echo ">>> SFT v7 | BF16 LoRA | 严格 quota sampler (cryptarithm 2/batch)"
    echo "============================================================"

    MODEL_PATH="${MODEL_PATH_OVERRIDE:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
    DATA_PATH="${DATA_PATH_OVERRIDE:-data/v2/train_corpus_aug_sampled_train_v2.csv}"
    OUTPUT_DIR="${SCRIPT_DIR}/output/nemotron_v7_bf16_$(date +%Y%m%d_%H%M%S)"

    NUM_EPOCHS="${NUM_EPOCHS_OVERRIDE:-1}"
    BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-1}"
    GRAD_ACCUM="${GRAD_ACCUM_OVERRIDE:-32}"
    LEARNING_RATE="${LEARNING_RATE_OVERRIDE:-2e-4}"
    SCHEDULER="${SCHEDULER_OVERRIDE:-cosine_with_min_lr}"
    WARMUP_RATIO="${WARMUP_OVERRIDE:-0.05}"
    MAX_LENGTH="${MAX_LENGTH_OVERRIDE:-8192}"
    BOOST="${BOOST_OVERRIDE-cryptarithm:2}"
    LORA_R=32
    LORA_ALPHA=32
    LORA_DROPOUT=0.05
    WEIGHT_DECAY=0.01
    MAX_GRAD_NORM=1.0
    MIN_LR=0.00002

    mkdir -p "${OUTPUT_DIR}"

    echo "模型: ${MODEL_PATH}"
    echo "数据: ${DATA_PATH}  (必须 type=7 类)"
    echo "输出: ${OUTPUT_DIR}"
    echo "序列长度: ${MAX_LENGTH}"
    echo "有效Batch: $((BATCH_SIZE * GRAD_ACCUM))"
    echo "Epochs: ${NUM_EPOCHS}"
    echo "LR: ${LEARNING_RATE} | Scheduler: ${SCHEDULER} | Min LR: ${MIN_LR} | Warmup: ${WARMUP_RATIO}"
    echo "LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
    echo "Grad Clip: ${MAX_GRAD_NORM} | Weight Decay: ${WEIGHT_DECAY}"
    echo "类别采样: v7 严格 quota (${BOOST:-OFF}) + 其他自然分布"
    echo "============================================================"

    python ${SCRIPT_DIR}/nemotron_lora_sft_v7.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --max_length ${MAX_LENGTH} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --lr_scheduler_type ${SCHEDULER} \
        --min_lr ${MIN_LR} \
        --warmup_ratio ${WARMUP_RATIO} \
        --weight_decay ${WEIGHT_DECAY} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --class_target_per_batch "${BOOST}" \
        --logging_steps 1 \
        --save_steps 50 \
        --save_total_limit 6 \
        --gradient_checkpointing \
        --log_dir ${OUTPUT_DIR}/logs \
        --num_proc 8 \
        --seed 42 \
        ${NO_BALANCED}

# ============================================================
# QLoRA 模式 (RTX 3090 24GB)
# ============================================================
elif [ "${MODE}" = "qlora" ]; then
    echo "============================================================"
    echo ">>> SFT v7 | QLoRA 4-bit | 严格 quota sampler (cryptarithm 2/batch)"
    echo "============================================================"

    MODEL_PATH="${MODEL_PATH_OVERRIDE:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
    DATA_PATH="${DATA_PATH_OVERRIDE:-data/v2/train_corpus_aug_sampled_train_v2.csv}"
    OUTPUT_DIR="${SCRIPT_DIR}/output/nemotron_v7_qlora_$(date +%Y%m%d_%H%M%S)"

    NUM_EPOCHS="${NUM_EPOCHS_OVERRIDE:-1}"
    BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-1}"
    GRAD_ACCUM="${GRAD_ACCUM_OVERRIDE:-32}"
    LEARNING_RATE="${LEARNING_RATE_OVERRIDE:-2e-5}"
    SCHEDULER="${SCHEDULER_OVERRIDE:-cosine_with_min_lr}"
    WARMUP_RATIO="${WARMUP_OVERRIDE:-0.05}"
    MAX_LENGTH="${MAX_LENGTH_OVERRIDE:-4096}"
    BOOST="${BOOST_OVERRIDE-cryptarithm:2}"
    LORA_R=32
    LORA_ALPHA=32
    LORA_DROPOUT=0.05
    WEIGHT_DECAY=0.01
    MAX_GRAD_NORM=1.0
    MIN_LR=0.000002

    mkdir -p "${OUTPUT_DIR}"

    echo "模型: ${MODEL_PATH}"
    echo "数据: ${DATA_PATH}  (必须 type=7 类)"
    echo "输出: ${OUTPUT_DIR}"
    echo "量化: 4-bit NF4 + 双重量化"
    echo "序列长度: ${MAX_LENGTH}"
    echo "有效Batch: $((BATCH_SIZE * GRAD_ACCUM))"
    echo "Epochs: ${NUM_EPOCHS}"
    echo "LR: ${LEARNING_RATE} | Scheduler: ${SCHEDULER}"
    echo "类别采样: v7 严格 quota (${BOOST:-OFF}) + 其他自然分布"
    echo "============================================================"

    python ${SCRIPT_DIR}/nemotron_lora_sft_v7.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --max_length ${MAX_LENGTH} \
        --use_4bit \
        --bnb_4bit_quant_type nf4 \
        --bnb_4bit_compute_dtype bfloat16 \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --lr_scheduler_type ${SCHEDULER} \
        --min_lr ${MIN_LR} \
        --warmup_ratio ${WARMUP_RATIO} \
        --weight_decay ${WEIGHT_DECAY} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --class_target_per_batch "${BOOST}" \
        --logging_steps 1 \
        --save_steps 50 \
        --save_total_limit 6 \
        --gradient_checkpointing \
        --log_dir ${OUTPUT_DIR}/logs \
        --num_proc 8 \
        --seed 42 \
        ${NO_BALANCED}

else
    echo "错误: 未知模式 '${MODE}' (支持: bf16 / qlora)"
    exit 1
fi

echo "============================================================"
echo "训练完成！模型: ${OUTPUT_DIR}"
echo "============================================================"
