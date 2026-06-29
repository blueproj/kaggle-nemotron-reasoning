"""
配置文件 - 统一管理 CoT 确定性推理生成的所有参数
=================================================

本模块集中管理：
- 数据文件路径
- 输出目录
- 模型/tokenizer 相关配置
- 各推理器的参数配置
- 生成流程控制参数
"""

import os

# ============================================================
# 项目根目录（自动推断）
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 数据路径配置
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# ============================================================
# 输出路径配置
# ============================================================
COT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(COT_DIR, "output")

# 推理结果输出（每个类别独立的 JSONL 文件）
REASONING_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "reasoning")

# 增强数据输出
AUGMENTATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "augmentation")

# 最终训练语料输出（合并后的 JSONL）
CORPUS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "train_corpus.jsonl")

# ============================================================
# 模型/Tokenizer 配置
# ============================================================
# 竞赛指定模型，用于确定 token 长度限制
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# 最大生成 token 数（竞赛评测时 max_tokens=7680）
MAX_TOKENS = 7680

# 最大模型输入长度（竞赛评测时 max_model_len=8192）
MAX_MODEL_LEN = 8192

# ============================================================
# 题型分类关键词配置
# ============================================================
# 每种题型通过 prompt 中的关键词进行识别
CATEGORY_PATTERNS = {
    "bit_manipulation": ["bit manipulation", "8-bit binary"],
    "gravity": ["gravitational constant", "d = 0.5*g*t^2", "gravitational"],
    "unit_conversion": ["unit conversion", "m becomes", "convert the following measurement"],
    "cipher": ["encryption rules", "decrypt the following"],
    "numeral": ["numeral system", "wonderland numeral"],
    "cryptarithm": ["transformation rules", "equations"],
}

# ============================================================
# 各推理器参数配置
# ============================================================

# 重力推理器配置
GRAVITY_CONFIG = {
    # 用于拟合 g 的最小样本数
    "min_examples": 2,
    # 答案保留小数位数
    "decimal_places": 2,
    # 容差范围（验证用）
    "tolerance": 0.01,
}

# 单位转换推理器配置
UNIT_CONVERSION_CONFIG = {
    # 最小样本数
    "min_examples": 2,
    # 答案保留小数位数
    "decimal_places": 2,
    # 容差范围
    "tolerance": 0.01,
}

# 密码推理器配置
CIPHER_CONFIG = {
    # 英文字母表
    "alphabet": "abcdefghijklmnopqrstuvwxyz",
}

# 位操作推理器配置
BIT_MANIPULATION_CONFIG = {
    # 位宽
    "bit_width": 8,
    # 最大搜索参数范围（用于暴力搜索 XOR 常数等）
    "max_param_search": 256,
}

# 数字系统推理器配置
NUMERAL_CONFIG = {
    # 支持的进制系统
    "supported_bases": list(range(2, 37)),
}

# 密码算术推理器配置（equation/cryptarithm）
CRYPTARITHM_CONFIG = {
    # 支持的运算符
    "operators": ["+", "-", "*", "/", "|", "\\", "^"],
}

# ============================================================
# 数据增强配置
# ============================================================
AUGMENTATION_CONFIG = {
    # 每种题型的增强数据目标数量
    # 注：5 个保持不变的类继续使用粗类 key（augmenter 接口未改）；
    #     cryptarithm 粗类 key 保留向下兼容；同时按 9 类细分追加 4 个子类 key，
    #     当前 augmenter 暂未实现，target=0 仅作占位。
    "target_count_per_category": {
        "bit_manipulation": 5000,
        "gravity": 3000,
        "unit_conversion": 3000,
        "cipher": 3000,
        "numeral": 3000,
        # 粗类 key（向下兼容旧 augmentation.py）
        "cryptarithm": 3000,
        # 9 类细分占位（augmenter 未实现，先置 0）
        "cryptarithm_deduce": 0,
        "cryptarithm_guess": 0,
        "equation_numeric_deduce": 0,
        "equation_numeric_guess": 0,
    },
    # 随机种子
    "seed": 42,
}

# ============================================================
# 语料生成配置
# ============================================================
CORPUS_CONFIG = {
    # 是否包含增强数据
    "include_augmented": True,
    # 是否对无法确定性求解的题目生成简单回复
    "include_fallback": True,
    # 用户消息后缀（格式提示）
    "user_suffix": "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`",
}
