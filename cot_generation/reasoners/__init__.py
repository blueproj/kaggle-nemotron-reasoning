"""
推理器包 - 各题型确定性推理器的集合 (V1 baseline only, code/new_cot 隔离副本)
======================================================================

每个推理器负责：
1. 从题目 prompt 中解析示例和查询
2. 根据示例推断规则
3. 应用规则计算答案
4. 生成详细的推理过程文本（用于 CoT 训练）

注: 本副本明确仅注册 V1 baseline reasoner。bit_manipulation_v2 / equation_numeric_v2
    均因 GT 引导泄漏 / 教学性丧失被审计判定不可用，已不在此搬迁，也不在此 import。
"""

from reasoners.gravity import GravityReasoner
from reasoners.unit_conversion import UnitConversionReasoner
from reasoners.cipher import CipherReasoner
from reasoners.bit_manipulation import BitManipulationReasoner  # V1 baseline (85.14%, CoT 与 GT 完全独立)
from reasoners.numeral import NumeralReasoner
from reasoners.cryptarithm import CryptarithmReasoner
from reasoners.equation_numeric import EquationNumericReasoner  # v1 原版（CoT 不依赖 GT，纯算法搜索）

# 推理器注册表：题型名称 -> 推理器类
REASONER_REGISTRY = {
    "gravity": GravityReasoner,
    "unit_conversion": UnitConversionReasoner,
    "cipher": CipherReasoner,
    "bit_manipulation": BitManipulationReasoner,
    "numeral": NumeralReasoner,
    "cryptarithm": CryptarithmReasoner,
    "equation_numeric": EquationNumericReasoner,
}


def get_reasoner(category: str):
    """根据题型名称获取对应的推理器实例。

    Args:
        category: 题型名称

    Returns:
        推理器实例，如果题型不支持则返回 None
    """
    reasoner_class = REASONER_REGISTRY.get(category)
    if reasoner_class is None:
        return None
    return reasoner_class()
