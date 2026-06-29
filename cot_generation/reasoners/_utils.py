"""
reasoners 共享工具
==================
所有 reasoner 共享的答案提取等工具函数。
"""

import re


def extract_last_boxed(text: str) -> str:
    r"""提取 text 中最后一个 \boxed{...} 的内容（v15 兼容版）。

    与 Kaggle 官方 metric v15 的 extract_final_answer 中 boxed 提取逻辑保持一致：
    多阶段定位所有 \boxed{ 起点，每段取段内最后一个 } 作为右边界。
    可正确处理:
      - 嵌套 LaTeX，如 \boxed{\frac{1}{2}}
      - 答案本身含 } 字符，如 \boxed{}52}  → "}52"

    Args:
        text: 推理文本

    Returns:
        最后一个非空 boxed 的内容，若无则返回空串
    """
    if not text:
        return ""
    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    if not boxed_starts:
        return ""
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    non_empty = [m.strip() for m in matches if m.strip()]
    if non_empty:
        return non_empty[-1]
    return matches[-1].strip() if matches else ""
