"""
重力推理器 - 确定性求解 d = k * t^2 类题目
=============================================

移植自原始方案的核心算法：
- 使用 long_division 逐步计算 k = d / t^2
- 使用 long_multiplication 逐步计算 t^2 和最终结果
- 使用**中位数**而非平均值来确定 k 值
- 结果截断到3位小数
- 全程使用原始字符串进行算术，避免浮点精度损失

确定性保证：使用精确的字符串算术
"""

import re
import sys
import os
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from store_types import (
    Problem, ReasoningResult, parse_gravity_examples_str,
    cast_dp_pair, long_division_lines, long_multiplication_lines, truncate_3dp,
)


class GravityReasoner:
    """重力/自由落体题推理器（忠实移植原始算法）"""
    
    def solve(self, problem: Problem) -> ReasoningResult:
        """
        求解重力题。
        
        算法流程（与原方案完全一致）：
        1. 对每个示例：计算 t^2（long_multiplication），计算 k = d / t^2（long_division）
        2. 对所有 k 值排序，取中位数（偶数个取较小的中间值）
        3. 计算 query_t^2（long_multiplication）
        4. 计算 d = k * t^2（long_multiplication）
        5. 截断到3位小数作为最终答案
        """
        try:
            # 使用字符串版本的解析器，保留原始数值格式
            examples, query_str = parse_gravity_examples_str(problem.prompt)
            
            if not examples or not query_str:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="gravity",
                    success=False,
                    error_message="无法解析示例或查询值"
                )
            
            # 调用核心推理算法
            reasoning_text = self._reasoning_gravity(examples, query_str)
            
            if reasoning_text is None:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="gravity",
                    success=False,
                    error_message="推理过程失败"
                )
            
            # 从 \boxed{} 中提取答案（v15 兼容：支持嵌套大括号 / 含 } 答案）
            from reasoners._utils import extract_last_boxed
            predicted = extract_last_boxed(reasoning_text)
            
            return ReasoningResult(
                problem_id=problem.id,
                category="gravity",
                success=True,
                predicted_answer=predicted,
                reasoning_text=reasoning_text,
            )
            
        except Exception as e:
            return ReasoningResult(
                problem_id=problem.id,
                category="gravity",
                success=False,
                error_message=f"推理异常: {str(e)}"
            )
    
    def _reasoning_gravity(
        self, examples: List[Tuple[str, str]], query_str: str
    ) -> Optional[str]:
        """
        核心推理算法（忠实移植原始方案）。
        
        全程使用字符串操作，与原始代码逻辑完全一致。
        """
        lines: List[str] = []
        lines.append(
            "We need to determine the falling distance using d = k*t^2. "
            "Let me find k from the examples."
        )
        lines.append("I will put my final answer inside \\boxed{}.")
        lines.append("")
        
        k_strs: List[str] = []
        for t_str, d_str_raw in examples:
            t = float(t_str)
            if t > 0:
                # 计算 t^2（使用精确浮点再转回字符串）
                t_squared = round(t * t, 4)
                t_sq_full = str(t_squared)
                t_sq_str = truncate_3dp(t_sq_full)
                d_str = truncate_3dp(d_str_raw)
                
                lines.append(f"t = {t_str}s, d = {d_str_raw}m:")
                lines.append(f"t^2 = {t_str} * {t_str}:")
                # 使用 long_multiplication 显示 t^2 的计算步骤
                sq_lines, sq_result = long_multiplication_lines(t_str, t_str)
                lines.extend(sq_lines)
                if sq_result != t_sq_full:
                    lines.append(f"= {t_sq_full}")
                
                # 将 d 和 t^2 补齐到相同小数位
                d_cast, tsq_cast, _, _ = cast_dp_pair(d_str, t_sq_str)
                lines.append(
                    f"k = {d_str_raw} / {t_str}^2 "
                    f"= {d_str} / {t_sq_full} = {d_cast} / {tsq_cast}"
                )
                # 使用 long_division 显示 k 的计算步骤
                div_lines, k_str = long_division_lines(d_cast, tsq_cast)
                lines.extend(div_lines)
                lines.append(f"= {k_str}")
                k_strs.append(k_str)
                lines.append("")
        
        if not k_strs:
            return None
        
        k_values = [float(s) for s in k_strs]
        
        # 列出所有 k 值并取中位数（偶数个时取较小的中间值）
        k_list_str = ", ".join(k_strs)
        lines.append(f"k values: {k_list_str}")
        paired = sorted(zip(k_values, k_strs))
        sorted_k_str = ", ".join(s for _, s in paired)
        lines.append(f"k values (sorted): {sorted_k_str}")
        if len(paired) % 2 == 0 and len(paired) >= 2:
            _, k_fit_str = paired[len(paired) // 2 - 1]
        else:
            mid = len(paired) // 2
            _, k_fit_str = paired[mid]
        lines.append(f"The median k is {k_fit_str}.")
        
        # 计算查询时间的 t^2
        lines.append("")
        lines.append(f"For t = {query_str}:")
        lines.append(f"t^2 = {query_str} * {query_str}:")
        sq_lines, t_sq_str = long_multiplication_lines(query_str, query_str)
        lines.extend(sq_lines)
        lines.append(f"= {t_sq_str}")
        lines.append("")
        
        # 计算 d = k * t^2
        k_display = k_fit_str.rstrip("0").rstrip(".")
        lines.append(f"d = {k_display} * {t_sq_str}:")
        mult_lines, mult_result = long_multiplication_lines(k_display, t_sq_str)
        lines.extend(mult_lines)
        # 截断到3位小数
        dot = mult_result.index(".")
        boxed_answer = mult_result[: dot + 4]
        lines.append(f"= {boxed_answer}")
        
        lines.append("")
        lines.append("I will now return the answer in \\boxed{}")
        lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{boxed_answer}}}")
        return "\n".join(lines)
