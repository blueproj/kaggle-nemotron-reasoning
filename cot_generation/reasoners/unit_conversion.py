"""
单位转换推理器 - 确定性求解线性单位转换题（output = factor * input）
=====================================================================

移植自原始方案的核心算法：
- 使用 long_division 逐步计算 factor = output / input
- 使用 long_multiplication 逐步计算最终结果
- 使用**中位数**而非平均值来确定 factor
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
    Problem, ReasoningResult, parse_unit_conversion_examples_str,
    cast_dp_pair, long_division_lines, long_multiplication_lines, truncate_3dp,
)


class UnitConversionReasoner:
    """单位转换题推理器（忠实移植原始算法）"""
    
    def solve(self, problem: Problem) -> ReasoningResult:
        """
        求解单位转换题。
        
        算法流程（与原方案完全一致）：
        1. 对每个示例计算 factor = output / input（使用 long_division）
        2. 对所有 factor 值排序，取中位数（偶数个取较小的中间值）
        3. 计算 result = query * factor（使用 long_multiplication）
        4. 截断到3位小数作为最终答案
        """
        try:
            # 使用字符串版本的解析器
            examples, query_str = parse_unit_conversion_examples_str(problem.prompt)
            
            if not examples or not query_str:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="unit_conversion",
                    success=False,
                    error_message="无法解析示例或查询值"
                )
            
            # 调用核心推理算法
            reasoning_text = self._reasoning_unit_conversion(examples, query_str)
            
            if reasoning_text is None:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="unit_conversion",
                    success=False,
                    error_message="推理过程失败"
                )
            
            # 从 \boxed{} 中提取答案（v15 兼容：支持嵌套大括号 / 含 } 答案）
            from reasoners._utils import extract_last_boxed
            predicted = extract_last_boxed(reasoning_text)
            
            return ReasoningResult(
                problem_id=problem.id,
                category="unit_conversion",
                success=True,
                predicted_answer=predicted,
                reasoning_text=reasoning_text,
            )
            
        except Exception as e:
            return ReasoningResult(
                problem_id=problem.id,
                category="unit_conversion",
                success=False,
                error_message=f"推理异常: {str(e)}"
            )
    
    def _reasoning_unit_conversion(
        self, examples: List[Tuple[str, str]], query_str: str
    ) -> Optional[str]:
        """
        核心推理算法（忠实移植原始方案）。
        
        全程使用字符串操作，与原始代码逻辑完全一致。
        """
        lines: List[str] = []
        lines.append(
            "We need to find a conversion rule that maps the inputs to outputs. "
            "Let me check if it's a linear factor."
        )
        lines.append("I will put my final answer inside \\boxed{}.")
        lines.append("")
        
        factor_strs: List[str] = []
        for inp_str, out_str_raw in examples:
            inp = float(inp_str)
            if inp != 0:
                out_str = truncate_3dp(out_str_raw)
                inp_str_trunc = truncate_3dp(inp_str)
                
                lines.append(f"{inp_str} -> {out_str_raw}")
                # 将 input 和 output 补齐到相同小数位
                inp_cast, out_cast, inp_dp, out_dp = cast_dp_pair(inp_str_trunc, out_str)
                lines.append(
                    f"Casting input to {inp_dp} decimal places, "
                    f"output to {out_dp} decimal places: "
                    f"{inp_cast} -> {out_cast}"
                )
                lines.append(f"factor = {out_cast} / {inp_cast}")
                # 使用 long_division 显示 factor 的计算步骤
                div_lines, factor_str = long_division_lines(out_cast, inp_cast)
                lines.extend(div_lines)
                lines.append(f"= {factor_str}")
                factor_strs.append(factor_str)
                lines.append("")
        
        if not factor_strs:
            return None
        
        factors = [float(s) for s in factor_strs]
        
        # 列出所有 factor 值并取中位数（偶数个时取较小的中间值）
        f_list_str = ", ".join(factor_strs)
        lines.append(f"factor values: {f_list_str}")
        paired = sorted(zip(factors, factor_strs))
        sorted_str = ", ".join(s for _, s in paired)
        lines.append(f"factor values (sorted): {sorted_str}")
        if len(paired) % 2 == 0 and len(paired) >= 2:
            _, med_factor_str = paired[len(paired) // 2 - 1]
        else:
            mid = len(paired) // 2
            _, med_factor_str = paired[mid]
        lines.append(f"The median factor is {med_factor_str}.")
        
        # 计算最终结果
        med_display = med_factor_str.rstrip("0").rstrip(".")
        lines.append("")
        lines.append(f"Converting {query_str}:")
        lines.append(f"{query_str} * {med_display}:")
        mult_lines, mult_result = long_multiplication_lines(query_str, med_display)
        lines.extend(mult_lines)
        # 截断到3位小数
        dot = mult_result.index(".")
        boxed_answer = mult_result[: dot + 4]
        lines.append(f"= {boxed_answer}")
        
        lines.append("")
        lines.append("I will now return the answer in \\boxed{}")
        lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{boxed_answer}}}")
        return "\n".join(lines)
