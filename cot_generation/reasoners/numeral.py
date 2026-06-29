"""
数字系统推理器 - 确定性求解进制/数字系统转换题
================================================

设计思路：
- 题目给出数字到某种表示的转换示例
- 常见模式包括：罗马数字、自定义进制、自定义符号系统
- 通过示例推断转换规则，然后应用到查询值

支持的数字系统类型：
1. 罗马数字 (I, V, X, L, C, D, M 等)
2. 自定义进制系统（如 base-N）
3. 自定义符号映射

确定性保证：
- 罗马数字有标准规则
- 自定义进制可通过示例唯一确定基数
- 符号映射可通过足够示例唯一确定
"""

import sys
import os
import re
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from store_types import Problem, ReasoningResult, parse_numeral_examples


class NumeralReasoner:
    """数字系统转换推理器"""
    
    # 罗马数字映射表
    ROMAN_VALUES = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    
    def solve(self, problem: Problem) -> ReasoningResult:
        """
        求解数字系统转换题。
        
        算法流程：
        1. 解析示例 (input_number, output_representation)
        2. 判断输出格式类型（罗马数字 / 自定义进制 / 其他）
        3. 根据类型应用对应的转换算法
        """
        try:
            examples, query_number = parse_numeral_examples(problem.prompt)
            
            if not examples or not query_number:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="numeral",
                    success=False,
                    error_message="无法解析示例或查询值"
                )
            
            # 判断是否为罗马数字
            if self._is_roman_numeral_system(examples):
                return self._solve_roman(problem, examples, query_number)
            
            # 尝试自定义进制系统
            result = self._solve_custom_base(problem, examples, query_number)
            if result.success:
                return result
            
            # 尝试自定义符号映射
            return self._solve_symbol_mapping(problem, examples, query_number)
            
        except Exception as e:
            return ReasoningResult(
                problem_id=problem.id,
                category="numeral",
                success=False,
                error_message=f"推理异常: {str(e)}"
            )
    
    def _is_roman_numeral_system(self, examples: List[Tuple[str, str]]) -> bool:
        """
        判断示例是否使用罗马数字系统。
        通过检查输出是否只包含罗马数字字符来判断。
        """
        roman_chars = set('IVXLCDM')
        for _, output in examples:
            if not all(c in roman_chars for c in output):
                return False
        return True
    
    def _solve_roman(self, problem: Problem, examples: List[Tuple[str, str]], query: str) -> ReasoningResult:
        """
        使用罗马数字规则求解。
        标准罗马数字转换是完全确定性的。
        """
        try:
            num = int(query)
        except ValueError:
            return ReasoningResult(
                problem_id=problem.id,
                category="numeral",
                success=False,
                error_message=f"无法将查询 '{query}' 解析为整数"
            )
        
        # 验证：用示例确认确实是标准罗马数字
        for inp_str, expected_output in examples:
            try:
                inp_num = int(inp_str)
                computed = self._int_to_roman(inp_num)
                if computed != expected_output:
                    # 不是标准罗马数字，可能是变体
                    return self._solve_custom_base(problem, examples, query)
            except ValueError:
                continue
        
        predicted = self._int_to_roman(num)
        reasoning = self._generate_roman_reasoning(examples, num, predicted)
        
        return ReasoningResult(
            problem_id=problem.id,
            category="numeral",
            success=True,
            predicted_answer=predicted,
            reasoning_text=reasoning,
        )
    
    def _int_to_roman(self, num: int) -> str:
        """整数转罗马数字"""
        result = []
        for value, symbol in self.ROMAN_VALUES:
            while num >= value:
                result.append(symbol)
                num -= value
        return ''.join(result)
    
    def _solve_custom_base(self, problem: Problem, examples: List[Tuple[str, str]], query: str) -> ReasoningResult:
        """
        尝试将转换解释为自定义进制系统。
        
        通过示例反推基数和数字符号。
        """
        # 尝试识别输出是否为某种进制表示
        # 首先检查输出是否为纯数字（可能是另一种进制的十进制表示）
        try:
            query_num = int(query)
        except ValueError:
            return ReasoningResult(
                problem_id=problem.id,
                category="numeral",
                success=False,
                error_message="无法解析查询数字"
            )
        
        # 尝试不同的进制 (2-36)
        for base in range(2, 37):
            all_match = True
            for inp_str, expected in examples:
                try:
                    inp_num = int(inp_str)
                    converted = self._int_to_base(inp_num, base)
                    if converted.upper() != expected.upper():
                        all_match = False
                        break
                except (ValueError, Exception):
                    all_match = False
                    break
            
            if all_match:
                predicted = self._int_to_base(query_num, base)
                reasoning = self._generate_base_reasoning(examples, base, query_num, predicted)
                return ReasoningResult(
                    problem_id=problem.id,
                    category="numeral",
                    success=True,
                    predicted_answer=predicted,
                    reasoning_text=reasoning,
                )
        
        return ReasoningResult(
            problem_id=problem.id,
            category="numeral",
            success=False,
            error_message="无法确定进制系统"
        )
    
    def _int_to_base(self, num: int, base: int) -> str:
        """将整数转换为指定进制的字符串表示"""
        if num == 0:
            return "0"
        
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = []
        n = abs(num)
        
        while n > 0:
            result.append(digits[n % base])
            n //= base
        
        if num < 0:
            result.append('-')
        
        return ''.join(reversed(result))
    
    def _solve_symbol_mapping(self, problem: Problem, examples: List[Tuple[str, str]], query: str) -> ReasoningResult:
        """
        尝试通过符号映射表求解。
        
        如果输出使用自定义符号，尝试建立数值到符号的映射规则。
        """
        # 这是一个兜底方案：如果前面的方法都失败了
        # 尝试直接在示例中找到查询值的对应
        for inp_str, output in examples:
            if inp_str == query:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="numeral",
                    success=True,
                    predicted_answer=output,
                    reasoning_text=f"Direct lookup: {query} -> {output}",
                )
        
        return ReasoningResult(
            problem_id=problem.id,
            category="numeral",
            success=False,
            error_message="无法确定数字系统转换规则"
        )
    
    def _generate_roman_reasoning(self, examples, query_num, predicted) -> str:
        """生成罗马数字转换的 CoT 推理文本"""
        lines = []
        lines.append("I can see from the examples that this is a Roman numeral conversion system.")
        lines.append("")
        lines.append("Let me verify with the examples:")
        for inp_str, output in examples[:4]:
            lines.append(f"  {inp_str} -> {output} ✓")
        lines.append("")
        lines.append("Roman numeral rules:")
        lines.append("  I=1, V=5, X=10, L=50, C=100, D=500, M=1000")
        lines.append("  Subtractive notation: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900")
        lines.append("")
        lines.append(f"Converting {query_num} to Roman numerals:")
        
        # 展示分解过程
        remaining = query_num
        parts = []
        for value, symbol in self.ROMAN_VALUES:
            while remaining >= value:
                parts.append(f"{symbol}({value})")
                remaining -= value
        lines.append(f"  {query_num} = {' + '.join(parts)}")
        lines.append(f"  Result: {predicted}")
        
        return "\n".join(lines)
    
    def _generate_base_reasoning(self, examples, base, query_num, predicted) -> str:
        """生成进制转换的 CoT 推理文本"""
        lines = []
        lines.append(f"I need to identify the numeral system used in Wonderland.")
        lines.append("")
        lines.append("Analyzing the examples:")
        for inp_str, output in examples[:4]:
            lines.append(f"  {inp_str} -> {output}")
        lines.append("")
        lines.append(f"This appears to be a base-{base} numeral system.")
        lines.append("")
        lines.append(f"Converting {query_num} to base {base}:")
        
        # 展示转换过程
        n = query_num
        steps = []
        while n > 0:
            steps.append(f"  {n} ÷ {base} = {n // base} remainder {n % base}")
            n //= base
        for step in steps:
            lines.append(step)
        
        lines.append(f"  Result: {predicted}")
        
        return "\n".join(lines)
