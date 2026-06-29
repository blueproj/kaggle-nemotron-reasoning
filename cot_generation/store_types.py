"""
数据类型与辅助函数
==================

定义 CoT 推理流程中使用的数据结构和通用工具函数。
包括：
- 题目数据结构
- 推理结果数据结构
- 题型分类函数
- 通用解析工具
- 长乘法/长除法显示步骤（用于gravity/unit_conversion推理器）
"""

import re
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

from config import CATEGORY_PATTERNS


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class Problem:
    """
    单道题目的数据结构。
    
    Attributes:
        id: 题目唯一标识符
        prompt: 题目原文
        answer: 标准答案
        category: 题型分类
        examples: 从 prompt 中解析出的输入输出示例
        query: 需要求解的输入
    """
    id: str
    prompt: str
    answer: str
    category: str = ""
    examples: List[Tuple[str, str]] = field(default_factory=list)
    query: str = ""


@dataclass
class ReasoningResult:
    """
    推理结果数据结构。
    
    Attributes:
        problem_id: 对应题目 ID
        category: 题型分类
        success: 是否成功求解
        predicted_answer: 预测答案
        reasoning_text: 推理过程文本 (CoT)
        error_message: 如果失败，记录错误信息
    """
    problem_id: str
    category: str
    success: bool
    predicted_answer: str = ""
    reasoning_text: str = ""
    error_message: str = ""


# ============================================================
# 题型分类函数
# ============================================================

def classify_problem(prompt: str) -> str:
    """
    根据 prompt 中的关键词判断题目类型。
    
    设计意图：
    - 竞赛中每种题型的 prompt 有固定的描述模板
    - 通过简单的关键词匹配即可高精度分类
    - 优先匹配更具体的关键词，避免误分类
    
    Args:
        prompt: 题目原文
        
    Returns:
        题型字符串，如 "bit_manipulation", "gravity" 等
        无法分类时返回 "unknown"
    """
    prompt_lower = prompt.lower()
    
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(p.lower() in prompt_lower for p in patterns):
            return category
    
    return "unknown"


# ============================================================
# 通用解析工具
# ============================================================

def parse_gravity_examples(prompt: str) -> Tuple[List[Tuple[float, float]], float]:
    """
    解析重力题的示例和查询值。
    
    示例格式: "For t = 1.37s, distance = 14.92 m"
    查询格式: "determine the falling distance for t = 4.41s"
    
    Returns:
        (examples_list, query_t) - 示例列表[(t, d), ...] 和查询时间
    """
    examples = []
    
    # 匹配示例: For t = X.XXs, distance = Y.YY m
    example_pattern = r'[Ff]or t\s*=\s*([\d.]+)\s*s?,\s*distance\s*=\s*([\d.]+)\s*m?'
    for match in re.finditer(example_pattern, prompt):
        t = float(match.group(1))
        d = float(match.group(2))
        examples.append((t, d))
    
    # 匹配查询: determine the falling distance for t = X.XXs
    query_pattern = r'determine the falling distance for t\s*=\s*([\d.]+)\s*s?'
    query_match = re.search(query_pattern, prompt)
    query_t = float(query_match.group(1)) if query_match else 0.0
    
    return examples, query_t


def parse_gravity_examples_str(prompt: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    解析重力题的示例和查询值（保留原始字符串格式）。
    
    与 parse_gravity_examples 类似，但返回字符串而非浮点数，
    以避免浮点精度损失。
    
    Returns:
        (examples_list, query_t_str) - 示例列表[(t_str, d_str), ...] 和查询时间字符串
    """
    examples = []
    
    example_pattern = r'[Ff]or t\s*=\s*([\d.]+)\s*s?,\s*distance\s*=\s*([\d.]+)\s*m?'
    for match in re.finditer(example_pattern, prompt):
        t_str = match.group(1)
        d_str = match.group(2)
        examples.append((t_str, d_str))
    
    query_pattern = r'determine the falling distance for t\s*=\s*([\d.]+)\s*s?'
    query_match = re.search(query_pattern, prompt)
    query_str = query_match.group(1) if query_match else ""
    
    return examples, query_str


def parse_unit_conversion_examples(prompt: str) -> Tuple[List[Tuple[float, float]], float]:
    """
    解析单位转换题的示例和查询值。
    
    示例格式: "10.08 m becomes 6.69" 或 "10.08 m becomes 6.69"
    查询格式: "convert the following measurement: 25.09 m"
    
    Returns:
        (examples_list, query_value) - 示例列表[(input, output), ...] 和查询值
    """
    examples = []
    
    # 匹配示例: X.XX m becomes Y.YY
    example_pattern = r'([\d.]+)\s*m\s+becomes\s+([\d.]+)'
    for match in re.finditer(example_pattern, prompt):
        input_val = float(match.group(1))
        output_val = float(match.group(2))
        examples.append((input_val, output_val))
    
    # 匹配查询: convert the following measurement: X.XX m
    query_pattern = r'convert the following measurement:\s*([\d.]+)\s*m?'
    query_match = re.search(query_pattern, prompt)
    query_value = float(query_match.group(1)) if query_match else 0.0
    
    return examples, query_value


def parse_unit_conversion_examples_str(prompt: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    解析单位转换题的示例和查询值（保留原始字符串格式）。
    
    Returns:
        (examples_list, query_str) - 示例列表[(input_str, output_str), ...] 和查询值字符串
    """
    examples = []
    
    example_pattern = r'([\d.]+)\s*m\s+becomes\s+([\d.]+)'
    for match in re.finditer(example_pattern, prompt):
        input_str = match.group(1)
        output_str = match.group(2)
        examples.append((input_str, output_str))
    
    query_pattern = r'convert the following measurement:\s*([\d.]+)\s*m?'
    query_match = re.search(query_pattern, prompt)
    query_str = query_match.group(1) if query_match else ""
    
    return examples, query_str


def parse_cipher_examples(prompt: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    解析密码替换题的示例和查询文本。
    
    示例格式: "encrypted text -> decrypted text"
    查询格式: "decrypt the following text: ..."
    
    Returns:
        (examples_list, query_text) - 示例列表[(加密, 明文), ...] 和查询密文
    """
    examples = []
    
    # 分行解析
    lines = prompt.strip().split('\n')
    
    # 查找包含 " -> " 的行作为示例
    for line in lines:
        if ' -> ' in line:
            parts = line.strip().split(' -> ')
            if len(parts) == 2:
                encrypted = parts[0].strip()
                decrypted = parts[1].strip()
                examples.append((encrypted, decrypted))
    
    # 匹配查询: "decrypt the following text: ..."
    query_pattern = r'decrypt the following text:\s*(.+?)$'
    query_match = re.search(query_pattern, prompt, re.MULTILINE)
    query_text = query_match.group(1).strip() if query_match else ""
    
    return examples, query_text


def parse_bit_manipulation_examples(prompt: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    解析位操作题的示例和查询值。
    
    示例格式: "01010001 -> 11011101"
    查询格式: "determine the output for: 00110100"
    
    Returns:
        (examples_list, query_bits) - 示例列表[(input_bits, output_bits), ...] 和查询位串
    """
    examples = []
    
    # 匹配 8-bit 示例: XXXXXXXX -> YYYYYYYY
    example_pattern = r'([01]{8})\s*->\s*([01]{8})'
    for match in re.finditer(example_pattern, prompt):
        input_bits = match.group(1)
        output_bits = match.group(2)
        examples.append((input_bits, output_bits))
    
    # 匹配查询: "determine the output for: XXXXXXXX"
    query_pattern = r'determine the output for:\s*([01]{8})'
    query_match = re.search(query_pattern, prompt)
    query_bits = query_match.group(1) if query_match else ""
    
    return examples, query_bits


def parse_numeral_examples(prompt: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    解析数字系统转换题的示例和查询值。
    
    示例格式: "11 -> XI" 或 "15 -> XV"
    查询格式: "write the number 38 in the Wonderland numeral system"
    
    Returns:
        (examples_list, query_number) - 示例列表[(input, output), ...] 和查询数字
    """
    examples = []
    
    lines = prompt.strip().split('\n')
    
    # 匹配示例行: "XX -> YY"
    for line in lines:
        line = line.strip()
        if ' -> ' in line:
            parts = line.split(' -> ')
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
    
    # 匹配查询: "write the number XX in the Wonderland numeral system"
    query_pattern = r'write the number\s+(\S+)\s+in'
    query_match = re.search(query_pattern, prompt)
    query_number = query_match.group(1) if query_match else ""
    
    return examples, query_number


def parse_cryptarithm_examples(prompt: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    解析密码算术/方程变换题的示例和查询。
    
    示例格式: "表达式 = 结果"（使用各种符号字符）
    查询格式: "determine the result for: ..."
    
    Returns:
        (examples_list, query_expr) - 示例列表[(expression, result), ...] 和查询表达式
    """
    examples = []
    
    lines = prompt.strip().split('\n')
    
    # 查找包含 " = " 的行（但不是描述性文字）
    for line in lines:
        line = line.strip()
        # 跳过描述性行
        if line.startswith("In Alice") or line.startswith("Now,") or line.startswith("Below"):
            continue
        if ' = ' in line and not line.startswith("In "):
            # 最后一个 ' = ' 作为分隔符（因为表达式本身可能包含 = ）
            idx = line.rfind(' = ')
            if idx > 0:
                expr = line[:idx].strip()
                result = line[idx+3:].strip()
                examples.append((expr, result))
    
    # 匹配查询: "determine the result for: ..."
    query_pattern = r'determine the result for:\s*(.+?)$'
    query_match = re.search(query_pattern, prompt, re.MULTILINE)
    query_expr = query_match.group(1).strip() if query_match else ""
    
    return examples, query_expr


def load_problems(csv_path: str) -> List[Problem]:
    """
    从 CSV 文件加载题目列表，并自动分类和解析。
    
    Args:
        csv_path: CSV 文件路径（需包含 id, prompt, answer 列）
        
    Returns:
        Problem 对象列表
    """
    df = pd.read_csv(csv_path)
    problems = []
    
    for _, row in df.iterrows():
        problem = Problem(
            id=str(row["id"]),
            prompt=str(row["prompt"]),
            answer=str(row["answer"]),
        )
        # 自动分类
        problem.category = classify_problem(problem.prompt)
        problems.append(problem)
    
    return problems


# ============================================================
# 长乘法/长除法辅助函数（移植自原始方案）
# 用于 gravity 和 unit_conversion 推理器生成详细步骤
# ============================================================

def _fmt_int_with_dp(value: int, dp: int) -> str:
    """将整数格式化为带 dp 位小数的字符串。"""
    if dp == 0:
        return str(value)
    s = str(value).zfill(dp + 1)
    s = s[: len(s) - dp] + "." + s[len(s) - dp:]
    # 去除前导零，但保留小数点前至少一位
    s = s.lstrip("0") or "0"
    if s.startswith("."):
        s = "0" + s
    return s


def truncate_3dp(s: str) -> str:
    """将小数字符串截断到最多3位小数（不四舍五入）。"""
    if "." not in s:
        return s
    integer, frac = s.split(".")
    if len(frac) <= 3:
        return s
    return integer + "." + frac[:3]


def _dp_count(s: str) -> int:
    """计算小数位数。"""
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def pad_dp(s: str, n: int) -> str:
    """将小数字符串补齐到恰好 n 位小数。"""
    if "." not in s:
        s = s + "."
    integer, frac = s.split(".")
    return integer + "." + frac.ljust(n, "0")


def cast_dp_pair(a: str, b: str) -> Tuple[str, str, int, int]:
    """
    将两个小数字符串补齐到相同的小数位数。
    
    Returns:
        (a_padded, b_padded, target_dp, target_dp)
    """
    da, db = _dp_count(a), _dp_count(b)
    target = max(da, db)
    return pad_dp(a, target), pad_dp(b, target), target, target


def long_multiplication_lines(a_str: str, b_str: str) -> Tuple[List[str], str]:
    """
    生成逐步长乘法过程。
    
    将 b 分解为按位值的分量，逐一与 a 相乘，然后展示累加过程。
    
    Args:
        a_str: 被乘数（可含小数点）
        b_str: 乘数（可含小数点）
        
    Returns:
        (步骤行列表, 最终结果字符串)
    """
    # 计算小数位数
    a_dp = len(a_str.split(".")[1]) if "." in a_str else 0
    b_dp = len(b_str.split(".")[1]) if "." in b_str else 0
    total_dp = a_dp + b_dp

    # 转为整数计算
    a_int = int(a_str.replace(".", ""))
    b_int = int(b_str.replace(".", ""))

    lines: List[str] = []

    # 将 b 分解为各位值分量（从最低位开始）
    b_digits_str = str(abs(b_int))
    b_num_digits = len(b_digits_str)

    # (分量显示, 乘积整数, 乘积显示)
    components: List[Tuple[str, int, str]] = []
    for i in range(b_num_digits - 1, -1, -1):
        d = int(b_digits_str[i])
        if d == 0:
            continue
        # 分量值（按 10^b_dp 缩放）
        comp_scaled = d * (10 ** (b_num_digits - 1 - i))
        comp_display = _fmt_int_with_dp(comp_scaled, b_dp)
        if b_dp > 0:
            comp_display = pad_dp(comp_display, b_dp)

        product_int = a_int * comp_scaled  # 按 10^total_dp 缩放
        product_display = _fmt_int_with_dp(product_int, total_dp)
        if total_dp > 0:
            product_display = pad_dp(product_display, total_dp)

        components.append((comp_display, product_int, product_display))

    # 乘法行: a * 分量 = 乘积
    for comp_display, _, product_display in components:
        lines.append(f"{a_str} * {comp_display} = {product_display}")

    # 累加过程（从最小到最大）
    if len(components) >= 2:
        running = components[0][1]
        for i in range(1, len(components)):
            running_display = _fmt_int_with_dp(running, total_dp)
            if total_dp > 0:
                running_display = pad_dp(running_display, total_dp)
            running += components[i][1]
            sum_display = _fmt_int_with_dp(running, total_dp)
            if total_dp > 0:
                sum_display = pad_dp(sum_display, total_dp)
            lines.append(f"{running_display} + {components[i][2]} = {sum_display}")

    # 计算最终结果
    total = a_int * b_int
    result_str = _fmt_int_with_dp(total, total_dp)
    return lines, result_str


def long_division_lines(
    numerator_str: str, denominator_str: str, max_decimal_digits: int = 3
) -> Tuple[List[str], str]:
    """
    生成长除法步骤（通过反复减法实现）。
    
    Args:
        numerator_str: 被除数字符串
        denominator_str: 除数字符串
        max_decimal_digits: 最大小数位数
        
    Returns:
        (步骤行列表, 结果字符串)
    """
    n_dp: int = len(numerator_str.split(".")[1]) if "." in numerator_str else 0
    d_dp: int = len(denominator_str.split(".")[1]) if "." in denominator_str else 0
    max_dp: int = max(n_dp, d_dp)

    num: int = int(round(float(numerator_str) * 10**max_dp))
    den: int = int(round(float(denominator_str) * 10**max_dp))

    lines: List[str] = []
    acc: int = 0  # 累加器（整数形式）；真实值 = acc / 10^decimal_digits
    decimal_digits: int = 0

    def fmt_acc() -> str:
        if decimal_digits == 0:
            return str(acc)
        s = str(acc).zfill(decimal_digits + 1)
        return s[:-decimal_digits] + "." + s[-decimal_digits:]

    def fmt_scale() -> str:
        if decimal_digits == 0:
            return "1"
        return "0." + "0" * (decimal_digits - 1) + "1"

    def fmt_line(n: int) -> str:
        return f"= {fmt_acc()} + {fmt_scale()} * {n} / {den}"

    lines.append(fmt_line(num))

    while decimal_digits <= max_decimal_digits:
        if num >= den:
            num -= den
            acc += 1
            lines.append(fmt_line(num))
        else:
            decimal_digits += 1
            if decimal_digits > max_decimal_digits:
                break
            num *= 10
            acc *= 10
            lines.append(fmt_line(num))

    # 如果 decimal_digits 超过了 max 则恢复
    if decimal_digits > max_decimal_digits:
        decimal_digits = max_decimal_digits
    return lines, fmt_acc()
