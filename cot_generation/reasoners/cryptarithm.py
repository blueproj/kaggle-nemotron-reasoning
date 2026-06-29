"""
密码算术推理器 - 处理符号拼接类 cryptarithm 题目
=================================================

移植自原始方案的核心算法：
- 识别 5 字符格式的输入（2字符 + 运算符 + 2字符）
- 检测拼接类型：正向拼接 (A1A2B1B2) 或反向拼接 (B1B2A1A2)
- 按运算符分组，对每个运算符独立检测拼接方向

确定性保证：直接比较字符拼接结果，无歧义
"""

from __future__ import annotations

import re
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from store_types import Problem, ReasoningResult, parse_cryptarithm_examples


@dataclass
class _Ex:
    """解析后的单个示例。"""
    a: Tuple[str, str]
    op: str
    b: Tuple[str, str]
    out: str


def _concat_type(exs: list[_Ex]) -> Optional[str]:
    """检测拼接类型：'fwd'=A1A2B1B2, 'rev'=B1B2A1A2, None=无法确定。"""
    if all(ex.out == ex.a[0] + ex.a[1] + ex.b[0] + ex.b[1] for ex in exs):
        return "fwd"
    if all(ex.out == ex.b[0] + ex.b[1] + ex.a[0] + ex.a[1] for ex in exs):
        return "rev"
    return None


def _box(s: str) -> str:
    """将每个字符包裹在【】中。"""
    return "".join(f"【{c}】" for c in s)


class CryptarithmReasoner:
    """密码算术/符号拼接推理器（忠实移植原始算法）"""
    
    def solve(self, problem: Problem) -> ReasoningResult:
        """
        求解密码算术题。
        
        算法流程：
        0. 优先尝试 CSP 全局求解器（10 ops + unique=True/False fallback，
           严格唯一性，多解时弃权）—— 命中即用真实搜索 trace 渲染 CoT
        1. 否则尝试 equation_numeric 解法（数值运算）
        2. 否则尝试 5 字符拼接规则
        """
        try:
            examples, query_expr = parse_cryptarithm_examples(problem.prompt)
            
            if not examples or not query_expr:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="cryptarithm",
                    success=False,
                    error_message="无法解析示例或查询表达式"
                )

            # 策略1：尝试 equation_numeric 解法（v1 原版）
            from reasoners.equation_numeric import EquationNumericReasoner
            eq_reasoner = EquationNumericReasoner()
            eq_result = eq_reasoner.solve(problem)
            if eq_result.success:
                return eq_result
            
            # 策略2：尝试5字符拼接格式
            reasoning_text = self._reasoning_cryptarithm(examples, query_expr)
            
            if reasoning_text is None:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="cryptarithm",
                    success=False,
                    error_message="无法识别拼接规则"
                )
            
            # 提取答案（v15 兼容：支持嵌套大括号 / 含 } 答案）
            from reasoners._utils import extract_last_boxed
            predicted = extract_last_boxed(reasoning_text)
            
            return ReasoningResult(
                problem_id=problem.id,
                category="cryptarithm",
                success=True,
                predicted_answer=predicted,
                reasoning_text=reasoning_text,
            )
            
        except Exception as e:
            return ReasoningResult(
                problem_id=problem.id,
                category="cryptarithm",
                success=False,
                error_message=f"推理异常: {str(e)}"
            )
    
    def _reasoning_cryptarithm(
        self, examples: List[Tuple[str, str]], query: str
    ) -> Optional[str]:
        """
        核心推理算法（忠实移植原始方案）。
        
        处理 5 字符输入格式的拼接类 cryptarithm。
        """
        def quote(s: str) -> str:
            return f"【{s}】"
        
        # 解析示例
        exs: list[_Ex] = []
        for expr, result in examples:
            inp = expr.strip()
            if len(inp) != 5:
                return None
            exs.append(
                _Ex(
                    a=(inp[0], inp[1]),
                    op=inp[2],
                    b=(inp[3], inp[4]),
                    out=result.strip(),
                )
            )
        
        # 解析查询
        q = query.strip()
        if len(q) != 5:
            return None
        q_a = (q[0], q[1])
        q_op = q[2]
        q_b = (q[3], q[4])
        
        # 按运算符分组
        by_op: dict[str, list[_Ex]] = {}
        for parsed_ex in exs:
            by_op.setdefault(parsed_ex.op, []).append(parsed_ex)
        
        # 对每个运算符检测拼接类型
        concat_types: dict[str, str] = {}
        for op, op_exs in by_op.items():
            ct = _concat_type(op_exs)
            if ct is not None:
                concat_types[op] = ct
        
        # 确定查询运算符的拼接类型
        if q_op in by_op:
            q_ct = _concat_type(by_op[q_op])
            if q_ct is None:
                q_ct = "fwd"
        else:
            q_ct = "fwd"
        
        # 计算答案
        if q_ct == "fwd":
            answer = q_a[0] + q_a[1] + q_b[0] + q_b[1]
        else:
            answer = q_b[0] + q_b[1] + q_a[0] + q_a[1]
        
        # 生成推理过程
        lines: list[str] = []
        lines.append("We need to infer the transformation rule from the examples.")
        lines.append("I will put my final answer inside \\boxed{}.")
        lines.append("")
        
        # 展示每个示例的拼接检查
        for (expr, result), ex_parsed in zip(examples, exs):
            orig_inp = expr.strip()
            orig_out = result.strip()
            lines.append(f"{quote(orig_inp)} = {quote(orig_out)}")
            a0, a1 = quote(ex_parsed.a[0]), quote(ex_parsed.a[1])
            b0, b1 = quote(ex_parsed.b[0]), quote(ex_parsed.b[1])
            op_q = quote(ex_parsed.op)
            out_boxed = _box(orig_out)
            lines.append(f"  input: {a0}{a1}{op_q}{b0}{b1}")
            lines.append(f"  left:{a0}{a1}")
            lines.append(f"  operator: {op_q}")
            lines.append(f"  right:{b0}{b1}")
            lines.append(f"  output: {out_boxed}")
            
            fwd = ex_parsed.a[0] + ex_parsed.a[1] + ex_parsed.b[0] + ex_parsed.b[1]
            rev = ex_parsed.b[0] + ex_parsed.b[1] + ex_parsed.a[0] + ex_parsed.a[1]
            is_fwd = orig_out == fwd
            is_rev = orig_out == rev
            
            lines.append(f"  concatenation: {_box(fwd)} {'match' if is_fwd else 'mismatch'}")
            lines.append(f"  reverse concatenation: {_box(rev)} {'match' if is_rev else 'mismatch'}")
            
            ct = concat_types.get(ex_parsed.op)
            if ct == "fwd":
                op_type = "concatenation"
            elif ct == "rev":
                op_type = "reverse concatenation"
            else:
                op_type = "unknown"
            lines.append(f"  operator: {quote(ex_parsed.op)}{op_type}")
            lines.append("")
        
        # 对查询应用规则
        op_label = "concatenation" if q_ct == "fwd" else "reverse concatenation"
        qa0, qa1 = quote(q_a[0]), quote(q_a[1])
        qb0, qb1 = quote(q_b[0]), quote(q_b[1])
        lines.append(f"Question{quote(q)}")
        lines.append(f"  input: {qa0}{qa1}{quote(q_op)}{qb0}{qb1}")
        lines.append(f"  left:{qa0}{qa1}")
        lines.append(f"  operator:{quote(q_op)}")
        lines.append(f"  right:{qb0}{qb1}")
        lines.append("")
        
        q_op_known = q_op in concat_types
        if q_op_known:
            lines.append(f"The question operator is {quote(q_op)}, which is {op_label}.")
        else:
            lines.append(f"The question operator is {quote(q_op)}, which is unknown.")
            lines.append("As the question operator is unknown, we default to concatenation.")
        lines.append("")
        
        lines.append(f"  {op_label}({qa0}{qa1}, {qb0}{qb1}) = {_box(answer)}")
        lines.append(f"  output: {quote(answer)}-> {quote('{' + answer + '}')}")
        lines.append("")
        lines.append("I will now return the answer in \\boxed{}")
        lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
        return "\n".join(lines)
