"""
位操作推理器 - 确定性求解 8-bit 位变换题
==========================================

移植自原始方案的核心算法：
- 覆盖完整操作族：对称（XOR/OR/AND）、非对称（AND-NOT/XOR-NOT/OR-NOT）、
  一元（I/NOT）、常数（0/1）
- 基于列匹配的规则发现
- 左右最长 stride 链搜索
- 中间位通过完美匹配填充

确定性保证：通过穷举所有操作族找到精确匹配的变换规则
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from store_types import Problem, ReasoningResult, parse_bit_manipulation_examples

N_BITS = 8

# 操作族定义
SYM_FAMILIES = ("XOR", "OR", "AND")
ASYM_FAMILIES = ("AND-NOT", "XOR-NOT", "OR-NOT")
PAIR_FAMILIES = SYM_FAMILIES + ASYM_FAMILIES
UNARY_FAMILIES = ("I", "NOT")
CONSTANT_FAMILIES = ("0", "1")
DEFAULT_FAMILY: RuleFamily = "DEFAULT"

# 段处理顺序
SECTION_ORDER = (
    "Identity", "NOT", "Constant",
    "AND", "OR", "XOR",
    "AND-NOT", "OR-NOT", "XOR-NOT",
)

# 段名到操作族的映射
_SECTION_TO_FAMILIES = {
    "Identity": ("I",),
    "NOT": ("NOT",),
    "Constant": ("0", "1"),
}

# 反向映射：操作族 → 段名
_FAMILY_TO_SECTION: dict[str, str] = {}
for _section in SECTION_ORDER:
    for _fam in _SECTION_TO_FAMILIES.get(_section, (_section,)):
        _FAMILY_TO_SECTION[_fam] = _section

# 规则族类型
RuleFamily = Literal[
    "I", "NOT", "0", "1",
    "XOR", "OR", "AND",
    "AND-NOT", "XOR-NOT", "OR-NOT",
    "DEFAULT",
]


@dataclass(frozen=True)
class RuleCandidate:
    """一条候选规则（描述输出某位如何从输入位计算）。"""
    family: RuleFamily
    primary: Optional[int]
    secondary: Optional[int]
    expr: str
    primary_stride: Optional[int] = None
    secondary_stride: Optional[int] = None
    primary_offset: Optional[int] = None
    secondary_offset: Optional[int] = None

    @property
    def is_default(self) -> bool:
        return self.family == DEFAULT_FAMILY


@dataclass(frozen=True)
class Record:
    """原始列记录。"""
    label: str
    col: str
    hash_: str
    matches: Tuple[int, ...]


def _normalize_bits(value: str) -> str:
    """从字符串提取8位二进制。"""
    bits = "".join(ch for ch in str(value) if ch in {"0", "1"})
    if len(bits) != N_BITS:
        return ""
    return bits


def _column_bits(values: Sequence[str], bit: int) -> str:
    """提取所有样本在指定位的列向量。"""
    return "".join(v[bit] for v in values)


def _bit_not(bit: str) -> str:
    return "1" if bit == "0" else "0"


def _invert(bits: str) -> str:
    return "".join(_bit_not(b) for b in bits)


def _column_hash(bits: str, total_examples: int) -> str:
    """列哈希：全0或全1返回'a'，否则返回1的计数的十六进制。"""
    ones = bits.count("1")
    if ones == 0 or ones == total_examples:
        return "a"
    return format(ones, "x")


def _evaluate_binary(a: str, b: str, family: str) -> str:
    """计算二元位操作结果。"""
    if family in ("AND", "AND-NOT"):
        return "1" if a == "1" and b == "1" else "0"
    if family in ("OR", "OR-NOT"):
        return "1" if a == "1" or b == "1" else "0"
    if family in ("XOR", "XOR-NOT"):
        return "1" if a != b else "0"
    raise ValueError(f"不支持的操作族 {family}")


def _apply_family(
    a_bits: str, b_bits: str, family: str, invert_second: bool = False
) -> str:
    """对两列位向量应用指定操作族。"""
    b_eff = _invert(b_bits) if invert_second else b_bits
    out = []
    for x, y in zip(a_bits, b_eff):
        out.append(_evaluate_binary(x, y, family))
    return "".join(out)


def _find_match(
    candidates: List[RuleCandidate], fam: str, ep: Optional[int], es: Optional[int]
) -> Optional[RuleCandidate]:
    """在候选列表中查找匹配 (fam, ep, es) 的规则。"""
    for c in candidates:
        if c.family != fam:
            continue
        if c.primary == ep and (fam not in PAIR_FAMILIES or c.secondary == es):
            return c
    return None


def _exists_anywhere(
    all_matches: List[List[RuleCandidate]],
    fam: str, ep: Optional[int], es: Optional[int],
) -> bool:
    """检查操作数对 (ep, es) 是否在任何位位置存在。"""
    for bit_cands in all_matches:
        if _find_match(bit_cands, fam, ep, es) is not None:
            return True
    return False


def _fail_suffix(
    all_matches: List[List[RuleCandidate]],
    fam: str, ep: Optional[int], es: Optional[int],
) -> str:
    """如果操作数存在于其他位置返回'y'，否则返回'x'。"""
    if _exists_anywhere(all_matches, fam, ep, es):
        return "y"
    return "x"


def _find_all_left_runs(
    all_matches: List[List[RuleCandidate]],
) -> List[Tuple[List[RuleCandidate], Optional[str]]]:
    """从 bit 0 开始的所有 stride 一致链。"""
    if not all_matches or not all_matches[0]:
        return []
    runs: List[Tuple[List[RuleCandidate], Optional[str]]] = []
    for start_cand in all_matches[0]:
        fam = start_cand.family
        strides = [(1, 1)]
        for p_step, s_step in strides:
            chain = [start_cand]
            cur_p = start_cand.primary
            cur_s = start_cand.secondary
            failed_next: Optional[str] = None
            for b in range(1, len(all_matches)):
                ep = (cur_p + p_step) % N_BITS if cur_p is not None else None
                es = (cur_s + s_step) % N_BITS if cur_s is not None else None
                found = _find_match(all_matches[b], fam, ep, es)
                if found is None:
                    suffix = _fail_suffix(all_matches, fam, ep, es)
                    if ep is not None and es is not None:
                        failed_next = f"{ep}{es}{suffix}"
                    elif ep is not None:
                        failed_next = f"{ep}{suffix}"
                    break
                chain.append(found)
                cur_p, cur_s = ep, es
            runs.append((chain, failed_next))
    return runs


def _find_all_right_runs(
    all_matches: List[List[RuleCandidate]],
) -> List[Tuple[List[RuleCandidate], Optional[str]]]:
    """从最后一位开始向前的所有 stride 一致链。"""
    n = len(all_matches)
    if not all_matches or not all_matches[-1]:
        return []
    runs: List[Tuple[List[RuleCandidate], Optional[str]]] = []
    for end_cand in all_matches[-1]:
        fam = end_cand.family
        strides = [(1, 1)]
        for p_step, s_step in strides:
            chain = [end_cand]
            cur_p = end_cand.primary
            cur_s = end_cand.secondary
            failed_next: Optional[str] = None
            for k in range(1, n):
                b = n - 1 - k
                pp = (cur_p - p_step) % N_BITS if cur_p is not None else None
                ps = (cur_s - s_step) % N_BITS if cur_s is not None else None
                found = _find_match(all_matches[b], fam, pp, ps)
                if found is None:
                    suffix = _fail_suffix(all_matches, fam, pp, ps)
                    if pp is not None and ps is not None:
                        failed_next = f"{pp}{ps}{suffix}"
                    elif pp is not None:
                        failed_next = f"{pp}{suffix}"
                    break
                chain.insert(0, found)
                cur_p, cur_s = pp, ps
            runs.append((chain, failed_next))
    return runs


def _lr_from_matches(
    all_matches: List[List[RuleCandidate]],
) -> Tuple[List[str], str, List[str], str]:
    """计算 Left/Right 最长链。"""
    all_left_runs = _find_all_left_runs(all_matches)
    all_right_runs = _find_all_right_runs(all_matches)
    left_run = max(all_left_runs, key=lambda t: len(t[0])) if all_left_runs else ([], None)
    right_run = max(all_right_runs, key=lambda t: len(t[0])) if all_right_runs else ([], None)

    left_lines = (
        [_format_list(chain, failed=failed) for chain, failed in all_left_runs]
        if all_left_runs else ["none"]
    )
    left_best = _format_list(left_run[0], with_count=True)
    right_lines = (
        [_format_list(list(reversed(chain)), failed=failed) for chain, failed in all_right_runs]
        if all_right_runs else ["none"]
    )
    right_best = _format_list(list(reversed(right_run[0])), with_count=True)
    return left_lines, left_best, right_lines, right_best


def _format_list(
    cands: List[RuleCandidate], with_count: bool = False, failed: Optional[str] = None,
) -> str:
    if not cands:
        return "none"
    if with_count:
        parts = []
        for i, c in enumerate(cands):
            if i == 0:
                parts.append(c.expr)
            else:
                parts.append(_compact_rule(c))
        return " ".join(parts) + f": {len(cands)}"
    parts = [_compact_rule(c) for c in cands]
    if failed:
        parts.append(failed)
    return " ".join(parts)


def _compact_rule(c: RuleCandidate) -> str:
    """紧凑显示：只显示操作数索引。"""
    if c.primary is not None and c.secondary is not None:
        return f"{c.primary}{c.secondary}"
    if c.primary is not None:
        return str(c.primary)
    return c.family


def _evaluate_rule(bits: str, rule: RuleCandidate) -> str:
    """对输入位串应用规则，返回输出位。"""
    if rule.family == "DEFAULT":
        return "1"
    if rule.family == "0":
        return "0"
    if rule.family == "1":
        return "1"
    if rule.family == "I":
        assert rule.primary is not None
        return bits[rule.primary]
    if rule.family == "NOT":
        assert rule.primary is not None
        return _bit_not(bits[rule.primary])
    if rule.family in PAIR_FAMILIES:
        assert rule.primary is not None and rule.secondary is not None
        a = bits[rule.primary]
        b = bits[rule.secondary]
        if "-NOT" in rule.family:
            b = _bit_not(b)
        return _evaluate_binary(a, b, rule.family)
    raise ValueError(f"未知操作族 {rule.family}")


def _emit_apply(
    lines: List[str], question_bits: str, vector: List[RuleCandidate]
) -> None:
    """应用规则向量到查询输入，生成最终答案。"""
    lines.append(f"Applying to {question_bits}")
    lines.append("Input")
    for i, bit in enumerate(question_bits):
        lines.append(f"{i} {bit}")
    lines.append("Output")

    answer_bits: List[str] = []
    for i, rule in enumerate(vector):
        if rule.family == "DEFAULT":
            lines.append(f"{i} default 1 = 1")
            answer_bits.append("1")
            continue
        if rule.family in CONSTANT_FAMILIES:
            lines.append(f"{i} {rule.expr} = {rule.family}")
            answer_bits.append(rule.family)
            continue
        if rule.family == "I":
            assert rule.primary is not None
            val = question_bits[rule.primary]
            lines.append(f"{i} {rule.expr} = {val}")
            answer_bits.append(val)
            continue
        if rule.family == "NOT":
            assert rule.primary is not None
            val = question_bits[rule.primary]
            nval = _bit_not(val)
            lines.append(f"{i} {rule.expr} = NOT({val}) = {nval}")
            answer_bits.append(nval)
            continue

        assert rule.primary is not None and rule.secondary is not None
        a = question_bits[rule.primary]
        b = question_bits[rule.secondary]
        if rule.family in SYM_FAMILIES:
            result = _evaluate_rule(question_bits, rule)
            lines.append(f"{i} {rule.expr} = {rule.family}({a},{b}) = {result}")
            answer_bits.append(result)
            continue

        base = rule.family.split("-")[0]
        result = _evaluate_rule(question_bits, rule)
        lines.append(f"{i} {rule.expr} = {base}({a},NOT({b})) = {result}")
        answer_bits.append(result)

    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{''.join(answer_bits)}}}")


def _reasoning_bit_manipulation(
    inputs: List[str], outputs: List[str], question_bits: str
) -> Optional[str]:
    """
    核心推理算法（忠实移植原始方案）。
    
    步骤：
    1. 构建输出列向量和输入列向量
    2. 对每个操作族（Identity, NOT, Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT）
       计算匹配情况
    3. 寻找左右最长 stride 链
    4. 填充中间位
    5. 应用最终规则向量到查询输入
    """
    n_examples = len(outputs)
    
    # 构建列向量
    output_columns = [_column_bits(outputs, i) for i in range(N_BITS)]
    input_columns = [_column_bits(inputs, i) for i in range(N_BITS)]
    input_inverted = [_invert(col) for col in input_columns]

    all_records: Dict[str, List[Record]] = {name: [] for name in SECTION_ORDER}
    all_matches: Dict[str, List[List[RuleCandidate]]] = {
        name: [[] for _ in range(N_BITS)] for name in SECTION_ORDER
    }

    # 构建一元和常数的记录与匹配
    for out_idx, out_col in enumerate(output_columns):
        for i_col, in_col in enumerate(input_columns):
            if in_col == out_col:
                all_matches["Identity"][out_idx].append(
                    RuleCandidate("I", i_col, None, f"I{i_col}")
                )
            if input_inverted[i_col] == out_col:
                all_matches["NOT"][out_idx].append(
                    RuleCandidate("NOT", i_col, None, f"NOT{i_col}")
                )
        if out_col.count("1") == 0:
            all_matches["Constant"][out_idx].append(
                RuleCandidate("0", None, None, "C0")
            )
        if out_col.count("1") == n_examples:
            all_matches["Constant"][out_idx].append(
                RuleCandidate("1", None, None, "C1")
            )

    # 构建一元原始记录
    for label, col in zip([str(i) for i in range(N_BITS)], input_columns):
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["Identity"].append(
            Record(label=label, col=col, hash_=_column_hash(col, n_examples), matches=matches)
        )
    for label, col in zip([str(i) for i in range(N_BITS)], input_inverted):
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["NOT"].append(
            Record(label=label, col=col, hash_=_column_hash(col, n_examples), matches=matches)
        )
    for val in ("0", "1"):
        col = val * n_examples
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["Constant"].append(
            Record(label=val, col=col, hash_=_column_hash(col, n_examples), matches=matches)
        )

    # 构建对称二元操作的记录和匹配
    fam: RuleFamily
    for fam in ("XOR", "OR", "AND"):
        for circ_diff in range(1, N_BITS // 2 + 1):
            n_pairs = N_BITS // 2 if circ_diff == N_BITS // 2 else N_BITS
            for a in range(n_pairs):
                b = (a + circ_diff) % N_BITS
                lo, hi = min(a, b), max(a, b)
                col = _apply_family(input_columns[lo], input_columns[hi], fam)
                matches = tuple(
                    i for i, out_col in enumerate(output_columns) if col == out_col
                )
                all_records[fam].append(
                    Record(label=f"{a}{b} {b}{a}", col=col,
                           hash_=_column_hash(col, n_examples), matches=matches)
                )
                for out_idx in matches:
                    all_matches[fam][out_idx].append(
                        RuleCandidate(fam, a, b, f"{fam}{a}{b}")
                    )
                    all_matches[fam][out_idx].append(
                        RuleCandidate(fam, b, a, f"{fam}{b}{a}")
                    )

    # 构建非对称二元操作的记录和匹配
    for fam in ("AND-NOT", "XOR-NOT", "OR-NOT"):
        for diff in range(1, N_BITS):
            for a in range(N_BITS):
                b = (a + diff) % N_BITS
                col = _apply_family(
                    input_columns[a], input_columns[b], fam, invert_second=True
                )
                matches = tuple(
                    i for i, out_col in enumerate(output_columns) if col == out_col
                )
                all_records[fam].append(
                    Record(label=f"{a}{b}", col=col,
                           hash_=_column_hash(col, n_examples), matches=matches)
                )
                for out_idx in matches:
                    all_matches[fam][out_idx].append(
                        RuleCandidate(fam, a, b, f"{fam}{a}{b}")
                    )

    # 排序记录
    for name in ("Identity", "NOT", "Constant"):
        all_records[name].sort(key=lambda r: r.label)

    lines: List[str] = []

    # 1) 头部
    lines.append("We need to deduce the transformation by matching the example outputs.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # 2) 输出示例
    for i, out in enumerate(outputs):
        lines.append(f"Output {i}: {out}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {out[bit]}")
        lines.append("")

    # 3) 输出位列（带哈希）
    lines.append("Output bit columns (with bitsum as hash)")
    for bit in range(N_BITS):
        lines.append(
            f"{bit} {output_columns[bit]} {_column_hash(output_columns[bit], n_examples)}"
        )

    # 4) 输入示例
    lines.append("")
    for i, inp in enumerate(inputs):
        lines.append(f"Input {i}: {inp}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {inp[bit]}")
        lines.append("")

    # 5) 各操作段：原始数据 + 匹配 + LRM
    lines.append("When matching output")
    lines.append("x: not in operator")
    lines.append("y: wrong position")
    lines.append("")
    section_lefts: list[tuple[str, str]] = []
    section_rights: list[tuple[str, str]] = []

    def _add_section(name: str) -> None:
        records = all_records[name]
        per_bit = all_matches[name]
        lines.append(name)
        prev_diff = None
        for rec in records:
            if (len(rec.label) >= 2 and rec.label[0].isdigit() and rec.label[1].isdigit()):
                diff = (int(rec.label[1]) - int(rec.label[0])) % N_BITS
                if prev_diff is not None and diff != prev_diff:
                    lines.append("")
                prev_diff = diff
            line = f"{rec.label} {rec.col} {rec.hash_}"
            if rec.matches:
                line += " match " + " ".join(str(i) for i in rec.matches)
            lines.append(line)
        lines.append("")
        lines.append("Matching output")
        for i in range(N_BITS):
            cands = per_bit[i]
            if cands:
                def _compact(c: RuleCandidate) -> str:
                    if c.primary is not None and c.secondary is not None:
                        return f"{c.primary}{c.secondary}"
                    if c.primary is not None:
                        return str(c.primary)
                    return c.expr
                lines.append(f"{i} " + " ".join(_compact(c) for c in cands))
            else:
                lines.append(f"{i} absent")
        lines.append("")
        left_lines, left_best, right_lines, right_best = _lr_from_matches(per_bit)
        section_lefts.append((name, left_best))
        section_rights.append((name, right_best))
        lines.append("Left")
        for ll in left_lines:
            lines.append(ll)
        lines.append(f"Best: {left_best}")
        lines.append("")
        lines.append("Right")
        for rl in right_lines:
            lines.append(rl)
        lines.append(f"Best: {right_best}")
        lines.append("")

    for name in all_records:
        _add_section(name)

    # 7) 选择规则块
    lines.append("Selecting")
    lines.append("")

    def _parse_count(val: str) -> int:
        if val == "none":
            return 0
        try:
            return int(val.rsplit(": ", 1)[-1])
        except ValueError:
            return 0

    def _pick_winner(entries: list[tuple[str, str]]) -> tuple[Optional[str], str, int]:
        best_name: Optional[str] = None
        best_text = "none"
        best_count = 0
        for name, val in entries:
            count = _parse_count(val)
            if count > best_count:
                best_count = count
                best_name = name
                best_text = val
        return best_name, best_text, best_count

    left_winner_name, left_winner_text, left_winner_count = _pick_winner(section_lefts)
    right_winner_name, right_winner_text, right_winner_count = _pick_winner(section_rights)

    def _get_section_run(winner_name: Optional[str], direction: str) -> List[RuleCandidate]:
        if winner_name is None:
            return []
        per_bit = all_matches[winner_name]
        if direction == "left":
            runs = _find_all_left_runs(per_bit)
        else:
            runs = _find_all_right_runs(per_bit)
        if not runs:
            return []
        best_chain, _ = max(runs, key=lambda t: len(t[0]))
        return best_chain

    left_run = _get_section_run(left_winner_name, "left")
    right_run = _get_section_run(right_winner_name, "right")

    lines.append("Lefts")
    for name, lb in section_lefts:
        lines.append(f"{name} {lb}")
    lines.append("")
    lines.append("Rights")
    for name, rb in section_rights:
        lines.append(f"{name} {rb}")
    lines.append("")
    lines.append(f"Left longest: {left_winner_count}")
    lines.append(f"Right longest: {right_winner_count}")
    lines.append("")

    def _matching_line(label: str, winner_name: Optional[str], entries: list[tuple[str, str]]) -> str:
        parts = []
        for name, _val in entries:
            parts.append(f"{name} {'yes' if name == winner_name else 'no'}")
        return f"{label} winner: {', '.join(parts)}"

    if right_winner_count > left_winner_count:
        lines.append(_matching_line("Right", right_winner_name, section_rights))
        lines.append(_matching_line("Left", left_winner_name, section_lefts))
        lines.append("")
        lines.append(f"Best right: {right_winner_text}")
        lines.append(f"Best left: {left_winner_text}")
    else:
        lines.append(_matching_line("Left", left_winner_name, section_lefts))
        lines.append(_matching_line("Right", right_winner_name, section_rights))
        lines.append("")
        lines.append(f"Best left: {left_winner_text}")
        lines.append(f"Best right: {right_winner_text}")
    lines.append("")

    # 截断处理
    left_len_final = left_winner_count
    right_len_final = right_winner_count
    if left_len_final + right_len_final > N_BITS:
        if right_len_final > left_len_final:
            left_len_final = N_BITS - right_len_final
            left_run = left_run[:left_len_final]
        else:
            right_len_final = N_BITS - left_len_final
            right_run = right_run[-right_len_final:] if right_len_final else []

    left_was_truncated = left_len_final < left_winner_count
    right_was_truncated = right_len_final < right_winner_count
    trunc_left = f"Truncated left: {_format_list(left_run, with_count=True)}"
    if left_was_truncated:
        trunc_left += " truncated"
    trunc_right = f"Truncated right: {_format_list(list(reversed(right_run)), with_count=True)}"
    if right_was_truncated:
        trunc_right += " truncated"
    if right_winner_count > left_winner_count:
        lines.append(trunc_right)
        lines.append(trunc_left)
    else:
        lines.append(trunc_left)
        lines.append(trunc_right)
    lines.append("")

    right_start_final = N_BITS - right_len_final
    lines.append("Tentative from right")
    for i in range(N_BITS - 1, -1, -1):
        if i >= right_start_final and right_run:
            lines.append(f"{i} {right_run[i - right_start_final].expr}")
        else:
            lines.append(f"{i} pending")
    lines.append("")
    lines.append("Tentative")
    for i in range(N_BITS):
        if i < left_len_final:
            lines.append(f"{i} {left_run[i].expr}")
        elif i >= right_start_final and right_run:
            lines.append(f"{i} {right_run[i - right_start_final].expr}")
        else:
            lines.append(f"{i} pending")
    lines.append("")

    # 外推填充
    def _extrap_from(
        run: List[RuleCandidate], bit: int, run_start_bit: int, side: str = "left",
    ) -> Optional[str]:
        if not run:
            return None
        r = run[0]
        p = r.primary
        s = r.secondary
        if p is not None:
            p_off = (p - run_start_bit) % N_BITS
            ep = (p_off + bit) % N_BITS
        else:
            ep = None
        if s is not None:
            s_off = (s - run_start_bit) % N_BITS
            es = (s_off + bit) % N_BITS
        else:
            es = None
        if ep is not None and es is not None:
            return f"?{ep}{es}"
        if ep is not None:
            if side == "left":
                return f"?{ep}?"
            else:
                return f"??{ep}"
        return None

    left_fam = left_run[0].family if left_run else None
    right_fam = right_run[0].family if right_run else None
    left_is_const = left_fam in CONSTANT_FAMILIES if left_fam else False
    right_is_const = right_fam in CONSTANT_FAMILIES if right_fam else False
    left_is_binary = left_fam in PAIR_FAMILIES if left_fam else False
    right_is_binary = right_fam in PAIR_FAMILIES if right_fam else False
    left_is_unary = left_fam in UNARY_FAMILIES if left_fam else False
    right_is_unary = right_fam in UNARY_FAMILIES if right_fam else False

    # 构建 preferred 向量
    if right_winner_count > left_winner_count:
        preferred: list[str] = []
        for i in range(N_BITS):
            if i >= right_start_final and right_run:
                preferred.append(right_run[i - right_start_final].expr)
            elif i < left_len_final:
                preferred.append(left_run[i].expr)
            elif right_is_binary or right_is_unary:
                preferred.append(
                    _extrap_from(right_run, i, right_start_final, "right") or "pending"
                )
            else:
                preferred.append("pending")

        lines.append("Preferred from right")
        for i in range(N_BITS - 1, -1, -1):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")

        for i in range(N_BITS):
            if preferred[i] == "pending":
                if left_is_binary or left_is_unary:
                    preferred[i] = _extrap_from(left_run, i, 0, "left") or "?"
                else:
                    preferred[i] = "?"
            elif "?" in preferred[i][1:] and left_is_unary:
                el = _extrap_from(left_run, i, 0, "left")
                if el:
                    merged = list(preferred[i])
                    el_chars = list(el)
                    for j in range(1, min(len(merged), len(el_chars))):
                        if merged[j] == "?" and el_chars[j] != "?":
                            merged[j] = el_chars[j]
                    preferred[i] = "".join(merged)

        lines.append("Preferred from left")
        for i in range(N_BITS):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")
    else:
        preferred = []
        for i in range(N_BITS):
            if i < left_len_final:
                preferred.append(left_run[i].expr)
            elif i >= right_start_final and right_run:
                preferred.append(right_run[i - right_start_final].expr)
            elif left_is_binary or left_is_unary:
                preferred.append(
                    _extrap_from(left_run, i, 0, "left") or "pending"
                )
            else:
                preferred.append("pending")

        lines.append("Preferred from left")
        for i in range(N_BITS):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")

        for i in range(N_BITS):
            if preferred[i] == "pending":
                if right_is_binary or right_is_unary:
                    preferred[i] = _extrap_from(right_run, i, right_start_final, "right") or "?"
                else:
                    preferred[i] = "?"
            elif "?" in preferred[i][1:] and right_is_unary:
                er = _extrap_from(right_run, i, right_start_final, "right")
                if er:
                    merged = list(preferred[i])
                    er_chars = list(er)
                    for j in range(1, min(len(merged), len(er_chars))):
                        if merged[j] == "?" and er_chars[j] != "?":
                            merged[j] = er_chars[j]
                    preferred[i] = "".join(merged)

        lines.append("Preferred from right")
        for i in range(N_BITS - 1, -1, -1):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")

    lines.append("Preferred")
    for i, pref in enumerate(preferred):
        if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?":
            lines.append(f"{i} {pref} ?{pref[2]}{pref[1]}")
        else:
            lines.append(f"{i} {pref}")
    lines.append("")

    # 构建最终规则向量
    default_cand = RuleCandidate(DEFAULT_FAMILY, None, None, "default 1")
    best: List[RuleCandidate] = [default_cand] * N_BITS

    for i, rc in enumerate(left_run):
        best[i] = rc
    for i, rc in enumerate(right_run):
        best[right_start_final + i] = rc

    # 填充中间 pending 位
    lines.append("Matching")
    pending_indices: list[int] = []
    per_bit_cat: dict[str, dict[int, list[RuleCandidate]]] = {
        name: {} for name in SECTION_ORDER
    }

    for i in range(N_BITS):
        pref = preferred[i]
        if not pref.startswith("?") or pref == "?":
            lines.append(f"{i} {best[i].expr}")
            continue

        pending_indices.append(i)
        digits_str = pref[1:]
        pref_digits = [int(d) for d in digits_str if d != "?"]

        checks: list[str] = []
        for section_name in SECTION_ORDER:
            cands = all_matches[section_name][i]
            if section_name in ("Identity", "NOT"):
                found = [c for c in cands if c.primary in pref_digits]
                if found:
                    checks.append(section_name + " " + " ".join(c.expr for c in found))
                    per_bit_cat[section_name][i] = found
                else:
                    checks.append(f"{section_name} absent")
            elif section_name == "Constant":
                if cands:
                    checks.append("Constant " + " ".join(c.expr for c in cands))
                    per_bit_cat["Constant"][i] = list(cands)
                else:
                    checks.append("Constant absent")
            else:
                found_c: Optional[RuleCandidate] = None
                orderings = []
                want_p = int(pref[1]) if len(pref) > 1 and pref[1] != "?" else None
                want_s = int(pref[2]) if len(pref) > 2 and pref[2] != "?" else None
                orderings.append((want_p, want_s))
                if want_p is not None and want_s is not None and want_p != want_s:
                    orderings.append((want_s, want_p))
                for wp, ws in orderings:
                    for c in cands:
                        if (wp is None or c.primary == wp) and (ws is None or c.secondary == ws):
                            found_c = c
                            break
                    if found_c is not None:
                        break
                if found_c is not None:
                    checks.append(found_c.expr)
                    per_bit_cat[section_name][i] = [found_c]
                else:
                    checks.append(f"{section_name} absent")
        if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?":
            pref_display = f"{pref} ?{pref[2]}{pref[1]}"
        else:
            pref_display = pref
        lines.append(f"{i} {pref_display} - {', '.join(checks)}")
    lines.append("")

    # 完美匹配：第一个覆盖所有 pending 位的段获胜
    lines.append("Perfect match")
    chosen_cat: Optional[str] = None
    for cat in SECTION_ORDER:
        is_perfect = (
            chosen_cat is None
            and bool(pending_indices)
            and all(i in per_bit_cat[cat] for i in pending_indices)
        )
        lines.append(f"{cat} {'yes' if is_perfect else 'no'}")
        if is_perfect:
            chosen_cat = cat
    lines.append("")

    # 使用完美匹配的段填充 pending 位
    pending_set = set(pending_indices)
    lines.append("Matched")
    for i in range(N_BITS):
        if i in pending_set:
            if chosen_cat and i in per_bit_cat[chosen_cat]:
                best[i] = per_bit_cat[chosen_cat][i][0]
                lines.append(f"{i} {best[i].expr}")
            else:
                all_cands: list[RuleCandidate] = []
                for name in SECTION_ORDER:
                    if i in per_bit_cat[name]:
                        all_cands.extend(per_bit_cat[name][i])
                if all_cands:
                    lines.append(f"{i} " + " ".join(c.expr for c in all_cands))
                    best[i] = all_cands[0]
                else:
                    lines.append(f"{i} none")
                    best[i] = default_cand
        else:
            lines.append(f"{i} {best[i].expr}")
    lines.append("")

    # 检查是否有非默认规则
    if all(r.is_default for r in best):
        return None

    lines.append("Selected")
    for i, rule in enumerate(best):
        lines.append(f"{i} {rule.expr}")

    # 应用到查询
    lines.append("")
    _emit_apply(lines, question_bits, best)

    return "\n".join(lines)


class BitManipulationReasoner:
    """位操作题推理器（忠实移植原始算法）"""
    
    def solve(self, problem: Problem) -> ReasoningResult:
        """
        求解位操作题。
        
        使用完整操作族（XOR/OR/AND/AND-NOT/XOR-NOT/OR-NOT/I/NOT/0/1）
        通过列匹配和 stride 链搜索找到变换规则。
        """
        try:
            examples, query_bits = parse_bit_manipulation_examples(problem.prompt)
            
            if not examples or not query_bits:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="bit_manipulation",
                    success=False,
                    error_message="无法解析示例或查询值"
                )
            
            # 标准化输入输出
            inputs = [_normalize_bits(inp) for inp, _ in examples]
            outputs = [_normalize_bits(out) for _, out in examples]
            question = _normalize_bits(query_bits)
            
            if any(not bits for bits in inputs + outputs) or not question:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="bit_manipulation",
                    success=False,
                    error_message="位串格式不正确"
                )
            
            # 调用核心推理算法
            reasoning_text = _reasoning_bit_manipulation(inputs, outputs, question)
            
            if reasoning_text is None:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="bit_manipulation",
                    success=False,
                    error_message="无法找到匹配规则"
                )
            
            # 从 \boxed{} 中提取答案
            from reasoners._utils import extract_last_boxed
            # v15 兼容：支持嵌套大括号 / 含 } 答案
            predicted = extract_last_boxed(reasoning_text)
            
            return ReasoningResult(
                problem_id=problem.id,
                category="bit_manipulation",
                success=True,
                predicted_answer=predicted,
                reasoning_text=reasoning_text,
            )
            
        except Exception as e:
            return ReasoningResult(
                problem_id=problem.id,
                category="bit_manipulation",
                success=False,
                error_message=f"推理异常: {str(e)}"
            )
