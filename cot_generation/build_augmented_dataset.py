"""
扩充训练集 - 替换式 N-1 倍扩充
=================================
对每条原样本:
  原: examples = [e_1, ..., e_N], query = q, answer = a
  扩充第 k 条 (k = 1..N):
    examples_new = [..., e_{k-1}, (q, a), e_{k+1}, ...]   # 用 (q,a) 替换 e_k
    query_new    = e_k.input
    answer_new   = e_k.output

每个原样本产生 N - 1 条扩充(k 不取扩充自身),加 1 条原样本 = N 条;
但因为 (q,a) 替换原 e_k,新 examples 数 = N(数量不变,语义重排)。

输入:  data/train_full_9500_7cats.csv (9500 行, 列: id, prompt, answer, fine_category, type)
输出:  data/train_augmented.csv (列: id, prompt, answer, fine_category, type, is_augmented, source_id)

用法:
    cd code/new_cot && python build_augmented_dataset.py
"""
import os
import sys
import re
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(PROJECT_ROOT, "data/train_full_9500_7cats.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/train_augmented.csv")


# ========================================================================
# 各类 prompt 模板的解析器/格式化器
# ========================================================================
# 每个 type 提供:
#   parse_example(line) -> (input_str, output_str) | None    # 不是 example 行返回 None
#   format_example(input_str, output_str) -> str             # 反向构造 example 行
#   parse_now(line) -> query_str | None                      # 不是 Now 行返回 None
#   format_now(query_str) -> str                             # 反向构造 Now 行

# ---- bit_manipulation ----
_BIT_NOW_PREFIX = "Now, determine the output for: "

def _bit_parse_example(line: str):
    if " -> " not in line:
        return None
    parts = line.split(" -> ", 1)
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    if not (re.fullmatch(r"[01]{8}", a) and re.fullmatch(r"[01]{8}", b)):
        return None
    return (a, b)

def _bit_format_example(inp, out):
    return f"{inp} -> {out}"

def _bit_parse_now(line: str):
    if line.startswith(_BIT_NOW_PREFIX):
        return line[len(_BIT_NOW_PREFIX):].strip()
    return None

def _bit_format_now(q):
    return f"{_BIT_NOW_PREFIX}{q}"


# ---- cipher ----
_CIPHER_NOW_PREFIX = "Now, decrypt the following text: "

def _cipher_parse_example(line: str):
    if " -> " not in line or line.startswith("Now,"):
        return None
    parts = line.split(" -> ", 1)
    if len(parts) != 2:
        return None
    return (parts[0].strip(), parts[1].strip())

def _cipher_format_example(inp, out):
    return f"{inp} -> {out}"

def _cipher_parse_now(line: str):
    if line.startswith(_CIPHER_NOW_PREFIX):
        return line[len(_CIPHER_NOW_PREFIX):].strip()
    return None

def _cipher_format_now(q):
    return f"{_CIPHER_NOW_PREFIX}{q}"


# ---- numeral ----
_NUM_NOW_PREFIX = "Now, write the number "
_NUM_NOW_SUFFIX = " in the Wonderland numeral system."

def _num_parse_example(line: str):
    if " -> " not in line or line.startswith("Now,"):
        return None
    parts = line.split(" -> ", 1)
    if len(parts) != 2:
        return None
    return (parts[0].strip(), parts[1].strip())

def _num_format_example(inp, out):
    return f"{inp} -> {out}"

def _num_parse_now(line: str):
    if line.startswith(_NUM_NOW_PREFIX) and line.endswith(_NUM_NOW_SUFFIX):
        mid = line[len(_NUM_NOW_PREFIX):-len(_NUM_NOW_SUFFIX)]
        return mid.strip()
    return None

def _num_format_now(q):
    return f"{_NUM_NOW_PREFIX}{q}{_NUM_NOW_SUFFIX}"


# ---- unit_conversion ----
_UNIT_NOW_PREFIX = "Now, convert the following measurement: "
_UNIT_NOW_SUFFIX = " m"

def _unit_parse_example(line: str):
    if " m becomes " not in line:
        return None
    parts = line.split(" m becomes ", 1)
    if len(parts) != 2:
        return None
    return (parts[0].strip(), parts[1].strip())

def _unit_format_example(inp, out):
    return f"{inp} m becomes {out}"

def _unit_parse_now(line: str):
    if line.startswith(_UNIT_NOW_PREFIX) and line.endswith(_UNIT_NOW_SUFFIX):
        mid = line[len(_UNIT_NOW_PREFIX):-len(_UNIT_NOW_SUFFIX)]
        return mid.strip()
    return None

def _unit_format_now(q):
    return f"{_UNIT_NOW_PREFIX}{q}{_UNIT_NOW_SUFFIX}"


# ---- gravity ----
_GRAV_EX_RE = re.compile(r"^For t = (.+?)s, distance = (.+?) m$")
_GRAV_NOW_RE = re.compile(r"^Now, determine the falling distance for t = (.+?)s given d = 0\.5\*g\*t\^2\.$")

def _grav_parse_example(line: str):
    m = _GRAV_EX_RE.match(line)
    if not m:
        return None
    return (m.group(1).strip(), m.group(2).strip())

def _grav_format_example(inp, out):
    return f"For t = {inp}s, distance = {out} m"

def _grav_parse_now(line: str):
    m = _GRAV_NOW_RE.match(line)
    if not m:
        return None
    return m.group(1).strip()

def _grav_format_now(q):
    return f"Now, determine the falling distance for t = {q}s given d = 0.5*g*t^2."


# ---- cryptarithm / equation_numeric (相同模板) ----
_EQ_NOW_PREFIX = "Now, determine the result for: "

def _eq_parse_example(line: str):
    if " = " not in line or line.startswith("Now,"):
        return None
    parts = line.split(" = ", 1)
    if len(parts) != 2:
        return None
    return (parts[0].strip(), parts[1].strip())

def _eq_format_example(inp, out):
    return f"{inp} = {out}"

def _eq_parse_now(line: str):
    if line.startswith(_EQ_NOW_PREFIX):
        return line[len(_EQ_NOW_PREFIX):].strip()
    return None

def _eq_format_now(q):
    return f"{_EQ_NOW_PREFIX}{q}"


PARSERS = {
    "bit_manipulation": (_bit_parse_example, _bit_format_example, _bit_parse_now, _bit_format_now),
    "cipher": (_cipher_parse_example, _cipher_format_example, _cipher_parse_now, _cipher_format_now),
    "numeral": (_num_parse_example, _num_format_example, _num_parse_now, _num_format_now),
    "unit_conversion": (_unit_parse_example, _unit_format_example, _unit_parse_now, _unit_format_now),
    "gravity": (_grav_parse_example, _grav_format_example, _grav_parse_now, _grav_format_now),
    "cryptarithm": (_eq_parse_example, _eq_format_example, _eq_parse_now, _eq_format_now),
    "equation_numeric": (_eq_parse_example, _eq_format_example, _eq_parse_now, _eq_format_now),
}


def parse_prompt(prompt: str, type_key: str):
    """解析 prompt,返回 (lines, ex_indices, now_idx, examples, query)。

    Returns:
        lines: List[str]                 - prompt 按 \n split 的所有行(保留空行)
        ex_indices: List[int]            - examples 在 lines 中的下标列表
        now_idx: int                     - Now 行在 lines 中的下标
        examples: List[(input, output)]  - 解析出的 example 元组
        query: str                       - 解析出的 query 字符串
    """
    pe, _fe, pn, _fn = PARSERS[type_key]
    lines = prompt.split("\n")
    ex_indices = []
    examples = []
    now_idx = None
    query = None
    for i, ln in enumerate(lines):
        # 优先识别 Now 行(它可能也含 ' -> ' 之类,但前缀更明确)
        nq = pn(ln)
        if nq is not None and now_idx is None:
            now_idx = i
            query = nq
            continue
        ex = pe(ln)
        if ex is not None:
            ex_indices.append(i)
            examples.append(ex)
    return lines, ex_indices, now_idx, examples, query


def build_augmented_prompt(lines, ex_indices, now_idx, k, q, a, examples, type_key):
    """构造扩充第 k 条样本的 prompt。

    用 (q, a) 替换第 k 个 example 的行;Now 行的 query 改为 examples[k].input。
    """
    _pe, fe, _pn, fn = PARSERS[type_key]
    new_lines = list(lines)
    # 替换第 k 个 example
    new_lines[ex_indices[k]] = fe(q, a)
    # 替换 Now 行
    new_query = examples[k][0]
    new_lines[now_idx] = fn(new_query)
    return "\n".join(new_lines), new_query, examples[k][1]


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"[load] {INPUT_CSV} -> {len(df)} rows")
    if not {"id", "prompt", "answer", "fine_category", "type"}.issubset(df.columns):
        raise ValueError(f"Input columns mismatch: {list(df.columns)}")

    # 解析阶段:对每条样本提取 examples 列表 + query
    out_rows = []
    parse_fail = []
    type_stats = {}

    for _, r in df.iterrows():
        rid = r["id"]
        prompt = r["prompt"]
        gt = str(r["answer"])
        type_key = r["type"]
        fine_cat = r["fine_category"]

        if type_key not in PARSERS:
            raise ValueError(f"unknown type {type_key} for id {rid}")

        lines, ex_indices, now_idx, examples, query = parse_prompt(prompt, type_key)

        if now_idx is None or len(examples) < 2 or query is None:
            parse_fail.append({
                "id": rid, "type": type_key, "fine_category": fine_cat,
                "n_examples": len(examples), "now_idx": now_idx, "query": query,
            })
            # 兜底:仍写出原样本(不扩充)
            out_rows.append({
                "id": rid, "prompt": prompt, "answer": gt,
                "fine_category": fine_cat, "type": type_key,
                "is_augmented": 0, "source_id": rid,
            })
            continue

        # 写原样本
        out_rows.append({
            "id": rid, "prompt": prompt, "answer": gt,
            "fine_category": fine_cat, "type": type_key,
            "is_augmented": 0, "source_id": rid,
        })

        # 扩充 N 条(k = 0..N-1):用 (q, gt) 替换第 k 个 example,新 query = examples[k].input
        N = len(examples)
        for k in range(N):
            new_prompt, new_query, new_answer = build_augmented_prompt(
                lines, ex_indices, now_idx, k, query, gt, examples, type_key
            )
            out_rows.append({
                "id": f"{rid}_aug_{k}",
                "prompt": new_prompt,
                "answer": new_answer,
                "fine_category": fine_cat,
                "type": type_key,
                "is_augmented": 1,
                "source_id": rid,
            })

        ts = type_stats.setdefault(type_key, {"n_orig": 0, "n_aug": 0, "n_examples_sum": 0})
        ts["n_orig"] += 1
        ts["n_aug"] += N
        ts["n_examples_sum"] += N

    # 写出
    df_out = pd.DataFrame(
        out_rows, columns=["id", "prompt", "answer", "fine_category", "type", "is_augmented", "source_id"]
    )
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print()
    print("=" * 70)
    print("扩充训练集生成完成")
    print("=" * 70)
    print(f"[write] {OUTPUT_CSV}")
    print(f"        rows={len(df_out)}, "
          f"size={os.path.getsize(OUTPUT_CSV) / 1024 / 1024:.2f} MB")
    print()
    print(f"解析失败样本: {len(parse_fail)}")
    if parse_fail:
        print("(前 5 条):")
        for f in parse_fail[:5]:
            print(f"  {f}")
    print()
    print("各 type 扩充统计(原样本数 / 扩充数 / 平均 examples 数):")
    for k, v in sorted(type_stats.items()):
        avg = v["n_examples_sum"] / max(v["n_orig"], 1)
        print(f"  {k:20s} orig={v['n_orig']:>5d}  aug={v['n_aug']:>6d}  avg_examples={avg:.2f}")
    print()
    print("is_augmented 分布:")
    print(df_out["is_augmented"].value_counts())
    print()
    print("type 分布(含原样本+扩充):")
    print(df_out["type"].value_counts())


if __name__ == "__main__":
    main()
