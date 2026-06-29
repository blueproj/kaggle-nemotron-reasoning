"""
密码/替换密码推理器 - 确定性求解字符替换加密题
================================================

移植自原始方案的核心算法：
- 使用 Wonderland 单词列表做词典约束
- 通过 word_pattern 匹配候选词
- 回溯搜索未知映射字符
- 维护双射映射的一致性

确定性保证：
- 单表替换密码一旦映射关系确定就是唯一的
- Wonderland 词典约束大幅缩小搜索空间
"""

import sys
import os
from typing import List, Tuple, Dict, Optional, Set
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from store_types import Problem, ReasoningResult, parse_cipher_examples

# Wonderland 单词列表路径
_WONDERLAND_PATH = Path(__file__).parent / "wonderland.txt"


@lru_cache(maxsize=1)
def _load_wonderland() -> List[str]:
    """加载 Wonderland 单词列表（已排序）。"""
    with _WONDERLAND_PATH.open() as f:
        words = [line.strip() for line in f if line.strip()]
    return sorted(words)


def _word_pattern(word: str) -> Tuple[int, ...]:
    """
    计算单词的字符模式（用于筛选候选词）。
    相同字符获得相同编号，不同字符获得不同编号。
    例如 "hello" -> (0, 1, 2, 2, 3)
    """
    seen: Dict[str, int] = {}
    pattern: List[int] = []
    for char in word:
        if char not in seen:
            seen[char] = len(seen)
        pattern.append(seen[char])
    return tuple(pattern)


def _candidate_words_for_partial(
    partial: str,
    cipher_to_plain: Dict[str, str],
    plain_to_cipher: Dict[str, str],
    cipher_word: str,
) -> List[str]:
    """
    寻找与部分解密结果匹配的 Wonderland 候选词。
    
    候选词必须：
    1. 长度匹配
    2. 字符模式匹配（相同密文字符必须映射到相同明文字符）
    3. 已知映射位置的字符必须一致
    4. 与现有 cipher_to_plain 映射兼容
    """
    candidates: List[str] = []
    target_len = len(partial)
    target_pattern = _word_pattern(cipher_word)

    for word in _load_wonderland():
        if len(word) != target_len:
            continue
        if _word_pattern(word) != target_pattern:
            continue
        # 检查已知位置是否匹配
        match = True
        for i, ch in enumerate(partial):
            if ch != "?" and ch != word[i]:
                match = False
                break
        if not match:
            continue
        # 检查前向一致性（cipher->plain）
        consistent = True
        for cc, wc in zip(cipher_word, word):
            if cc in cipher_to_plain and cipher_to_plain[cc] != wc:
                consistent = False
                break
        if consistent:
            candidates.append(word)

    candidates.sort()
    return candidates


class CipherReasoner:
    """密码替换题推理器（忠实移植原始算法）"""
    
    def solve(self, problem: Problem) -> ReasoningResult:
        """
        求解密码替换题。
        
        算法流程（与原方案一致）：
        1. 从示例逐字符建立 cipher->plain 映射
        2. 对查询密文中已知映射的字符直接解密
        3. 对未知字符，使用 Wonderland 词典约束 + 回溯搜索
        4. 选择第一个在 Wonderland 列表中且与现有映射兼容的候选词
        """
        try:
            examples, query_text = parse_cipher_examples(problem.prompt)
            
            if not examples or not query_text:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="cipher",
                    success=False,
                    error_message="无法解析示例或查询文本"
                )
            
            # 调用核心推理算法
            reasoning_text = self._reasoning_cipher(examples, query_text)
            
            if reasoning_text is None:
                return ReasoningResult(
                    problem_id=problem.id,
                    category="cipher",
                    success=False,
                    error_message="推理过程失败（无法解密所有字符）"
                )
            
            # 从 \boxed{} 中提取答案
            import re
            from reasoners._utils import extract_last_boxed
            # v15 兼容：支持嵌套大括号 / 含 } 答案
            predicted = extract_last_boxed(reasoning_text)
            
            return ReasoningResult(
                problem_id=problem.id,
                category="cipher",
                success=True,
                predicted_answer=predicted,
                reasoning_text=reasoning_text,
            )
            
        except Exception as e:
            return ReasoningResult(
                problem_id=problem.id,
                category="cipher",
                success=False,
                error_message=f"推理异常: {str(e)}"
            )
    
    def _reasoning_cipher(
        self, examples: List[Tuple[str, str]], query_text: str
    ) -> Optional[str]:
        """
        核心推理算法（忠实移植原始方案）。
        
        1. 从示例按单词对齐建立映射
        2. 对查询文本逐单词解密
        3. 未知字符通过 Wonderland 词典约束求解
        """
        dash = "–"
        lines: List[str] = []
        lines.append(
            "We need to find the encryption mapping from the examples. "
            "It looks like a substitution cipher."
        )
        lines.append("I will put my final answer inside \\boxed{}.")
        
        # 列出所有输入单词
        lines.append("")
        lines.append("Listing the input words:")
        for enc, dec in examples:
            cipher_words = enc.split()
            lines.append("")
            lines.append(f"【{enc}】")
            for j, w in enumerate(cipher_words):
                lines.append(f"{'' if j == 0 else ' '}{w}")
        question_words_list = query_text.split()
        lines.append("")
        lines.append(f"【 {query_text}】")
        for w in question_words_list:
            lines.append(f" {w}")
        
        # 逐字符拆分
        lines.append("")
        lines.append("Breaking down into characters:")
        for enc, dec in examples:
            lines.append("")
            lines.append(f"【{enc}】")
            for w in enc.split():
                lines.append(f"{dash.join(w)}")
        lines.append("")
        lines.append(f"【 {query_text}】")
        for w in question_words_list:
            lines.append(f"{dash.join(w)}")
        
        wonderland_words = _load_wonderland()
        
        # 从示例建立映射
        cipher_to_plain: Dict[str, str] = {}
        for enc, dec in examples:
            cipher_words = enc.split()
            plain_words = dec.split()
            if len(cipher_words) != len(plain_words):
                continue
            
            lines.append("")
            lines.append("")
            plain_quoted = " ".join(f"【{w}】" for w in plain_words)
            lines.append(f"【{enc}】 -> 【{dec}】 / {plain_quoted}:")
            lines.append("")
            
            for wi, (cw, pw) in enumerate(zip(cipher_words, plain_words)):
                if len(cw) != len(pw):
                    continue
                word_mappings: List[str] = []
                for cc, pc in zip(cw, pw):
                    if cc not in cipher_to_plain:
                        cipher_to_plain[cc] = pc
                    word_mappings.append(f"{cc}->{pc}")
                cw_display = f" {cw}" if wi > 0 else cw
                cipher_dashed = dash.join(cw)
                plain_dashed = dash.join(pw)
                nl_mappings = "\n".join(word_mappings)
                if wi > 0:
                    lines.append("")
                lines.append(
                    f"【{cw_display}】->【{pw}】\n{cipher_dashed}->{plain_dashed}\n{nl_mappings}"
                )
        
        # 展示当前映射状态
        lines.append("")
        all_mappings = "\n".join(
            f"{c}->{cipher_to_plain.get(c, '?')}" for c in "abcdefghijklmnopqrstuvwxyz"
        )
        lines.append(f"Mapping so far\n{all_mappings}")
        plain_to_cipher_all = {v: k for k, v in cipher_to_plain.items()}
        inverse_mappings = "\n".join(
            f"{c}->{plain_to_cipher_all.get(c, '?')}" for c in "abcdefghijklmnopqrstuvwxyz"
        )
        lines.append(f"Inverse mapping\n{inverse_mappings}")
        unknown_mapping_chars = [
            c for c in "abcdefghijklmnopqrstuvwxyz" if c not in cipher_to_plain
        ]
        mapped_targets = set(cipher_to_plain.values())
        unmapped_targets = sorted(
            c for c in "abcdefghijklmnopqrstuvwxyz" if c not in mapped_targets
        )
        nl_unknown = "\n".join(unknown_mapping_chars)
        nl_unmapped = "\n".join(unmapped_targets)
        lines.append(f"Unknown characters\n{nl_unknown}")
        lines.append(f"Unmapped target letters\n{nl_unmapped}")
        
        # 解密查询文本
        lines.append("")
        question_words = query_text.split()
        lines.append(f"Now decrypting 【 {query_text}】:")
        plain_to_cipher: Dict[str, str] = {v: k for k, v in cipher_to_plain.items()}
        decoded_words: List[str] = [""] * len(question_words)
        unknown_words: List[Tuple[int, str, str, str, str]] = []
        
        for i, cw in enumerate(question_words):
            decrypted_chars: List[str] = []
            display_chars: List[str] = []
            mapping_steps: List[str] = []
            has_unknown = False
            for cc in cw:
                if cc in cipher_to_plain:
                    decrypted_chars.append(cipher_to_plain[cc])
                    display_chars.append(cipher_to_plain[cc])
                    mapping_steps.append(f"{cc}->{cipher_to_plain[cc]}")
                else:
                    decrypted_chars.append("?")
                    display_chars.append(f"({cc})")
                    mapping_steps.append(f"{cc}->?")
                    has_unknown = True
            
            partial = "".join(decrypted_chars)
            display_partial = "".join(display_chars)
            step_str = "\n".join(mapping_steps)
            cipher_dashed = dash.join(cw)
            plain_dashed = dash.join(display_chars)
            if i > 0:
                lines.append("")
            if has_unknown:
                lines.append(
                    f"【 {cw}】\n{cipher_dashed}\n{step_str}\n{plain_dashed}->【{display_partial}】-> {display_partial}"
                )
                orig_dashed = dash.join(display_chars)
                unknown_words.append((i, cw, partial, display_partial, orig_dashed))
            else:
                lines.append(
                    f"【 {cw}】\n{cipher_dashed}\n{step_str}\n{plain_dashed}->【{partial}】-> {partial}"
                )
                decoded_words[i] = partial
        
        # 收集查询中所有未知密文字符
        all_unknown_chars: Set[str] = set()
        for _, cw, _, _, _ in unknown_words:
            for cc in cw:
                if cc not in cipher_to_plain:
                    all_unknown_chars.add(cc)
        
        # 显示当前句子状态
        sentence_parts = []
        for i, cw in enumerate(question_words):
            if decoded_words[i]:
                sentence_parts.append(decoded_words[i])
            else:
                display = dash.join(
                    f"({cc})" if cc not in cipher_to_plain else cipher_to_plain[cc]
                    for cc in cw
                )
                sentence_parts.append(display)
        lines.append("")
        lines.append("The sentence currently is")
        lines.append(" ".join(sentence_parts))
        
        lines.append("")
        if all_unknown_chars:
            iter_parts = "\n".join(
                f"{c} {'yes' if c in all_unknown_chars else 'no'}"
                for c in sorted(unknown_mapping_chars)
            )
            lines.append(
                f"Iterating over the unknown letters to see if they are in the question\n{iter_parts}"
            )
            unknown_in_question = sorted(all_unknown_chars)
            lines.append("")
            nl_unknown_q = "\n".join(unknown_in_question)
            lines.append(f"The unknown letters\n{nl_unknown_q}")
            lines.append("")
            lines.append("Let me find the best matching wonderland words:")
        else:
            lines.append(
                "Iterating over the unknown letters to see if they are in the question: no unknown letters"
            )
        
        # 使用 Wonderland 词典约束解决未知单词
        if unknown_words:
            wonderland_set = set(wonderland_words)
            initial_c2p = dict(cipher_to_plain)
            
            for idx, cw, _partial_orig, display_partial, orig_dashed in unknown_words:
                # 用当前映射重新计算部分解密结果
                partial = "".join(
                    cipher_to_plain[cc] if cc in cipher_to_plain else "?" for cc in cw
                )
                candidates = _candidate_words_for_partial(
                    partial, cipher_to_plain, plain_to_cipher, cw
                )
                display_candidates = sorted(candidates)
                if not display_candidates:
                    return None
                
                display_dashed = dash.join(
                    f"({cc})" if cc not in cipher_to_plain else cipher_to_plain[cc]
                    for cc in cw
                )
                accumulated_new = [
                    f"【({cc})】->【{cipher_to_plain[cc]}】"
                    for cc in sorted(cipher_to_plain)
                    if cc not in initial_c2p
                ]
                lines.append("")
                lines.append(f"【{orig_dashed}】")
                if accumulated_new:
                    lines.append(f"New mappings: {', '.join(accumulated_new)}")
                else:
                    lines.append("New mappings: none")
                lines.append(f"【{display_dashed}】")
                
                # 遍历 Wonderland 单词列表，逐个检查匹配
                target_len = len(cw)
                lines.append(f"The length of the word is {target_len}.")
                for word in wonderland_words:
                    wlen = len(word)
                    if wlen != target_len:
                        lines.append(f"{word} {wlen} length")
                        continue
                    # 逐字符比较，遇到不匹配即提前停止
                    word_dashed = dash.join(word)
                    comparisons: List[str] = []
                    mismatch_found = False
                    tentative: Dict[str, str] = {}
                    mapped_plain = set(cipher_to_plain.values())
                    for pos, (wi_char, cc) in enumerate(zip(word, cw)):
                        if cc in cipher_to_plain:
                            pc = cipher_to_plain[cc]
                            if wi_char == pc:
                                comparisons.append(f"{pos}【{wi_char}】【{pc}】match")
                            else:
                                comparisons.append(f"{pos}【{pc}】【{wi_char}】unmatchable")
                                mismatch_found = True
                                break
                        else:
                            if cc in tentative:
                                if tentative[cc] == wi_char:
                                    comparisons.append(f"{pos}【{wi_char}】【({cc})】consistent")
                                else:
                                    comparisons.append(f"{pos}【{wi_char}】【({cc})】contradiction")
                                    mismatch_found = True
                                    break
                            else:
                                if wi_char in mapped_plain:
                                    comparisons.append(f"{pos}【{wi_char}】【({cc})】untargeted")
                                    mismatch_found = True
                                    break
                                tentative[cc] = wi_char
                                comparisons.append(f"{pos}【{wi_char}】【({cc})】matchable")
                    comp_str = ", ".join(comparisons)
                    if not mismatch_found:
                        comp_str += f", {len(cw)} all match"
                    lines.append(f"{word} {wlen} 【{word_dashed}】, {comp_str}")
                
                # 过滤候选词：拒绝那些将未知密文字符映射到已占用明文字符的候选
                unmapped = {
                    c for c in "abcdefghijklmnopqrstuvwxyz"
                    if c not in cipher_to_plain.values()
                }
                remaining = []
                for c in display_candidates:
                    bad = False
                    for ci, wi in zip(cw, c):
                        if ci not in cipher_to_plain and wi not in unmapped:
                            bad = True
                            break
                    if not bad:
                        remaining.append(c)
                
                # 优先选择在 Wonderland 列表中的候选词
                wonderland_remaining = [c for c in remaining if c in wonderland_set]
                if wonderland_remaining:
                    chosen = wonderland_remaining[0]
                elif remaining:
                    chosen = remaining[0]
                else:
                    return None
                lines.append(f"Best match: 【{chosen}】")
                decoded_words[idx] = chosen
                
                # 显示解析后的字符
                resolved_dashed = dash.join(chosen)
                lines.append(f"【{display_dashed}】->【{resolved_dashed}】")
                
                # 逐字符比较：已知映射 vs 新映射
                pending_mappings: List[Tuple[str, str]] = []
                for cc, pc in zip(cw, chosen):
                    if cc in cipher_to_plain:
                        known_plain = cipher_to_plain[cc]
                        lines.append(f"【{known_plain}】->【{pc}】same")
                    else:
                        lines.append(f"【({cc})】->【{pc}】 new")
                        if (cc, pc) not in pending_mappings:
                            pending_mappings.append((cc, pc))
                # 更新映射
                new_mappings: List[str] = []
                for cc, pc in pending_mappings:
                    cipher_to_plain[cc] = pc
                    plain_to_cipher[pc] = cc
                    new_mappings.append(f"【({cc})】->【{pc}】")
                if new_mappings:
                    nl_new = "\n".join(new_mappings)
                    lines.append(f"Added mappings\n{nl_new}")
        
        # 检查是否所有单词都已解密
        if any(w == "" for w in decoded_words):
            return None
        
        computed = " ".join(decoded_words)
        lines.append("")
        lines.append("I will now return the answer in \\boxed{}")
        lines.append("The answer in \\boxed{–} is \\boxed{%s}" % computed)
        return "\n".join(lines)
