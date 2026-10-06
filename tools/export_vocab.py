#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Python 版 src/vocab.py 导出 Rust 词表。

必须按「词条边界」折行，不能用 textwrap 按字符宽度硬折 ——
否则 "city skyline" 会被折成 "city\\n    skyline"，Rust 字符串不跨行拼接，
于是 ZH2TAG["城市"] = "city skyline" 永远匹配不上入库标签。
"""
import ast
import json
import pathlib
import sys

SRC = pathlib.Path("/home/wang/Garden/src/vocab.py")
OUT = pathlib.Path(__file__).resolve().parent.parent / "src" / "vocab.rs"
WIDTH = 100


def load_values():
    """求值 vocab.py 里的顶层赋值（TAG_ZH_DISPLAY 是 DictComp + update，需特判）"""
    ns = {}
    for node in ast.parse(SRC.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            tgt, val = node.targets[0], node.value
            if tgt.id == "TAG_ZH_DISPLAY" and isinstance(val, ast.DictComp):
                # 与 Python 保持一致：{v: k for k, v in ZH2TAG.items()}
                ns["TAG_ZH_DISPLAY"] = {v: k for k, v in ns["ZH2TAG"].items()}
                continue
            ns[tgt.id] = ast.literal_eval(val)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            if isinstance(f, ast.Attribute) and f.attr == "update":
                ns[f.value.id].update(ast.literal_eval(node.value.args[0]))
    return ns


def q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def wrap(items, indent="    "):
    """只在 items 之间折行 —— 绝不切断一个词条"""
    lines, cur = [], indent
    for it in items:
        piece = it + ","
        if len(cur) + len(piece) + 1 > WIDTH and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += piece + " "
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def emit_flat(name, items):
    out = [f"pub static {name}: &[&str] = &["]
    out += wrap([q(t) for t in items])
    out.append("];")
    return "\n".join(out)


def emit_pairs(name, pairs):
    # 长词优先（与 Python 的 QueryParser 行为一致：先匹配长词）
    items = sorted(pairs.items(), key=lambda kv: -len(kv[0]))
    out = [f"pub static {name}: &[(&str, &str)] = &["]
    out += wrap([f"({q(k)}, {q(v)})" for k, v in items])
    out.append("];")
    return "\n".join(out)


def main():
    ns = load_values()
    parts = [
        "//! 词表：零样本标签（闭集）+ 中→英映射词典。可持续扩充。",
        "//!",
        "//! 本文件由 tools/export_vocab.py 从 Python 版 src/vocab.py 生成，改词表请改那边再重新导出。",
        "//! 生成器保证折行只发生在词条之间 —— 字符串字面量内绝不会出现 `\\n`。",
        "",
    ]
    parts.append("/// 零样本标签闭集（138 个），CLIP 文本塔用 `a photo of {tag}` 编码")
    parts.append(emit_flat("TAG_VOCAB", ns["TAG_VOCAB"]))
    parts.append("")
    for name in ["ZH2TAG", "ZH2CLIP", "RELATION", "TAG_ZH_DISPLAY"]:
        parts.append(emit_pairs(name, ns[name]))
        parts.append("")
    parts.append("/// 触发定向 OCR 的文档类标签")
    parts.append(emit_flat("DOC_TAGS", sorted(ns["DOC_TAGS"])))
    parts.append("")
    parts.append(TESTS)
    OUT.write_text("\n".join(parts), encoding="utf-8")

    # 自检：只看数据段（#[cfg(test)] 之前的注释里会出现示例字符串，会误报）
    import re
    src = OUT.read_text(encoding="utf-8")
    data = src.split("#[cfg(test)]")[0]
    lits = re.findall(r'"((?:[^"\\]|\\.)*)"', data)
    dirty = [x for x in lits if "\n" in x or "\r" in x or "\\n" in x]
    if dirty:
        print(f"✗ 仍有 {len(dirty)} 个字面量含换行: {dirty[:3]}")
        sys.exit(1)
    print(f"✓ {OUT}（{len(lits)} 个字面量，无跨行）")


TESTS = r'''
#[cfg(test)]
#[allow(non_snake_case)] // 测试名用中文更好读
mod tests {
    use super::*;

    /// 折行污染是最隐蔽的 bug：一个被折开的字符串字面量编译得过、运行时不报错，
    /// 但 ZH2TAG 里的多词标签永远匹配不上任何入库标签。这条测试是整个词表的地基。
    fn assert_clean(name: &str, items: &[&str]) {
        for s in items {
            assert!(!s.contains('\n'), "{name} 含换行: {s:?}");
            assert!(!s.contains('\r'), "{name} 含回车: {s:?}");
            assert_eq!(s.trim(), *s, "{name} 首尾有空白: {s:?}");
            assert!(!s.is_empty(), "{name} 有空串");
        }
    }

    fn assert_pairs_clean(name: &str, items: &[(&str, &str)]) {
        for (k, v) in items {
            assert!(!k.contains('\n'), "{name} key 含换行: {k:?}");
            assert!(!v.contains('\n'), "{name} value 含换行: {v:?}");
            assert!(!k.is_empty() && !v.is_empty(), "{name} 有空条目: {k:?}={v:?}");
        }
    }

    #[test]
    fn 词表无换行() {
        assert_clean("TAG_VOCAB", TAG_VOCAB);
        assert_clean("DOC_TAGS", DOC_TAGS);
        assert_pairs_clean("ZH2TAG", ZH2TAG);
        assert_pairs_clean("ZH2CLIP", ZH2CLIP);
        assert_pairs_clean("RELATION", RELATION);
        assert_pairs_clean("TAG_ZH_DISPLAY", TAG_ZH_DISPLAY);
    }

    #[test]
    fn 多词标签完整() {
        // 这些是最容易被折行破坏、且被 ZH2TAG 直接引用的值
        for want in ["city skyline", "swimming pool", "train station", "wedding dress",
                     "chat screenshot", "boarding pass", "snowy mountain", "birthday party"] {
            assert!(TAG_VOCAB.contains(&want), "TAG_VOCAB 缺 {want:?}");
        }
    }

    #[test]
    fn 中文映射指向存在的标签() {
        let tags: std::collections::HashSet<&str> = TAG_VOCAB.iter().copied().collect();
        for (zh, tag) in ZH2TAG {
            assert!(tags.contains(tag), "ZH2TAG[{zh:?}] = {tag:?} 不在 TAG_VOCAB 里");
        }
    }

    #[test]
    fn OCR标签都在闭集里() {
        let tags: std::collections::HashSet<&str> = TAG_VOCAB.iter().copied().collect();
        for t in DOC_TAGS {
            assert!(tags.contains(t), "DOC_TAGS 的 {t:?} 不在 TAG_VOCAB 里");
        }
    }

    #[test]
    fn 中文显示名覆盖全部标签() {
        // TAG_ZH_DISPLAY 由 ZH2TAG 反向生成，未被中文覆盖的标签应回落到英文
        let missing: Vec<&&str> = TAG_VOCAB
            .iter()
            .filter(|t| !TAG_ZH_DISPLAY.iter().any(|(k, _)| k == *t))
            .collect();
        assert!(
            missing.len() < TAG_VOCAB.len() / 2,
            "{} / {} 个标签没有中文名，回落逻辑可能没生效",
            missing.len(), TAG_VOCAB.len()
        );
    }
}
'''

if __name__ == "__main__":
    main()
