# -*- coding: utf-8 -*-
"""查询解析器：中文自然语言 → 结构化 DSL（时间文法 + 词典 + 修饰词，零 LLM）"""
import re
import calendar
from datetime import datetime, timedelta, date
import jieba
from .vocab import ZH2TAG, ZH2CLIP, RELATION
STOP = set("""的 了 地 得 和 跟 与 及 或 在 是 有 啊 吧 呢 嘛 呀 哦 这 那 都 就 还 又 再 很 非常
所有 全部 照片 图片 相片 相册 帮我 给我 找 找找 搜索 查 查查 看看 显示 列出 来 张 些 时候 时间 里面
什么 哪些 那些 这些 一下 帮 帮忙 一点 里 边 去 上""".split())
NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
SEASON = {"春": [3, 4, 5], "夏": [6, 7, 8], "秋": [9, 10, 11], "冬": [12, 1, 2]}
FESTIVAL = {  # 农历节日为近似窗口，可按年份精确化
    "元旦": (1, 1, 1, 3), "新年": (1, 1, 1, 3), "跨年": (12, 29, 1, 3),
    "春节": (1, 15, 2, 25), "过年": (1, 15, 2, 25),
    "情人节": (2, 13, 2, 15), "清明": (4, 3, 4, 6),
    "五一": (4, 29, 5, 5), "劳动节": (4, 29, 5, 5),
    "端午": (5, 20, 6, 25), "中秋": (9, 8, 10, 7),
    "国庆": (9, 29, 10, 8), "圣诞": (12, 23, 12, 26),
}
# ---- 修饰词 → DSL 子句 ----
def _p_no_stranger(d): d.setdefault("people", {})["unknown_count"] = 0
def _p_smile(d): d.setdefault("emotion", {})["smile"] = {"gte": 0.5}
def _p_bigsmile(d): d.setdefault("emotion", {})["smile"] = {"gte": 0.72}
def _p_nosmile(d): d.setdefault("emotion", {})["smile"] = {"lte": 0.25}
def _p_eyes(d): d.setdefault("emotion", {})["eyes_open"] = True
def _p_sharp(d): d.setdefault("quality", {})["sharpness"] = {"gte": 60}
def _p_hd(d): d.setdefault("quality", {})["sharpness"] = {"gte": 75}
def _p_blur(d): d.setdefault("quality", {})["sharpness"] = {"lte": 35}
def _p_pretty(d): d.setdefault("quality", {})["aesthetic"] = {"gte": 6.5}
def _p_great(d): d.setdefault("quality", {})["aesthetic"] = {"gte": 7.5}
def _p_two(d):
    pe = d.setdefault("people", {}); pe["count"] = {"eq": 2}; pe["unknown_count"] = 0
def _p_one(d):
    pe = d.setdefault("people", {}); pe["count"] = {"eq": 1}; pe["unknown_count"] = 0
def _p_group(d): d.setdefault("people", {}).setdefault("count", {"gte": 2})
MODIFIERS = {
    "没路人": _p_no_stranger, "没有路人": _p_no_stranger, "无路人": _p_no_stranger,
    "没有陌生人": _p_no_stranger, "没有其他人": _p_no_stranger, "没有别人": _p_no_stranger, "没有旁人": _p_no_stranger,
    "微笑": _p_smile, "笑了": _p_smile, "笑得自然": _p_smile, "笑容": _p_smile, "开心的笑": _p_smile,
    "笑": _p_smile, "开心": _p_smile,
    "大笑": _p_bigsmile, "笑得很开心": _p_bigsmile, "笑得开心": _p_bigsmile, "笑容灿烂": _p_bigsmile,
    "没笑": _p_nosmile, "不笑": _p_nosmile, "严肃": _p_nosmile,
    "没闭眼": _p_eyes, "没闭着眼": _p_eyes, "睁着眼": _p_eyes,
    "清晰": _p_sharp, "高清": _p_hd, "模糊": _p_blur,
    "好看": _p_pretty, "漂亮": _p_pretty, "拍得好": _p_pretty, "大片": _p_great, "精选": _p_great,
    "两个人": _p_two, "我们俩": _p_two, "咱俩": _p_two, "俩人": _p_two, "我们两个": _p_two,
    "一个人": _p_one, "独自": _p_one, "只有我": _p_one,
    "合照": _p_group, "合影": _p_group, "合影留念": _p_group, "多人": _p_group,
}
def _zh_num(s):
    return NUM.get(s) if len(s) == 1 and not s.isdigit() else (int(s) if s.isdigit() else None)
class QueryParser:
    def __init__(self, person_names=()):
        self.now = datetime.now()
        entries = {}
        for zh, tag in ZH2TAG.items(): entries.setdefault(zh, ("tag", tag))
        for zh, en in ZH2CLIP.items(): entries.setdefault(zh, ("clip", en))
        for zh, name in RELATION.items(): entries.setdefault(zh, ("person", name))
        for n in person_names: entries.setdefault(n, ("person", n))
        for zh, fn in MODIFIERS.items(): entries.setdefault(zh, ("mod", fn))
        keys = sorted(entries, key=len, reverse=True)      # 长词优先
        self.entries = entries
        self.phrase_re = re.compile("|".join(re.escape(k) for k in keys))
        self._compile_time()
    def _compile_time(self):
        R = [
            (r"(最近|过去|近)\s*(\d+|[一二两三四五六七八九十])\s*天", self._t_lastdays),
            (r"(前年|去年|今年)", self._t_year),
            (r"((?:19|20)\d{2})\s*年", self._t_yearabs),
            (r"(上个?月|这个?月|本月|当月|下个?月)", self._t_relmonth),
            (r"(上个?周|上个?星期|这周|本周|这星期|这礼拜)", self._t_relweek),
            (r"(春天|春季|夏天|夏季|秋天|秋季|冬天|冬季)", self._t_season),
            (r"(春节|过年|元旦|新年|跨年|情人节|清明|五一|劳动节|端午|中秋节?|国庆节?|圣诞节?)", self._t_festival),
            (r"(上半年|下半年|年初|年底|年末)", self._t_half),
            (r"(\d{1,2})\s*月\s*(\d{1,2})\s*[号日]", self._t_monthday),
            (r"(\d{1,2})\s*月", self._t_month),
            (r"(最近|近期|近来)", self._t_recent),
        ]
        self.time_rules = [(re.compile(p), fn) for p, fn in R]
    # ---- 时间处理器（写 parts，返回 chip 文本） ----
    def _t_year(self, m, p):
        p["year"] = self.now.year + {"今年": 0, "去年": -1, "前年": -2}[m.group()]; return m.group()
    def _t_yearabs(self, m, p):
        p["year"] = int(m.group(1)); return m.group()
    def _t_lastdays(self, m, p):
        n = _zh_num(m.group(2)); p["days_back"] = n or 30; return m.group()
    def _t_recent(self, m, p):
        p.setdefault("days_back", 30); return m.group()
    def _t_relmonth(self, m, p):
        w = m.group()
        off = -1 if w.startswith("上") else (1 if w.startswith("下") else 0)
        t = self.now.year * 12 + self.now.month - 1 + off
        p["year"], p["months"] = t // 12, [t % 12 + 1]; return w
    def _t_relweek(self, m, p):
        off = -1 if m.group().startswith("上") else 0
        monday = self.now.date() - timedelta(days=self.now.weekday()) + timedelta(days=7 * off)
        p["abs"] = (monday, monday + timedelta(days=6)); return m.group()
    def _t_season(self, m, p):
        k = m.group()[0]
        p["months"] = list(SEASON[k])
        if "year" not in p:
            p["year"] = self.now.year - (0 if (k == "冬" and self.now.month == 12) else (1 if k == "冬" else 0))
        return m.group()
    def _t_festival(self, m, p):
        for k, v in FESTIVAL.items():
            if m.group().startswith(k): p["window"] = v; return m.group()
        return m.group()
    def _t_half(self, m, p):
        p["months"] = {"上半年": [1, 2, 3, 4, 5, 6], "下半年": [7, 8, 9, 10, 11, 12],
                       "年初": [1, 2]}.get(m.group(), [11, 12]); return m.group()
    def _t_monthday(self, m, p):
        p["months"] = [int(m.group(1))]; p["day"] = int(m.group(2)); return m.group()
    def _t_month(self, m, p):
        p["months"] = [int(m.group(1))]; return m.group()
    # ---- 时间合成 ----
    def _compose(self, p):
        now = self.now
        if "abs" in p:
            return self._rng(*p["abs"])
        if "days_back" in p:
            return self._rng((now - timedelta(days=p["days_back"])).date(), now.date())
        year = p.get("year", now.year)
        if "window" in p:
            m1, d1, m2, d2 = p["window"]
            return self._rng(date(year, m1, d1), date(year + (1 if m2 < m1 else 0), m2, d2))
        months = p.get("months")
        if not months:
            return self._rng(date(year, 1, 1), date(year, 12, 31))
        if sorted(months) == [1, 2, 12]:   # 冬季跨年：Y年12月~(Y+1)年2月
            return self._rng(date(year, 12, 1), date(year + 1, 2, 28))
        day = p.get("day")
        if day and len(months) == 1:
            s = date(year, months[0], min(day, calendar.monthrange(year, months[0])[1]))
            return self._rng(s, s)
        m0, m1 = min(months), max(months)
        return self._rng(date(year, m0, 1), date(year, m1, calendar.monthrange(year, m1)[1]))
    @staticmethod
    def _rng(s, e):
        return {"gte": f"{s} 00:00:00", "lte": f"{e} 23:59:59"}
    # ---- 主入口 ----
    def parse(self, text):
        s = re.sub(r"[，,、。\s]+", " ", text.strip())
        parts, chips, dsl = {}, [], {}
        # 1) 时间文法（先长模式：最近N天 → 年份 → 季节/节日/月份）
        for rx, fn in self.time_rules:
            for m in list(rx.finditer(s)):
                chips.append(("time", fn(m, parts)))
                s = s[:m.start()] + " " + s[m.end():]
        # 2) 短语扫描：tag / clip / person / 修饰词（长词优先）
        spans = []
        for m in self.phrase_re.finditer(s):
            kind, payload = self.entries[m.group()]
            spans.append(m.span()); chips.append((kind, m.group()))
            if kind == "tag":
                dsl.setdefault("tags", {}).setdefault("all", []).append(payload)
            elif kind == "clip":
                dsl.setdefault("clip_terms", []).append(payload)
            elif kind == "person":
                dsl.setdefault("people", {}).setdefault("all", []).append(payload)
            else:
                payload(dsl)
        for a, b in sorted(spans, reverse=True):
            s = s[:a] + " " + s[b:]
        # 3) 剩余碎片 → jieba 兜底（英文→开放词汇；中文→BM25 关键词）
        for t in jieba.cut(s):
            t = t.strip()
            if not t or t in STOP or (len(t) < 2 and not t.isascii()):
                continue
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9 ]*", t):
                dsl.setdefault("clip_terms", []).append(t)
            else:
                dsl.setdefault("keywords", []).append(t)
                chips.append(("kw", t))
        # 4) 组装
        if parts:
            dsl["time"] = self._compose(parts)
        if dsl.get("clip_terms"):
            dsl["clip_text"] = " ".join(dsl.pop("clip_terms"))
        return dsl, chips
