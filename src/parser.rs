//! 查询解析器：中文自然语言 → 结构化 DSL（时间文法 + 词典 + 修饰词，零 LLM）

use chrono::{Datelike, Duration, Local, NaiveDate};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::vocab::{RELATION, ZH2CLIP, ZH2TAG};

pub static STOP: &[&str] = &[
    "的", "了", "地", "得", "和", "跟", "与", "及", "或", "在", "是", "有", "啊", "吧", "呢", "嘛", "呀", "哦",
    "这", "那", "都", "就", "还", "又", "再", "很", "非常", "所有", "全部", "照片", "图片", "相片", "相册",
    "帮我", "给我", "找", "找找", "搜索", "查", "查查", "看看", "显示", "列出", "来", "张", "些", "时候",
    "时间", "里面", "什么", "哪些", "那些", "这些", "一下", "帮", "帮忙", "一点", "里", "边", "去", "上",
    // 「第一次」这类序数词不是 BM25 关键词，留着只会污染 FTS MATCH
    "第一次", "上次", "上次见", "最近一次",
];

fn num_zh(s: &str) -> Option<u32> {
    Some(match s {
        "一" => 1, "二" | "两" => 2, "三" => 3, "四" => 4, "五" => 5, "六" => 6,
        "七" => 7, "八" => 8, "九" => 9, "十" => 10,
        _ => s.parse().ok()?,
    })
}

fn season_months(k: char) -> Vec<u32> {
    match k {
        '春' => vec![3, 4, 5],
        '夏' => vec![6, 7, 8],
        '秋' => vec![9, 10, 11],
        _ => vec![12, 1, 2],
    }
}

/// 节日窗口（近似值；农历节日可按年份精确化）
fn festival_window(w: &str) -> Option<(u32, u32, u32, u32)> {
    const TABLE: &[(&str, (u32, u32, u32, u32))] = &[
        ("元旦", (1, 1, 1, 3)),
        ("新年", (1, 1, 1, 3)),
        ("跨年", (12, 29, 1, 3)),
        ("春节", (1, 15, 2, 25)),
        ("过年", (1, 15, 2, 25)),
        ("情人节", (2, 13, 2, 15)),
        ("清明", (4, 3, 4, 6)),
        ("五一", (4, 29, 5, 5)),
        ("劳动节", (4, 29, 5, 5)),
        ("端午", (5, 20, 6, 25)),
        ("中秋", (9, 8, 10, 7)),
        ("国庆", (9, 29, 10, 8)),
        ("圣诞", (12, 23, 12, 26)),
    ];
    TABLE.iter().find(|(k, _)| w.starts_with(k)).map(|(_, v)| *v)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Range {
    pub gte: Option<String>,
    pub lte: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PeopleClause {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub all: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub none: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unknown_count: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<HashMap<String, i64>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TagsClause {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub all: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub none: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Dsl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<Range>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<TagsClause>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub people: Option<PeopleClause>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<HashMap<String, HashMap<String, f64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotion: Option<HashMap<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_text: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub exclude_screenshot: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone, Serialize)]
pub struct Chip {
    pub kind: String,
    pub text: String,
}

// ---------------------------------------------------------------- 修饰词

type ModFn = fn(&mut Dsl);

fn set_smile(d: &mut Dsl, gte: Option<f64>, lte: Option<f64>) {
    let e = d.emotion.get_or_insert_with(Default::default);
    let sm = e
        .entry("smile".to_string())
        .or_insert_with(|| Value::Object(Default::default()));
    if let Value::Object(o) = sm {
        if let Some(v) = gte {
            o.insert("gte".into(), serde_json::json!(v));
        }
        if let Some(v) = lte {
            o.insert("lte".into(), serde_json::json!(v));
        }
    }
}

fn quality(d: &mut Dsl, field: &str, op: &str, v: f64) {
    let q = d.quality.get_or_insert_with(Default::default);
    q.entry(field.to_string()).or_default().insert(op.to_string(), v);
}

fn people_mut(d: &mut Dsl) -> &mut PeopleClause {
    d.people.get_or_insert_with(Default::default)
}

fn m_no_stranger(d: &mut Dsl) {
    people_mut(d).unknown_count = Some(0);
}
fn m_smile(d: &mut Dsl) {
    set_smile(d, Some(0.5), None);
}
fn m_bigsmile(d: &mut Dsl) {
    set_smile(d, Some(0.72), None);
}
fn m_nosmile(d: &mut Dsl) {
    set_smile(d, None, Some(0.25));
}
fn m_eyes(d: &mut Dsl) {
    d.emotion
        .get_or_insert_with(Default::default)
        .insert("eyes_open".into(), Value::Bool(true));
}
fn m_sharp(d: &mut Dsl) {
    quality(d, "sharpness", "gte", 60.0);
}
fn m_hd(d: &mut Dsl) {
    quality(d, "sharpness", "gte", 75.0);
}
fn m_blur(d: &mut Dsl) {
    quality(d, "sharpness", "lte", 35.0);
}
fn m_pretty(d: &mut Dsl) {
    quality(d, "aesthetic", "gte", 6.5);
}
fn m_great(d: &mut Dsl) {
    quality(d, "aesthetic", "gte", 7.5);
}
fn m_two(d: &mut Dsl) {
    let p = people_mut(d);
    p.unknown_count = Some(0);
    p.count.get_or_insert_with(Default::default).insert("eq".into(), 2);
}
fn m_one(d: &mut Dsl) {
    let p = people_mut(d);
    p.unknown_count = Some(0);
    p.count.get_or_insert_with(Default::default).insert("eq".into(), 1);
}
fn m_group(d: &mut Dsl) {
    people_mut(d)
        .count
        .get_or_insert_with(Default::default)
        .insert("gte".into(), 2);
}

fn modifiers() -> &'static [(&'static str, ModFn)] {
    use std::sync::OnceLock;
    static M: OnceLock<Vec<(&'static str, ModFn)>> = OnceLock::new();
    M.get_or_init(|| {
        vec![
            ("没有陌生人", m_no_stranger as ModFn), ("没有其他人", m_no_stranger),
            ("没有旁人", m_no_stranger), ("没有路人", m_no_stranger),
            ("没有别人", m_no_stranger), ("无路人", m_no_stranger),
            ("笑得自然", m_smile), ("开心的笑", m_smile), ("笑容灿烂", m_bigsmile),
            ("笑得很开心", m_bigsmile), ("笑得开心", m_bigsmile),
            ("没闭着眼", m_eyes), ("睁着眼", m_eyes), ("没闭眼", m_eyes),
            ("笑了", m_smile), ("笑", m_smile), ("笑容", m_smile), ("开心", m_smile),
            ("大笑", m_bigsmile), ("没笑", m_nosmile), ("不笑", m_nosmile), ("严肃", m_nosmile),
            ("清晰", m_sharp), ("高清", m_hd), ("模糊", m_blur),
            ("好看", m_pretty), ("漂亮", m_pretty), ("拍得好", m_pretty),
            ("大片", m_great), ("精选", m_great),
            ("我们两个", m_two), ("两个人", m_two), ("我们俩", m_two),
            ("咱俩", m_two), ("俩人", m_two),
            ("一个人", m_one), ("独自", m_one), ("只有我", m_one),
            ("合影留念", m_group), ("合照", m_group), ("合影", m_group), ("多人", m_group),
        ]
    })
}

// ---------------------------------------------------------------- 时间文法

#[derive(Debug, Default, Clone)]
struct TimeParts {
    year: Option<i32>,
    months: Option<Vec<u32>>,
    day: Option<u32>,
    days_back: Option<i64>,
    window: Option<(u32, u32, u32, u32)>,
    abs: Option<(NaiveDate, NaiveDate)>,
}

type RuleFn = fn(&str, &mut TimeParts) -> Option<String>;

fn r_lastdays(s: &str, p: &mut TimeParts) -> Option<String> {
    let re = Regex::new(r"(\d+|[一二两三四五六七八九十])\s*天").unwrap();
    let m = re.captures(s)?;
    p.days_back = num_zh(&m[1]).map(|n| n as i64).or(Some(30));
    Some(s.to_string())
}

fn r_year(s: &str, p: &mut TimeParts) -> Option<String> {
    let now = Local::now();
    let y = if s.contains("今年") {
        now.year()
    } else if s.contains("去年") {
        now.year() - 1
    } else {
        now.year() - 2
    };
    p.year = Some(y);
    Some(s.to_string())
}

fn r_yearabs(s: &str, p: &mut TimeParts) -> Option<String> {
    let re = Regex::new(r"((?:19|20)\d{2})").unwrap();
    let m = re.captures(s)?;
    p.year = m[1].parse().ok();
    Some(s.to_string())
}

fn r_relmonth(s: &str, p: &mut TimeParts) -> Option<String> {
    let now = Local::now();
    let off = if s.starts_with('上') {
        -1
    } else if s.starts_with('下') {
        1
    } else {
        0
    };
    let t = now.year() * 12 + now.month0() as i32 + off;
    p.year = Some(t.div_euclid(12));
    p.months = Some(vec![t.rem_euclid(12) as u32 + 1]);
    Some(s.to_string())
}

fn r_relweek(s: &str, p: &mut TimeParts) -> Option<String> {
    let now = Local::now().date_naive();
    let off = if s.starts_with('上') { -7 } else { 0 };
    let monday = now - Duration::days(now.weekday().num_days_from_monday() as i64) + Duration::days(off);
    p.abs = Some((monday, monday + Duration::days(6)));
    Some(s.to_string())
}

fn r_season(s: &str, p: &mut TimeParts) -> Option<String> {
    let k = s.chars().next()?;
    p.months = Some(season_months(k));
    if p.year.is_none() {
        let now = Local::now();
        p.year = Some(if k == '冬' && now.month() == 12 { now.year() } else { now.year() - 1 });
    }
    Some(s.to_string())
}

fn r_festival(s: &str, p: &mut TimeParts) -> Option<String> {
    p.window = Some(festival_window(s)?);
    Some(s.to_string())
}

fn r_half(s: &str, p: &mut TimeParts) -> Option<String> {
    p.months = Some(match s {
        "上半年" => vec![1, 2, 3, 4, 5, 6],
        "下半年" => vec![7, 8, 9, 10, 11, 12],
        "年初" => vec![1, 2],
        _ => vec![11, 12],
    });
    Some(s.to_string())
}

fn r_monthday(s: &str, p: &mut TimeParts) -> Option<String> {
    let re = Regex::new(r"(\d{1,2})\s*月\s*(\d{1,2})").unwrap();
    let m = re.captures(s)?;
    p.months = Some(vec![m[1].parse().ok()?]);
    p.day = Some(m[2].parse().ok()?);
    Some(s.to_string())
}

fn r_month(s: &str, p: &mut TimeParts) -> Option<String> {
    let re = Regex::new(r"(\d{1,2})\s*月").unwrap();
    let m = re.captures(s)?;
    p.months = Some(vec![m[1].parse().ok()?]);
    Some(s.to_string())
}

fn r_recent(s: &str, p: &mut TimeParts) -> Option<String> {
    p.days_back.get_or_insert(30);
    Some(s.to_string())
}

fn rng(s: NaiveDate, e: NaiveDate) -> Range {
    Range {
        gte: Some(format!("{} 00:00:00", s.format("%Y-%m-%d"))),
        lte: Some(format!("{} 23:59:59", e.format("%Y-%m-%d"))),
    }
}

fn compose(p: &TimeParts, now: chrono::DateTime<Local>) -> Option<Range> {
    if let Some((a, b)) = p.abs {
        return Some(rng(a, b));
    }
    if let Some(d) = p.days_back {
        let end = now.date_naive();
        return Some(rng(end - Duration::days(d), end));
    }
    let year = p.year.unwrap_or_else(|| now.year());
    if let Some((m1, d1, m2, d2)) = p.window {
        let y2 = if m2 < m1 { year + 1 } else { year };
        return Some(rng(
            NaiveDate::from_ymd_opt(year, m1, d1)?,
            NaiveDate::from_ymd_opt(y2, m2, d2)?,
        ));
    }
    let Some(months) = p.months.clone() else {
        return Some(rng(
            NaiveDate::from_ymd_opt(year, 1, 1)?,
            NaiveDate::from_ymd_opt(year, 12, 31)?,
        ));
    };
    let mut sorted = months.clone();
    sorted.sort_unstable();
    // 冬季跨年：Y年12月 ~ (Y+1)年2月
    if sorted == [1, 2, 12] {
        return Some(rng(NaiveDate::from_ymd_opt(year, 12, 1)?, NaiveDate::from_ymd_opt(year + 1, 2, 28)?));
    }
    let m0 = *sorted.first()?;
    let m1 = *sorted.last()?;
    if let (Some(day), true) = (p.day, sorted.len() == 1) {
        let last = days_in_month(year, m0);
        let s = NaiveDate::from_ymd_opt(year, m0, day.min(last))?;
        return Some(rng(s, s));
    }
    Some(rng(
        NaiveDate::from_ymd_opt(year, m0, 1)?,
        NaiveDate::from_ymd_opt(year, m1, days_in_month(year, m1))?,
    ))
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    }
}

// ---------------------------------------------------------------- Parser

#[derive(Clone)]
enum Kind {
    Tag,
    Clip,
    Person,
    Mod(ModFn),
}

pub struct QueryParser {
    now: chrono::DateTime<Local>,
    entries: HashMap<String, (Kind, String)>,
    phrase_re: Regex,
    rules: Vec<(Regex, RuleFn)>,
}

impl QueryParser {
    pub fn new(person_names: &[String]) -> Self {
        let now = Local::now();
        let mut entries: HashMap<String, (Kind, String)> = HashMap::new();
        for (zh, tag) in ZH2TAG {
            entries.entry((*zh).to_string()).or_insert((Kind::Tag, tag.to_string()));
        }
        for (zh, en) in ZH2CLIP {
            entries.entry((*zh).to_string()).or_insert((Kind::Clip, en.to_string()));
        }
        for (zh, name) in RELATION {
            entries.entry((*zh).to_string()).or_insert((Kind::Person, name.to_string()));
        }
        for n in person_names {
            entries.entry(n.clone()).or_insert((Kind::Person, n.clone()));
        }
        for (zh, f) in modifiers() {
            entries.entry((*zh).to_string()).or_insert((Kind::Mod(*f), String::new()));
        }
        // 长词优先，避免「合照」被「照片」抢先
        let mut keys: Vec<String> = entries.keys().cloned().collect();
        keys.sort_by_key(|k| std::cmp::Reverse(k.chars().count()));
        let pat = keys
            .iter()
            .map(|k| regex::escape(k))
            .collect::<Vec<_>>()
            .join("|");
        let phrase_re = Regex::new(&pat).expect("词典正则应可编译");
        let rules: Vec<(Regex, RuleFn)> = vec![
            (Regex::new(r"(最近|过去|近)\s*(\d+|[一二两三四五六七八九十])\s*天").unwrap(), r_lastdays),
            (Regex::new(r"((?:19|20)\d{2})\s*年").unwrap(), r_yearabs),
            (Regex::new(r"(前年|去年|今年)").unwrap(), r_year),
            (Regex::new(r"(上个?月|这个?月|本月|当月|下个?月)").unwrap(), r_relmonth),
            (Regex::new(r"(上个?周|上个?星期|这周|本周|这星期|这礼拜)").unwrap(), r_relweek),
            (Regex::new(r"(春天|春季|夏天|夏季|秋天|秋季|冬天|冬季)").unwrap(), r_season),
            (Regex::new(r"(春节|过年|元旦|新年|跨年|情人节|清明|五一|劳动节|端午|中秋节?|国庆节?|圣诞节?)").unwrap(), r_festival),
            (Regex::new(r"(上半年|下半年|年初|年底|年末)").unwrap(), r_half),
            (Regex::new(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[号日]").unwrap(), r_monthday),
            (Regex::new(r"(\d{1,2})\s*月").unwrap(), r_month),
            (Regex::new(r"(最近|近期|近来)").unwrap(), r_recent),
        ];
        Self {
            now,
            entries,
            phrase_re,
            rules,
        }
    }

    pub fn now(&self) -> chrono::DateTime<Local> {
        self.now
    }

    pub fn parse(&self, text: &str) -> (Dsl, Vec<Chip>) {
        let mut s: String = text
            .trim()
            .chars()
            .map(|c| match c {
                '，' | ',' | '、' | '。' | '\t' | '\n' | '\r' => ' ',
                _ => c,
            })
            .collect();
        let mut dsl = Dsl::default();
        let mut chips: Vec<Chip> = Vec::new();
        let mut parts = TimeParts::default();

        // 1) 时间文法
        for (rx, rule) in &self.rules {
            let mut guard = 0;
            loop {
                guard += 1;
                if guard > 16 {
                    break;
                }
                let Some(m) = rx.find(&s).map(|m| (m.start(), m.end(), m.as_str().to_string())) else {
                    break;
                };
                match rule(&m.2, &mut parts) {
                    Some(txt) => {
                        chips.push(Chip { kind: "time".into(), text: txt });
                        s.replace_range(m.0..m.1, " ");
                    }
                    None => {
                        // 规则不认这个片段；挪到末尾防止死循环
                        let frag = s[m.0..m.1].to_string();
                        s.replace_range(m.0..m.1, " ");
                        s.push_str(&frag);
                    }
                }
            }
        }

        // 2) 短语扫描
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for caps in self.phrase_re.captures_iter(&s) {
            let whole = caps.get(0).unwrap();
            let Some((kind, payload)) = self.entries.get(whole.as_str()) else {
                continue;
            };
            spans.push((whole.start(), whole.end()));
            let kind_name = match kind {
                Kind::Tag => "tag",
                Kind::Clip => "clip",
                Kind::Person => "person",
                Kind::Mod(_) => "mod",
            };
            chips.push(Chip {
                kind: kind_name.into(),
                text: whole.as_str().to_string(),
            });
            match kind {
                Kind::Tag => dsl.tags.get_or_insert_with(Default::default).all.push(payload.clone()),
                Kind::Clip => push_clip(&mut dsl, payload.clone()),
                Kind::Person => dsl.people.get_or_insert_with(Default::default).all.push(payload.clone()),
                Kind::Mod(f) => f(&mut dsl),
            }
        }
        for (a, b) in spans.into_iter().rev() {
            s.replace_range(a..b, " ");
        }

        // 3) 剩余碎片 → jieba 兜底
        for tok in jieba().cut(&s, true) {
            let t = tok.trim();
            if t.is_empty() || STOP.contains(&t) {
                continue;
            }
            let all_ascii = t.chars().all(|c| c.is_ascii_alphanumeric() || c == ' ');
            let has_alpha = t.chars().any(|c| c.is_ascii_alphabetic());
            if all_ascii && has_alpha {
                push_clip(&mut dsl, t.to_string());
            } else if t.chars().count() >= 2 || t.is_ascii() {
                if !dsl.keywords.iter().any(|k| k == t) {
                    dsl.keywords.push(t.to_string());
                }
                chips.push(Chip { kind: "kw".into(), text: t.to_string() });
            }
        }

        // 4) 组装
        dsl.time = compose(&parts, self.now);
        (dsl, chips)
    }
}

/// jieba-rs 0.7 只有 Jieba::cut，没有全局函数；懒加载一个全局实例
fn jieba() -> &'static jieba_rs::Jieba {
    use once_cell::sync::Lazy;
    static J: Lazy<jieba_rs::Jieba> = Lazy::new(|| jieba_rs::Jieba::new());
    &J
}

fn push_clip(d: &mut Dsl, term: String) {
    match d.clip_text.as_mut() {
        Some(t) => {
            t.push(' ');
            t.push_str(&term);
        }
        None => d.clip_text = Some(term),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> QueryParser {
        QueryParser::new(&[])
    }

    #[test]
    fn 时间文法_绝对年月() {
        let (d, _) = p().parse("2024年7月 海边");
        assert_eq!(
            d.time.unwrap(),
            Range {
                gte: Some("2024-07-01 00:00:00".into()),
                lte: Some("2024-07-31 23:59:59".into()),
            }
        );
    }

    #[test]
    fn 词典命中标签() {
        let (d, chips) = p().parse("海边");
        assert_eq!(d.tags.unwrap().all, vec!["beach".to_string()]);
        assert!(chips.iter().any(|c| c.kind == "tag"));
    }

    #[test]
    fn 修饰词转质量与人物约束() {
        let (d, _) = p().parse("2026年 清晰 没路人 两个人");
        assert_eq!(d.quality.unwrap().get("sharpness").unwrap()["gte"], 60.0);
        assert_eq!(d.people.unwrap().unknown_count, Some(0));
    }

    #[test]
    fn 序数词不进BM25() {
        let (d, _) = p().parse("第一次去海边");
        assert!(!d.keywords.contains(&"第一次".to_string()));
        assert_eq!(d.tags.unwrap().all, vec!["beach".to_string()]);
    }

    #[test]
    fn 冬季跨年合并() {
        let (d, _) = p().parse("2025年冬天");
        let t = d.time.unwrap();
        assert_eq!(t.gte.unwrap(), "2025-12-01 00:00:00");
        assert_eq!(t.lte.unwrap(), "2026-02-28 23:59:59");
    }

    #[test]
    fn 开放词汇进CLIP() {
        let (d, _) = p().parse("红裙");
        assert_eq!(d.clip_text.unwrap(), "a red dress");
    }
}
