//! 查询路径：SQL 预过滤 + CLIP 向量 + BM25 → RRF 融合 → 后过滤

use anyhow::Result;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::db::{DB, Photo};
use crate::models::Hub;
use crate::parser::{Dsl, QueryParser};
use crate::vocab::RELATION;

/// RRF（Reciprocal Rank Fusion）：k=60 是原论文的经验值
pub fn rrf(rank_lists: &[Vec<i64>], k: f64) -> HashMap<i64, f64> {
    let mut out: HashMap<i64, f64> = HashMap::new();
    for rl in rank_lists {
        for (r, pid) in rl.iter().enumerate() {
            *out.entry(*pid).or_insert(0.0) += 1.0 / (k + r as f64 + 1.0);
        }
    }
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub dsl: Dsl,
    pub chips: Vec<crate::parser::Chip>,
    pub results: Vec<Photo>,
    /// 纯结构化查询没有向量/BM25 排序，按时间倒序
    pub structural_only: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind")]
pub enum Answer {
    Error { message: String },
    LastMeeting {
        person: String,
        last_time: String,
        total_photos: i64,
        photo: Photo,
    },
    Search(SearchResult),
}

pub struct SearchEngine {
    db: DB,
    hub: Arc<Hub>,
    parser: QueryParser,
}

impl SearchEngine {
    pub fn new(db: DB, hub: Arc<Hub>) -> Result<Self> {
        let names: Vec<String> = db
            .persons()?
            .iter()
            .filter(|p| !is_auto_name(&p.name))
            .map(|p| p.name.clone())
            .collect();
        let parser = QueryParser::new(&names);
        Ok(Self { db, hub, parser })
    }

    pub fn db(&self) -> &DB {
        &self.db
    }

    pub fn search(&self, text: &str, topk: usize) -> Result<Answer> {
        if let Some(name) = detect_last_meeting(text) {
            return self.last_meeting(&name);
        }
        let (dsl, chips) = self.parser.parse(text);
        let results = self.search_dsl(&dsl, topk)?;
        Ok(Answer::Search(SearchResult {
            dsl,
            chips,
            results,
            structural_only: false,
        }))
    }

    /// SQL 预过滤：把 DSL 编译成 WHERE 子句
    fn filter_sql(&self, dsl: &Dsl) -> Result<(String, Vec<Box<dyn rusqlite::ToSql>>)> {
        let mut w: Vec<String> = vec!["1=1".into()];
        let mut p: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        macro_rules! push {
            ($v:expr) => {
                p.push(Box::new($v))
            };
        }
        if let Some(t) = &dsl.time {
            if let Some(g) = &t.gte {
                w.push("taken_at >= ?".into());
                push!(g.clone());
            }
            if let Some(l) = &t.lte {
                w.push("taken_at <= ?".into());
                push!(l.clone());
            }
        }
        if let Some(tags) = &dsl.tags {
            for tag in &tags.all {
                w.push("id IN (SELECT photo_id FROM tags WHERE tag=?)".into());
                push!(tag.clone());
            }
            for tag in &tags.none {
                w.push("id NOT IN (SELECT photo_id FROM tags WHERE tag=?)".into());
                push!(tag.clone());
            }
        }
        if let Some(pe) = &dsl.people {
            for name in &pe.all {
                let pid = self
                    .db
                    .person_id_by_name(name)?
                    .or(self.db.person_id_by_name(rel_canon(name))?);
                match pid {
                    Some(id) => {
                        w.push("id IN (SELECT photo_id FROM faces WHERE person_id=?)".into());
                        push!(id);
                    }
                    None => w.push("0".into()), // 人物未命名 → 空结果
                }
            }
            for name in &pe.none {
                if let Some(id) = self.db.person_id_by_name(name)? {
                    w.push("id NOT IN (SELECT photo_id FROM faces WHERE person_id=?)".into());
                    push!(id);
                }
            }
            if let Some(n) = pe.unknown_count {
                w.push("unknown_face_count = ?".into());
                push!(n);
            }
            if let Some(c) = &pe.count {
                let col = "known_face_count + unknown_face_count";
                for (op, sql) in [("gte", ">="), ("lte", "<="), ("eq", "=")] {
                    if let Some(v) = c.get(op) {
                        w.push(format!("{col} {sql} ?"));
                        push!(*v);
                    }
                }
            }
        }
        if let Some(em) = &dsl.emotion {
            if let Some(sm) = em.get("smile").and_then(|v| v.as_object()) {
                w.push("avg_smile IS NOT NULL".into());
                if let Some(v) = sm.get("gte").and_then(|x| x.as_f64()) {
                    w.push("avg_smile >= ?".into());
                    push!(v);
                }
                if let Some(v) = sm.get("lte").and_then(|x| x.as_f64()) {
                    w.push("avg_smile <= ?".into());
                    push!(v);
                }
            }
            if em.get("eyes_open").and_then(|v| v.as_bool()).unwrap_or(false) {
                w.push("has_closed_eyes = 0".into());
            }
        }
        if let Some(q) = &dsl.quality {
            for col in ["sharpness", "aesthetic", "exposure"] {
                let Some(rng) = q.get(col) else { continue };
                for (op, sql) in [("gte", ">="), ("lte", "<="), ("eq", "=")] {
                    if let Some(v) = rng.get(op) {
                        w.push(format!("{col} {sql} ?"));
                        push!(*v);
                    }
                }
            }
        }
        if dsl.exclude_screenshot {
            w.push("is_screenshot = 0".into());
        }
        Ok((w.join(" AND "), p))
    }

    pub fn search_dsl(&self, dsl: &Dsl, topk: usize) -> Result<Vec<Photo>> {
        let (where_sql, params) = self.filter_sql(dsl)?;
        let refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref()).collect();
        let cand = self.db.query(
            &format!("SELECT * FROM photos WHERE {where_sql}"),
            refs.as_slice(),
            row_to_photo,
        )?;
        if cand.is_empty() {
            return Ok(Vec::new());
        }
        let allow: HashSet<i64> = cand.iter().map(|p| p.id).collect();
        let mut rankings: Vec<Vec<i64>> = Vec::new();

        // 开放词汇 → CLIP 文本向量（带缓存，命中后 <0.1ms）
        if let Some(ct) = &dsl.clip_text {
            let qv = self.hub.embed_text_cached(ct)?;
            rankings.push(
                self.db
                    .vectors
                    .search(&qv, 600, Some(&allow))
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect(),
            );
        }
        // 精确关键词 → BM25
        if !dsl.keywords.is_empty() {
            rankings.push(
                self.db
                    .fts_search(&dsl.keywords, 2000, Some(&allow))
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect(),
            );
        }

        let mut byid: HashMap<i64, Photo> = cand.iter().map(|p| (p.id, p.clone())).collect();
        let order: Vec<i64> = if rankings.is_empty() {
            // 纯结构化 → 时间倒序
            let mut v = cand;
            v.sort_by(|a, b| b.taken_at.cmp(&a.taken_at));
            v.into_iter().take(topk).map(|p| p.id).collect()
        } else {
            let fused = rrf(&rankings, 60.0);
            let mut v: Vec<(i64, f64)> = allow
                .iter()
                .map(|i| (*i, fused.get(i).copied().unwrap_or(0.0)))
                .collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            v.into_iter().take(topk).map(|(i, _)| i).collect()
        };
        let tmap = self.db.tags_of(&order)?;
        let mut out = Vec::with_capacity(order.len());
        for pid in order {
            if let Some(p) = byid.get_mut(&pid) {
                p.tags = tmap
                    .get(&pid)
                    .map(|v| v.iter().map(|(t, _)| t.clone()).collect())
                    .unwrap_or_default();
                out.push(p.clone());
            }
        }
        Ok(out)
    }

    /// 关系图谱：上次见某人是什么时候（纯 SQL，毫秒级）
    pub fn last_meeting(&self, name: &str) -> Result<Answer> {
        let Some(pid) = self
            .db
            .person_id_by_name(rel_canon(name))?
            .or(self.db.person_id_by_name(name)?)
        else {
            return Ok(Answer::Error {
                message: format!("未找到人物「{name}」，请先用 person 命令命名"),
            });
        };
        let me = self.db.person_id_by_name("我")?;
        let mut sql = "SELECT p.* FROM photos p WHERE p.id IN (SELECT photo_id FROM faces WHERE person_id=?)".to_string();
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(pid)];
        if let Some(m) = me {
            if m != pid {
                sql.push_str(" AND p.id IN (SELECT photo_id FROM faces WHERE person_id=?)");
                params.push(Box::new(m));
            }
        }
        sql.push_str(" ORDER BY p.taken_at DESC LIMIT 1");
        let refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref()).collect();
        let rows = self.db.query(&sql, refs.as_slice(), row_to_photo)?;
        let Some(photo) = rows.into_iter().next() else {
            return Ok(Answer::Error {
                message: format!("没有找到和「{name}」的合照"),
            });
        };
        let total = self.db.query(
            "SELECT COUNT(DISTINCT photo_id) c FROM faces WHERE person_id=?",
            rusqlite::params![pid],
            |r| r.get::<_, i64>(0),
        )?;
        Ok(Answer::LastMeeting {
            person: name.to_string(),
            last_time: photo.taken_at.clone().unwrap_or_default(),
            total_photos: total.first().copied().unwrap_or(0),
            photo,
        })
    }
}

fn rel_canon(name: &str) -> &str {
    RELATION
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| *v)
        .unwrap_or(name)
}

fn is_auto_name(n: &str) -> bool {
    n.strip_prefix("人物").is_some_and(|r| !r.is_empty() && r.chars().all(|c| c.is_ascii_digit()))
}

/// "上次见爸爸是什么时候" → Some("爸爸")
fn detect_last_meeting(text: &str) -> Option<String> {
    let t = text.trim();
    for marker in ["上次见", "最近一次见", "上次遇到", "上次碰到", "上次见到"] {
        if let Some(i) = t.find(marker) {
            let rest = &t[i + marker.len()..];
            let name: String = rest
                .chars()
                .take_while(|c| {
                    let s = *c;
                    s.is_alphanumeric() || RELATION.iter().any(|(k, _)| k.contains(s))
                })
                .collect();
            if !name.is_empty() {
                // 去掉时间后缀
                let name = name
                    .split("是")
                    .next()
                    .unwrap_or(&name)
                    .trim()
                    .to_string();
                return Some(name);
            }
        }
    }
    None
}

pub use crate::db::row_to_photo;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_两路都靠前的排前面() {
        let a = vec![1i64, 2, 3];
        let b = vec![3i64, 1, 2];
        let f = rrf(&[a, b], 60.0);
        assert_eq!(*f.get(&1).unwrap() > *f.get(&3).unwrap(), false);
        // 1 在两路都是第 1/第 2，3 是第 3/第 1，总分应接近
        assert!((f[&1] - f[&3]).abs() < 0.02);
    }

    #[test]
    fn 识别上次见问句() {
        assert_eq!(detect_last_meeting("上次见爸爸是什么时候"), Some("爸爸".into()));
        assert_eq!(detect_last_meeting("最近一次见到妈妈"), Some("妈妈".into()));
        assert_eq!(detect_last_meeting("海边"), None);
    }
}
