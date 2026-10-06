//! F. 语义相册：DSL + 缓存文本向量，新照片入库一次点积自动收录

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

use crate::db::{DB, Photo};
use crate::models::Hub;
use crate::vocab::TAG_ZH_DISPLAY;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlbumSpec {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags_any: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags_all: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_avg_smile: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_aesthetic: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_clip: Option<f32>,
    /// kind=firsts 时不入库筛选，按需现算
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub person: Option<String>,
    #[serde(skip)]
    pub id: i64,
    #[serde(skip)]
    pub vec: Option<Vec<f32>>,
}

pub fn builtin_albums() -> Vec<AlbumSpec> {
    vec![
        AlbumSpec {
            name: "所有发票".into(),
            tags_any: Some(vec![
                "invoice".into(), "receipt".into(), "ticket".into(),
                "boarding pass".into(), "banknote".into(),
            ]),
            tags_all: None, min_avg_smile: None, min_aesthetic: None,
            clip_text: None, min_clip: None, kind: None, person: None, id: 0, vec: None,
        },
        AlbumSpec {
            name: "笑脸精选".into(),
            tags_any: None, tags_all: None,
            min_avg_smile: Some(0.55), min_aesthetic: Some(6.0),
            clip_text: None, min_clip: None, kind: None, person: None, id: 0, vec: None,
        },
        AlbumSpec {
            name: "风景美图".into(),
            tags_any: None, tags_all: None,
            min_avg_smile: None, min_aesthetic: Some(5.0),
            clip_text: Some("beautiful scenery landscape".into()),
            min_clip: Some(0.17), kind: None, person: None, id: 0, vec: None,
        },
        AlbumSpec {
            name: "宝宝第一次".into(),
            tags_any: None, tags_all: None,
            min_avg_smile: None, min_aesthetic: None,
            clip_text: None, min_clip: None,
            kind: Some("firsts".into()), person: Some("宝宝".into()), id: 0, vec: None,
        },
    ]
}

pub struct AlbumEngine {
    db: DB,
    hub: Arc<Hub>,
    pub albums: Vec<AlbumSpec>,
}

impl AlbumEngine {
    pub fn new(db: DB, hub: Arc<Hub>) -> Result<Self> {
        for a in builtin_albums() {
            let dsl = serde_json::to_string(&a)?;
            db.save_album(&a.name, &dsl)?;
        }
        let mut albums = Vec::new();
        for r in db.albums()? {
            let mut spec: AlbumSpec = serde_json::from_str(&r.dsl)
                .map_err(|e| anyhow::anyhow!("相册 {} 的 DSL 解析失败: {e}", r.name))?;
            spec.id = r.id;
            if let Some(ct) = &spec.clip_text {
                spec.vec = Some(hub.embed_text_cached(ct)?);
            }
            albums.push(spec);
        }
        Ok(Self { db, hub, albums })
    }

    pub fn db(&self) -> &DB {
        &self.db
    }

    /// 新照片入库：每相册一次点积，微秒级
    pub fn add_photo(&self, pid: i64, row: &Photo, tags: &[(String, f64)], vec: &[f32]) -> Result<()> {
        let tagset: Vec<&str> = tags.iter().map(|(t, _)| t.as_str()).collect();
        for alb in &self.albums {
            if alb.kind.as_deref() == Some("firsts") {
                continue;
            }
            if let Some(score) = self.score(alb, row, &tagset, Some(vec)) {
                self.db.album_add(alb.id, pid, (score * 1000.0).round() / 1000.0)?;
            }
        }
        Ok(())
    }

    fn score(&self, alb: &AlbumSpec, row: &Photo, tagset: &[&str], vec: Option<&[f32]>) -> Option<f64> {
        if let Some(any) = &alb.tags_any {
            if !any.iter().any(|t| tagset.contains(&t.as_str())) {
                return None;
            }
        }
        if let Some(all) = &alb.tags_all {
            if !all.iter().all(|t| tagset.contains(&t.as_str())) {
                return None;
            }
        }
        if let Some(m) = alb.min_avg_smile {
            match row.avg_smile {
                Some(v) if v >= m => {}
                _ => return None,
            }
        }
        if let Some(m) = alb.min_aesthetic {
            match row.aesthetic {
                Some(v) if v >= m => {}
                _ => return None,
            }
        }
        let mut v = 1.0f64;
        if let Some(av) = &alb.vec {
            let Some(iv) = vec else { return None };
            let s: f32 = av.iter().zip(iv.iter()).map(|(a, b)| a * b).sum();
            if s < alb.min_clip.unwrap_or(0.2) {
                return None;
            }
            v = s as f64;
        }
        Some(v)
    }

    pub fn firsts(&self, person: &str) -> Result<Vec<FirstItem>> {
        let Some(pid) = self.db.person_id_by_name(person)? else {
            return Ok(Vec::new());
        };
        let rows: Vec<Photo> = self.db.query(
            "SELECT p.* FROM photos p JOIN faces f ON f.photo_id=p.id \
             WHERE f.person_id=? ORDER BY p.taken_at",
            rusqlite::params![pid],
            crate::db::row_to_photo,
        )?;
        let ids: Vec<i64> = rows.iter().map(|r| r.id).collect();
        let tmap = self.db.tags_of(&ids)?;
        let mut seen: HashMap<String, ()> = HashMap::new();
        let mut out = Vec::new();
        // CLIP 噪声标签（"第一次鹿"没有意义），只认明确的里程碑
        const SKIP: &[&str] = &[
            "man", "woman", "child", "selfie", "portrait", "baby",
            "outdoor", "indoor", "sunny", "cloudy", "rainy", "zoo", "toy",
            "grass", "field", "garden", "park", "painting", "calligraphy",
        ];
        for r in &rows {
            let Some(list) = tmap.get(&r.id) else { continue };
            for (t, _) in list.iter().take(6) {
                if seen.contains_key(t) || SKIP.contains(&t.as_str()) {
                    continue;
                }
                seen.insert(t.clone(), ());
                let zh = TAG_ZH_DISPLAY
                    .iter()
                    .find(|(k, _)| k == t)
                    .map(|(_, v)| v.to_string())
                    .unwrap_or_else(|| t.clone());
                out.push(FirstItem {
                    title: format!("第一次{zh}"),
                    photo_id: r.id,
                    taken_at: r.taken_at.clone().unwrap_or_default(),
                    path: r.path.clone(),
                    tag: t.clone(),
                });
                break;
            }
        }
        Ok(out)
    }

    /// 相册 → 实际收录的照片（带标签，供 GUI 展示）
    pub fn album_photos(&self, aid: i64, limit: usize) -> Result<Vec<Photo>> {
        let rows = self.db.album_photos(aid, limit)?;
        let ids: Vec<i64> = rows.iter().map(|r| r.id).collect();
        let tmap = self.db.tags_of(&ids)?;
        Ok(rows
            .into_iter()
            .map(|mut p| {
                p.tags = tmap
                    .get(&p.id)
                    .map(|v| v.iter().take(6).map(|(t, _)| t.clone()).collect())
                    .unwrap_or_default();
                p
            })
            .collect())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FirstItem {
    pub title: String,
    pub photo_id: i64,
    pub taken_at: String,
    pub path: String,
    pub tag: String,
}

/// 相册 DSL 的可读写法（GUI 建相册时用）
pub fn parse_dsl(name: &str, raw: &str) -> Result<Value> {
    let v: Value = serde_json::from_str(raw)
        .map_err(|e| anyhow::anyhow!("相册 DSL 不是合法 JSON: {e}"))?;
    if !v.is_object() {
        anyhow::bail!("相册 DSL 必须是 JSON 对象");
    }
    Ok(json!({ "name": name, "dsl": v }))
}
