//! 智能择优与清理：只给建议和置信度，绝不自动删除

use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;

use crate::config as C;
use crate::db::{DB, Photo};
use crate::metadata::phash_hamming;

/// 连拍择优：美学0.35 + 技术0.15 + 清晰度0.15 + 曝光0.15 + 人脸质量0.15 + 微笑0.10
pub fn burst_score(r: &Photo) -> f64 {
    let aes = r.aesthetic.unwrap_or(5.0);
    let tec = r.technical.unwrap_or(5.0);
    let shp = r.sharpness.unwrap_or(50.0);
    let exp = r.exposure.unwrap_or(70.0);
    let fq = if r.known_face_count > 0 || r.unknown_face_count > 0 {
        (r.best_face_area.unwrap_or(0.05) * 10.0).min(1.0)
    } else {
        0.3
    };
    let sm = r.avg_smile.unwrap_or(0.4);
    C::W_AESTHETIC * aes / 10.0
        + C::W_TECH * tec / 10.0
        + C::W_SHARP * shp / 100.0
        + C::W_EXPO * exp / 100.0
        + C::W_FACE * fq
        + C::W_SMILE * sm
}

pub fn recompute_burst_best(db: &DB) -> Result<()> {
    for b in db.bursts()? {
        if b.1 < 2 {
            continue;
        }
        let rows = db.photos_in_burst(b.0)?;
        if rows.len() < 2 {
            continue;
        }
        let Some(best) = rows.iter().max_by(|a, b| burst_score(a).total_cmp(&burst_score(b))) else {
            continue;
        };
        for r in &rows {
            db.mark_burst_best(r.id, r.id == best.id)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct Suggestion {
    #[serde(flatten)]
    pub photo: Photo,
    #[serde(rename = "type")]
    pub kind: String,
    pub confidence: f64,
    pub reason: String,
}

pub fn cleanup_suggestions(db: &DB) -> Result<Vec<Suggestion>> {
    let mut sug: Vec<Suggestion> = Vec::new();
    let all = db.all_photos()?;

    // 1) 完全重复（同 phash 同尺寸）
    let mut groups: HashMap<(String, i64, i64), Vec<&Photo>> = HashMap::new();
    let mut dated: Vec<&Photo> = all.iter().filter(|p| p.phash.is_some()).collect();
    dated.sort_by(|a, b| a.taken_at.cmp(&b.taken_at));
    for p in dated {
        groups
            .entry((p.phash.clone().unwrap(), p.width, p.height))
            .or_default()
            .push(p);
    }
    for g in groups.values() {
        if g.len() > 1 {
            for p in &g[1..] {
                sug.push(Suggestion {
                    photo: (*p).clone(),
                    kind: "duplicate".into(),
                    confidence: 0.95,
                    reason: format!("与 {} 内容完全重复", g[0].path),
                });
            }
        }
    }

    // 2) 连拍冗余（保留最优）
    for (bid, cnt) in db.bursts()? {
        if cnt <= 1 {
            continue;
        }
        let rows = db.photos_in_burst(bid)?;
        let Some(best) = rows.iter().find(|r| r.burst_best != 0).or(rows.first()) else {
            continue;
        };
        let bs = burst_score(best);
        for r in &rows {
            if r.id == best.id {
                continue;
            }
            let gap = bs - burst_score(r);
            sug.push(Suggestion {
                photo: r.clone(),
                kind: "burst_redundant".into(),
                confidence: (0.5 + gap * 0.5).min(0.9).round_to_2(),
                reason: format!("连拍冗余，更优为 {}", best.path),
            });
        }
    }

    // 3)+4) 一次遍历同时判模糊与双低分（原来两个独立 for 都遍历 all）
    for p in &all {
        let no_face = p.known_face_count == 0 && p.unknown_face_count == 0;
        if !no_face {
            continue;
        }
        if p.sharpness.unwrap_or(100.0) < 25.0 {
            sug.push(Suggestion {
                photo: p.clone(),
                kind: "blurry".into(),
                confidence: 0.65,
                reason: "画面模糊且无人脸".into(),
            });
        }
        if p.aesthetic.unwrap_or(10.0) < 3.5 && p.technical.unwrap_or(10.0) < 3.5 {
            sug.push(Suggestion {
                photo: p.clone(),
                kind: "low_quality".into(),
                confidence: 0.6,
                reason: "美学分与技术分均低且无人脸".into(),
            });
        }
    }

    // 5) 票据/截图堆积（同目录同 phash 近似但非同图，OCR 已提取完）
    let mut by_prefix: HashMap<String, Vec<&Photo>> = HashMap::new();
    for p in &all {
        if p.ocr_text.is_none() && p.is_screenshot == 0 {
            continue;
        }
        let dir = std::path::Path::new(&p.path)
            .parent()
            .map(|d| d.to_string_lossy().to_string())
            .unwrap_or_default();
        by_prefix.entry(dir).or_default().push(p);
    }
    for (dir, g) in by_prefix {
        if g.len() < 5 {
            continue;
        }
        // 同目录下 5 张以上票据，提示可归档
        for p in g.iter().take(1) {
            sug.push(Suggestion {
                photo: (*p).clone(),
                kind: "archive_bills".into(),
                confidence: 0.55,
                reason: format!("{} 下有 {} 张票据/截图，建议归档到单独目录", short_dir(&dir), g.len()),
            });
        }
    }

    sug.sort_by(|a, b| b.confidence.total_cmp(&a.confidence).then(a.photo.id.cmp(&b.photo.id)));
    Ok(sug)
}

fn short_dir(d: &str) -> String {
    std::path::Path::new(d)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| d.to_string())
}

trait Round2 {
    fn round_to_2(self) -> f64;
}
impl Round2 for f64 {
    fn round_to_2(self) -> f64 {
        (self * 100.0).round() / 100.0
    }
}

/// 供 events 层复用：两张图的 phash 距离
pub fn hamming(a: &str, b: &str) -> u32 {
    phash_hamming(a, b)
}
