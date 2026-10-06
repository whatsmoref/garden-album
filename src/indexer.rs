//! 流水线编排：扫描 → 元数据 → 视觉 → 人脸 → OCR → 相册
//!
//! 与 Python 版的差异：
//! - 用 rayon 做批内并行（ORT 会自己用线程，故 rayon 线程数压到 2）
//! - 每批 commit，避免中途退出丢数据（Python 版原本缺 commit）

use anyhow::Result;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::albums::AlbumEngine;
use crate::config as C;
use crate::db::{DB, Photo};
use crate::face::{FacePipeline, PersonStore};
use crate::metadata::{extract_metadata, load_image_rgb};
use crate::models::{Hub, init_ort};
use crate::ocr::TargetedOcr;
use crate::quality;
use crate::visual::VisualAnalyzer;
use crate::{events, tags_of_doc};

/// 阶段 1（解码 + CLIP + NIMA）的产物；无副作用所以能并行
struct PreResult {
    meta: crate::db::PhotoMeta,
    vec: Vec<f32>,
    tags: Vec<(String, f32)>,
    quality: (f64, f64, f64, f64),
    rgb: Vec<u8>,
    w: usize,
    h: usize,
    stem: String,
}

pub struct Indexer {
    pub db: DB,
    pub hub: Arc<Hub>,
    pub visual: VisualAnalyzer,
    pub persons: Mutex<PersonStore>,
    pub faces: FacePipeline,
    pub ocr: TargetedOcr,
    pub albums: AlbumEngine,
}

impl Indexer {
    pub fn new(db: DB) -> Result<Self> {
        init_ort();
        let hub = Arc::new(Hub::new());
        let visual = VisualAnalyzer::new(hub.clone())?;
        let persons = PersonStore::load(&db)?;
        let faces = FacePipeline::new(hub.clone())?;
        let ocr = TargetedOcr::new(hub.clone());
        let albums = AlbumEngine::new(db_placeholder(&db)?, hub.clone())?;
        Ok(Self {
            db,
            hub,
            visual,
            persons: Mutex::new(persons),
            faces,
            ocr,
            albums,
        })
    }

    /// 增量扫描：库里已有 path 的跳过
    pub fn scan(&self, root: &Path) -> Result<Vec<PathBuf>> {
        let known: HashSet<String> = self
            .db
            .all_photos()?
            .into_iter()
            .map(|p| p.path)
            .collect();
        let mut out = Vec::new();
        for entry in walkdir(root) {
            let p = entry;
            if !p.is_file() {
                continue;
            }
            let Some(ext) = p.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            if !C::is_image_ext(ext) {
                continue;
            }
            if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.')) {
                continue;
            }
            if known.contains(&p.to_string_lossy().to_string()) {
                continue;
            }
            out.push(p);
        }
        out.sort();
        Ok(out)
    }

    pub fn index_folder(&self, root: &Path, watch: bool) -> Result<()> {
        loop {
            let paths = self.scan(root)?;
            println!("[index] 发现 {} 张新照片", paths.len());
            if !paths.is_empty() {
                let t0 = Instant::now();
                let n = self.index_batch(&paths)?;
                println!(
                    "[index] 完成 {n} 张，用时 {:.1}s（{:.2}s/张）",
                    t0.elapsed().as_secs_f64(),
                    t0.elapsed().as_secs_f64() / n.max(1) as f64
                );
                events::recompute(&self.db)?;
                quality::recompute_burst_best(&self.db)?;
                let live: HashSet<i64> = self.db.all_photos()?.into_iter().map(|p| p.id).collect();
                let dropped = self.db.vectors.prune(&live);
                self.db.commit()?;
                self.db.vectors.save(&C::vec_path())?;
                if dropped > 0 {
                    println!("[index] 清理 {dropped} 个孤儿向量");
                }
                println!("[index] 事件 / 连拍 / 向量已更新");
            }
            if !watch {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
        Ok(())
    }

    /// 批处理：CLIP/NIMA 并行，人脸聚类串行（要按顺序写 persons 表）
    pub fn index_batch(&self, paths: &[PathBuf]) -> Result<usize> {
        let mut done = 0usize;
        for chunk in paths.chunks(32) {
            self.db.begin()?;
            // 阶段 1：解码 + CLIP + NIMA（纯计算，可并行）
            // 注意：rusqlite 的 Connection 不是 Sync，去重必须在进 rayon 之前串行做完
            let todo: Vec<&PathBuf> = chunk
                .iter()
                .filter(|p| self.db.photo_id(&p.to_string_lossy()).ok().flatten().is_none())
                .collect();
            let pre: Vec<(PathBuf, PreResult)> = todo
                .par_iter()
                .filter_map(|p| self.prepare(p).ok().map(|r| ((*p).clone(), r)))
                .collect();
            // 阶段 2：串行落库（faces/persons/albums 有顺序依赖）
            for (path, pre) in pre {
                match self.persist(&path, pre) {
                    Ok(Some(pid)) => {
                        done += 1;
                        println!("  ✓ {}", path.file_name().unwrap_or_default().to_string_lossy());
                    }
                    Ok(None) => {}
                    Err(e) => eprintln!("[index] ✗ {}：{e}", path.display()),
                }
            }
            self.db.commit()?;
        }
        Ok(done)
    }

    /// 纯计算：不碰 DB（rusqlite 的 Connection 不是 Sync，rayon 里用不了）
    fn prepare(&self, path: &Path) -> Result<PreResult> {
        let meta = extract_metadata(path)?;
        let (rgb, w, h) = load_image_rgb(path, 1024)?;
        let vec = self.visual.embed(&rgb, w, h)?;
        let tags = self.visual.top_tags(&vec, 8, None);
        let q = self.visual.quality(&rgb, w, h)?;
        Ok(PreResult {
            meta,
            vec,
            tags,
            quality: q,
            rgb,
            w,
            h,
            stem: path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
        })
    }

    /// 阶段 2：落库 + 人脸 + OCR + 相册
    fn persist(&self, path: &Path, pre: PreResult) -> Result<Option<i64>> {
        let pid = self.db.insert_photo(&pre.meta, path)?;
        self.db.set_quality(pid, pre.quality.0, pre.quality.1, pre.quality.2, pre.quality.3)?;
        let mut tags = pre.tags.clone();
        self.db.set_tags(pid, &tags)?;

        // 人脸
        self.db.delete_faces_of_photo(pid)?;
        self.db.update_photo(
            pid,
            &[
                ("known_face_count", 0.into()),
                ("unknown_face_count", 0.into()),
                ("avg_smile", rusqlite::types::Value::Null),
                ("has_closed_eyes", 0.into()),
                ("best_face_area", rusqlite::types::Value::Null),
            ],
        )?;
        let faces = self.faces.analyze(&pre.rgb, pre.w, pre.h, None)?;
        if !faces.is_empty() {
            let mut known = 0i64;
            let mut smiles: Vec<f64> = Vec::new();
            let mut closed = false;
            let mut best_area = 0.0f64;
            for f in &faces {
                let (person_id, _sim) = self.persons.lock().assign(&self.db, &f.emb)?;
                let is_known = self.persons.lock().named().contains_key(&person_id);
                if is_known {
                    known += 1;
                }
                self.db.insert_face(
                    pid, person_id,
                    &f.bbox, &f.kps_flat(), &f.emb,
                    0.0, f.eyes_open, f.area,
                )?;
                smiles.push(0.0);
                closed |= !f.eyes_open;
                best_area = best_area.max(f.area);
            }
            let n = faces.len() as i64;
            self.db.update_photo(
                pid,
                &[
                    ("known_face_count", known.into()),
                    ("unknown_face_count", (n - known).into()),
                    ("avg_smile", rusqlite::types::Value::Null),
                    ("has_closed_eyes", i64::from(closed).into()),
                    ("best_face_area", best_area.into()),
                ],
            )?;
        }

        // FTS：标签 + 文件名
        let doc = tags_of_doc(&tags, &pre.stem);
        self.db.fts_set(pid, &doc)?;

        // 定向 OCR（仅疑似票据/截图）
        if TargetedOcr::need(pre.meta.is_screenshot != 0, &tags) {
            match self.ocr.run(path) {
                Ok((full, extra)) if !full.is_empty() => {
                    for t in extra {
                        if !tags.iter().any(|(x, _)| *x == t) {
                            tags.push((t, 1.0));
                        }
                    }
                    self.db.set_tags(pid, &tags)?;
                    let clipped: String = full.chars().take(5000).collect();
                    self.db
                        .update_photo(pid, &[("ocr_text", clipped.clone().into())])?;
                    self.db.fts_set(pid, &format!("{doc} {clipped}"))?;
                }
                Ok(_) => {}
                Err(e) => log::warn!("OCR 失败 {}: {e}", path.display()),
            }
        }

        // 向量 + 相册
        self.db.vectors.add(pid, &pre.vec);
        let row: Photo = self
            .db
            .query("SELECT * FROM photos WHERE id=?", rusqlite::params![pid], crate::db::row_to_photo)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("刚落库的 photo 读不回来"))?;
        self.albums.add_photo(pid, &row, &tags, &pre.vec)?;
        Ok(Some(pid))
    }
}

/// 相册引擎需要一个自己的 DB 句柄（rusqlite 连接不可跨线程共享成 &self）
/// 这里复用同一文件开第二个连接，避免把 AlbumEngine 改成泛型
fn db_placeholder(db: &DB) -> Result<DB> {
    DB::open_at(db.path().to_path_buf())
}

/// 递归遍历（不引 walkdir crate）
fn walkdir(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}

/// 把标签映射成中文，供 GUI 展示
pub fn tag_zh(t: &str) -> String {
    crate::vocab::TAG_ZH_DISPLAY
        .iter()
        .find(|(k, _)| *k == t)
        .map(|(_, v)| v.to_string())
        .unwrap_or_else(|| t.to_string())
}

/// 索引进度摘要
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub photos: i64,
    pub tags: i64,
    pub faces: i64,
    pub persons: i64,
    pub events: i64,
    pub bursts: i64,
    pub vectors: usize,
    pub with_ocr: i64,
    pub screenshots: i64,
}

impl Indexer {
    pub fn stats(&self) -> Result<IndexStats> {
        let one = |sql: &str| -> Result<i64> {
            Ok(self
                .db
                .query(sql, [], |r| r.get::<_, i64>(0))?
                .into_iter()
                .next()
                .unwrap_or(0))
        };
        Ok(IndexStats {
            photos: one("SELECT COUNT(*) FROM photos")?,
            tags: one("SELECT COUNT(*) FROM tags")?,
            faces: one("SELECT COUNT(*) FROM faces")?,
            persons: one("SELECT COUNT(*) FROM persons")?,
            events: one("SELECT COUNT(*) FROM events")?,
            bursts: one("SELECT COUNT(*) FROM bursts")?,
            vectors: self.db.vectors.len(),
            with_ocr: one("SELECT COUNT(*) FROM photos WHERE ocr_text IS NOT NULL")?,
            screenshots: one("SELECT COUNT(*) FROM photos WHERE is_screenshot=1")?,
        })
    }
}

/// 按 tag 聚合（GUI 侧边栏"标签云"用）
pub fn tag_counts(db: &DB) -> Result<Vec<(String, i64)>> {
    db.tag_counts()
}

/// 人脸数排行
pub fn face_counts(db: &DB) -> Result<Vec<(i64, i64)>> {
    db.face_counts()
}

pub type NamedMap = HashMap<i64, String>;
