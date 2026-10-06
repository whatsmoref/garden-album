//! SQLite（结构化 + FTS5 BM25）+ 暴力向量索引。
//! `VectorStore` 的接口与 zvec / sqlite-vec 兼容，规模上去后可无缝替换实现。

use anyhow::{Context, Result};
use parking_lot::RwLock;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// `SELECT *` 的列顺序一旦在 SCHEMA 里变动就会静默错位（Photo.from_row 按序号取）。
/// 这里显式列出列名，`all_photos` / `photos_by_ids` 都用它。
pub const PHOTO_COLS: &str = "id, path, filename, taken_at, gps_lat, gps_lon, device, \
width, height, is_screenshot, phash, aesthetic, technical, sharpness, exposure, \
known_face_count, unknown_face_count, avg_smile, has_closed_eyes, best_face_area, \
event_id, burst_id, burst_best, ocr_text, added_at";

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS photos(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE, filename TEXT,
  taken_at TEXT, gps_lat REAL, gps_lon REAL, device TEXT,
  width INTEGER, height INTEGER, is_screenshot INTEGER DEFAULT 0, phash TEXT,
  aesthetic REAL, technical REAL, sharpness REAL, exposure REAL,
  known_face_count INTEGER DEFAULT 0, unknown_face_count INTEGER DEFAULT 0,
  avg_smile REAL, has_closed_eyes INTEGER DEFAULT 0, best_face_area REAL,
  event_id INTEGER, burst_id INTEGER, burst_best INTEGER DEFAULT 0,
  ocr_text TEXT, added_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_photos_taken ON photos(taken_at);
CREATE INDEX IF NOT EXISTS idx_photos_event ON photos(event_id);
CREATE TABLE IF NOT EXISTS tags(photo_id INTEGER, tag TEXT, score REAL, PRIMARY KEY(photo_id, tag));
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE TABLE IF NOT EXISTS persons(id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE IF NOT EXISTS faces(
  id INTEGER PRIMARY KEY, photo_id INTEGER, person_id INTEGER,
  bbox TEXT, kps TEXT, embedding BLOB, smile REAL, eyes_open INTEGER, face_area REAL
);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, start TEXT, end TEXT, title TEXT, city TEXT,
  photo_count INTEGER, device_count INTEGER, top_tags TEXT);
CREATE TABLE IF NOT EXISTS bursts(id INTEGER PRIMARY KEY, photo_count INTEGER);
CREATE TABLE IF NOT EXISTS albums(id INTEGER PRIMARY KEY, name TEXT UNIQUE, dsl TEXT);
CREATE TABLE IF NOT EXISTS album_photos(album_id INTEGER, photo_id INTEGER, score REAL,
  PRIMARY KEY(album_id, photo_id));
CREATE VIRTUAL TABLE IF NOT EXISTS photo_fts USING fts5(text, tokenize='unicode61');
"#;

// ---------------------------------------------------------------- 向量索引

pub struct VectorStore {
    dim: usize,
    inner: RwLock<VectorInner>,
}

struct VectorInner {
    ids: Vec<i64>,
    vecs: Vec<Vec<f32>>,
    idx: HashMap<i64, usize>,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            inner: RwLock::new(VectorInner {
                ids: Vec::new(),
                vecs: Vec::new(),
                idx: HashMap::new(),
            }),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add(&self, pid: i64, vec: &[f32]) {
        let mut g = self.inner.write();
        if g.idx.contains_key(&pid) {
            return;
        }
        let n = g.ids.len();
        g.idx.insert(pid, n);
        g.ids.push(pid);
        g.vecs.push(vec.to_vec());
    }

    /// 丢掉 photos 表里已不存在的向量，避免检索召回幽灵结果
    pub fn prune(&self, keep: &HashSet<i64>) -> usize {
        let mut g = self.inner.write();
        let before = g.ids.len();
        let pairs: Vec<(i64, Vec<f32>)> = g
            .ids
            .iter()
            .copied()
            .zip(g.vecs.iter().cloned())
            .filter(|(i, _)| keep.contains(i))
            .collect();
        g.ids = pairs.iter().map(|(i, _)| *i).collect();
        g.vecs = pairs.into_iter().map(|(_, v)| v).collect();
        g.idx = g.ids.iter().enumerate().map(|(n, i)| (*i, n)).collect();
        before - g.ids.len()
    }

    pub fn ids(&self) -> Vec<i64> {
        self.inner.read().ids.clone()
    }

    /// 返回 (photo_id, 余弦相似度)，按相似度降序
    pub fn search(&self, q: &[f32], k: usize, allow: Option<&HashSet<i64>>) -> Vec<(i64, f32)> {
        let g = self.inner.read();
        if g.ids.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(i64, f32)> = Vec::with_capacity(g.ids.len());
        for (i, id) in g.ids.iter().enumerate() {
            if let Some(a) = allow {
                if !a.contains(id) {
                    continue;
                }
            }
            let v = &g.vecs[i];
            let mut s = 0f32;
            for (x, y) in v.iter().zip(q.iter()) {
                s += x * y;
            }
            scored.push((*id, s));
        }
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored
    }

    /// 自定义二进制格式（非压缩，比 npz 快很多），布局：
    /// magic "ALB1" | u32 count | u32 dim | count*(i64 id | dim*f32)
    pub fn save(&self, path: &Path) -> Result<()> {
        use std::io::Write;
        let g = self.inner.read();
        if g.ids.is_empty() {
            return Ok(());
        }
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let f = std::fs::File::create(path)
            .with_context(|| format!("创建向量文件失败 {}", path.display()))?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(b"ALB1")?;
        w.write_all(&(g.ids.len() as u32).to_le_bytes())?;
        w.write_all(&(self.dim as u32).to_le_bytes())?;
        // 逐分量写小端字节：不用 unsafe from_raw_parts，
        // 否则 big-endian 平台上与 load() 的 from_le_bytes 不兼容
        let mut buf = [0u8; 4];
        for (i, id) in g.ids.iter().enumerate() {
            w.write_all(&id.to_le_bytes())?;
            for v in &g.vecs[i] {
                buf.copy_from_slice(&v.to_le_bytes());
                w.write_all(&buf)?;
            }
        }
        w.flush()?;
        Ok(())
    }

    pub fn load(&self, path: &Path) -> Result<usize> {
        if !path.exists() {
            return Ok(0);
        }
        let data = std::fs::read(path).with_context(|| format!("读取向量文件失败 {}", path.display()))?;
        if data.len() < 12 || &data[0..4] != b"ALB1" {
            anyhow::bail!("向量文件格式不正确：{}", path.display());
        }
        let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        if dim != self.dim {
            // 换模型（CLIP_DIM 变了）却沿用旧文件，会用错误的维度解释每一行，
            // 检索结果全是噪声且没有任何报错 —— 必须显式拒绝
            anyhow::bail!(
                "向量文件维度 {dim} 与当前 CLIP_DIM {} 不符，请删除 {} 重新索引",
                self.dim,
                path.display()
            );
        }
        let stride = 8 + dim * 4;
        let mut g = self.inner.write();
        for i in 0..count {
            let off = 12 + i * stride;
            if off + stride > data.len() {
                break;
            }
            let id = i64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            let mut v = vec![0f32; dim];
            for j in 0..dim {
                let b = off + 8 + j * 4;
                v[j] = f32::from_le_bytes(data[b..b + 4].try_into().unwrap());
            }
            let n = g.ids.len();
            g.idx.insert(id, n);
            g.ids.push(id);
            g.vecs.push(v);
        }
        Ok(g.ids.len())
    }
}


// ---------------------------------------------------------------- 行类型

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Photo {
    pub id: i64,
    pub path: String,
    pub filename: String,
    pub taken_at: Option<String>,
    pub gps_lat: Option<f64>,
    pub gps_lon: Option<f64>,
    pub device: Option<String>,
    pub width: i64,
    pub height: i64,
    pub is_screenshot: i64,
    pub phash: Option<String>,
    pub aesthetic: Option<f64>,
    pub technical: Option<f64>,
    pub sharpness: Option<f64>,
    pub exposure: Option<f64>,
    pub known_face_count: i64,
    pub unknown_face_count: i64,
    pub avg_smile: Option<f64>,
    pub has_closed_eyes: i64,
    pub best_face_area: Option<f64>,
    pub event_id: Option<i64>,
    pub burst_id: Option<i64>,
    pub burst_best: i64,
    pub ocr_text: Option<String>,
    pub added_at: Option<String>,
    /// 搜索/相册时附加，不落库
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PhotoMeta {
    pub taken_at: Option<String>,
    pub gps_lat: Option<f64>,
    pub gps_lon: Option<f64>,
    pub device: Option<String>,
    pub width: i64,
    pub height: i64,
    pub is_screenshot: i64,
    pub phash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Person {
    pub id: i64,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceRec {
    pub id: i64,
    pub photo_id: i64,
    pub person_id: i64,
    pub bbox: Vec<f32>,
    pub kps: Vec<f32>,
    pub smile: f64,
    pub eyes_open: bool,
    pub face_area: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventRow {
    pub id: i64,
    pub start: String,
    pub end: String,
    pub title: String,
    pub city: String,
    pub photo_count: i64,
    pub device_count: i64,
    pub top_tags: String,
}

/// 按序号取列，必须与 `PHOTO_COLS` 的顺序严格一致（所有查询已改用显式列名）
pub fn row_to_photo(r: &rusqlite::Row) -> rusqlite::Result<Photo> {
    Ok(Photo {
        id: r.get(0)?, path: r.get(1)?, filename: r.get(2)?,
        taken_at: r.get(3)?, gps_lat: r.get(4)?, gps_lon: r.get(5)?,
        device: r.get(6)?, width: r.get(7)?, height: r.get(8)?,
        is_screenshot: r.get(9)?, phash: r.get(10)?,
        aesthetic: r.get(11)?, technical: r.get(12)?,
        sharpness: r.get(13)?, exposure: r.get(14)?,
        known_face_count: r.get(15)?, unknown_face_count: r.get(16)?,
        avg_smile: r.get(17)?, has_closed_eyes: r.get(18)?, best_face_area: r.get(19)?,
        event_id: r.get(20)?, burst_id: r.get(21)?, burst_best: r.get(22)?,
        ocr_text: r.get(23)?, added_at: r.get(24)?,
        tags: Vec::new(),
    })
}

fn row_to_event(r: &rusqlite::Row) -> rusqlite::Result<EventRow> {
    Ok(EventRow {
        id: r.get(0)?, start: r.get(1)?, end: r.get(2)?, title: r.get(3)?,
        city: r.get(4)?, photo_count: r.get(5)?, device_count: r.get(6)?, top_tags: r.get(7)?,
    })
}

fn row_to_album(r: &rusqlite::Row) -> rusqlite::Result<AlbumRow> {
    Ok(AlbumRow { id: r.get(0)?, name: r.get(1)?, dsl: r.get(2)? })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlbumRow {
    pub id: i64,
    pub name: String,
    pub dsl: String,
}

// ---------------------------------------------------------------- DB

pub struct DB {
    pub conn: Connection,
    pub vectors: VectorStore,
    path: PathBuf,
}

impl DB {
    pub fn open() -> Result<Self> {
        Self::open_at(crate::config::db_path())
    }

    pub fn open_at(path: PathBuf) -> Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let conn = Connection::open(&path)
            .with_context(|| format!("打开数据库失败 {}", path.display()))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA foreign_keys=ON;
             PRAGMA temp_store=MEMORY;",
        )?;
        conn.execute_batch(SCHEMA)?;
        let vectors = VectorStore::new(crate::config::CLIP_DIM);
        vectors.load(&crate::config::vec_path())?;
        Ok(Self {
            conn,
            vectors,
            path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn execute(&self, sql: &str, params: &[&dyn rusqlite::ToSql]) -> Result<usize> {
        Ok(self.conn.execute(sql, params)?)
    }

    /// 通用查询：闭包把 `&Row` 映射成任意类型。
    /// （rusqlite 0.32 没有 FromRow trait，故用闭包而不是泛型 trait。）
    pub fn query<T, P, F>(&self, sql: &str, params: P, mut f: F) -> Result<Vec<T>>
    where
        P: rusqlite::Params,
        F: FnMut(&rusqlite::Row) -> rusqlite::Result<T>,
    {
        let mut st = self.conn.prepare(sql)?;
        let rows = st.query_map(params, |r| f(r))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn commit(&self) -> Result<()> {
        self.conn.execute_batch("COMMIT")?;
        Ok(())
    }

    // ---------- photos ----------

    pub fn photo_id(&self, path: &str) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row("SELECT id FROM photos WHERE path=?", [path], |r| r.get(0))
            .optional()?)
    }

    pub fn insert_photo(&self, m: &PhotoMeta, path: &Path) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO photos(path,filename,taken_at,gps_lat,gps_lon,device,width,height,\
             is_screenshot,phash) VALUES(?,?,?,?,?,?,?,?,?,?)",
            rusqlite::params![
                path.to_string_lossy(),
                path.file_name().unwrap_or_default().to_string_lossy(),
                m.taken_at,
                m.gps_lat,
                m.gps_lon,
                m.device,
                m.width,
                m.height,
                m.is_screenshot,
                m.phash,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn set_quality(&self, pid: i64, aes: f64, tec: f64, shp: f64, exp: f64) -> Result<()> {
        self.conn.execute(
            "UPDATE photos SET aesthetic=?, technical=?, sharpness=?, exposure=? WHERE id=?",
            rusqlite::params![aes, tec, shp, exp, pid],
        )?;
        Ok(())
    }

    pub fn update_photo(&self, pid: i64, fields: &[(&str, rusqlite::types::Value)]) -> Result<()> {
        if fields.is_empty() {
            return Ok(());
        }
        let sets: Vec<String> = fields.iter().map(|(k, _)| format!("{k}=?")).collect();
        let sql = format!(
            "UPDATE photos SET {} WHERE id=?",
            sets.join(",")
        );
        let mut vals: Vec<Box<dyn rusqlite::ToSql>> =
            fields.iter().map(|(_, v)| Box::new(v.clone()) as Box<dyn rusqlite::ToSql>).collect();
        vals.push(Box::new(pid));
        let refs: Vec<&dyn rusqlite::ToSql> = vals.iter().map(|b| b.as_ref()).collect();
        self.conn.execute(&sql, refs.as_slice())?;
        Ok(())
    }

    pub fn all_photos(&self) -> Result<Vec<Photo>> {
        self.query(&format!("SELECT {PHOTO_COLS} FROM photos ORDER BY id"), [], row_to_photo)
    }

    pub fn photos_by_ids(&self, ids: &[i64]) -> Result<HashMap<i64, Photo>> {
        let mut out = HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        for chunk in ids.chunks(500) {
            let qs: Vec<String> = chunk.iter().map(|_| "?".to_string()).collect();
            let sql = format!("SELECT {PHOTO_COLS} FROM photos WHERE id IN ({})", qs.join(","));
            let rows: Vec<Photo> = self.query(&sql, rusqlite::params_from_iter(chunk), row_to_photo)?;
            for r in rows {
                out.insert(r.id, r);
            }
        }
        Ok(out)
    }

    // ---------- tags ----------

    pub fn set_tags(&self, pid: i64, tags: &[(String, f64)]) -> Result<()> {
        self.conn
            .execute("DELETE FROM tags WHERE photo_id=?", [pid])?;
        let mut st = self.conn.prepare_cached(
            "INSERT OR REPLACE INTO tags(photo_id,tag,score) VALUES(?,?,?)",
        )?;
        for (t, s) in tags {
            st.execute(rusqlite::params![pid, t, s])?;
        }
        Ok(())
    }

    /// → { photo_id: [(tag, score) 按 score 降序] }
    pub fn tags_of(&self, pids: &[i64]) -> Result<HashMap<i64, Vec<(String, f64)>>> {
        let mut out: HashMap<i64, Vec<(String, f64)>> = HashMap::new();
        if pids.is_empty() {
            return Ok(out);
        }
        for chunk in pids.chunks(500) {
            let qs: Vec<String> = chunk.iter().map(|_| "?".to_string()).collect();
            let sql = format!(
                "SELECT photo_id, tag, score FROM tags WHERE photo_id IN ({}) ORDER BY score DESC",
                qs.join(",")
            );
            let rows: Vec<(i64, String, f64)> =
                self.query(&sql, rusqlite::params_from_iter(chunk), |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
            for (pid, tag, score) in rows {
                out.entry(pid).or_default().push((tag, score));
            }
        }
        Ok(out)
    }

    pub fn tag_counts(&self) -> Result<Vec<(String, i64)>> {
        self.query(
            "SELECT tag, COUNT(*) c FROM tags GROUP BY tag ORDER BY c DESC",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
    }

    pub fn photos_with_tag(&self, tag: &str) -> Result<Vec<i64>> {
        self.query(
            "SELECT photo_id FROM tags WHERE tag=?",
            rusqlite::params![tag],
            |r| r.get(0),
        )
    }

    // ---------- persons / faces ----------

    pub fn persons(&self) -> Result<Vec<Person>> {
        self.query("SELECT id, name FROM persons ORDER BY id", [], |r| Ok(Person { id: r.get(0)?, name: r.get(1)? }))
    }

    pub fn person_id_by_name(&self, name: &str) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row("SELECT id FROM persons WHERE name=?", [name], |r| r.get(0))
            .optional()?)
    }

    pub fn add_person(&self, name: &str) -> Result<i64> {
        self.conn
            .execute("INSERT INTO persons(name) VALUES(?)", [name])?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn rename_person(&self, pid: i64, name: &str) -> Result<()> {
        self.conn
            .execute("UPDATE persons SET name=? WHERE id=?", rusqlite::params![name, pid])?;
        Ok(())
    }

    pub fn face_counts(&self) -> Result<Vec<(i64, i64)>> {
        self.query(
            "SELECT person_id, COUNT(*) c FROM faces GROUP BY person_id ORDER BY c DESC",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
    }

    pub fn insert_face(
        &self,
        photo_id: i64,
        person_id: i64,
        bbox: &[f32],
        kps: &[f32],
        embedding: &[f32],
        smile: f64,
        eyes_open: bool,
        face_area: f64,
    ) -> Result<i64> {
        // embedding 以 f32 小端二进制存入 BLOB
        let mut blob = Vec::with_capacity(embedding.len() * 4);
        for v in embedding {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        self.conn.execute(
            "INSERT INTO faces(photo_id,person_id,bbox,kps,embedding,smile,eyes_open,face_area) \
             VALUES(?,?,?,?,?,?,?,?)",
            rusqlite::params![
                photo_id,
                person_id,
                fmt_f32_vec(bbox),
                fmt_f32_vec(kps),
                blob,
                smile,
                eyes_open as i64,
                face_area
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn delete_faces_of_photo(&self, photo_id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM faces WHERE photo_id=?", [photo_id])?;
        Ok(())
    }

    pub fn all_face_vecs(&self) -> Result<Vec<(i64, i64, Vec<f32>)>> {
        let mut st = self
            .conn
            .prepare("SELECT id, person_id, embedding FROM faces ORDER BY id")?;
        let mut rows = st.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let pid: i64 = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            out.push((id, pid, decode_f32_blob(&blob)));
        }
        Ok(out)
    }

    /// 每张照片出现过的人脸 id（跨设备事件合并时用来判断"共享人物"）
    pub fn photo_persons(&self) -> Result<HashMap<i64, HashSet<i64>>> {
        let mut out: HashMap<i64, HashSet<i64>> = HashMap::new();
        let mut st = self
            .conn
            .prepare("SELECT photo_id, person_id FROM faces")?;
        let mut rows = st.query([])?;
        while let Some(row) = rows.next()? {
            let pid: i64 = row.get(0)?;
            let per: i64 = row.get(1)?;
            out.entry(pid).or_default().insert(per);
        }
        Ok(out)
    }

    pub fn faces_of_photo(&self, photo_id: i64) -> Result<Vec<FaceRec>> {
        let rows: Vec<(i64, i64, i64, String, String, Option<f64>, i64, Option<f64>)> = self
            .query(
                "SELECT id, photo_id, person_id, bbox, kps, smile, eyes_open, face_area \
                 FROM faces WHERE photo_id=?",
                rusqlite::params![photo_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?)),
            )?;
        Ok(rows
            .into_iter()
            .map(|(id, pid, per, bbox, kps, smile, eyes, area)| FaceRec {
                id,
                photo_id: pid,
                person_id: per,
                bbox: parse_f32_vec(&bbox),
                kps: parse_f32_vec(&kps),
                smile: smile.unwrap_or(0.0),
                eyes_open: eyes != 0,
                face_area: area.unwrap_or(0.0),
            })
            .collect())
    }

    // ---------- FTS5 (BM25) ----------

    pub fn fts_set(&self, pid: i64, text: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM photo_fts WHERE rowid=?", [pid])?;
        if !text.is_empty() {
            self.conn
                .execute("INSERT INTO photo_fts(rowid, text) VALUES(?,?)", (pid, text))?;
        }
        Ok(())
    }

    pub fn fts_search(
        &self,
        words: &[String],
        limit: usize,
        allow: Option<&HashSet<i64>>,
    ) -> Vec<(i64, f64)> {
        if words.is_empty() {
            return Vec::new();
        }
        let q = words
            .iter()
            .map(|w| format!("\"{}\"", w.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR ");
        let mut st = match self
            .conn
            .prepare("SELECT rowid, bm25(photo_fts) s FROM photo_fts WHERE photo_fts MATCH ? ORDER BY s LIMIT ?")
        {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let mut rows = match st.query(rusqlite::params![q, limit as i64]) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::new();
        while let Ok(Some(row)) = rows.next() {
            let rid: i64 = row.get(0).unwrap_or(0);
            let s: f64 = row.get(1).unwrap_or(0.0);
            if let Some(a) = allow {
                if !a.contains(&rid) {
                    continue;
                }
            }
            out.push((rid, -s));
        }
        out
    }

    // ---------- albums ----------

    pub fn albums(&self) -> Result<Vec<AlbumRow>> {
        self.query("SELECT id, name, dsl FROM albums ORDER BY id", [], row_to_album)
    }

    pub fn save_album(&self, name: &str, dsl: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO albums(name, dsl) VALUES(?,?) \
             ON CONFLICT(name) DO UPDATE SET dsl=excluded.dsl",
            rusqlite::params![name, dsl],
        )?;
        Ok(self
            .conn
            .query_row("SELECT id FROM albums WHERE name=?", [name], |r| r.get(0))?)
    }

    pub fn album_add(&self, aid: i64, pid: i64, score: f64) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO album_photos(album_id,photo_id,score) VALUES(?,?,?)",
            rusqlite::params![aid, pid, score],
        )?;
        Ok(())
    }

    pub fn album_photos(&self, aid: i64, limit: usize) -> Result<Vec<Photo>> {
        self.query(
            "SELECT p.id, p.path, p.filename, p.taken_at, p.gps_lat, p.gps_lon, p.device, \
             p.width, p.height, p.is_screenshot, p.phash, p.aesthetic, p.technical, \
             p.sharpness, p.exposure, p.known_face_count, p.unknown_face_count, p.avg_smile, \
             p.has_closed_eyes, p.best_face_area, p.event_id, p.burst_id, p.burst_best, \
             p.ocr_text, p.added_at \
             FROM album_photos ap JOIN photos p ON p.id=ap.photo_id \
             WHERE ap.album_id=? ORDER BY p.taken_at DESC LIMIT ?",
            rusqlite::params![aid, limit as i64],
            row_to_photo,
        )
    }

    pub fn album_count(&self, aid: i64) -> Result<i64> {
        Ok(self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM album_photos WHERE album_id=?",
                [aid],
                |r| r.get(0),
            )
            .unwrap_or(0))
    }

    // ---------- events / bursts ----------

    pub fn events(&self, limit: usize) -> Result<Vec<EventRow>> {
        self.query(
            "SELECT id, start, end, title, city, photo_count, device_count, top_tags \
             FROM events ORDER BY start DESC LIMIT ?",
            rusqlite::params![limit as i64],
            row_to_event,
        )
    }

    pub fn clear_events(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM events; DELETE FROM bursts;")?;
        Ok(())
    }

    pub fn insert_event(&self, e: &EventRow) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO events(start,end,title,city,photo_count,device_count,top_tags) \
             VALUES(?,?,?,?,?,?,?)",
            rusqlite::params![e.start, e.end, e.title, e.city, e.photo_count, e.device_count, e.top_tags],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn set_photo_event(&self, pid: i64, eid: i64) -> Result<()> {
        self.conn
            .execute("UPDATE photos SET event_id=? WHERE id=?", rusqlite::params![eid, pid])?;
        Ok(())
    }

    pub fn reset_event_burst(&self) -> Result<()> {
        self.conn
            .execute_batch("UPDATE photos SET event_id=NULL, burst_id=NULL, burst_best=0")?;
        Ok(())
    }

    pub fn insert_burst(&self, count: usize) -> Result<i64> {
        self.conn
            .execute("INSERT INTO bursts(photo_count) VALUES(?)", [count as i64])?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn set_photo_burst(&self, pid: i64, bid: i64, best: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE photos SET burst_id=?, burst_best=? WHERE id=?",
            rusqlite::params![bid, best as i64, pid],
        )?;
        Ok(())
    }

    pub fn bursts(&self) -> Result<Vec<(i64, i64)>> {
        self.query("SELECT id, photo_count FROM bursts", [], |r| Ok((r.get(0)?, r.get(1)?)))
    }

    pub fn photos_in_burst(&self, bid: i64) -> Result<Vec<Photo>> {
        self.query(
            &format!("SELECT {PHOTO_COLS} FROM photos WHERE burst_id=? ORDER BY id"),
            rusqlite::params![bid],
            row_to_photo,
        )
    }

    pub fn mark_burst_best(&self, pid: i64, best: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE photos SET burst_best=? WHERE id=?",
            rusqlite::params![best as i64, pid],
        )?;
        Ok(())
    }

    // ---------- 事务 ----------

    pub fn begin(&self) -> Result<()> {
        self.conn.execute_batch("BEGIN")?;
        Ok(())
    }

    pub fn close(&self) -> Result<()> {
        self.vectors.save(&crate::config::vec_path())?;
        self.conn.execute_batch("COMMIT")?;
        Ok(())
    }
}

fn fmt_f32_vec(v: &[f32]) -> String {
    v.iter()
        .map(|x| format!("{x:.4}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_f32_vec(s: &str) -> Vec<f32> {
    s.split(',')
        .filter_map(|x| x.trim().parse::<f32>().ok())
        .collect()
}

pub fn decode_f32_blob(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
