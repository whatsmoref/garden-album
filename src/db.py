# -*- coding: utf-8 -*-
"""SQLite（结构化 + FTS5 BM25）+ NumPy 向量索引。
VectorStore 接口与 zvec / sqlite-vec 兼容，规模上去后可无缝替换实现。"""
import sqlite3
import numpy as np
from pathlib import Path
from . import config as C
SCHEMA = """
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
CREATE INDEX IF NOT EXISTS idx_photos_taken ON photos(taken_at DESC);
CREATE INDEX IF NOT EXISTS idx_photos_event ON photos(event_id);
CREATE INDEX IF NOT EXISTS idx_photos_shot ON photos(is_screenshot);
CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash);
CREATE TABLE IF NOT EXISTS tags(photo_id INTEGER, tag TEXT, score REAL,
  source TEXT DEFAULT 'auto', PRIMARY KEY(photo_id, tag));
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_tags_tag_photo ON tags(tag, photo_id);
CREATE INDEX IF NOT EXISTS idx_tags_photo_tag ON tags(photo_id, tag);
CREATE TABLE IF NOT EXISTS tag_hidden(photo_id INTEGER, tag TEXT, PRIMARY KEY(photo_id, tag));
CREATE TABLE IF NOT EXISTS tag_blocklist(tag TEXT PRIMARY KEY, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS persons(id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE IF NOT EXISTS faces(
  id INTEGER PRIMARY KEY, photo_id INTEGER, person_id INTEGER,
  bbox TEXT, kps TEXT, embedding BLOB, smile REAL, eyes_open INTEGER, face_area REAL
);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_area ON faces(person_id, face_area DESC);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, start TEXT, end TEXT, title TEXT, city TEXT,
  photo_count INTEGER, device_count INTEGER, top_tags TEXT);
CREATE TABLE IF NOT EXISTS bursts(id INTEGER PRIMARY KEY, photo_count INTEGER);
CREATE TABLE IF NOT EXISTS albums(id INTEGER PRIMARY KEY, name TEXT UNIQUE, dsl TEXT);
CREATE TABLE IF NOT EXISTS album_photos(album_id INTEGER, photo_id INTEGER, score REAL,
  PRIMARY KEY(album_id, photo_id));
CREATE INDEX IF NOT EXISTS idx_album_photos_photo ON album_photos(photo_id);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS photo_fts USING fts5(text, tokenize='unicode61');
"""
class VectorStore:
    """万级规模暴力点积 <10ms；更大规模换 zvec/sqlite-vec，接口不变"""
    def __init__(self, dim):
        self.dim = dim; self.ids = []; self.vecs = []; self.idx = {}; self._mat = None
    def add(self, pid, vec):
        if pid in self.idx: return
        self.idx[pid] = len(self.ids); self.ids.append(pid)
        self.vecs.append(np.asarray(vec, np.float32)); self._mat = None
    def prune(self, keep_ids):
        """丢掉 photos 表里已不存在的向量，避免检索召回幽灵结果"""
        keep = set(keep_ids)
        pairs = [(i, v) for i, v in zip(list(self.ids), list(self.vecs)) if i in keep]
        self.ids = [i for i, _ in pairs]
        self.vecs = [v for _, v in pairs]
        self.idx = {i: n for n, i in enumerate(self.ids)}
        self._mat = None
        return len(self.ids)
    def mat(self):
        if self._mat is None:
            self._mat = np.stack(self.vecs) if self.vecs else np.zeros((0, self.dim), np.float32)
        return self._mat
    def search(self, q, k=100, allow=None):
        if not self.ids: return []
        if allow is not None and len(allow) * 2 < len(self.ids):
            rows = [self.idx[p] for p in allow if p in self.idx]
            if not rows: return []
            sims = self.mat()[rows] @ q; ids = [self.ids[r] for r in rows]
        else:
            sims = self.mat() @ q; ids = self.ids
        order = np.argsort(-sims)
        return [(ids[i], float(sims[i])) for i in order[:k] if allow is None or ids[i] in allow]
    def save(self, path):
        if self.ids:
            np.savez_compressed(path, ids=np.asarray(self.ids), vecs=np.stack(self.vecs))
    def load(self, path):
        if Path(path).exists():
            z = np.load(path)
            for i, v in zip(z["ids"].tolist(), z["vecs"]):
                self.add(i, v)
class DB:
    def __init__(self, path=None):
        self.path = Path(path or C.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(tags)")}
        if "source" not in cols:
            try:
                self.conn.execute("ALTER TABLE tags ADD COLUMN source TEXT DEFAULT 'auto'")
            except sqlite3.OperationalError:
                pass
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=8000")
        self.vectors = VectorStore(C.CLIP_DIM)
        self.vectors.load(C.VEC_PATH)
    def execute(self, sql, params=()):
        return self.conn.execute(sql, params)
    def query(self, sql, params=()):
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]
    def one(self, sql, params=()):
        r = self.conn.execute(sql, params).fetchone()
        return dict(r) if r else None
    # ---------- photos ----------
    def photo_id(self, path):
        return (self.one("SELECT id FROM photos WHERE path=?", (str(path),)) or {}).get("id")
    def insert_photo(self, **kw):
        cols, ph = ",".join(kw), ",".join("?" * len(kw))
        return self.conn.execute(f"INSERT INTO photos({cols}) VALUES({ph})", list(kw.values())).lastrowid
    def update_photo(self, pid, **kw):
        sets = ",".join(f"{k}=?" for k in kw)
        self.conn.execute(f"UPDATE photos SET {sets} WHERE id=?", [*kw.values(), pid])
    # ---------- tags ----------
    def set_tags(self, pid, tags):
        """写入自动标签；保留用户手动标签，并跳过隐藏/全库屏蔽的词。"""
        manual = self.query("SELECT tag, score FROM tags WHERE photo_id=? AND source='manual'",
                            (pid,))
        manual_tags = {t for t, _ in manual}
        blocked = {r["tag"] for r in self.query("SELECT tag FROM tag_blocklist")}
        hidden = {r["tag"] for r in self.query("SELECT tag FROM tag_hidden WHERE photo_id=?", (pid,))}
        final, seen = [], set()
        for t, s in tags:
            if not t or t in seen or t in blocked or t in hidden or t in manual_tags:
                continue
            final.append((t, s)); seen.add(t)
        self.conn.execute("DELETE FROM tags WHERE photo_id=? AND source='auto'", (pid,))
        self.conn.executemany(
            "INSERT OR REPLACE INTO tags(photo_id,tag,score,source) VALUES(?,?,?,'auto')",
            [(pid, t, s) for t, s in final])
        return final + [(t, s) for t, s in manual if t not in seen]

    def tag_names(self, pid):
        return [r["tag"] for r in self.query(
            "SELECT tag FROM tags WHERE photo_id=? ORDER BY source='manual' DESC, score DESC", (pid,))]

    def tags_manual(self, pid):
        return [r["tag"] for r in self.query(
            "SELECT tag FROM tags WHERE photo_id=? AND source='manual' ORDER BY score DESC, tag", (pid,))]

    def add_manual_tag(self, pid, tag, score=1.0):
        self.conn.execute("DELETE FROM tag_hidden WHERE photo_id=? AND tag=?", (pid, tag))
        self.conn.execute("INSERT OR REPLACE INTO tags(photo_id,tag,score,source) "
                          "VALUES(?,?,?,'manual')", (pid, tag, score))

    def remove_tag(self, pid, tag):
        self.conn.execute("DELETE FROM tags WHERE photo_id=? AND tag=?", (pid, tag))
        self.conn.execute("INSERT OR REPLACE INTO tag_hidden(photo_id,tag) VALUES(?,?)", (pid, tag))

    def block_tag(self, tag):
        self.conn.execute("INSERT OR IGNORE INTO tag_blocklist(tag) VALUES(?)", (tag,))
        self.conn.execute("DELETE FROM tags WHERE tag=?", (tag,))
        self.conn.execute("DELETE FROM tag_hidden WHERE tag=?", (tag,))

    def refresh_fts(self, pid):
        row = self.one("SELECT filename, ocr_text FROM photos WHERE id=?", (pid,))
        if not row:
            return
        text = " ".join(self.tag_names(pid)) + " " + (row.get("filename") or "")
        if row.get("ocr_text"):
            text += " " + row["ocr_text"][:5000]
        self.fts_set(pid, text)

    def tags_of(self, pids):
        """→ {photo_id: [(tag, score) 按 score 降序]}"""
        out = {}
        for i in range(0, len(pids), 500):
            chunk = pids[i:i + 500]
            qs = ",".join("?" * len(chunk))
            for r in self.query(f"SELECT photo_id, tag, score FROM tags "
                                f"WHERE photo_id IN ({qs}) ORDER BY score DESC", chunk):
                out.setdefault(r["photo_id"], []).append((r["tag"], r["score"]))
        return out
    # ---------- persons / faces ----------
    def persons(self):
        return self.query("SELECT id, name FROM persons ORDER BY id")
    def person_id_by_name(self, name):
        return (self.one("SELECT id FROM persons WHERE name=?", (name,)) or {}).get("id")
    def add_person(self, name):
        return self.conn.execute("INSERT INTO persons(name) VALUES(?)", (name,)).lastrowid
    def rename_person(self, pid, name):
        self.conn.execute("UPDATE persons SET name=? WHERE id=?", (name, pid))
    # ---------- meta ----------
    def get_meta(self, key, default=None):
        r = self.one("SELECT value FROM meta WHERE key=?", (key,))
        return r["value"] if r else default
    def set_meta(self, key, value):
        self.conn.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (key, str(value)))
    def insert_face(self, photo_id, person_id, bbox, kps, embedding, smile, eyes_open, face_area):
        return self.conn.execute(
            "INSERT INTO faces(photo_id,person_id,bbox,kps,embedding,smile,eyes_open,face_area) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (photo_id, person_id, str(bbox), str(kps),
             np.asarray(embedding, np.float32).tobytes(), smile, int(eyes_open), face_area)).lastrowid
    def all_face_vecs(self):
        for r in self.conn.execute("SELECT id, person_id, embedding FROM faces ORDER BY id"):
            yield r["id"], r["person_id"], np.frombuffer(r["embedding"], np.float32)
    # ---------- FTS5 (BM25) ----------
    def fts_set(self, pid, text):
        self.conn.execute("DELETE FROM photo_fts WHERE rowid=?", (pid,))
        if text:
            self.conn.execute("INSERT INTO photo_fts(rowid, text) VALUES(?,?)", (pid, text))
    def fts_search(self, words, limit=1000, allow=None):
        q = " OR ".join(f'"{w}"' for w in words)
        try:
            rows = self.query("SELECT rowid, bm25(photo_fts) s FROM photo_fts "
                              "WHERE photo_fts MATCH ? ORDER BY s LIMIT ?", (q, limit))
        except sqlite3.OperationalError:
            return []
        return [(r["rowid"], -r["s"]) for r in rows if allow is None or r["rowid"] in allow]
    # ---------- albums ----------
    def albums(self):
        return self.query("SELECT * FROM albums ORDER BY id")
    def save_album(self, name, dsl_json):
        self.conn.execute("INSERT INTO albums(name, dsl) VALUES(?,?) "
                          "ON CONFLICT(name) DO UPDATE SET dsl=excluded.dsl", (name, dsl_json))
        return self.one("SELECT id FROM albums WHERE name=?", (name,))["id"]
    def album(self, aid):
        return self.one("SELECT * FROM albums WHERE id=?", (int(aid),))
    def album_add(self, aid, pid, score=1.0):
        self.conn.execute("INSERT OR REPLACE INTO album_photos VALUES(?,?,?)", (aid, pid, score))
    def album_add_many(self, aid, pids):
        self.conn.executemany("INSERT OR REPLACE INTO album_photos VALUES(?,?,1.0)",
                              [(aid, int(p)) for p in pids])
    def album_remove_many(self, aid, pids):
        self.conn.executemany("DELETE FROM album_photos WHERE album_id=? AND photo_id=?",
                              [(aid, int(p)) for p in pids])
    def album_delete(self, aid):
        self.conn.execute("DELETE FROM album_photos WHERE album_id=?", (int(aid),))
        self.conn.execute("DELETE FROM albums WHERE id=?", (int(aid),))
    def album_rename(self, aid, name):
        self.conn.execute("UPDATE albums SET name=? WHERE id=?", (name, int(aid)))
    def album_photos(self, aid, limit=200, offset=0):
        return self.query(
            "SELECT p.* FROM album_photos ap JOIN photos p ON p.id=ap.photo_id "
            "WHERE ap.album_id=? ORDER BY ap.score DESC, p.taken_at DESC LIMIT ? OFFSET ?",
            (aid, limit, offset))
    def album_count(self, aid):
        return self.one("SELECT COUNT(*) c FROM album_photos WHERE album_id=?", (aid,))["c"]
    def photo_albums(self, pid):
        return self.query(
            "SELECT a.id, a.name, a.dsl FROM albums a "
            "JOIN album_photos ap ON ap.album_id=a.id "
            "WHERE ap.photo_id=? ORDER BY a.name", (pid,))
    def commit(self):
        self.conn.commit()
    def close(self):
        self.vectors.save(C.VEC_PATH)
        self.conn.commit(); self.conn.close()
