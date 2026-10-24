# -*- coding: utf-8 -*-
"""本地 HTTP API，给 Tauri 壳 / 浏览器前端调用。

设计目标：首屏快、滚动稳、只在用户点刷新时重新取数。
- 缩略图走磁盘 + 内存 LRU，带 ETag / immutable Cache-Control；
- CLIP 文本向量和标签矩阵落盘，重启不再重算；
- 结构化接口带短 TTL 缓存，索引/改标后统一 bump generation 失效；
- JSON 大响应自动 gzip。
"""
import argparse
import ast
import gzip
import hashlib
import io
import json
import mimetypes
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config as C
from .vocab import TAG_ZH_DISPLAY

# ── 进程内共享状态 ──────────────────────────────────────────────
_state = {
    "db": None,
    "hub": None,
    "search": None,
    "albums": None,
    "model_lock": threading.Lock(),   # ONNX 会话串行化，避免并发踩
    "index": {"running": False, "kind": None, "phase": "idle",
              "done": 0, "total": 0, "error": None, "started_at": None},
    "generation": 0,
    "cache": {},
    "cache_lock": threading.Lock(),
    "db_lock": threading.Lock(),
}

_START = time.time()


def db():
    if _state["db"] is None:
        with _state["db_lock"]:
            if _state["db"] is None:
                from .db import DB
                _state["db"] = DB()
    return _state["db"]


def hub():
    if _state["hub"] is None:
        from .models import ModelHub
        _state["hub"] = ModelHub.get()
    return _state["hub"]


def search_engine():
    """检索引擎要 CLIP 文本塔，首次调用才会加载模型。"""
    if _state["search"] is None:
        from .search import SearchEngine
        with _state["model_lock"]:
            if _state["search"] is None:
                _state["search"] = SearchEngine(db(), hub())
    return _state["search"]


def albums_engine():
    if _state["albums"] is None:
        from .albums import AlbumEngine
        _state["albums"] = AlbumEngine(db(), hub())
    return _state["albums"]


def bump_generation():
    _state["generation"] += 1
    with _state["cache_lock"]:
        _state["cache"].clear()
    _state["albums"] = None


def cached(key, ttl, fn):
    now = time.time()
    gen = _state["generation"]
    with _state["cache_lock"]:
        hit = _state["cache"].get(key)
        if hit and hit[0] > now and hit[1] == gen:
            return hit[2]
    value = fn()
    with _state["cache_lock"]:
        _state["cache"][key] = (now + ttl, gen, value)
        if len(_state["cache"]) > 256:
            _state["cache"].clear()
    return value


def jsonable(o):
    """sqlite3.Row / bytes → 可 JSON 序列化的结构。"""
    if isinstance(o, (bytes, bytearray, memoryview)):
        return None
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def _bbox(v):
    try:
        return ast.literal_eval(v) if isinstance(v, str) else list(v)
    except Exception:
        return None


# ── 业务处理 ──────────────────────────────────────────────────
def api_health(_q):
    return {
        "ok": True,
        "ready": _state["search"] is not None,
        "uptime": round(time.time() - _START, 1),
        "db": str(C.DB_PATH),
        "data_dir": str(C.DATA_DIR),
        "db_exists": C.DB_PATH.exists(),
        "models_loaded": _state["search"] is not None,
        "index": _state["index"],
    }


def api_stats(_q):
    d = db()
    one = lambda s, p=(): d.one(s, p)["c"]

    def build():
        return {
            "photos": one("SELECT COUNT(*) c FROM photos"),
            "tags": one("SELECT COUNT(DISTINCT tag) c FROM tags"),
            "faces": one("SELECT COUNT(*) c FROM faces"),
            "persons": one("SELECT COUNT(*) c FROM persons"),
            "events": one("SELECT COUNT(*) c FROM events"),
            "memories": one("SELECT COUNT(*) c FROM events WHERE photo_count >= 3"),
            "places": one("SELECT COUNT(DISTINCT city) c FROM events WHERE city IS NOT NULL AND city<>''"),
            "bursts": one("SELECT COUNT(*) c FROM bursts"),
            "albums": one("SELECT COUNT(*) c FROM albums"),
            "with_ocr": one("SELECT COUNT(*) c FROM photos WHERE ocr_text IS NOT NULL AND ocr_text<>''"),
            "screenshots": one("SELECT COUNT(*) c FROM photos WHERE is_screenshot=1"),
            "tag_version": d.get_meta("tag_vocab_version", ""),
            "current_tag_version": C.TAG_VOCAB_VERSION,
            "index": dict(_state["index"]),
        }
    return cached("stats", 3, build)


def api_setup_status(_q):
    """首次启动引导：是否已有可展示的照片库。"""
    d = db()
    photos = d.one("SELECT COUNT(*) c FROM photos")["c"]
    library = (d.get_meta("library_dir") or "").strip()
    return {
        "configured": photos > 0,
        "photos": photos,
        "library": library,
        "library_exists": bool(library) and Path(library).is_dir(),
    }


def _resolve_folder(raw):
    raw = str(raw or "").strip()
    if not raw:
        return None, "请选择照片文件夹"
    try:
        p = Path(raw).expanduser().resolve(strict=True)
    except Exception:
        return None, "文件夹不存在或无法访问"
    if not p.is_dir():
        return None, "请选择文件夹，而不是文件"
    return str(p), None


def api_setup(body):
    """首次选择照片文件夹：记住路径并开始建立索引。"""
    if _state["index"]["running"]:
        return {"error": "已有任务在进行中，请稍后再试"}
    folder, err = _resolve_folder(body.get("dir"))
    if err:
        return {"error": err}
    d = db()
    d.set_meta("library_dir", folder)
    d.commit()
    bump_generation()
    threading.Thread(target=_index_worker, args=(folder, "setup"), daemon=True).start()
    return {"ok": True, "dir": folder, "kind": "setup"}


def api_search(q):
    query = (q.get("q") or [""])[0].strip()
    n = min(int((q.get("n") or ["80"])[0]), 300)
    if not query:
        return {"query": "", "chips": [], "dsl": {}, "results": []}
    t0 = time.time()
    res = search_engine().search(query, topk=n)
    if "error" in res:
        return {"query": query, "error": res["error"], "chips": [], "dsl": {}, "results": []}
    if "last_time" in res:                       # 「上次见XX」型，结构不同
        res = {
            "query": query,
            "chips": [("time", res["last_time"])],
            "dsl": {},
            "results": [res["photo"]],
            "last_meeting": {k: v for k, v in res.items() if k != "photo"},
        }
    res["query"] = query
    res["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonable(res)


def _photo_rows(where, params, limit, offset, summary=False):
    d = db()
    sql = ("SELECT p.*, e.city city, e.title event_title FROM photos p "
           f"LEFT JOIN events e ON e.id=p.event_id WHERE {' AND '.join(where)} "
           "ORDER BY p.taken_at DESC, p.id DESC LIMIT ? OFFSET ?")
    rows = d.query(sql, params + [limit, offset])
    total = d.one(f"SELECT COUNT(*) c FROM photos p "
                  f"LEFT JOIN events e ON e.id=p.event_id WHERE {' AND '.join(where)}",
                  params)["c"]
    if not rows:
        return rows, total
    ids = [r["id"] for r in rows]
    tmap = d.tags_of(ids)
    counts = {}
    qs = ",".join("?" * len(ids))
    for r in d.query(f"SELECT photo_id, COUNT(DISTINCT person_id) c FROM faces "
                     f"WHERE photo_id IN ({qs}) GROUP BY photo_id", ids):
        counts[r["photo_id"]] = r["c"]
    for r in rows:
        tags = [t for t, _ in tmap.get(r["id"], [])]
        r["tags"] = tags if not summary else []
        r["tags_zh"] = [TAG_ZH_DISPLAY.get(t, t) for t in tags[:6]]
        r["people_count"] = counts.get(r["id"], 0)
        r["has_ocr"] = bool((r.get("ocr_text") or "").strip())
        if summary:
            r.pop("ocr_text", None)
            r.pop("phash", None)
    return rows, total


def api_photos(q):
    limit = min(int((q.get("limit") or ["200"])[0]), 1000)
    offset = int((q.get("offset") or ["0"])[0])
    summary = (q.get("summary") or ["1"])[0] not in ("0", "false")
    where, params = ["1=1"], []
    for key in ("event", "person", "album", "city"):
        v = (q.get(key) or [""])[0]
        if not v:
            continue
        if key == "person":
            where.append("p.id IN (SELECT photo_id FROM faces WHERE person_id=?)")
            params.append(int(v))
        elif key == "album":
            where.append("p.id IN (SELECT photo_id FROM album_photos WHERE album_id=?)")
            params.append(int(v))
        elif key == "city":
            where.append("e.city=?")
            params.append(v)
        else:
            where.append("p.event_id=?")
            params.append(int(v))
    tag = (q.get("tag") or [""])[0].strip()
    if tag:
        where.append("p.id IN (SELECT photo_id FROM tags WHERE tag=? COLLATE NOCASE)")
        params.append(tag)
    if (q.get("ocr") or [""])[0]:
        where.append("p.ocr_text IS NOT NULL AND p.ocr_text<>''")
    if (q.get("screenshot") or [""])[0]:
        where.append("p.is_screenshot=1")
    rows, total = _photo_rows(where, params, limit, offset, summary)
    return {"total": total, "offset": offset, "photos": jsonable(rows)}


def api_photo(q, pid):
    d = db()
    row = d.one("SELECT p.*, e.city city, e.title event_title FROM photos p "
                "LEFT JOIN events e ON e.id=p.event_id WHERE p.id=?", (int(pid),))
    if not row:
        return {"error": f"没有 id={pid} 的照片"}
    tmap = d.tags_of([row["id"]])
    tags = [t for t, _ in tmap.get(row["id"], [])]
    row["tags"] = tags
    row["tags_zh"] = [TAG_ZH_DISPLAY.get(t, t) for t in tags]
    row["tags_manual"] = d.tags_manual(row["id"])
    row["albums"] = []
    for a in d.photo_albums(row["id"]):
        try:
            kind = json.loads(a.get("dsl") or "{}").get("kind", "smart")
        except Exception:
            kind = "smart"
        row["albums"].append({"id": a["id"], "name": a["name"], "kind": kind})
    faces = d.query(
        "SELECT f.id id, f.bbox bbox, f.smile smile, f.eyes_open eyes_open, "
        "f.face_area face_area, p.name person FROM faces f "
        "LEFT JOIN persons p ON p.id=f.person_id WHERE f.photo_id=? "
        "ORDER BY f.face_area DESC", (int(pid),))
    for f in faces:
        f["bbox"] = _bbox(f["bbox"])
    row["faces"] = jsonable(faces)
    return jsonable(row)


def api_events(q):
    limit = min(int((q.get("limit") or ["100"])[0]), 500)

    def build():
        rows = db().query(
            "SELECT e.*, "
            "(SELECT p.id FROM photos p WHERE p.event_id=e.id "
            " ORDER BY COALESCE(p.aesthetic,0) DESC, p.taken_at DESC LIMIT 1) cover_id "
            "FROM events e ORDER BY e.start DESC LIMIT ?", (limit,))
        for r in rows:
            r["top_tags_zh"] = [TAG_ZH_DISPLAY.get(t, t)
                                for t in (r.get("top_tags") or "").split(",") if t]
        return {"events": jsonable(rows)}
    return cached(f"events:{limit}", C.API_CACHE_TTL, build)


def api_memories(q):
    """回忆/地点共用：一次返回事件 + 每个事件的前 N 张照片，避免前端 N+1 请求。"""
    limit = min(int((q.get("limit") or ["60"])[0]), 200)
    per = min(int((q.get("per_event") or ["10"])[0]), 30)

    def build():
        events = db().query("SELECT * FROM events ORDER BY start DESC LIMIT ?", (limit,))
        for e in events:
            photos, _ = _photo_rows(["p.event_id=?"], [e["id"]], per, 0, True)
            e["photos"] = jsonable(photos)
            e["top_tags_zh"] = [TAG_ZH_DISPLAY.get(t, t)
                                for t in (e.get("top_tags") or "").split(",") if t]
        return {"events": jsonable(events)}
    return cached(f"memories:{limit}:{per}", C.API_CACHE_TTL, build)


def api_bursts(q):
    rows = db().query(
        "SELECT b.id id, b.photo_count photo_count, "
        "(SELECT id FROM photos WHERE burst_id=b.id AND burst_best=1) best_id, "
        "(SELECT path FROM photos WHERE burst_id=b.id AND burst_best=1) best_path "
        "FROM bursts b ORDER BY b.photo_count DESC")
    return {"bursts": jsonable(rows)}


def api_persons(_q):
    def build():
        rows = db().query(
            "SELECT p.id id, p.name name, COUNT(f.id) faces, "
            "COUNT(DISTINCT f.photo_id) photos, "
            "(SELECT f2.id FROM faces f2 WHERE f2.person_id=p.id "
            " ORDER BY COALESCE(f2.face_area,0) DESC LIMIT 1) cover_id "
            "FROM persons p LEFT JOIN faces f ON f.person_id=p.id "
            "GROUP BY p.id ORDER BY photos DESC, faces DESC")
        return {"persons": jsonable(rows)}
    return cached("persons", C.API_CACHE_TTL, build)


def api_albums(_q):
    def build():
        rows = db().query(
            "SELECT a.id id, a.name name, a.dsl dsl, COUNT(ap.photo_id) photo_count, "
            "(SELECT ap2.photo_id FROM album_photos ap2 WHERE ap2.album_id=a.id "
            " ORDER BY ap2.score DESC, ap2.photo_id DESC LIMIT 1) cover_id "
            "FROM albums a LEFT JOIN album_photos ap ON ap.album_id=a.id "
            "GROUP BY a.id ORDER BY a.id")
        for r in rows:
            try:
                r["kind"] = json.loads(r.get("dsl") or "{}").get("kind", "smart")
            except Exception:
                r["kind"] = "smart"
            r.pop("dsl", None)
        return {"albums": jsonable(rows)}
    return cached("albums", C.API_CACHE_TTL, build)


def api_tags(q):
    text = (q.get("q") or [""])[0].strip()
    limit = min(int((q.get("limit") or ["5000"])[0]), 10000)
    where, params = "", []
    if text:
        where, params = "WHERE tag LIKE ?", [f"%{text}%"]
    def build():
        rows = db().query(
            f"SELECT tag, COUNT(*) n, "
            f"SUM(CASE WHEN source='manual' THEN 1 ELSE 0 END) manual "
            f"FROM tags {where} GROUP BY tag ORDER BY n DESC, tag LIMIT ?",
            params + [limit])
        for r in rows:
            r["zh"] = TAG_ZH_DISPLAY.get(r["tag"], r["tag"])
        total = db().one(f"SELECT COUNT(DISTINCT tag) c FROM tags {where}", params)["c"]
        return {"tags": jsonable(rows), "total": total}
    return cached(f"tags:{text}:{limit}", 5, build)


# ── 标签 / 相簿 / 人物 ────────────────────────────────────────
def _norm_tag(raw):
    t = " ".join(str(raw or "").strip().split())
    if not t:
        return ""
    if len(t) > 40:
        t = t[:40].strip()
        from .vocab import ZH2TAG
        t = ZH2TAG.get(t, t)
    return t.lower() if t.isascii() else t


def _manual_album(aid):
    a = db().album(aid)
    if not a:
        return None
    try:
        kind = json.loads(a.get("dsl") or "{}").get("kind", "smart")
    except Exception:
        kind = "smart"
    return a if kind == "manual" else None


def api_photo_tags(_q, pid, body):
    action = (body.get("action") or "").strip()
    tag = _norm_tag(body.get("tag"))
    pid = int(pid)
    if not tag:
        return {"error": "标签不能为空"}
    if not db().one("SELECT id FROM photos WHERE id=?", (pid,)):
        return {"error": f"没有 id={pid} 的照片"}
    if action == "add":
        db().add_manual_tag(pid, tag)
    elif action == "remove":
        db().remove_tag(pid, tag)
    else:
        return {"error": "action 必须是 add 或 remove"}
    db().refresh_fts(pid)
    db().commit()
    bump_generation()
    return {"ok": True, "photo": api_photo({}, pid)}


def api_tag_delete(_q, body):
    tag = _norm_tag(body.get("tag"))
    if not tag:
        return {"error": "标签不能为空"}
    db().block_tag(tag)
    from .events import recompute
    recompute(db())
    from .albums import AlbumEngine
    AlbumEngine(db(), hub()).rebuild_all()
    db().commit()
    bump_generation()
    return {"ok": True, "tag": tag}


def api_create_album(_q, body):
    name = " ".join(str(body.get("name") or "").strip().split())
    if not name:
        return {"error": "相簿名不能为空"}
    if len(name) > 40:
        name = name[:40]
    if db().one("SELECT id FROM albums WHERE name=?", (name,)):
        return {"error": f"相簿「{name}」已存在"}
    aid = db().save_album(name, json.dumps({"kind": "manual"}, ensure_ascii=False))
    db().commit()
    bump_generation()
    return {"ok": True, "id": aid, "name": name}


def api_merge_tags_album(_q, body):
    name = " ".join(str(body.get("name") or "").strip().split())
    tags = [_norm_tag(t) for t in (body.get("tags") or []) if _norm_tag(t)]
    tags = list(dict.fromkeys(tags))[:80]
    if not name:
        return {"error": "相簿名不能为空"}
    if not tags:
        return {"error": "请选择至少一个标签"}
    r = api_create_album({}, {"name": name})
    if r.get("error"):
        return r
    qs = ",".join("?" * len(tags))
    rows = db().query(f"SELECT DISTINCT photo_id FROM tags WHERE tag IN ({qs})", tags)
    ids = [int(x["photo_id"]) for x in rows]
    if ids:
        db().album_add_many(r["id"], ids)
    db().commit()
    bump_generation()
    return {"ok": True, "id": r["id"], "name": name, "count": len(ids), "tags": tags}


def _photo_ids_under(folder):
    root = Path(folder).expanduser().resolve()
    rows = db().query("SELECT id, path FROM photos")
    out = []
    for r in rows:
        p = Path(r["path"])
        if not p.is_absolute():
            p = C.ROOT / p
        try:
            p = p.resolve()
        except OSError:
            p = p.absolute()
        if p == root or root in p.parents:
            out.append(int(r["id"]))
    return out


def api_import_album(_q, body):
    st = _state["index"]
    if st["running"]:
        return {"error": "已有扫描/导入任务在进行中"}
    raw = str(body.get("dir") or "").strip()
    if not raw:
        return {"error": "请选择文件夹"}
    folder = Path(raw).expanduser()
    try:
        folder = folder.resolve()
    except OSError:
        folder = folder.absolute()
    if not folder.is_dir():
        return {"error": f"文件夹不存在：{folder}"}
    name = " ".join(str(body.get("name") or "").strip().split())[:40]
    if not name:
        name = folder.name or "导入相簿"
    r = api_create_album({}, {"name": name})
    if r.get("error"):
        return r
    aid = int(r["id"])
    ids = _photo_ids_under(folder)
    if ids:
        db().album_add_many(aid, ids)
        db().commit()
        bump_generation()
    threading.Thread(target=_index_worker,
                     args=(str(folder), "import", aid), daemon=True).start()
    return {"ok": True, "id": aid, "name": name, "dir": str(folder),
            "count": len(ids), "indexing": True}


def api_album_update(_q, aid, body):
    a = _manual_album(aid)
    if not a:
        return {"error": "只能修改手动相簿"}
    action = (body.get("action") or "").strip()
    if action == "rename":
        name = " ".join(str(body.get("name") or "").strip().split())
        if not name:
            return {"error": "相簿名不能为空"}
        if len(name) > 40:
            name = name[:40]
        dup = db().one("SELECT id FROM albums WHERE name=? AND id<>?", (name, int(aid)))
        if dup:
            return {"error": f"相簿「{name}」已存在"}
        db().album_rename(aid, name)
        db().commit()
        bump_generation()
        return {"ok": True, "id": int(aid), "name": name}
    if action == "delete":
        db().album_delete(aid)
        db().commit()
        bump_generation()
        return {"ok": True, "id": int(aid)}
    return {"error": "action 必须是 rename 或 delete"}


def api_album_photos(_q, aid, body):
    a = _manual_album(aid)
    if not a:
        return {"error": "只能编辑手动相簿"}
    action = (body.get("action") or "add").strip()
    ids = body.get("ids") or []
    if isinstance(ids, (int, str)):
        ids = [ids]
    try:
        ids = [int(x) for x in ids]
    except Exception:
        return {"error": "ids 必须是照片 id 列表"}
    if not ids:
        return {"error": "没有选择照片"}
    if action == "add":
        db().album_add_many(aid, ids)
    elif action == "remove":
        db().album_remove_many(aid, ids)
    else:
        return {"error": "action 必须是 add 或 remove"}
    db().commit()
    bump_generation()
    return {"ok": True, "id": int(aid), "count": db().album_count(aid)}


def api_album_add_tags(_q, aid, body):
    a = _manual_album(aid)
    if not a:
        return {"error": "只能编辑手动相簿"}
    tags = [_norm_tag(t) for t in (body.get("tags") or []) if _norm_tag(t)]
    tags = list(dict.fromkeys(tags))[:80]
    if not tags:
        return {"error": "请选择至少一个标签"}
    qs = ",".join("?" * len(tags))
    rows = db().query(f"SELECT DISTINCT photo_id FROM tags WHERE tag IN ({qs})", tags)
    ids = [int(x["photo_id"]) for x in rows]
    if not ids:
        return {"error": "这些标签下还没有照片"}
    db().album_add_many(int(aid), ids)
    db().commit()
    bump_generation()
    return {"ok": True, "id": int(aid), "name": a["name"], "count": len(ids), "tags": tags}


def api_album_import(_q, aid, body):
    a = _manual_album(aid)
    if not a:
        return {"error": "只能编辑手动相簿"}
    st = _state["index"]
    if st["running"]:
        return {"error": "已有扫描/导入任务在进行中"}
    raw = str(body.get("dir") or "").strip()
    folder = Path(raw).expanduser()
    try:
        folder = folder.resolve()
    except OSError:
        folder = folder.absolute()
    if not raw or not folder.is_dir():
        return {"error": f"文件夹不存在：{folder}"}
    ids = _photo_ids_under(folder)
    if ids:
        db().album_add_many(int(aid), ids)
        db().commit()
        bump_generation()
    threading.Thread(target=_index_worker,
                     args=(str(folder), "import", int(aid)), daemon=True).start()
    return {"ok": True, "id": int(aid), "name": a["name"], "dir": str(folder),
            "count": len(ids), "indexing": True}


def api_firsts(q):
    person = (q.get("person") or ["宝宝"])[0]
    try:
        return jsonable({"person": person, "items": albums_engine().firsts(person)})
    except Exception as e:
        return {"person": person, "items": [], "error": str(e)}


def api_cleanup(_q):
    def build():
        from .quality import cleanup_suggestions
        items = cleanup_suggestions(db())
        for it in items:
            tags = [t for t, _ in db().tags_of([it["photo"]["id"]]).get(it["photo"]["id"], [])]
            it["photo"]["tags_zh"] = [TAG_ZH_DISPLAY.get(t, t) for t in tags[:4]]
            it["photo"].pop("ocr_text", None)
        return {"items": jsonable(items)}
    return cached("cleanup", 30, build)


def api_rename(q, pid, body):
    name = (body.get("name") or "").strip()
    if not name:
        return {"error": "名字不能为空"}
    db().rename_person(int(pid), name)
    db().commit()
    bump_generation()
    return {"ok": True, "id": int(pid), "name": name}


# ── 后台任务 ──────────────────────────────────────────────────
def _index_worker(folder, kind="index", album_id=None):
    st = _state["index"]
    st.update({"running": True, "kind": kind, "phase": "loading",
               "done": 0, "total": 0, "error": None, "started_at": time.time(),
               "album_id": album_id})

    def cb(done, total):
        st["done"], st["total"] = done, total
        if done and total:
            st["phase"] = "running"

    try:
        from .indexer import Indexer
        idx = Indexer()
        if folder:
            n = idx.index_folder(folder, progress=cb)
            st["found"] = n if isinstance(n, int) else 0
        else:
            idx.retag_all(progress=cb)
        st.update({"running": False, "phase": "done", "finished_at": time.time(),
                   "error": None})
        db().commit()
        from .events import recompute
        recompute(db())
        db().commit()
        from .albums import AlbumEngine
        AlbumEngine(db(), hub()).rebuild_all()
        db().commit()
        bump_generation()
    except Exception as e:
        st.update({"running": False, "phase": "error", "error": str(e),
                   "finished_at": time.time()})
        traceback.print_exc()


def api_start_index(body):
    st = _state["index"]
    if st["running"]:
        return {"error": "已有扫描/导入任务在进行中"}
    raw = str(body.get("dir") or "").strip()
    if not raw:
        return {"error": "请选择文件夹"}
    folder = Path(raw).expanduser()
    try:
        folder = folder.resolve()
    except OSError:
        folder = folder.absolute()
    if not folder.is_dir():
        return {"error": f"文件夹不存在：{folder}"}
    threading.Thread(target=_index_worker, args=(str(folder), "index"), daemon=True).start()
    return {"ok": True, "dir": str(folder), "kind": "index"}


def api_start_retag(body):
    st = _state["index"]
    if st["running"]:
        return {"error": "已有扫描/导入任务在进行中"}
    threading.Thread(target=_index_worker, args=(None, "retag"), daemon=True).start()
    return {"ok": True, "kind": "retag"}


def api_index_status(_q):
    return dict(_state["index"])


# ── 预热 ──────────────────────────────────────────────────────
def _prewarm():
    time.sleep(0.35)
    try:
        d = db()
        # 空库首次启动直接进入选择文件夹页，不预热模型
        if d.one("SELECT COUNT(*) c FROM photos")["c"] == 0:
            return
        # 标签体系升级后，用已有向量重打标，不重新解码图片
        if (d.get_meta("tag_vocab_version") != C.TAG_VOCAB_VERSION
                and not _state["index"]["running"]):
            threading.Thread(target=_index_worker, args=(None, "retag"), daemon=True).start()
    except Exception:
        traceback.print_exc()
    try:
        search_engine()          # 把文本塔和 tokenizer 提前加载好
    except Exception:
        traceback.print_exc()
    time.sleep(0.8)
    try:
        from .thumbs import warm_library
        warm_library(db(), C.THUMB_PREWARM_SIZES)
    except Exception:
        pass


# ── HTTP ──────────────────────────────────────────────────────
ROUTES_GET = {
    "/api/health": api_health,
    "/api/setup": api_setup_status,
    "/api/stats": api_stats,
    "/api/search": api_search,
    "/api/photos": api_photos,
    "/api/events": api_events,
    "/api/memories": api_memories,
    "/api/bursts": api_bursts,
    "/api/persons": api_persons,
    "/api/albums": api_albums,
    "/api/tags": api_tags,
    "/api/firsts": api_firsts,
    "/api/cleanup": api_cleanup,
    "/api/index/status": api_index_status,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "GardenAlbum/0.2"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # 别把访问日志糊满终端
        if os.environ.get("GARDEN_HTTP_LOG"):
            super().log_message(fmt, *args)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, If-None-Match")
        self.send_header("Access-Control-Expose-Headers", "ETag")

    def _gzip_ok(self):
        return "gzip" in (self.headers.get("Accept-Encoding") or "").lower()

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8",
              cache=None, etag=None, compress=True):
        headers = [("Content-Type", ctype)]
        if compress and self._gzip_ok() and len(body) > 1024:
            body = gzip.compress(body, 5)
            headers.append(("Content-Encoding", "gzip"))
            headers.append(("Vary", "Accept-Encoding"))
        headers.append(("Content-Length", str(len(body))))
        if cache:
            headers.append(("Cache-Control", cache))
        if etag:
            headers.append(("ETag", etag))
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self._cors()
        self.end_headers()
        if code != 304:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _json(self, obj, code=200, cache="no-store"):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        etag = '"' + hashlib.md5(body).hexdigest()[:20] + '"'
        self._send(code, body, cache=cache, etag=etag)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _serve_thumb(self, q):
        pid = int((q.get("id") or ["0"])[0])
        size = max(64, min(int((q.get("size") or ["320"])[0]), 2048))
        row = db().one("SELECT path FROM photos WHERE id=?", (pid,))
        if not row:
            return self._json({"error": "not found"}, 404)
        from .thumbs import thumb_bytes
        data = thumb_bytes(row["path"], size)
        etag = '"' + hashlib.md5(data).hexdigest()[:16] + '"'
        return self._send(200, data, "image/jpeg",
                          cache="public, max-age=31536000, immutable",
                          etag=etag, compress=False)

    def _serve_face(self, q):
        fid = int((q.get("id") or ["0"])[0])
        size = max(48, min(int((q.get("size") or ["160"])[0]), 512))
        row = db().one("SELECT f.photo_id, f.bbox, p.path FROM faces f "
                       "JOIN photos p ON p.id=f.photo_id WHERE f.id=?", (fid,))
        if not row:
            return self._json({"error": "not found"}, 404)
        bbox = _bbox(row["bbox"]) or [0, 0, 1, 1]
        from .thumbs import avatar_bytes
        data = avatar_bytes(row["path"], bbox, size)
        etag = '"' + hashlib.md5(data).hexdigest()[:16] + '"'
        return self._send(200, data, "image/jpeg",
                          cache="public, max-age=31536000, immutable",
                          etag=etag, compress=False)

    def _serve_original(self, q):
        pid = int((q.get("id") or ["0"])[0])
        row = db().one("SELECT path FROM photos WHERE id=?", (pid,))
        if not row:
            return self._json({"error": "not found"}, 404)
        p = Path(row["path"])
        if not p.exists() or p.suffix.lower() not in C.IMAGE_EXTS:
            return self._json({"error": "not found"}, 404)
        try:
            st = p.stat()
            etag = f'"{st.st_size:x}-{int(st.st_mtime):x}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            data = p.read_bytes()
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            return self._send(200, data, ctype,
                              cache="public, max-age=31536000, immutable",
                              etag=etag, compress=False)
        except OSError:
            return self._json({"error": "read failed"}, 500)

    def do_GET(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        try:
            if path == "/api/thumb":
                return self._serve_thumb(q)
            if path == "/api/face":
                return self._serve_face(q)
            if path == "/api/original":
                return self._serve_original(q)
            if path.startswith("/api/photo/"):
                return self._json(api_photo(q, path.rsplit("/", 1)[1]))
            fn = ROUTES_GET.get(path)
            if not fn:
                return self._json({"error": f"未知接口 {path}"}, 404)
            return self._json(fn(q))
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        path = u.path
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except Exception:
            body = {}
        try:
            if path == "/api/setup":
                r = api_setup(body)
                return self._json(r, 200 if "error" not in r else 400)
            if path == "/api/index/start":
                r = api_start_index(body)
                return self._json(r, 200 if "error" not in r else 409)
            if path == "/api/retag/start":
                r = api_start_retag(body)
                return self._json(r, 200 if "error" not in r else 409)
            if path == "/api/albums":
                r = api_create_album({}, body)
                return self._json(r, 200 if "error" not in r else 409)
            if path == "/api/albums/merge":
                r = api_merge_tags_album({}, body)
                return self._json(r, 200 if "error" not in r else 400)
            if path == "/api/albums/import":
                r = api_import_album({}, body)
                return self._json(r, 200 if "error" not in r else 400)
            if path == "/api/tags/delete":
                r = api_tag_delete({}, body)
                return self._json(r, 200 if "error" not in r else 400)
            if path.startswith("/api/photo/") and path.endswith("/tags"):
                pid = path.split("/")[3]
                r = api_photo_tags({}, pid, body)
                return self._json(r, 200 if "error" not in r else 400)
            if path.startswith("/api/album/"):
                parts = [x for x in path.split("/") if x]
                aid = parts[2]
                if len(parts) >= 4 and parts[3] == "photos":
                    r = api_album_photos({}, aid, body)
                elif len(parts) >= 4 and parts[3] == "tags":
                    r = api_album_add_tags({}, aid, body)
                elif len(parts) >= 4 and parts[3] == "import":
                    r = api_album_import({}, aid, body)
                else:
                    r = api_album_update({}, aid, body)
                return self._json(r, 200 if "error" not in r else 400)
            if path.startswith("/api/person/"):
                return self._json(api_rename({}, path.rsplit("/", 1)[1], body))
            return self._json({"error": f"未知接口 {path}"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(port: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    actual = srv.server_address[1]
    print(f"PORT {actual}", flush=True)
    print(f"相册 API: http://127.0.0.1:{actual}/api/health", flush=True)
    threading.Thread(target=_prewarm, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    serve(ap.parse_args().port)
