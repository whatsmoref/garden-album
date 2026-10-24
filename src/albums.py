# -*- coding: utf-8 -*-
"""F. 语义相册：DSL + 缓存文本向量，新照片入库一次点积自动收录（微秒级）"""
import json
from .models import ModelHub
from .vocab import TAG_ZH_DISPLAY
BUILTINS = [
    {"name": "所有票据", "tags_any": ["invoice", "receipt", "ticket", "boarding pass",
                                    "banknote", "id card", "credit card", "business card"]},
    {"name": "风景旅行", "tags_any": ["beach", "seaside", "coast", "mountain", "snowy mountain",
                                     "lake", "river", "forest", "waterfall", "sunset", "sunrise",
                                     "starry sky", "city skyline", "park", "island", "desert",
                                     "grassland", "hot spring", "hiking trail", "camping tent"]},
    {"name": "美食时刻", "tags_any": ["food", "meal", "breakfast", "lunch", "dinner", "dessert",
                                     "cake", "birthday cake", "hotpot", "barbecue", "noodles",
                                     "coffee", "milk tea", "fruit", "pizza", "sushi"]},
    {"name": "高分精选", "min_aesthetic": 5.8, "min_technical": 4.5},
    {"name": "笑脸精选", "min_avg_smile": 0.55, "min_aesthetic": 6.0},
    {"name": "宝宝第一次", "kind": "firsts", "person": "宝宝"},
]
OLD_BUILTINS = {"所有发票", "风景美图", "毛孩子", "家人朋友"}


class AlbumEngine:
    def __init__(self, db, hub=None):
        self.db = db
        self.hub = hub or ModelHub.get()
        for a in BUILTINS:
            row = self.db.one("SELECT dsl FROM albums WHERE name=?", (a["name"],))
            try:
                is_manual = row and json.loads(row["dsl"] or "{}").get("kind") == "manual"
            except Exception:
                is_manual = False
            if not is_manual:
                self.db.save_album(a["name"], json.dumps(a, ensure_ascii=False))
        # 迁移：旧版本的语义相册已由更通用的新版替代，避免侧栏出现重复项
        for name in OLD_BUILTINS:
            row = self.db.one("SELECT id FROM albums WHERE name=?", (name,))
            if row:
                self.db.execute("DELETE FROM album_photos WHERE album_id=?", (row["id"],))
                self.db.execute("DELETE FROM albums WHERE id=?", (row["id"],))
        self.db.commit()
        self.albums = []
        for r in self.db.albums():
            spec = json.loads(r["dsl"]); spec["_id"] = r["id"]
            if spec.get("kind") == "manual":
                continue
            if spec.get("clip_text"):
                spec["_vec"] = self.hub.embed_text(spec["clip_text"])
            self.albums.append(spec)
    def add_photo(self, pid, row, tags, vec):
        tagset = {t for t, _ in tags}
        for alb in self.albums:
            if alb.get("kind") in ("firsts", "manual"): continue
            score = self._score(alb, row, tagset, vec)
            if score is not None:
                self.db.album_add(alb["_id"], pid, round(score, 3))

    def rebuild_all(self):
        """标签体系更新后，用库里的向量重算全部语义相册（秒级）。"""
        rows = self.db.query("SELECT * FROM photos")
        vecs = dict(zip(self.db.vectors.ids, self.db.vectors.vecs))
        for alb in self.albums:
            if alb.get("kind") in ("firsts", "manual"):
                continue
            self.db.execute("DELETE FROM album_photos WHERE album_id=?", (alb["_id"],))
        tmap = self.db.tags_of([r["id"] for r in rows])
        n = 0
        for row in rows:
            tags = tmap.get(row["id"], [])
            tagset = {t for t, _ in tags}
            vec = vecs.get(row["id"])
            for alb in self.albums:
                if alb.get("kind") in ("firsts", "manual"):
                    continue
                score = self._score(alb, row, tagset, vec)
                if score is not None:
                    self.db.album_add(alb["_id"], row["id"], round(score, 3))
                    n += 1
        self.db.commit()
        return n

    def _score(self, alb, row, tagset, vec):
        if alb.get("kind") == "manual": return None
        if "tags_any" in alb and not (set(alb["tags_any"]) & tagset): return None
        if "tags_all" in alb and not set(alb["tags_all"]) <= tagset: return None
        if alb.get("min_avg_smile") is not None and \
           (row.get("avg_smile") is None or row["avg_smile"] < alb["min_avg_smile"]): return None
        if alb.get("min_aesthetic") is not None and \
           (row.get("aesthetic") is None or row["aesthetic"] < alb["min_aesthetic"]): return None
        if alb.get("min_technical") is not None and \
           (row.get("technical") is None or row["technical"] < alb["min_technical"]): return None
        if alb.get("min_people") is not None:
            people = (row.get("known_face_count") or 0) + (row.get("unknown_face_count") or 0)
            if people < alb["min_people"]: return None
        v = 1.0
        if "_vec" in alb:
            if vec is None: return None
            v = float(vec @ alb["_vec"])
            if v < alb.get("min_clip", 0.2): return None
        return v
    def firsts(self, person="宝宝"):
        """按标签分组取最早一张：第一次笑/第一次走路/第一次去海边…"""
        pid = self.db.person_id_by_name(person)
        if pid is None: return []
        rows = self.db.query("SELECT p.* FROM photos p JOIN faces f ON f.photo_id=p.id "
                             "WHERE f.person_id=? ORDER BY p.taken_at", (pid,))
        tmap = self.db.tags_of([r["id"] for r in rows])
        seen, out = set(), []
        skip = {"man", "woman", "child", "smiling", "selfie", "portrait"}
        for r in rows:
            for t, _ in tmap.get(r["id"], [])[:4]:
                if t in seen or t in skip: continue
                seen.add(t)
                out.append({"title": f"第一次{TAG_ZH_DISPLAY.get(t, t)}", "photo": r})
                break
        return out
