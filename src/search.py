# -*- coding: utf-8 -*-
"""查询路径：SQL 预过滤 + CLIP 向量 ANN + BM25 → RRF 融合 → 后过滤（全程 <100ms）"""
import re
from .parser import QueryParser, RELATION
def rrf(rank_lists, k=60):
    out = {}
    for rl in rank_lists:
        for r, pid in enumerate(rl):
            out[pid] = out.get(pid, 0.0) + 1.0 / (k + r + 1)
    return out
class SearchEngine:
    def __init__(self, db, hub=None):
        self.db = db
        from .models import ModelHub
        self.hub = hub or ModelHub.get()
        names = [r["name"] for r in db.persons() if not re.match(r"^人物\d+$", r["name"])]
        self.parser = QueryParser(names)
    def search(self, text, topk=60):
        m = re.search(r"(?:上次|最近一次|最近)(?:见|遇到|碰到|见到)(.+?)(?:是什么时候|啥时候|的时间)?$", text.strip())
        if m:
            return self.last_meeting(m.group(1).strip())
        dsl, chips = self.parser.parse(text)
        # 手动标签可能不在词典里：整句正好是库内标签时，直接按标签过滤。
        raw = text.strip()
        if raw and raw.lower() not in {str(c[1]).lower() for c in chips if c[0] == "tag"}:
            hit = self.db.one(
                "SELECT tag FROM tags WHERE tag=? OR LOWER(tag)=? LIMIT 1",
                (raw, raw.lower()))
            if hit:
                # 整句正好是库内标签：只按这个标签查，避免「旅行测试」被拆成
                # 「旅行」+「测试」后与标签本身互相冲突。
                dsl = {"tags": {"all": [hit["tag"]]}}
                chips = [("tag", raw)]
        return {"dsl": dsl, "chips": chips, "results": self._search_dsl(dsl, topk)}
    def _filter_sql(self, dsl):
        w, p = ["1=1"], []
        t = dsl.get("time")
        if t:
            if t.get("gte"): w.append("taken_at >= ?"); p.append(t["gte"])
            if t.get("lte"): w.append("taken_at <= ?"); p.append(t["lte"])
        for tag in dsl.get("tags", {}).get("all", []):
            w.append("id IN (SELECT photo_id FROM tags WHERE tag=?)"); p.append(tag)
        for tag in dsl.get("tags", {}).get("none", []):
            w.append("id NOT IN (SELECT photo_id FROM tags WHERE tag=?)"); p.append(tag)
        pe = dsl.get("people", {})
        for name in pe.get("all", []):
            pid = self.db.person_id_by_name(name) or \
                  self.db.person_id_by_name(RELATION.get(name, name))
            if pid is None: w.append("0")        # 人物未命名 → 空结果
            else:
                w.append("id IN (SELECT photo_id FROM faces WHERE person_id=?)"); p.append(pid)
        if "unknown_count" in pe:
            w.append("unknown_face_count = ?"); p.append(pe["unknown_count"])
        c = pe.get("count")
        if c:
            col = "known_face_count + unknown_face_count"
            for op, sql in (("gte", ">="), ("lte", "<="), ("eq", "=")):
                if c.get(op) is not None:
                    w.append(f"{col} {sql} ?"); p.append(c[op])
        em = dsl.get("emotion", {})
        sm = em.get("smile")
        if sm:
            w.append("avg_smile IS NOT NULL")
            if "gte" in sm: w.append("avg_smile >= ?"); p.append(sm["gte"])
            if "lte" in sm: w.append("avg_smile <= ?"); p.append(sm["lte"])
        if em.get("eyes_open"):
            w.append("has_closed_eyes = 0")
        q = dsl.get("quality", {})
        for col in ("sharpness", "aesthetic", "exposure"):
            rng = q.get(col)
            if not rng: continue
            for op, sql in (("gte", ">="), ("lte", "<="), ("eq", "=")):
                if rng.get(op) is not None:
                    w.append(f"{col} {sql} ?"); p.append(rng[op])
        if dsl.get("exclude_screenshot"):
            w.append("is_screenshot = 0")
        return " AND ".join(w), p
    def _search_dsl(self, dsl, topk):
        where, params = self._filter_sql(dsl)
        cand = self.db.query(f"SELECT * FROM photos WHERE {where}", params)
        if not cand: return []
        allow = {r["id"] for r in cand}
        from . import config as C
        rankings = []
        has_text = bool(dsl.get("clip_text") or dsl.get("keywords"))
        structured = bool(dsl.get("tags") or dsl.get("people") or dsl.get("time")
                          or dsl.get("quality") or dsl.get("emotion")
                          or dsl.get("exclude_screenshot"))
        if dsl.get("clip_text"):                       # 开放词汇 → CLIP 文本向量 → ANN
            qv = self.hub.embed_text(dsl["clip_text"])
            hits = self.db.vectors.search(qv, k=600, allow=allow)
            if hits and not structured:
                top = hits[0][1]
                fifth = hits[min(4, len(hits) - 1)][1]
                if top < C.SEARCH_CLIP_MIN_TOP or top - fifth < C.SEARCH_CLIP_MIN_SPREAD:
                    hits = []
                else:
                    hits = [(pid, sim) for pid, sim in hits
                            if sim >= top - C.SEARCH_CLIP_WINDOW][:C.SEARCH_CLIP_LIMIT]
            if hits:
                rankings.append([pid for pid, _ in hits])
        if dsl.get("keywords"):                        # 精确关键词 → BM25
            rankings.append([pid for pid, _ in self.db.fts_search(dsl["keywords"], limit=2000, allow=allow)])
        if rankings:                                   # RRF 融合
            fused = rrf(rankings)
            order = sorted(allow, key=lambda i: -fused.get(i, 0))[:topk]
        elif has_text and not structured:              # 语义查询没有可信结果，不要回退成整库
            return []
        else:                                          # 纯结构化 → 时间倒序
            cand.sort(key=lambda r: r["taken_at"] or "", reverse=True)
            order = [r["id"] for r in cand[:topk]]
        byid = {r["id"]: r for r in cand}
        tmap = self.db.tags_of(order)
        for pid in order:
            byid[pid]["tags"] = [t for t, _ in tmap.get(pid, [])]
        return [byid[pid] for pid in order if pid in byid]
    def last_meeting(self, name):
        """关系图谱：上次见某人是什么时候（纯 SQL，毫秒级）"""
        pid = self.db.person_id_by_name(RELATION.get(name, name))
        if pid is None:
            return {"error": f"未找到人物「{name}」，请先用 person 命令命名"}
        me = self.db.person_id_by_name("我")
        sql = "SELECT p.* FROM photos p WHERE p.id IN (SELECT photo_id FROM faces WHERE person_id=?)"
        params = [pid]
        if me is not None and me != pid:
            sql += " AND p.id IN (SELECT photo_id FROM faces WHERE person_id=?)"
            params.append(me)
        row = self.db.one(sql + " ORDER BY p.taken_at DESC LIMIT 1", params)
        if not row:
            return {"error": f"没有找到和「{name}」的合照"}
        total = self.db.one("SELECT COUNT(DISTINCT photo_id) c FROM faces WHERE person_id=?", (pid,))["c"]
        return {"person": name, "last_time": row["taken_at"], "total_photos": total, "photo": row}
