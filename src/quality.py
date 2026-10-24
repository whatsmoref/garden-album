# -*- coding: utf-8 -*-
"""智能择优与清理：只给建议和置信度，绝不自动删除"""
from . import config as C
def burst_score(r):
    """连拍择优：美学0.35 + 技术0.15 + 清晰度0.15 + 曝光0.15 + 人脸质量0.15 + 微笑0.10"""
    aes = r.get("aesthetic") or 5.0
    tec = r.get("technical") or 5.0
    shp = r.get("sharpness") or 50.0
    exp = r.get("exposure") or 70.0
    if r.get("known_face_count") or r.get("unknown_face_count"):
        fq = min(1.0, (r.get("best_face_area") or 0.05) * 10)
    else:
        fq = 0.3
    smile = r.get("avg_smile")
    sm = smile if smile is not None else 0.4
    return (C.W_AESTHETIC * aes / 10 + C.W_TECH * tec / 10 + C.W_SHARP * shp / 100 +
            C.W_EXPO * exp / 100 + C.W_FACE * fq + C.W_SMILE * sm)
def recompute_burst_best(db):
    for b in db.query("SELECT id FROM bursts"):
        rows = db.query("SELECT * FROM photos WHERE burst_id=? ORDER BY taken_at", (b["id"],))
        if len(rows) < 2: continue
        best = max(rows, key=burst_score)["id"]
        for r in rows:
            db.execute("UPDATE photos SET burst_best=? WHERE id=?", (int(r["id"] == best), r["id"]))
def cleanup_suggestions(db):
    sug = []
    # 1) 完全重复（同 phash 同尺寸）
    rows = db.query("SELECT id, path, phash, width, height FROM photos WHERE phash IS NOT NULL ORDER BY taken_at")
    groups = {}
    for r in rows:
        groups.setdefault((r["phash"], r["width"], r["height"]), []).append(r)
    for g in groups.values():
        if len(g) > 1:
            for r in g[1:]:
                sug.append({"photo": r, "type": "duplicate", "confidence": 0.95,
                            "reason": f"与 {g[0]['path']} 内容完全重复"})
    # 2) 连拍冗余（保留最优）
    for b in db.query("SELECT id FROM bursts WHERE photo_count > 1"):
        rows = db.query("SELECT * FROM photos WHERE burst_id=? ORDER BY id", (b["id"],))
        best = next((r for r in rows if r["burst_best"]), rows[0])
        for r in rows:
            if r["id"] == best["id"]: continue
            gap = burst_score(best) - burst_score(r)
            sug.append({"photo": r, "type": "burst_redundant",
                        "confidence": round(min(0.9, 0.5 + gap * 0.5), 2),
                        "reason": f"连拍冗余，更优为 {best['path']}"})
    # 3) 模糊且无人脸
    for r in db.query("SELECT * FROM photos WHERE sharpness IS NOT NULL AND sharpness < 25 "
                      "AND known_face_count=0 AND unknown_face_count=0"):
        sug.append({"photo": r, "type": "blurry", "confidence": 0.65, "reason": "画面模糊且无人脸"})
    # 4) 双低分且无人脸
    for r in db.query("SELECT * FROM photos WHERE aesthetic IS NOT NULL AND aesthetic < 3.5 "
                      "AND technical < 3.5 AND known_face_count=0 AND unknown_face_count=0"):
        sug.append({"photo": r, "type": "low_quality", "confidence": 0.6,
                    "reason": "美学分与技术分均低且无人脸"})
    sug.sort(key=lambda s: -s["confidence"])
    return sug
