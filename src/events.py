# -*- coding: utf-8 -*-
"""D. 事件层：时间/GPS 聚类 + 跨设备三方验证合并 + 模板命名 + 连拍分组"""
from datetime import datetime, timedelta, date
from collections import Counter
import numpy as np
from . import config as C
from .vocab import TAG_ZH_DISPLAY
CITIES = {  # 离线逆地理编码（可换 reverse_geocoder 扩充到全球）
    "北京": (39.90, 116.41), "上海": (31.23, 121.47), "广州": (23.13, 113.26), "深圳": (22.54, 114.06),
    "杭州": (30.27, 120.16), "南京": (32.06, 118.80), "成都": (30.57, 104.07), "重庆": (29.56, 106.55),
    "武汉": (30.59, 114.31), "西安": (34.34, 108.94), "长沙": (28.23, 112.94), "郑州": (34.75, 113.62),
    "青岛": (36.07, 120.38), "厦门": (24.48, 118.09), "昆明": (24.88, 102.83), "贵阳": (26.65, 106.63),
    "拉萨": (29.65, 91.13), "乌鲁木齐": (43.83, 87.62), "兰州": (36.06, 103.83), "西宁": (36.62, 101.78),
    "哈尔滨": (45.80, 126.53), "沈阳": (41.80, 123.43), "长春": (43.82, 125.32), "大连": (38.91, 121.61),
    "天津": (39.13, 117.20), "苏州": (31.30, 120.58), "合肥": (31.82, 117.23), "福州": (26.07, 119.30),
    "南宁": (22.82, 108.32), "海口": (20.04, 110.32), "三亚": (18.25, 109.51), "香港": (22.32, 114.17),
    "台北": (25.03, 121.57), "呼和浩特": (40.84, 111.75), "银川": (38.49, 106.23), "石家庄": (38.04, 114.51),
}
def haversine(lat1, lon1, lat2, lon2):
    if lat1 is None or lat2 is None: return 0.0
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))
def nearest_city(lat, lon):
    if lat is None or lon is None: return ""
    best, bd = "", 150.0
    for name, (la, lo) in CITIES.items():
        d = haversine(lat, lon, la, lo)
        if d < bd: best, bd = name, d
    return best
def phash_hamming(a, b):
    try: return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception: return 64
class _UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)
def _segment(photos):
    groups, cur = [], []
    for p in photos:
        if cur:
            dt = (p["dt"] - cur[-1]["dt"]).total_seconds() / 3600.0
            dist = haversine(cur[-1]["gps_lat"], cur[-1]["gps_lon"], p["gps_lat"], p["gps_lon"])
            if dt > C.EVENT_GAP_H or (dt > C.EVENT_GPS_GAP_H and dist > C.EVENT_GPS_KM) or dt < 0:
                groups.append(cur); cur = []
        cur.append(p)
    if cur: groups.append(cur)
    return groups
def _merge_cross_device(db, groups):
    """三方验证：时间重叠 + 共享人物 + phash 近重复 → 合并"""
    if len(groups) < 2: return groups
    persons = {}
    for r in db.query("SELECT photo_id, person_id FROM faces"):
        persons.setdefault(r["photo_id"], set()).add(r["person_id"])
    uf = _UF(len(groups))
    rng = [(g[0]["dt"], g[-1]["dt"]) for g in groups]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if rng[i][1] < rng[j][0] or rng[j][1] < rng[i][0]: continue   # 时间不重叠
            devs = {p["device"] for p in groups[i]} | {p["device"] for p in groups[j]}
            if len(devs) < 2: continue                                     # 无跨设备
            pi = set().union(*(persons.get(p["id"], set()) for p in groups[i]))
            pj = set().union(*(persons.get(p["id"], set()) for p in groups[j]))
            if not (pi & pj): continue                                     # 无共享人物
            near = any(phash_hamming(a["phash"], b["phash"]) <= C.MERGE_PHASH_HAMMING
                       for a in groups[i][:200] if a["phash"]
                       for b in groups[j][:200] if b["phash"])
            if near: uf.union(i, j)
    merged = {}
    for gi, g in enumerate(groups):
        merged.setdefault(uf.find(gi), []).extend(g)
    return sorted((sorted(v, key=lambda p: p["dt"]) for v in merged.values()),
                  key=lambda g: g[0]["dt"])
def _make_title(db, group, city):
    t0 = group[0]["taken_at"]; month = int(t0[5:7])
    tagc = Counter()
    for tags in db.tags_of([p["id"] for p in group]).values():
        for t, _ in tags: tagc[t] += 1
    zh = "/".join(TAG_ZH_DISPLAY.get(t, t) for t, _ in tagc.most_common(2))
    # 特殊主题必须达到组内占比，否则一张误标的圣诞树会把整段成都公园命名成圣诞。
    need = max(2, int(len(group) * 0.25))
    special = ""
    if tagc["birthday party"] >= need or tagc["birthday cake"] >= need:
        special = "🎂生日"
    elif tagc["wedding"] >= need or tagc["wedding dress"] >= need:
        special = "💍婚礼"
    elif tagc["christmas tree"] >= need:
        special = "🎄圣诞"
    elif tagc["fireworks"] >= need:
        special = "🎆烟花"
    return "·".join(x for x in (city, f"{month}月", zh) if x) + special, tagc
def recompute(db):
    """全量重算事件 + 连拍（纯内存/SQL，万张秒级）"""
    photos = db.query("SELECT id, taken_at, gps_lat, gps_lon, phash, device FROM photos "
                      "WHERE taken_at IS NOT NULL ORDER BY taken_at")
    for p in photos:
        p["dt"] = datetime.strptime(p["taken_at"], "%Y-%m-%d %H:%M:%S")
    groups = _merge_cross_device(db, _segment(photos)) if photos else []
    db.execute("DELETE FROM events"); db.execute("DELETE FROM bursts")
    db.execute("UPDATE photos SET event_id=NULL, burst_id=NULL, burst_best=0")
    for g in groups:
        lats = [p["gps_lat"] for p in g if p["gps_lat"]]
        lons = [p["gps_lon"] for p in g if p["gps_lon"]]
        city = nearest_city(float(np.mean(lats)), float(np.mean(lons))) if lats else ""
        title, tagc = _make_title(db, g, city)
        eid = db.execute(
            "INSERT INTO events(start,end,title,city,photo_count,device_count,top_tags) VALUES(?,?,?,?,?,?,?)",
            (g[0]["taken_at"], g[-1]["taken_at"], title, city, len(g),
             len({p["device"] for p in g}), ",".join(t for t, _ in tagc.most_common(5)))).lastrowid
        for p in g:
            db.execute("UPDATE photos SET event_id=? WHERE id=?", (eid, p["id"]))
    _rebuild_bursts(db, groups)
def _rebuild_bursts(db, groups):
    from .quality import burst_score    # 延迟导入避免环
    for g in groups:
        bursts, cur = [], []
        for p in g:
            if cur and ((p["dt"] - cur[-1]["dt"]).total_seconds() > C.BURST_DT_S or
                        phash_hamming(cur[-1]["phash"] or "0", p["phash"] or "0") > C.BURST_PHASH_HAMMING):
                if len(cur) >= 2: bursts.append(cur)
                cur = []
            cur.append(p)
        if len(cur) >= 2: bursts.append(cur)
        for b in bursts:
            bid = db.execute("INSERT INTO bursts(photo_count) VALUES(?)", (len(b),)).lastrowid
            qs = ",".join("?" * len(b))
            full = db.query(f"SELECT * FROM photos WHERE id IN ({qs})", [p["id"] for p in b])
            best = max(full, key=burst_score)["id"]
            for r in full:
                db.execute("UPDATE photos SET burst_id=?, burst_best=? WHERE id=?",
                           (bid, int(r["id"] == best), r["id"]))
