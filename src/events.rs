//! D. 事件层：时间/GPS 聚类 + 跨设备三方验证合并 + 模板命名 + 连拍分组

use anyhow::Result;
use chrono::{Datelike, NaiveDateTime};
use std::collections::{HashMap, HashSet};

use crate::config as C;
use crate::db::{DB, EventRow, Photo};
use crate::metadata::phash_hamming;
use crate::quality::burst_score;
use crate::vocab::TAG_ZH_DISPLAY;

/// 离线逆地理编码（国内主要城市；可换 reverse_geocoder 扩到全球）
pub static CITIES: &[(&str, f64, f64)] = &[
    ("北京", 39.90, 116.41), ("上海", 31.23, 121.47), ("广州", 23.13, 113.26),
    ("深圳", 22.54, 114.06), ("杭州", 30.27, 120.16), ("南京", 32.06, 118.80),
    ("成都", 30.57, 104.07), ("重庆", 29.56, 106.55), ("武汉", 30.59, 114.31),
    ("西安", 34.34, 108.94), ("长沙", 28.23, 112.94), ("郑州", 34.75, 113.62),
    ("青岛", 36.07, 120.38), ("厦门", 24.48, 118.09), ("昆明", 24.88, 102.83),
    ("贵阳", 26.65, 106.63), ("拉萨", 29.65, 91.13), ("乌鲁木齐", 43.83, 87.62),
    ("兰州", 36.06, 103.83), ("西宁", 36.62, 101.78), ("哈尔滨", 45.80, 126.53),
    ("沈阳", 41.80, 123.43), ("长春", 43.82, 125.32), ("大连", 38.91, 121.61),
    ("天津", 39.13, 117.20), ("苏州", 31.30, 120.58), ("合肥", 31.82, 117.23),
    ("福州", 26.07, 119.30), ("南宁", 22.82, 108.32), ("海口", 20.04, 110.32),
    ("三亚", 18.25, 109.51), ("香港", 22.32, 114.17), ("台北", 25.03, 121.57),
    ("呼和浩特", 40.84, 111.75), ("银川", 38.49, 106.23), ("石家庄", 38.04, 114.51),
];

pub fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6371.0;
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = (lat2 - lat1).to_radians();
    let dl = (lon2 - lon1).to_radians();
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * R * a.sqrt().asin()
}

pub fn nearest_city(lat: f64, lon: f64) -> String {
    let mut best = String::new();
    let mut bd = 150.0;
    for (name, la, lo) in CITIES {
        let d = haversine(lat, lon, *la, *lo);
        if d < bd {
            best = name.to_string();
            bd = d;
        }
    }
    best
}

/// 参与事件切分的最小信息
#[derive(Clone)]
struct Seg {
    id: i64,
    dt: NaiveDateTime,
    gps: Option<(f64, f64)>,
    device: String,
    phash: String,
}

/// 纯内存/SQL，万张秒级
pub fn recompute(db: &DB) -> Result<()> {
    let photos = db.all_photos()?;
    let mut segs: Vec<Seg> = Vec::with_capacity(photos.len());
    for p in &photos {
        let Some(dt) = p.taken_at.as_deref().and_then(parse_dt) else {
            continue;
        };
        segs.push(Seg {
            id: p.id,
            dt,
            gps: match (p.gps_lat, p.gps_lon) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            },
            device: p.device.clone().unwrap_or_default(),
            phash: p.phash.clone().unwrap_or_default(),
        });
    }
    db.clear_events()?;
    db.reset_event_burst()?;
    if segs.is_empty() {
        return Ok(());
    }
    segs.sort_by(|a, b| a.dt.cmp(&b.dt).then(a.id.cmp(&b.id)));
    let persons = db.photo_persons()?;
    let groups = merge_cross_device(&segs, &persons);

    for g in &groups {
        let ids: Vec<i64> = g.iter().map(|s| s.id).collect();
        let lats: Vec<f64> = g.iter().filter_map(|s| s.gps.map(|x| x.0)).collect();
        let lons: Vec<f64> = g.iter().filter_map(|s| s.gps.map(|x| x.1)).collect();
        let city = if lats.is_empty() {
            String::new()
        } else {
            nearest_city(
                lats.iter().sum::<f64>() / lats.len() as f64,
                lons.iter().sum::<f64>() / lons.len() as f64,
            )
        };
        let tagc = tag_histogram(db, &ids);
        let title = make_title(g[0].dt, &city, &tagc);
        let mut devices: Vec<&str> = g.iter().map(|s| s.device.as_str()).filter(|s| !s.is_empty()).collect();
        devices.sort_unstable();
        devices.dedup();
        let row = EventRow {
            id: 0,
            start: fmt_dt(g[0].dt),
            end: fmt_dt(g[g.len() - 1].dt),
            title,
            city,
            photo_count: g.len() as i64,
            device_count: devices.len() as i64,
            top_tags: tag_histogram_top(&tagc, 5).join(","),
        };
        let eid = db.insert_event(&row)?;
        for s in g {
            db.set_photo_event(s.id, eid)?;
        }
    }
    rebuild_bursts(db, &groups)
}

/// 时间切分：间隔 > 8h，或（> 2h 且跨 > 200km），或时间倒流
fn segment(segs: &[Seg]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for (i, s) in segs.iter().enumerate() {
        if let Some(&prev) = cur.last() {
            let p = &segs[prev];
            let dt = (s.dt - p.dt).num_seconds() as f64 / 3600.0;
            let dist = match (p.gps, s.gps) {
                (Some((a1, b1)), Some((a2, b2))) => haversine(a1, b1, a2, b2),
                _ => 0.0,
            };
            if dt > C::EVENT_GAP_H
                || (dt > C::EVENT_GPS_GAP_H && dist > C::EVENT_GPS_KM)
                || dt < 0.0
            {
                groups.push(std::mem::take(&mut cur));
            }
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

struct Dsu {
    p: Vec<usize>,
}
impl Dsu {
    fn new(n: usize) -> Self {
        Self { p: (0..n).collect() }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut x = x;
        while self.p[x] != x {
            self.p[x] = self.p[self.p[x]];
            x = self.p[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.p[ra] = rb;
        }
    }
}

/// 三方验证：时间重叠 + 跨设备 + 共享人物 + phash 近重复 → 合并
fn merge_cross_device(segs: &[Seg], persons: &HashMap<i64, HashSet<i64>>) -> Vec<Vec<Seg>> {
    let groups = segment(segs);
    if groups.len() < 2 {
        return groups.into_iter().map(|g| g.into_iter().map(|i| segs[i].clone()).collect()).collect();
    }
    let mut dsu = Dsu::new(groups.len());
    for i in 0..groups.len() {
        for j in i + 1..groups.len() {
            let (a0, a1) = (segs[groups[i][0]].dt, segs[*groups[i].last().unwrap()].dt);
            let (b0, b1) = (segs[groups[j][0]].dt, segs[*groups[j].last().unwrap()].dt);
            if a1 < b0 || b1 < a0 {
                continue; // 时间不重叠
            }
            let mut devs: HashSet<&str> = HashSet::new();
            for k in groups[i].iter().chain(groups[j].iter()) {
                let d = segs[*k].device.as_str();
                if !d.is_empty() {
                    devs.insert(d);
                }
            }
            if devs.len() < 2 {
                continue; // 无跨设备
            }
            let pi: HashSet<i64> = groups[i]
                .iter()
                .flat_map(|k| persons.get(&segs[*k].id).into_iter().flatten().copied())
                .collect();
            let pj: HashSet<i64> = groups[j]
                .iter()
                .flat_map(|k| persons.get(&segs[*k].id).into_iter().flatten().copied())
                .collect();
            if !pi.iter().any(|x| pj.contains(x)) {
                continue; // 无共享人物
            }
            // phash 近重复；各取前 200 张避免 O(n²) 爆炸
            let ha: Vec<&str> = groups[i]
                .iter()
                .take(200)
                .map(|k| segs[*k].phash.as_str())
                .filter(|s| !s.is_empty())
                .collect();
            let hb: Vec<&str> = groups[j]
                .iter()
                .take(200)
                .map(|k| segs[*k].phash.as_str())
                .filter(|s| !s.is_empty())
                .collect();
            if ha.iter().any(|a| hb.iter().any(|b| phash_hamming(a, b) <= C::MERGE_PHASH_HAMMING)) {
                dsu.union(i, j);
            }
        }
    }
    let mut merged: HashMap<usize, Vec<Seg>> = HashMap::new();
    for (gi, g) in groups.iter().enumerate() {
        let root = dsu.find(gi);
        let e = merged.entry(root).or_default();
        e.extend(g.iter().map(|k| segs[*k].clone()));
    }
    let mut out: Vec<Vec<Seg>> = merged
        .into_values()
        .map(|mut g| {
            g.sort_by(|a, b| a.dt.cmp(&b.dt).then(a.id.cmp(&b.id)));
            g
        })
        .collect();
    out.sort_by(|a, b| a[0].dt.cmp(&b[0].dt));
    out
}

fn tag_histogram(db: &DB, ids: &[i64]) -> HashMap<String, i64> {
    let tmap = db.tags_of(ids).unwrap_or_default();
    let mut c: HashMap<String, i64> = HashMap::new();
    for v in tmap.values() {
        for (t, _) in v {
            *c.entry(t.clone()).or_insert(0) += 1;
        }
    }
    c
}

fn tag_histogram_top(c: &HashMap<String, i64>, n: usize) -> Vec<String> {
    let mut v: Vec<(&String, &i64)> = c.iter().collect();
    v.sort_unstable_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    v.into_iter().take(n).map(|(k, _)| k.clone()).collect()
}

/// "杭州·9月·户外/晴天🎄圣诞"
fn make_title(dt: NaiveDateTime, city: &str, tagc: &HashMap<String, i64>) -> String {
    let month = dt.month();
    let total: i64 = tagc.values().sum();
    // 特殊事件要求组内占比 ≥15%：否则一张图带 christmas tree 就会把整个事件标成"圣诞"
    let ratio = |t: &str| -> f64 { tagc.get(t).copied().unwrap_or(0) as f64 / total.max(1) as f64 };
    let mut special = String::new();
    if ratio("christmas tree") >= 0.15 {
        special = "🎄圣诞".into();
    }
    if ratio("wedding") >= 0.15 || ratio("wedding dress") >= 0.15 {
        special = "💍婚礼".into();
    }
    if ratio("fireworks") >= 0.15 {
        special = "🎆跨年".into();
    }
    if ratio("cake") >= 0.2 || ratio("birthday party") >= 0.2 {
        special = "🎂生日".into();
    }
    let zh: Vec<String> = tag_histogram_top(tagc, 2)
        .into_iter()
        .map(|t| {
            TAG_ZH_DISPLAY
                .iter()
                .find(|(k, _)| *k == t)
                .map(|(_, v)| v.to_string())
                .unwrap_or(t)
        })
        .collect();
    let mut parts: Vec<String> = Vec::new();
    if !city.is_empty() {
        parts.push(city.to_string());
    }
    parts.push(format!("{month}月"));
    if !zh.is_empty() {
        parts.push(zh.join("/"));
    }
    format!("{}{}", parts.join("·"), special)
}

fn rebuild_bursts(db: &DB, groups: &[Vec<Seg>]) -> Result<()> {
    for g in groups {
        let mut bursts: Vec<Vec<i64>> = Vec::new();
        let mut cur: Vec<&Seg> = Vec::new();
        for s in g {
            if let Some(&prev) = cur.last() {
                let dt = (s.dt - prev.dt).num_milliseconds() as f64 / 1000.0;
                let hd = if prev.phash.is_empty() || s.phash.is_empty() {
                    64
                } else {
                    phash_hamming(&prev.phash, &s.phash)
                };
                if dt > C::BURST_DT_S || hd > C::BURST_PHASH_HAMMING {
                    if cur.len() >= 2 {
                        bursts.push(cur.drain(..).map(|s| s.id).collect());
                    } else {
                        cur.clear();
                    }
                }
            }
            cur.push(s);
        }
        if cur.len() >= 2 {
            bursts.push(cur.drain(..).map(|s| s.id).collect());
        }
        for ids in bursts {
            let bid = db.insert_burst(ids.len())?;
            let rows = db.photos_by_ids(&ids)?;
            let full: Vec<&Photo> = ids.iter().filter_map(|i| rows.get(i)).collect();
            let Some(best) = full.iter().max_by(|a, b| burst_score(a).total_cmp(&burst_score(b))) else {
                continue;
            };
            for p in &full {
                db.set_photo_burst(p.id, bid, p.id == best.id)?;
            }
        }
    }
    Ok(())
}

pub fn parse_dt(s: &str) -> Option<NaiveDateTime> {
    for f in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"] {
        if let Ok(t) = NaiveDateTime::parse_from_str(s, f) {
            return Some(t);
        }
        if let Ok(d) = chrono::NaiveDate::parse_from_str(s, f) {
            return d.and_hms_opt(0, 0, 0);
        }
    }
    None
}

pub fn fmt_dt(t: NaiveDateTime) -> String {
    t.format("%Y-%m-%d %H:%M:%S").to_string()
}
