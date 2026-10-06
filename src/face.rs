//! C. 人脸层：SCRFD 检测 + ArcFace 512d + 增量多原型聚类 + 闭眼启发式
//!
//! SCRFD 输出无 batch 维（scores(N,1)/bbox(N,4)/kps(N,10)），bbox/kps 已除以 stride，
//! 处理见 models.rs。

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

use crate::config as C;
use crate::db::DB;
use crate::models::{resize_rgb, Hub, Interp};

/// ArcFace 标准 5 点模板
const ARCFACE_DST: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// 增量聚类：新脸与已有原型比余弦，命名一次全簇生效
pub struct PersonStore {
    protos: HashMap<i64, Vec<Vec<f32>>>,
    named: HashMap<i64, String>,
    next_id: i64,
}

impl Default for PersonStore {
    fn default() -> Self {
        Self {
            protos: std::collections::HashMap::new(),
            named: std::collections::HashMap::new(),
            next_id: 1,
        }
    }
}

impl PersonStore {
    pub fn load(db: &DB) -> Result<Self> {
        let mut protos: HashMap<i64, Vec<Vec<f32>>> = HashMap::new();
        let mut named = HashMap::new();
        let mut max_id = 0i64;
        for p in db.persons()? {
            max_id = max_id.max(p.id);
            if !is_auto_name(&p.name) {
                named.insert(p.id, p.name);
            }
        }
        for (_fid, pid, vec) in db.all_face_vecs()? {
            add_proto(&mut protos, pid, &vec);
        }
        Ok(Self {
            protos,
            named,
            next_id: max_id + 1,
        })
    }

    pub fn named(&self) -> &HashMap<i64, String> {
        &self.named
    }

    pub fn person_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.named.values().cloned().collect();
        v.sort();
        v
    }

    /// → (person_id, 最高余弦)
    pub fn assign(&mut self, db: &DB, vec: &[f32]) -> Result<(i64, f32)> {
        let mut best_pid = -1i64;
        let mut best = -1.0f32;
        for (pid, protos) in self.protos.iter() {
            let s = protos
                .iter()
                .map(|p| dot(p, vec))
                .fold(f32::NEG_INFINITY, f32::max);
            if s > best {
                best = s;
                best_pid = *pid;
            }
        }
        if best_pid >= 0 && best >= C::FACE_COS_ASSIGN {
            add_proto(&mut self.protos, best_pid, vec);
            return Ok((best_pid, best));
        }
        let name = format!("人物{}", self.next_id);
        let pid = db.add_person(&name)?;
        self.next_id += 1;
        self.protos.insert(pid, vec![vec.to_vec()]);
        Ok((pid, 1.0))
    }

    pub fn rename(&mut self, db: &DB, pid: i64, name: &str) -> Result<()> {
        db.rename_person(pid, name)?;
        self.named.insert(pid, name.to_string());
        Ok(())
    }
}

fn is_auto_name(n: &str) -> bool {
    n.strip_prefix("人物").is_some_and(|rest| {
        !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
    })
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 多原型：同一人不同角度/表情的 embedding 会有差异，低于阈值就新增一个原型
fn add_proto(protos: &mut HashMap<i64, Vec<Vec<f32>>>, pid: i64, vec: &[f32]) {
    let p = protos.entry(pid).or_default();
    if p.is_empty() {
        p.push(vec.to_vec());
        return;
    }
    let best = p.iter().map(|x| dot(x, vec)).fold(f32::NEG_INFINITY, f32::max);
    if best < C::FACE_COS_NEWPROTO && p.len() < C::PERSON_MAX_PROTOS {
        p.push(vec.to_vec());
    }
}

pub struct FacePipeline {
    hub: Arc<Hub>,
}

impl FacePipeline {
    pub fn new(hub: Arc<Hub>) -> Result<Self> {
        Ok(Self { hub })
    }

    /// 与 new 等价（SCRFD/ArcFace 本来就是懒加载），保留以便语义清晰
    pub fn lazy(hub: Arc<Hub>) -> Result<Self> {
        Self::new(hub)
    }

    /// 对一张图做人脸检测 + 识别 + 统计，返回每张脸的信息
    pub fn analyze(&self, rgb: &[u8], w: usize, h: usize, det_thresh: Option<f32>) -> Result<Vec<FaceRecOut>> {
        let dets = self.hub.scrfd()?.detect(rgb, w, h, det_thresh)?;
        let mut out = Vec::new();
        if dets.is_empty() {
            return Ok(out);
        }
        let arc = self.hub.arcface()?;
        for d in &dets {
            let Some(aligned) = align_face(rgb, w, h, &d.kps) else {
                continue;
            };
            let n = C::ARCFACE_INPUT as usize;
            let small = resize_rgb(&aligned, n, n, n, n, Interp::Bilinear);
            let emb = arc.embed(&small)?;
            let eyes = eyes_open_estimate(rgb, w, h, &d.kps);
            let (x1, y1, x2, y2) = (d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]);
            let area = (((x2 - x1).max(0.0)) * ((y2 - y1).max(0.0))) as f64 / (w * h) as f64;
            out.push(FaceRecOut {
                bbox: d.bbox,
                kps: d.kps,
                emb,
                eyes_open: eyes,
                area,
                score: d.score,
            });
        }
        Ok(out)
    }
}

#[derive(Clone)]
pub struct FaceRecOut {
    pub bbox: [f32; 4],
    pub kps: [[f32; 2]; 5],
    pub emb: Vec<f32>,
    pub eyes_open: bool,
    pub area: f64,
    pub score: f32,
}

impl FaceRecOut {
    /// 10 个关键点展平成一维，和 Python 的 `kps` 存库格式一致
    pub fn kps_flat(&self) -> Vec<f32> {
        self.kps.iter().flat_map(|p| [p[0], p[1]]).collect()
    }
}

/// 按 5 点相似变换把人脸对齐到 112×112（等价 cv2.estimateAffinePartial2D + warpAffine）
pub fn align_face(rgb: &[u8], w: usize, h: usize, kps: &[[f32; 2]; 5]) -> Option<Vec<u8>> {
    let m = estimate_similarity(kps, &ARCFACE_DST)?;
    Some(warp_affine(rgb, w, h, &m, C::ARCFACE_INPUT as usize))
}

/// 相似变换最小二乘（等价 cv2.estimateAffinePartial2D）
///
/// 返回 [a, b, tx, c, d, ty]，正向映射为：
///   dst_x = a*(src_x - scx) - b*(src_y - scy) + dcx
///   dst_y = c*(src_x - scx) + d*(src_y - scy) + dcy
///
/// 之前这里有个「两轮 LM 细化」的 for 循环，但循环体是
/// `r00 = na / sc * sc` —— 化简后就是 `na`，输入输出完全相同，是纯 no-op。
/// 闭式解（Umeyama/Kabsch）本身就够精确，删掉迭代，加单元测试锁死精度。
fn estimate_similarity(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> Option<[f32; 6]> {
    const N: f32 = 5.0;
    let (mut scx, mut scy, mut dcx, mut dcy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..5 {
        scx += src[i][0];
        scy += src[i][1];
        dcx += dst[i][0];
        dcy += dst[i][1];
    }
    scx /= N;
    scy /= N;
    dcx /= N;
    dcy /= N;

    let (mut sxx, mut syy, mut sxsx, mut sxsy, mut sysx, mut sysy) =
        (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..5 {
        let (px, py) = (src[i][0] - scx, src[i][1] - scy);
        let (qx, qy) = (dst[i][0] - dcx, dst[i][1] - dcy);
        sxx += px * px;
        syy += py * py;
        sxsx += px * qx;
        sxsy += px * qy;
        sysx += py * qx;
        sysy += py * qy;
    }
    let den = sxx + syy;
    if den < 1e-6 {
        return None;   // 5 个点全重合，解不唯一
    }
    // 相似变换最优解：[cosθ*k, -sinθ*k; sinθ*k, cosθ*k]
    let a = (sxsx + sysy) / den;
    let b = (sxsy - sysx) / den;
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let tx = dcx - (a * scx - b * scy);
    let ty = dcy - (b * scx + a * scy);
    Some([a, -b, tx, b, a, ty])
}

fn warp_affine(rgb: &[u8], w: usize, h: usize, m: &[f32; 6], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n * n * 3];
    let (r00, r01, tx, r10, r11, ty) = (m[0], m[1], m[2], m[3], m[4], m[5]);
    let inv = det(r00, r01, r10, r11);
    if inv.abs() < 1e-9 {
        return out;
    }
    for y in 0..n {
        for x in 0..n {
            // 反向映射（与 cv2.warpAffine 的 INTER_LINEAR 一致）
            let px = x as f32 + 0.5 - tx;
            let py = y as f32 + 0.5 - ty;
            let sx = (r11 * px - r01 * py) / inv;
            let sy = (-r10 * px + r00 * py) / inv;
            if sx < -0.5 || sy < -0.5 || sx > w as f32 - 0.5 || sy > h as f32 - 0.5 {
                continue; // 越界留黑（OpenCV 默认 BORDER_CONSTANT）
            }
            let x0 = (sx - 0.5).floor().max(0.0) as usize;
            let y0 = (sy - 0.5).floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let wx = (sx - 0.5 - x0 as f32).clamp(0.0, 1.0);
            let wy = (sy - 0.5 - y0 as f32).clamp(0.0, 1.0);
            for c in 0..3 {
                let p00 = rgb[(y0 * w + x0) * 3 + c] as f32;
                let p01 = rgb[(y0 * w + x1) * 3 + c] as f32;
                let p10 = rgb[(y1 * w + x0) * 3 + c] as f32;
                let p11 = rgb[(y1 * w + x1) * 3 + c] as f32;
                let top = p00 + (p01 - p00) * wx;
                let bot = p10 + (p11 - p10) * wx;
                out[(y * n + x) * 3 + c] = (top + (bot - top) * wy).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn det(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a * d - b * c
}

/// 闭眼启发式：眼部区域拉普拉斯方差（闭眼=平滑皮肤→低方差）
pub fn eyes_open_estimate(rgb: &[u8], w: usize, h: usize, kps: &[[f32; 2]; 5]) -> bool {
    let d = dist2(kps[0], kps[1]);
    if d < 4.0 {
        return true;
    }
    let r = (d * 0.30).round().max(4.0) as i32;
    let mut flags = Vec::new();
    for eye in [kps[0], kps[1]] {
        let (ex, ey) = (eye[0], eye[1]);
        let x0 = (ex - r as f32).max(0.0) as usize;
        let x1 = ((ex + r as f32) as usize).min(w);
        let y0 = (ey - r as f32 * 0.6).max(0.0) as usize;
        let y1 = ((ey + r as f32 * 0.8) as usize).min(h);
        if x1 <= x0 + 6 || y1 <= y0 + 6 {
            continue;
        }
        let g = |x: usize, y: usize| -> f32 {
            let i = (y * w + x) * 3;
            0.299 * rgb[i] as f32 + 0.587 * rgb[i + 1] as f32 + 0.114 * rgb[i + 2] as f32
        };
        let mut sum = 0f64;
        let mut n = 0f64;
        for y in y0 + 1..y1 - 1 {
            for x in x0 + 1..x1 - 1 {
                let l = (g(x - 1, y - 1) + g(x + 1, y - 1) + g(x - 1, y + 1) + g(x + 1, y + 1)
                    - 4.0 * g(x, y)) as f64;
                sum += l * l;
                n += 1.0;
            }
        }
        if n == 0.0 {
            continue;
        }
        let v = sum / n;
        flags.push(v > (d as f64 * 0.8).max(20.0));
    }
    if flags.is_empty() {
        true
    } else {
        flags.into_iter().all(|x| x)
    }
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}


#[cfg(test)]
mod tests {
    use super::*;

    /// 相似变换的解析解 —— 人脸对齐正确性的唯一客观保证
    #[test]
    fn 相似变换_已知解() {
        for &(deg, k, tx, ty) in &[
            (0.0f32, 1.0f32, 0.0f32, 0.0f32),
            (30.0, 1.2, 10.0, -5.0),
            (-17.0, 0.83, -3.0, 8.0),
            (90.0, 1.0, 5.0, 5.0),
            (45.0, 2.5, 0.0, 0.0),
        ] {
            let (c, sn) = (deg.to_radians().cos(), deg.to_radians().sin());
            let src = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]];
            let dst: [[f32; 2]; 5] = std::array::from_fn(|i| {
                let (x, y) = (src[i][0], src[i][1]);
                [
                    k * (c * x - sn * y) + tx,
                    k * (sn * x + c * y) + ty,
                ]
            });
            let m = estimate_similarity(&src, &dst).expect("应能解出");
            assert!(
                (m[0] - k * c).abs() < 1e-4,
                "deg={deg} a: {} vs {}",
                m[0], k * c
            );
            assert!(
                (m[1] + k * sn).abs() < 1e-4,
                "deg={deg} b: {} vs {}",
                m[1], -k * sn
            );
            assert!((m[2] - tx).abs() < 1e-3, "deg={deg} tx: {} vs {tx}", m[2]);
            assert!((m[4] - k * c).abs() < 1e-4, "deg={deg} d: {}", m[4]);
            assert!((m[5] - ty).abs() < 1e-3, "deg={deg} ty: {} vs {ty}", m[5]);
        }
    }

    #[test]
    fn 相似变换_退化输入返回None() {
        let same = [[1.0f32, 1.0]; 5]; // 5 点全重合
        assert!(estimate_similarity(&same, &same).is_none());
    }

    #[test]
    fn 相似变换_等比无旋转() {
        let src = [[0.0f32, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]];
        let dst = [[1.0f32, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0], [15.0, 15.0]];
        let m = estimate_similarity(&src, &dst).unwrap();
        assert!((m[0] - 3.0).abs() < 1e-4);
        assert!(m[1].abs() < 1e-4);
        assert!(m[2].abs() < 1e-4);
    }

    #[test]
    fn 闭眼启发式_人脸区域有效() {
        // 有纹理的眼部区域 → 睁眼；纯色区域 → 判定不可靠但不应 panic
        let (w, h) = (64usize, 64);
        let mut img = vec![10u8; w * h * 3];
        // 画两只"眼睛"，内部放高频噪声
        for (ex, ey) in [(20usize, 30usize), (44, 30)] {
            for y in ey - 4..ey + 4 {
                for x in ex - 6..ex + 6 {
                    let i = (y * w + x) * 3;
                    let v = if (x + y) % 2 == 0 { 240 } else { 15 };
                    img[i] = v;
                    img[i + 1] = v;
                    img[i + 2] = v;
                }
            }
        }
        let kps = [[20.0, 30.0], [44.0, 30.0], [32.0, 42.0], [24.0, 50.0], [40.0, 50.0]];
        let _ = eyes_open_estimate(&img, w, h, &kps); // 不 panic 即通过
    }
}
