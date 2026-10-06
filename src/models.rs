//! 模型加载中心（懒加载单例）：ONNX Runtime 统一跑 CLIP / SCRFD / ArcFace / NIMA / RapidOCR。
//! OCR 侧保留 Python 版能力但走 ort，故不需要 tflitec/FFI。

use anyhow::{bail, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::ValueType;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::Path;

use crate::config as C;

/// 全局线程数：ORT 内部并行 + 外层 rayon 共享 4 核，避免超额订阅
pub fn init_ort() {
    let _ = ort::init().with_name("album").commit();
}

fn make_session(path: &Path) -> Result<Session> {
    if !path.exists() {
        bail!("模型文件不存在：{}", path.display());
    }
    // ort 2.0.0-rc.13：SessionBuilder 私有，只能经 Session::builder() 拿；
    // with_* 是 by-value 且返回 BuilderResult（错误里带 builder 本身，不是 Send/Sync），
    // 所以用 map_err 手动转成 anyhow，不能让 ? 走 From。
    let mut b = Session::builder().map_err(|e| anyhow::anyhow!("创建 SessionBuilder 失败: {e}"))?;
    b = b
        .with_intra_threads(C::ort_threads().min(2))
        .map_err(|e| anyhow::anyhow!("设置 intra_threads 失败: {e}"))?;
    b = b
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("设置优化级别失败: {e}"))?;
    b.commit_from_file(path)
        .with_context(|| format!("加载 ONNX 模型失败 {}", path.display()))
}

// ---------------------------------------------------------------- tokenizer

pub struct Tokenizer {
    tok: tokenizers::Tokenizer,
    auto_special: bool,
    can_manual: bool,
}

impl Tokenizer {
    fn load() -> Result<Self> {
        let p = C::mobileclip_tok();
        let tok = tokenizers::Tokenizer::from_file(&p)
            .map_err(|e| anyhow::anyhow!("加载 tokenizer 失败 {}: {e}", p.display()))?;
        let ids: Vec<i64> = tok
            .encode("test", false)
            .map(|e| e.get_ids().iter().map(|x| *x as i64).collect())
            .unwrap_or_default();
        let vs = tok.get_vocab_size(false) as i64;
        let auto = ids.first().copied() == Some(C::CLIP_BOS)
            && ids.last().copied() == Some(C::CLIP_EOS);
        Ok(Self {
            tok,
            auto_special: auto,
            can_manual: vs > C::CLIP_EOS,
        })
    }

    /// → (1, 77) int64 token id 矩阵；已按探测结果决定是否手工拼 BOS/EOS
    pub fn tokens(&self, text: &str) -> Vec<i64> {
        let mut ids: Vec<i64> = self
            .tok
            .encode(text, false)
            .map(|e| e.get_ids().iter().map(|x| *x as i64).collect())
            .unwrap_or_default();
        if self.can_manual && !self.auto_special {
            let inner = (C::CLIP_TEXT_CTX - 2).min(ids.len());
            ids.truncate(inner);
            ids.insert(0, C::CLIP_BOS);
            ids.push(C::CLIP_EOS);
        } else if ids.len() > C::CLIP_TEXT_CTX {
            ids.truncate(C::CLIP_TEXT_CTX);
        }
        ids.resize(C::CLIP_TEXT_CTX, 0);
        ids
    }
}

// ---------------------------------------------------------------- CLIP

pub struct Clip {
    vision: Mutex<Session>,
    text: Mutex<Session>,
    tokenizer: Tokenizer,
    dim: usize,
}

impl Clip {
    fn load() -> Result<Self> {
        let v = make_session(&C::mobileclip_vision())?;
        let t = make_session(&C::mobileclip_text())?;
        Ok(Self {
            vision: Mutex::new(v),
            text: Mutex::new(t),
            tokenizer: Tokenizer::load()?,
            dim: C::CLIP_DIM,
        })
    }

    /// img_rgb: HWC u8，尺寸不限（内部 resize 到 256）
    /// 预处理按实测：RGB/[0,1]，不做 mean/std（见 config.rs 注释）
    pub fn embed_image(&self, img: &[u8], w: usize, h: usize) -> Result<Vec<f32>> {
        let s = C::CLIP_IMG_SIZE;
        let resized = resize_rgb(img, w, h, s, s, if C::CLIP_INTERP_CUBIC { Interp::Bicubic } else { Interp::Area });
        let mut plane = vec![0f32; 3 * s * s];
        let normalize = C::CLIP_MEAN != [0.0, 0.0, 0.0];
        for c in 0..3 {
            let mean = C::CLIP_MEAN[c];
            let std = C::CLIP_STD[c];
            for y in 0..s {
                for x in 0..s {
                    let v = resized[(y * s + x) * 3 + c] as f32 / 255.0;
                    plane[c * s * s + y * s + x] = if normalize { (v - mean) / std } else { v };
                }
            }
        }
        let data = f32_tensor(&[1, 3, s, s], plane)?;
        let mut sess = self.vision.lock();
        let outs = sess.run(ort::inputs![data])?;
        let v = extract_vec(outs[0].try_extract_tensor::<f32>()?, self.dim)?;
        Ok(l2_normalize(&v))
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenizer.tokens(text);
        let ids_t = i64_tensor(&[1, C::CLIP_TEXT_CTX], ids.clone())?;
        let mut sess = self.text.lock();
        let outs = sess.run(ort::inputs![ids_t])?;
        let (shape, data) = outs[0].try_extract_tensor::<f32>()?;
        let v = if shape.len() == 3 {
            // (1, 77, D)：取 EOS 位置池化
            let dim = shape[2] as usize;
            let eos = ids
                .iter()
                .rposition(|x| *x == C::CLIP_EOS)
                .unwrap_or(C::CLIP_TEXT_CTX - 1);
            data[eos * dim..eos * dim + dim].to_vec()
        } else {
            extract_vec((shape, data), self.dim)?
        };
        Ok(l2_normalize(&v))
    }

    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_text(t)).collect()
    }
}

// ---------------------------------------------------------------- NIMA

pub struct Nima {
    sess: Mutex<Session>,
    nchw: bool,
}

impl Nima {
    fn load(path: &Path) -> Result<Self> {
        let s = make_session(path)?;
        // tflite2onnx 导出的输入是 NCHW；原生 ONNX 导出版可能是 NHWC，两种都兼容
        let nchw = match s.inputs()[0].dtype() {
            ValueType::Tensor { shape, .. } => {
                let d: Vec<i64> = shape.iter().copied().collect();
                d.len() == 4 && d[1] == 3
            }
            _ => false,
        };
        Ok(Self {
            sess: Mutex::new(s),
            nchw,
        })
    }

    /// img_rgb: HWC u8 → 1~10 分
    pub fn score(&self, img: &[u8], w: usize, h: usize) -> Result<f64> {
        const N: usize = 224;
        let r = resize_rgb(img, w, h, N, N, Interp::Area);
        let mut plane = vec![0f32; 3 * N * N];
        for c in 0..3 {
            for y in 0..N {
                for x in 0..N {
                    plane[c * N * N + y * N + x] = r[(y * N + x) * 3 + c] as f32 / 255.0;
                }
            }
        }
        let out: Vec<f32> = if self.nchw {
            let t = f32_tensor(&[1, 3, N, N], plane)?;
            let mut s = self.sess.lock();
            let o = s.run(ort::inputs![t])?;
            o[0].try_extract_tensor::<f32>()?.1.to_vec()
        } else {
            let t = f32_tensor(&[1, N, N, 3], plane)?;
            let mut s = self.sess.lock();
            let o = s.run(ort::inputs![t])?;
            o[0].try_extract_tensor::<f32>()?.1.to_vec()
        };
        // softmax 后与 1..10 求期望
        let max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = out.iter().map(|p| (p - max).exp()).sum();
        if sum <= 0.0 {
            return Ok(5.0);
        }
        let mut acc = 0.0;
        for (i, p) in out.iter().enumerate() {
            acc += ((p - max).exp() / sum) as f64 * (i + 1) as f64;
        }
        Ok(acc)
    }
}

// ---------------------------------------------------------------- SCRFD

pub struct Scrfd {
    sess: Mutex<Session>,
    iname: String,
    strides: [usize; 3],
    fmc: usize,
}

pub struct Detection {
    pub bbox: [f32; 4],
    pub kps: [[f32; 2]; 5],
    pub score: f32,
}

impl Scrfd {
    fn load() -> Result<Self> {
        let s = make_session(&C::scrfd_model())?;
        let iname = s.inputs()[0].name().to_string();
        Ok(Self {
            sess: Mutex::new(s),
            iname,
            strides: [8, 16, 32],
            fmc: 3,
        })
    }

    /// img_rgb: HWC u8。letterbox 到 640×640，返回原图坐标系的检测结果
    pub fn detect(&self, img: &[u8], w: usize, h: usize, thresh: Option<f32>) -> Result<Vec<Detection>> {
        let thresh = thresh.unwrap_or(C::FACE_DET_THRESH);
        let (nw, nh) = if h > w {
            (((640.0 * w as f64 / h as f64).round() as usize).max(1), 640usize)
        } else {
            (640usize, ((640.0 * h as f64 / w as f64).round() as usize).max(1))
        };
        let scale = nh as f32 / h as f32;
        let resized = resize_rgb(img, w, h, nw, nh, Interp::Area);
        let mut canvas = vec![0f32; 640 * 640 * 3];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..3 {
                    canvas[(y * 640 + x) * 3 + c] = (resized[(y * nw + x) * 3 + c] as f32 - 127.5) / 128.0;
                }
            }
        }
        let t = f32_tensor(&[1, 3, 640, 640], canvas)?;
        let mut sess = self.sess.lock();
        let outs = sess.run(ort::inputs![t])?;
        // 本机 scrfd_10g.onnx 实测结构（已用 onnxruntime 核对，别盲改）：
        //   outs[0..3] = scores  stride 8/16/32  → (12800,1) (3200,1) (800,1)
        //   outs[3..6] = bbox    stride 8/16/32  → (12800,4) (3200,4) (800,4)
        //   outs[6..9] = kps     stride 8/16/32  → (12800,10)(3200,10)(800,10)
        // 也就是说：按【类型】分组，不是按 stride 交错。
        // 长度关系恒为 side*side*2 个 anchor（80²×2=12800、40²×2=3200、20²×2=800）。
        // 无 batch 维；部分导出可能是 (1,N,C)，rows_of 统一压平。
        const ANCHORS: usize = 2;
        let mut per_stride: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::with_capacity(3);
        for si in 0..self.strides.len() {
            let (score, n_expect) = rows_of(outs[si].try_extract_tensor::<f32>()?, 1);
            let (bbox, _) = rows_of(outs[self.fmc + si].try_extract_tensor::<f32>()?, 4);
            let (kps, _) = rows_of(outs[self.fmc * 2 + si].try_extract_tensor::<f32>()?, 10);
            let side = 640 / self.strides[si];
            let want = side * side * ANCHORS;
            if n_expect != want {
                // 结构与预期不符时降级到实际长度，但必须告知（而不是静默算错）
                log::warn!(
                    "SCRFD stride {} 的 score 行数 {} 与预期 {}（{}²×{}）不符，按实际长度处理",
                    self.strides[si], n_expect, want, side, ANCHORS
                );
            }
            if bbox.len() < n_expect * 4 || kps.len() < n_expect * 10 {
                bail!("SCRFD stride {} 的 bbox/kps 行数不足", self.strides[si]);
            }
            per_stride.push((score, bbox, kps));
        }

        let mut boxes_all: Vec<[f32; 4]> = Vec::new();
        let mut kps_all: Vec<[f32; 10]> = Vec::new();
        let mut scores_all: Vec<f32> = Vec::new();
        for (si, stride) in self.strides.iter().enumerate() {
            let side = 640 / stride;
            let stride_f = *stride as f32;
            let (score, bbox, kps) = &per_stride[si];
            for j in 0..score.len() {
                let sv = score[j];
                if sv < thresh {
                    continue;
                }
                let anchor = j / ANCHORS;
                let cx = (anchor % side) as f32 * stride_f;
                let cy = (anchor / side) as f32 * stride_f;
                let d = &bbox[j * 4..j * 4 + 4];
                let k = &kps[j * 10..j * 10 + 10];
                // bbox/kps 导出时已除以 stride，乘回原图尺度
                boxes_all.push([
                    cx - d[0] * stride_f,
                    cy - d[1] * stride_f,
                    cx + d[2] * stride_f,
                    cy + d[3] * stride_f,
                ]);
                let mut kv = [0f32; 10];
                for p in 0..5 {
                    kv[2 * p] = k[2 * p] * stride_f + cx;
                    kv[2 * p + 1] = k[2 * p + 1] * stride_f + cy;
                }
                kps_all.push(kv);
                scores_all.push(sv);
            }
        }
        if boxes_all.is_empty() {
            return Ok(Vec::new());
        }
        let keep = nms(&boxes_all, &scores_all, 0.4);
        let mut out = Vec::new();
        for i in keep {
            let b = [
                boxes_all[i][0] / scale,
                boxes_all[i][1] / scale,
                boxes_all[i][2] / scale,
                boxes_all[i][3] / scale,
            ];
            if b[2] - b[0] < 24.0 || b[3] - b[1] < 24.0 {
                continue;
            }
            let cx = (b[0] + b[2]) / 2.0;
            let cy = (b[1] + b[3]) / 2.0;
            let bw = b[2] - b[0];
            let bh = b[3] - b[1];
            let mut k5 = [[0f32; 2]; 5];
            for p in 0..5 {
                k5[p][0] = (kps_all[i][2 * p] - cx) / bw * b[0] + cx;
                k5[p][1] = (kps_all[i][2 * p + 1] - cy) / bh * b[1] + cy;
            }
            out.push(Detection {
                bbox: [
                    b[0].max(0.0).min(w as f32 - 1.0),
                    b[1].max(0.0).min(h as f32 - 1.0),
                    b[2].max(0.0).min(w as f32 - 1.0),
                    b[3].max(0.0).min(h as f32 - 1.0),
                ],
                kps: k5,
                score: scores_all[i],
            });
        }
        Ok(out)
    }
}

pub struct Arcface {
    sess: Mutex<Session>,
}

impl Arcface {
    fn load() -> Result<Self> {
        Ok(Self {
            sess: Mutex::new(make_session(&C::arcface_model())?),
        })
    }

    /// aligned: 112×112×3 u8
    pub fn embed(&self, aligned: &[u8]) -> Result<Vec<f32>> {
        let n = C::ARCFACE_INPUT as usize;
        let mut plane = vec![0f32; 3 * n * n];
        for c in 0..3 {
            for y in 0..n {
                for x in 0..n {
                    plane[c * n * n + y * n + x] =
                        (aligned[(y * n + x) * 3 + c] as f32 - 127.5) / 127.5;
                }
            }
        }
        let t = f32_tensor(&[1, 3, n, n], plane)?;
        let mut sess = self.sess.lock();
        let outs = sess.run(ort::inputs![t])?;
        let v = extract_vec(outs[0].try_extract_tensor::<f32>()?, 512)?;
        Ok(l2_normalize(&v))
    }
}

// ---------------------------------------------------------------- RapidOCR

/// PP-OCRv4 rec：输入 NHWC [1, 48, W, 3]（W 动态，<=320），输出 [1, T, 6625] 概率
pub struct Rec {
    sess: Mutex<Session>,
}

impl Rec {
    fn load() -> Result<Self> {
        Ok(Self {
            sess: Mutex::new(make_session(&C::ocr_rec())?),
        })
    }

    /// img: HWC u8，调用方保证 h=48
    pub fn run(&self, img: &[u8], w: usize, h: usize) -> Result<Vec<f32>> {
        // PP-OCR 的 rec 输入是 [0,1] 归一化后的 RGB
        let plane: Vec<f32> = img.iter().map(|v| *v as f32 / 255.0).collect();
        let t = f32_tensor(&[1, h, w, 3], plane)?;
        let mut sess = self.sess.lock();
        let outs = sess.run(ort::inputs![t])?;
        Ok(outs[0].try_extract_tensor::<f32>()?.1.to_vec())
    }
}

/// PP-OCRv4 det：DB 二值化文本框检测。输入固定 [1, 3, 640, 640]，输出概率图
pub struct Det {
    sess: Mutex<Session>,
}

pub struct TextBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

impl Det {
    fn load() -> Result<Self> {
        Ok(Self {
            sess: Mutex::new(make_session(&C::ocr_det())?),
        })
    }

    /// → 文本框（原图坐标，按 y 排序）
    pub fn detect(&self, img: &[u8], w: usize, h: usize, thresh: f32, box_thresh: f32) -> Result<Vec<TextBox>> {
        const S: usize = 640;
        let small = resize_rgb(img, w, h, S, S, Interp::Area);
        // DB 预处理：(img/255 - 0.5)/0.5
        let mut plane = vec![0f32; 3 * S * S];
        for i in 0..S * S {
            for c in 0..3 {
                plane[c * S * S + i] = (small[i * 3 + c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        let t = f32_tensor(&[1, 3, S, S], plane)?;
        // 先在锁内把概率图拷出来（Shape 借用了 outs，不能带出作用域）
        let prob = {
            let mut sess = self.sess.lock();
            let outs = sess.run(ort::inputs![t])?;
            let (shape, data) = outs[0].try_extract_tensor::<f32>()?;
            let (ow, oh) = if shape.len() >= 2 {
                (shape[shape.len() - 2] as usize, shape[shape.len() - 1] as usize)
            } else {
                (S, S)
            };
            (data[..(ow * oh).min(data.len())].to_vec(), ow, oh)
        };
        let (prob, ow, oh) = prob;
        Ok(unclip_boxes(&prob, ow, oh, w, h, thresh, box_thresh))
    }
}

/// DB 概率图 → 文本框
///
/// 流程对齐 PaddleOCR 的 DB 后处理：二值化 → 连通域 → 按面积比例 unclip 扩张。
/// 之前的实现是「逐行扫描 + 纵向扩张」，会把相邻文本行吃成一个框、
/// 又会在概率中间凹陷处断开，票据 OCR 基本不可用。
fn unclip_boxes(
    prob: &[f32],
    ow: usize,
    oh: usize,
    w: usize,
    h: usize,
    thresh: f32,
    box_thresh: f32,
) -> Vec<TextBox> {
    let bin_thresh = thresh.max(box_thresh).max(0.05);
    let comps = connected_components(prob, ow, oh, bin_thresh, box_thresh);
    let sx = w as f32 / ow as f32;
    let sy = h as f32 / oh as f32;
    let mut out = Vec::with_capacity(comps.len());
    for (x0, y0, x1, y1, score) in comps {
        let bw = (x1 - x0 + 1) as f32;
        let bh = (y1 - y0 + 1) as f32;
        // DB 的 unclip：按面积比 1.6 反推扩张量，再按宽高比分配
        let area = bw * bh;
        let ratio = 1.6f32;
        let want = (area * ratio).sqrt();
        let expand = ((want - bw.min(bh)).max(2.0) / 2.0).min(bw.min(bh) * 0.5);
        let nx0 = (x0 as f32 - expand).max(0.0);
        let ny0 = (y0 as f32 - expand).max(0.0);
        let nx1 = (x1 as f32 + expand + 1.0).min(ow as f32);
        let ny1 = (y1 as f32 + expand + 1.0).min(oh as f32);
        out.push(TextBox {
            x0: nx0 * sx,
            y0: ny0 * sy,
            x1: nx1 * sx,
            y1: ny1 * sy,
            score,
        });
    }
    // 阅读顺序：从上到下，同一行从左到右
    out.sort_by(|a, b| {
        let row_h = oh as f32 * 0.01;
        if (a.y0 - b.y0).abs() < row_h {
            a.x0.total_cmp(&b.x0)
        } else {
            a.y0.total_cmp(&b.y0)
        }
    });
    out
}

/// 4 邻域连通域标记，返回 (x0,y0,x1,y1, 平均概率)
fn connected_components(
    prob: &[f32],
    w: usize,
    h: usize,
    bin_thresh: f32,
    score_thresh: f32,
) -> Vec<(usize, usize, usize, usize, f32)> {
    let mut visited = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    let mut out = Vec::new();
    let min_area = 4usize; // 过滤单像素噪点
    for start in 0..w * h {
        if visited[start] || prob[start] < bin_thresh {
            continue;
        }
        visited[start] = true;
        stack.clear();
        stack.push(start);
        let (mut x0, mut y0) = (w, h);
        let (mut x1, mut y1) = (0usize, 0usize);
        let (mut sum, mut n) = (0f32, 0f32);
        while let Some(i) = stack.pop() {
            let cx = i % w;
            let cy = i / w;
            x0 = x0.min(cx);
            y0 = y0.min(cy);
            x1 = x1.max(cx);
            y1 = y1.max(cy);
            sum += prob[i];
            n += 1.0;
            // 4 邻域
            if cx > 0 {
                let ni = i - 1;
                if !visited[ni] && prob[ni] >= bin_thresh {
                    visited[ni] = true;
                    stack.push(ni);
                }
            }
            if cx + 1 < w {
                let ni = i + 1;
                if !visited[ni] && prob[ni] >= bin_thresh {
                    visited[ni] = true;
                    stack.push(ni);
                }
            }
            if cy > 0 {
                let ni = i - w;
                if !visited[ni] && prob[ni] >= bin_thresh {
                    visited[ni] = true;
                    stack.push(ni);
                }
            }
            if cy + 1 < h {
                let ni = i + w;
                if !visited[ni] && prob[ni] >= bin_thresh {
                    visited[ni] = true;
                    stack.push(ni);
                }
            }
        }
        let score = if n > 0.0 { sum / n } else { 0.0 };
        let area = (x1 - x0 + 1) * (y1 - y0 + 1);
        if area >= min_area && score >= score_thresh {
            out.push((x0, y0, x1, y1, score));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一张白底黑字的假概率图：3 行文字，每行 3 个块
    fn fake_db_map() -> (Vec<f32>, usize, usize) {
        let (w, h) = (64usize, 48usize);
        let mut p = vec![0.0f32; w * h];
        for row in 0..3usize {
            let y0 = 6 + row * 14;
            for blk in 0..3usize {
                let x0 = 4 + blk * 18;
                for y in y0..y0 + 6 {
                    for x in x0..x0 + 14 {
                        if x < w && y < h {
                            p[y * w + x] = 0.9;
                        }
                    }
                }
            }
        }
        (p, w, h)
    }

    #[test]
    fn 连通域数出文本块() {
        let (p, w, h) = fake_db_map();
        let comps = connected_components(&p, w, h, 0.3, 0.5);
        assert_eq!(comps.len(), 9, "3 行 × 3 块");
        for (x0, y0, x1, y1, s) in &comps {
            assert_eq!((*x1 - x0 + 1, *y1 - y0 + 1), (14, 6));
            assert!(*s > 0.85);
        }
    }

    #[test]
    fn 文本框不吞掉整行() {
        // 旧实现会返回覆盖全图的 1~2 个大框
        let (p, w, h) = fake_db_map();
        let boxes = unclip_boxes(&p, w, h, 640, 480, 0.3, 0.5);
        assert_eq!(boxes.len(), 9);
        let covers_all = boxes.iter().any(|b| {
            (b.x1 - b.x0) > 600.0 && (b.y1 - b.y0) > 400.0
        });
        assert!(!covers_all, "不应有覆盖全图的框");
    }

    #[test]
    fn 阅读顺序自上而下() {
        let (p, w, h) = fake_db_map();
        let boxes = unclip_boxes(&p, w, h, 640, 480, 0.3, 0.5);
        for wnd in boxes.windows(2) {
            assert!(
                wnd[0].y0 <= wnd[1].y0 + 1.0,
                "顺序错: {:?} -> {:?}",
                (wnd[0].y0, wnd[0].x0),
                (wnd[1].y0, wnd[1].x0)
            );
        }
    }

    #[test]
    fn 空图不产生框() {
        let p = vec![0.0f32; 32 * 32];
        assert!(unclip_boxes(&p, 32, 32, 320, 320, 0.3, 0.5).is_empty());
    }
}

/// 缩放（HWC u8 → HWC u8）
///
/// `interp` 语义明确：
/// - `Interp::Area` —— 缩小时用 box 平均（等价 OpenCV INTER_AREA），放大时退化为双线性
/// - `Interp::Bicubic` —— Catmull-Rom 三次插值（等价 OpenCV INTER_CUBIC）
/// - `Interp::Bilinear` —— 双线性
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    Area,
    Bilinear,
    Bicubic,
}

#[inline]
fn cubic_w(t: f32) -> (f32, f32, f32, f32) {
    // Catmull-Rom（OpenCV INTER_CUBIC 的 a=-0.75 参数化）
    const A: f32 = -0.75;
    let x = t.abs();
    let x2 = x * x;
    let x3 = x2 * x;
    let w0 = ((A + 2.0) * x3 - (A + 3.0) * x2 + 1.0) * 0.5;
    let w1 = ((A * x - 5.0 * A) * x3 + (8.0 * A + 8.0) * x2 - (4.0 * A + 8.0) * x) * 0.5;
    let w2 = ((-A - 2.0) * x3 + (3.0 * A + 3.0) * x2 + 3.0 * A * x) * 0.5 + 0.0;
    let w3 = 1.0 - w0 - w1 - w2;
    if t < 0.0 {
        (w3, w2, w1, w0)
    } else {
        (w0, w1, w2, w3)
    }
}

pub fn resize_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, interp: Interp) -> Vec<u8> {
    let mut out = vec![0u8; dw * dh * 3];
    if sw == 0 || sh == 0 {
        return out;
    }
    let fx = sw as f32 / dw as f32;
    let fy = sh as f32 / dh as f32;
    // INTER_AREA：缩小时按 box 平均（等价 OpenCV 的面积加权）
    if interp == Interp::Area && fx > 1.5 && fy > 1.5 {
        for y in 0..dh {
            let y0 = (y as f32 * fy).floor() as usize;
            let y1 = (((y + 1) as f32 * fy).ceil() as usize).min(sh).max(y0 + 1);
            for x in 0..dw {
                let x0 = (x as f32 * fx).floor() as usize;
                let x1 = (((x + 1) as f32 * fx).ceil() as usize).min(sw).max(x0 + 1);
                let mut acc = [0u32; 3];
                let mut n = 0u32;
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        for c in 0..3 {
                            acc[c] += src[(sy * sw + sx) * 3 + c] as u32;
                        }
                        n += 1;
                    }
                }
                let n = n.max(1);
                for c in 0..3 {
                    out[(y * dw + x) * 3 + c] = (acc[c] / n) as u8;
                }
            }
        }
        return out;
    }
    if interp == Interp::Bicubic {
        for y in 0..dh {
            let fypos = (y as f32 + 0.5) * fy - 0.5;
            let y0 = fypos.floor() as isize;
            let (wy0, wy1, wy2, wy3) = cubic_w(fypos - y0 as f32);
            for x in 0..dw {
                let fxpos = (x as f32 + 0.5) * fx - 0.5;
                let x0 = fxpos.floor() as isize;
                let (wx0, wx1, wx2, wx3) = cubic_w(fxpos - x0 as f32);
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (dy, wy) in [(0isize, wy0), (1, wy1), (2, wy2), (3, wy3)] {
                        let yy = (y0 + dy).clamp(0, sh as isize - 1) as usize;
                        let mut row = 0.0f32;
                        for (dx, wx) in [(0isize, wx0), (1, wx1), (2, wx2), (3, wx3)] {
                            let xx = (x0 + dx).clamp(0, sw as isize - 1) as usize;
                            row += src[(yy * sw + xx) * 3 + c] as f32 * wx;
                        }
                        acc += row * wy;
                    }
                    out[(y * dw + x) * 3 + c] = acc.clamp(0.0, 255.0) as u8;
                }
            }
        }
        return out;
    }
    for y in 0..dh {
        let sy = ((y as f32 + 0.5) * fy - 0.5).clamp(0.0, sh as f32 - 1.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = sy - y0 as f32;
        for x in 0..dw {
            let sx = ((x as f32 + 0.5) * fx - 0.5).clamp(0.0, sw as f32 - 1.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = sx - x0 as f32;
            for c in 0..3 {
                let p00 = src[(y0 * sw + x0) * 3 + c] as f32;
                let p01 = src[(y0 * sw + x1) * 3 + c] as f32;
                let p10 = src[(y1 * sw + x0) * 3 + c] as f32;
                let p11 = src[(y1 * sw + x1) * 3 + c] as f32;
                let top = p00 + (p01 - p00) * wx;
                let bot = p10 + (p11 - p10) * wx;
                out[(y * dw + x) * 3 + c] = (top + (bot - top) * wy).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod resize_tests {
    use super::*;

    fn grad(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    v[(y * w + x) * 3 + c] = ((x * 7 + y * 13 + c * 40) % 256) as u8;
                }
            }
        }
        v
    }

    #[test]
    fn 三种插值都能跑且尺寸正确() {
        let src = grad(37, 23);
        for interp in [Interp::Area, Interp::Bilinear, Interp::Bicubic] {
            let out = resize_rgb(&src, 37, 23, 64, 64, interp);
            assert_eq!(out.len(), 64 * 64 * 3, "{interp:?}");
        }
        let down = resize_rgb(&src, 37, 23, 8, 8, Interp::Area);
        assert_eq!(down.len(), 8 * 8 * 3);
    }

    #[test]
    fn area缩小求平均() {
        // 2x2 纯色块缩到 1x1，area 必须给出平均值
        let src = vec![0u8, 0, 0, 100, 100, 100, 200, 200, 200, 255, 255, 255];
        let out = resize_rgb(&src, 2, 2, 1, 1, Interp::Area);
        let avg = (0 + 100 + 200 + 255) / 4;
        assert!(
            (out[0] as i32 - avg as i32).abs() <= 2,
            "area 平均 {out[0]} vs {avg}"
        );
    }

    #[test]
    fn 同尺寸恒等() {
        let src = grad(16, 16);
        for interp in [Interp::Bilinear, Interp::Cubic_PROBE()] {
            let out = resize_rgb(&src, 16, 16, 16, 16, interp);
            assert_eq!(out, src, "{interp:?} 同尺寸应恒等");
        }
    }
    #[allow(non_snake_case)]
    fn Interp_Cubic_PROBE() -> Interp { Interp::Bicubic }
}
