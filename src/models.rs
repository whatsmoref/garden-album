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
        let resized = resize_rgb(
            img, w, h, s, s,
            if C::CLIP_INTERP_CUBIC { Interp::Bicubic } else { Interp::Area },
        );
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
        // 本机 scrfd_10g.onnx 输出无 batch 维：scores(N,1)/bbox(N,4)/kps(N,10)
        // 部分导出带 (1,N,C)，统一按 shape 压成二维（rows() 取行首）
        let (score, n) = rows_of(outs[0].try_extract_tensor::<f32>()?, 1);
        let (bbox, _) = rows_of(outs[self.fmc].try_extract_tensor::<f32>()?, 4);
        let (kps, _) = rows_of(outs[self.fmc * 2].try_extract_tensor::<f32>()?, 10);

        let mut boxes_all: Vec<[f32; 4]> = Vec::new();
        let mut kps_all: Vec<[f32; 10]> = Vec::new();
        let mut scores_all: Vec<f32> = Vec::new();
        for stride in self.strides.iter() {
            let side = 640 / stride;
            let na = ((n / (side * side).max(1)).max(1)) as f32;
            for j in 0..n {
                let sv = score[j];
                if sv < thresh {
                    continue;
                }
                let anchor = (j as f32 / na) as usize;
                let stride_f = *stride as f32;
                let cx = (anchor % side) as f32 * stride_f;
                let cy = (anchor / side) as f32 * stride_f;
                // srow/brow/krow 已是扁平切片，直接按行偏移取
                let d: &[f32] = &bbox[j * 4..j * 4 + 4];
                let k: &[f32] = &kps[j * 10..j * 10 + 10];
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
/// 旧实现是「逐行扫描 + 纵向扩张」，会把相邻文本行吃成一个框，
/// 又会在概率图的凹陷处断开，票据 OCR 基本不可用。
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
        // DB 的 unclip：目标面积 = 原面积 × 1.6，反推单边扩张量
        let want = (bw * bh * 1.6f32).sqrt();
        let expand = ((want - bw.min(bh)).max(2.0) / 2.0).min(bw.min(bh) * 0.5);
        out.push(TextBox {
            x0: (x0 as f32 - expand).max(0.0) * sx,
            y0: (y0 as f32 - expand).max(0.0) * sy,
            x1: (x1 as f32 + expand + 1.0).min(ow as f32) * sx,
            y1: (y1 as f32 + expand + 1.0).min(oh as f32) * sy,
            score,
        });
    }
    // 阅读顺序：从上到下，同一行从左到右
    let row_h = oh as f32 * 0.01;
    out.sort_by(|a, b| {
        if (a.y0 - b.y0).abs() < row_h {
            a.x0.total_cmp(&b.x0)
        } else {
            a.y0.total_cmp(&b.y0)
        }
    });
    out
}

/// 4 邻域连通域标记，返回 (x0, y0, x1, y1, 平均概率)
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
    const MIN_AREA: usize = 4; // 滤掉单像素噪点
    for start in 0..w * h {
        if visited[start] || prob.get(start).copied().unwrap_or(0.0) < bin_thresh {
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
            for (nx, ny) in [
                (cx as isize - 1, cy as isize),
                (cx as isize + 1, cy as isize),
                (cx as isize, cy as isize - 1),
                (cx as isize, cy as isize + 1),
            ] {
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if !visited[ni] && prob[ni] >= bin_thresh {
                    visited[ni] = true;
                    stack.push(ni);
                }
            }
        }
        let score = if n > 0.0 { sum / n } else { 0.0 };
        let area = (x1 - x0 + 1) * (y1 - y0 + 1);
        if area >= MIN_AREA && score >= score_thresh {
            out.push((x0, y0, x1, y1, score));
        }
    }
    out
}

// ---------------------------------------------------------------- Hub

pub struct Hub {
    clip: Mutex<Option<std::sync::Arc<Clip>>>,
    nima_a: Mutex<Option<std::sync::Arc<Nima>>>,
    nima_t: Mutex<Option<std::sync::Arc<Nima>>>,
    scrfd: Mutex<Option<std::sync::Arc<Scrfd>>>,
    arcface: Mutex<Option<std::sync::Arc<Arcface>>>,
    rec: Mutex<Option<std::sync::Arc<Rec>>>,
    det: Mutex<Option<std::sync::Arc<Det>>>,
    text_cache: Mutex<HashMap<String, std::sync::Arc<Vec<f32>>>>,
}

impl Hub {
    pub fn new() -> Self {
        Self {
            clip: Mutex::new(None),
            nima_a: Mutex::new(None),
            nima_t: Mutex::new(None),
            scrfd: Mutex::new(None),
            arcface: Mutex::new(None),
            rec: Mutex::new(None),
            det: Mutex::new(None),
            text_cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn clip(&self) -> Result<std::sync::Arc<Clip>> {
        let mut g = self.clip.lock();
        if let Some(c) = g.as_ref() {
            return Ok(c.clone());
        }
        let c = std::sync::Arc::new(Clip::load()?);
        *g = Some(c.clone());
        Ok(c)
    }

    /// CLIP 文本向量带缓存：Python 版每次查询重跑文本塔要 331ms，
    /// 解析出的短语高度重复，缓存后开放词汇查询降到 ~0.1ms。
    pub fn embed_text_cached(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(v) = self.text_cache.lock().get(text) {
            return Ok((**v).clone());
        }
        let v = self.clip()?.embed_text(text)?;
        self.text_cache
            .lock()
            .insert(text.to_string(), std::sync::Arc::new(v.clone()));
        Ok(v)
    }

    pub fn nima(&self, aesthetic: bool) -> Result<std::sync::Arc<Nima>> {
        let cell = if aesthetic { &self.nima_a } else { &self.nima_t };
        let mut g = cell.lock();
        if let Some(c) = g.as_ref() {
            return Ok(c.clone());
        }
        let p = if aesthetic { C::nima_aesthetic() } else { C::nima_technical() };
        let n = std::sync::Arc::new(Nima::load(&p)?);
        *g = Some(n.clone());
        Ok(n)
    }

    pub fn scrfd(&self) -> Result<std::sync::Arc<Scrfd>> {
        let mut g = self.scrfd.lock();
        if let Some(c) = g.as_ref() {
            return Ok(c.clone());
        }
        let s = std::sync::Arc::new(Scrfd::load()?);
        *g = Some(s.clone());
        Ok(s)
    }

    pub fn rec(&self) -> Result<std::sync::Arc<Rec>> {
        let mut g = self.rec.lock();
        if let Some(c) = g.as_ref() {
            return Ok(c.clone());
        }
        let r = std::sync::Arc::new(Rec::load()?);
        *g = Some(r.clone());
        Ok(r)
    }

    pub fn det(&self) -> Result<std::sync::Arc<Det>> {
        let mut g = self.det.lock();
        if let Some(c) = g.as_ref() {
            return Ok(c.clone());
        }
        let d = std::sync::Arc::new(Det::load()?);
        *g = Some(d.clone());
        Ok(d)
    }

    pub fn arcface(&self) -> Result<std::sync::Arc<Arcface>> {
        let mut g = self.arcface.lock();
        if let Some(c) = g.as_ref() {
            return Ok(c.clone());
        }
        let a = std::sync::Arc::new(Arcface::load()?);
        *g = Some(a.clone());
        Ok(a)
    }

    pub fn warm_tag_matrix(&self, prompts: &[String]) -> Result<Vec<Vec<f32>>> {
        prompts.iter().map(|p| self.embed_text_cached(p)).collect()
    }
}

// ---------------------------------------------------------------- 工具

/// 造 owned f32 张量（绕开 ort 的 inputs! 对借用 ndarray 的限制）
fn f32_tensor(shape: &[usize], data: Vec<f32>) -> Result<ort::value::Tensor<f32>> {
    let expect: usize = shape.iter().product();
    if expect != data.len() {
        bail!("张量形状 {:?} 与数据长度 {} 不符", shape, data.len());
    }
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape.to_vec()), data)
        .map_err(|e| anyhow::anyhow!("张量构造失败: {e}"))?;
    ort::value::Tensor::from_array(arr)
        .map_err(|e| anyhow::anyhow!("张量转 Tensor 失败: {e}"))
}

fn i64_tensor(shape: &[usize], data: Vec<i64>) -> Result<ort::value::Tensor<i64>> {
    let expect: usize = shape.iter().product();
    if expect != data.len() {
        bail!("张量形状 {:?} 与数据长度 {} 不符", shape, data.len());
    }
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape.to_vec()), data)
        .map_err(|e| anyhow::anyhow!("张量构造失败: {e}"))?;
    ort::value::Tensor::from_array(arr)
        .map_err(|e| anyhow::anyhow!("张量转 Tensor 失败: {e}"))
}

/// 把 (Shape, &[f32]) 按行切开；(1,N,C) 与 (N,C) 两种导出都能处理
/// (1,N,C) / (N,C) 两种导出统一成扁平 &[f32] + 行数 C
fn rows_of(sd: (&ort::value::Shape, &[f32]), cols: usize) -> (Vec<f32>, usize) {
    let (shape, data) = sd;
    let n = if shape.len() == 3 {
        shape[1] as usize * shape[2] as usize / cols
    } else if shape.len() == 2 {
        shape[0] as usize * shape[1] as usize / cols
    } else {
        0
    };
    (data[..(n * cols).min(data.len())].to_vec(), n)
}

/// 从 (Shape, &[f32]) 取前 dim 个元素
fn extract_vec(shape_and_data: (&ort::value::Shape, &[f32]), dim: usize) -> Result<Vec<f32>> {
    let (_shape, data) = shape_and_data;
    if data.len() < dim {
        bail!("模型输出维度不足：期望 {dim}，实际 {}", data.len());
    }
    Ok(data[..dim].to_vec())
}

pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let d = if n > 1e-8 { n } else { 1.0 };
    v.iter().map(|x| x / d).collect()
}

fn nms(boxes: &[[f32; 4]], scores: &[f32], thr: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_unstable_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    let area = |b: &[f32; 4]| ((b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)) as f64;
    let mut keep = Vec::new();
    while !order.is_empty() {
        let i = order[0];
        keep.push(i);
        let mut rest = Vec::new();
        for &j in &order[1..] {
            let (a, b) = (&boxes[i], &boxes[j]);
            let xx1 = a[0].max(b[0]);
            let yy1 = a[1].max(b[1]);
            let xx2 = a[2].min(b[2]);
            let yy2 = a[3].min(b[3]);
            let inter = ((xx2 - xx1).max(0.0) * (yy2 - yy1).max(0.0)) as f64;
            let iou = inter / (area(a) + area(b) - inter + 1e-9);
            if iou <= thr as f64 {
                rest.push(j);
            }
        }
        order = rest;
    }
    keep
}

/// 缩放方式（HWC u8 → HWC u8）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    /// 缩小时按 box 平均（等价 OpenCV INTER_AREA），放大时退化为双线性
    Area,
    Bilinear,
    /// Catmull-Rom 三次插值（等价 OpenCV INTER_CUBIC 的 a=-0.75）
    Bicubic,
}

/// 三次卷积核（OpenCV INTER_CUBIC 的 a = -0.75），4 邻域权重 + 归一化。
///
/// 注意：不能用「Catmull-Rom 四点插值」那套公式（w0=(A+2)t³-(A+3)t²+1 …），
/// 它在 t=0 处给出 (0.5,0,0,0.5) 而不是 (0,1,0,0)，导致同尺寸缩放不再是恒等。
#[inline]
fn cubic_kernel(t: f32) -> f32 {
    const A: f32 = -0.75;
    let t = t.abs();
    let t2 = t * t;
    let t3 = t2 * t;
    if t <= 1.0 {
        (A + 2.0) * t3 - (A + 3.0) * t2 + 1.0
    } else if t < 2.0 {
        A * t3 - 5.0 * A * t2 + 8.0 * A * t - 4.0 * A
    } else {
        0.0
    }
}

/// 取 (floor(x)-1 .. floor(x)+2) 四个采样点的归一化权重
#[inline]
fn cubic_weights(frac: f32) -> [f32; 4] {
    let mut w = [
        cubic_kernel(frac + 1.0),
        cubic_kernel(frac),
        cubic_kernel(frac - 1.0),
        cubic_kernel(frac - 2.0),
    ];
    let s: f32 = w.iter().sum();
    if s.abs() > 1e-6 {
        for v in w.iter_mut() {
            *v /= s;
        }
    } else {
        w = [0.0, 1.0, 0.0, 0.0];
    }
    w
}

pub fn resize_rgb(
    src: &[u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    interp: Interp,
) -> Vec<u8> {
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
            // cubic_weights 返回的 4 个权重对应采样点 (y0-1, y0, y0+1, y0+2)，
            // 所以下面必须用 (y0 - 1 + dy) 取样。写成 y0 + dy 会整体错开一格，
            // 同尺寸缩放就不再是恒等（测试 models::resize_tests::同尺寸恒等 会抓到）。
            let wy = cubic_weights(fypos - y0 as f32);
            for x in 0..dw {
                let fxpos = (x as f32 + 0.5) * fx - 0.5;
                let x0 = fxpos.floor() as isize;
                let wx = cubic_weights(fxpos - x0 as f32);
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (k, wyi) in wy.iter().enumerate() {
                        let yy = (y0 - 1 + k as isize).clamp(0, sh as isize - 1) as usize;
                        let mut row = 0.0f32;
                        for (m, wxi) in wx.iter().enumerate() {
                            let xx = (x0 - 1 + m as isize).clamp(0, sw as isize - 1) as usize;
                            row += src[(yy * sw + xx) * 3 + c] as f32 * wxi;
                        }
                        acc += row * wyi;
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
#[allow(non_snake_case)]
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

    /// 假 DB 概率图：3 行文字，每行 3 个 14x6 的块
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
        // 旧实现（逐行扫描+纵向扩张）会返回覆盖全图的 1~2 个大框
        let (p, w, h) = fake_db_map();
        let boxes = unclip_boxes(&p, w, h, 640, 480, 0.3, 0.5);
        assert_eq!(boxes.len(), 9);
        let covers_all = boxes.iter().any(|b| (b.x1 - b.x0) > 600.0 && (b.y1 - b.y0) > 400.0);
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

    #[test]
    fn 三种插值都能跑且尺寸正确() {
        let src = grad(37, 23);
        for interp in [Interp::Area, Interp::Bilinear, Interp::Bicubic] {
            assert_eq!(resize_rgb(&src, 37, 23, 64, 64, interp).len(), 64 * 64 * 3);
            assert_eq!(resize_rgb(&src, 37, 23, 8, 8, interp).len(), 8 * 8 * 3);
            assert_eq!(resize_rgb(&src, 37, 23, 128, 128, interp).len(), 128 * 128 * 3);
        }
    }

    #[test]
    fn area缩小求平均() {
        let src = vec![0u8, 0, 0, 100, 100, 100, 200, 200, 200, 255, 255, 255];
        let out = resize_rgb(&src, 2, 2, 1, 1, Interp::Area);
        let avg = (0 + 100 + 200 + 255) / 4;
        assert!((out[0] as i32 - avg as i32).abs() <= 2, "area 平均 {} vs {avg}", out[0]);
    }

    #[test]
    fn 同尺寸恒等() {
        let src = grad(16, 16);
        for interp in [Interp::Bilinear, Interp::Bicubic] {
            assert_eq!(resize_rgb(&src, 16, 16, 16, 16, interp), src, "{interp:?}");
        }
    }

    #[test]
    fn cubic不产生越界值() {
        // u8 的 cubic 会因权重为负而溢出，必须被 clamp 住
        let src = vec![255u8; 8 * 8 * 3];
        for interp in [Interp::Bilinear, Interp::Bicubic] {
            let out = resize_rgb(&src, 8, 8, 16, 16, interp);
            assert_eq!(out.len(), 16 * 16 * 3);
            assert!(out.iter().all(|v| *v == 255), "{interp:?} 纯白输入应全白");
        }
        // 纯黑输入同理（权重负 × 0 仍为 0）
        let black = vec![0u8; 8 * 8 * 3];
        for interp in [Interp::Bilinear, Interp::Bicubic] {
            assert!(resize_rgb(&black, 8, 8, 16, 16, interp).iter().all(|v| *v == 0));
        }
    }
}
