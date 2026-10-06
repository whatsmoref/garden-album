//! B. 视觉层：MobileCLIP2 图文向量 + 零样本标签 + NIMA 质量分 + 清晰度/曝光

use anyhow::Result;
use std::sync::Arc;

use crate::config as C;
use crate::models::{resize_rgb, Hub};

pub struct VisualAnalyzer {
    hub: Arc<Hub>,
    tag_names: Vec<String>,
    /// (n_tags, 512) 行主序；每次 top_tags 是一次矩阵向量乘
    tag_mat: Vec<Vec<f32>>,
}

impl VisualAnalyzer {
    pub fn new(hub: Arc<Hub>) -> Result<Self> {
        Self::with_tags(hub, crate::vocab::TAG_VOCAB.iter().map(|s| s.to_string()).collect())
    }

    pub fn with_tags(hub: Arc<Hub>, tag_names: Vec<String>) -> Result<Self> {
        let prompts: Vec<String> = tag_names
            .iter()
            .map(|t| format!("a photo of {t}"))
            .collect();
        // 走带缓存的 embed_text：138 个标签只算一次，之后进程内命中缓存
        let tag_mat = hub.warm_tag_matrix(&prompts)?;
        Ok(Self {
            hub,
            tag_names,
            tag_mat,
        })
    }

    pub fn embed(&self, img: &[u8], w: usize, h: usize) -> Result<Vec<f32>> {
        self.hub.clip()?.embed_image(img, w, h)
    }

    /// 零样本标签，取分数超过阈值的 top-k
    pub fn top_tags(&self, vec: &[f32], k: usize, thresh: Option<f32>) -> Vec<(String, f32)> {
        let thresh = thresh.unwrap_or(C::TAG_THRESH);
        let mut sims: Vec<(usize, f32)> = self
            .tag_mat
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut s = 0f32;
                for (a, b) in row.iter().zip(vec.iter()) {
                    s += a * b;
                }
                (i, s)
            })
            .collect();
        sims.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        sims.into_iter()
            .take(k)
            .filter(|(_, s)| *s > thresh)
            .map(|(i, s)| (self.tag_names[i].clone(), (s * 1000.0).round() / 1000.0))
            .collect()
    }

    /// (美学, 技术, 清晰度, 曝光)
    pub fn quality(&self, img: &[u8], w: usize, h: usize) -> Result<(f64, f64, f64, f64)> {
        let aes = self.hub.nima(true)?.score(img, w, h)?;
        let tec = self.hub.nima(false)?.score(img, w, h)?;
        let (shp, exp) = sharpness_exposure(img, w, h);
        let r = |x: f64| (x * 100.0).round() / 100.0;
        let r1 = |x: f64| (x * 10.0).round() / 10.0;
        Ok((r(aes), r(tec), r1(shp), r1(exp)))
    }
}

/// 清晰度（拉普拉斯方差对数）与曝光（过曝/欠曝/均值偏离）
pub fn sharpness_exposure(rgb: &[u8], w: usize, h: usize) -> (f64, f64) {
    if w < 3 || h < 3 {
        return (50.0, 50.0);
    }
    // 灰度 + 限边长到 640（与 Python 一致）
    let (gw, gh) = if w.max(h) > 640 {
        let s = 640.0 / w.max(h) as f64;
        (((w as f64 * s).round() as usize).max(2), ((h as f64 * s).round() as usize).max(2))
    } else {
        (w, h)
    };
    let small = resize_rgb(rgb, w, h, gw, gh, false);
    let mut g = vec![0f32; gw * gh];
    for i in 0..gw * gh {
        g[i] = 0.299 * small[i * 3] as f32 + 0.587 * small[i * 3 + 1] as f32 + 0.114 * small[i * 3 + 2] as f32;
    }
    // 拉普拉斯核 [-1,1,-1,1,-4,1,-1,1,-1]
    let mut var = 0f64;
    let mut n = 0f64;
    for y in 1..gh - 1 {
        for x in 1..gw - 1 {
            let i = y * gw + x;
            let l = (g[i - gw - 1] + g[i - gw + 1] + g[i + gw - 1] + g[i + gw + 1]
                - 4.0 * g[i]) as f64;
            var += l * l;
            n += 1.0;
        }
    }
    var /= n.max(1.0);
    let sharp = (100.0 * (1.0 + var).log10() / 3001.0f64.log10()).clamp(0.0, 100.0);

    let over = g.iter().filter(|v| **v >= 250.0).count() as f64 / g.len() as f64;
    let under = g.iter().filter(|v| **v <= 5.0).count() as f64 / g.len() as f64;
    let mean = g.iter().map(|v| *v as f64).sum::<f64>() / g.len() as f64;
    let expo = (100.0 * (1.0 - over * 2.0 - under * 1.5 - (mean - 118.0).abs() / 118.0 * 0.6)).clamp(0.0, 100.0);
    (sharp, expo)
}
