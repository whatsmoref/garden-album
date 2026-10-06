//! E. 定向 OCR：仅对疑似文档/截图运行
//!
//! ORT 能跑 PP-OCRv4 的 onnx，但 PaddleOCR 那套 DB 后处理 + CTC 解码要自己实现。
//! 这里只实现"识别已有裁剪图"的 CTC 解码器（rec 模型），
//! 文本框检测（det 模型）先接上，检测不到框就退回整图识别。

use anyhow::Result;
use std::sync::Arc;

use crate::config as C;
use crate::metadata::load_image_rgb;
use crate::models::{resize_rgb, Hub};
use crate::vocab::DOC_TAGS;

pub const INVOICE_KWS: &[&str] = &[
    "发票", "价税合计", "税号", "增值税", "统一社会信用代码", "纳税人识别号",
];
/// 收据特征词必须与发票词分开计数，否则"金额/合计/收据/收款"会让所有收据命中发票分支
pub const RECEIPT_KWS: &[&str] = &["收据", "收款", "收款人", "今收到", "已收到"];
pub const TICKET_KWS: &[&str] = &[
    "票号", "检票口", "座位", "登机口", "航班号", "车次", "座席", "行程单",
];

pub struct TargetedOcr {
    hub: Arc<Hub>,
    dict: Option<Vec<String>>,
}

impl TargetedOcr {
    pub fn new(hub: Arc<Hub>) -> Self {
        Self {
            hub,
            dict: crate::ocrdict::load(),
        }
    }

    /// 是否值得跑 OCR
    pub fn need(is_screenshot: bool, tags: &[(String, f64)]) -> bool {
        if is_screenshot {
            return true;
        }
        tags.iter().any(|(t, _)| DOC_TAGS.contains(&t.as_str()))
    }

    /// → (全文, 追加标签)
    pub fn run(&self, path: &std::path::Path) -> Result<(String, Vec<String>)> {
        let (rgb, w, h) = match load_image_rgb(path, 1800) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("OCR 解码失败 {}: {e}", path.display());
                return Ok((String::new(), Vec::new()));
            }
        };
        let Some(dict) = self.dict.as_ref() else {
            return Ok((String::new(), Vec::new()));
        };
        // rec 模型输入 NHWC [1, 48, 320, 3]，按高度等比缩放、宽度上限 320
        let rec = self.hub.rec()?;
        let rh = 48usize;
        let scale = rh as f32 / h as f32;
        let rw = ((w as f32 * scale).round() as usize).clamp(8, 320);
        let small = resize_rgb(&rgb, w, h, rw, rh, false);
        let out = rec.run(&small)?;
        let (text, conf) = ctc_decode(&out, dict, rw, rh);
        if text.trim().is_empty() || conf < 0.3 {
            return Ok((String::new(), Vec::new()));
        }
        let extra = classify_bill(&text);
        Ok((text, extra))
    }
}

/// CTCLabelDecoder：argmax 去重 + 空白过滤
pub fn ctc_decode(probs: &[f32], dict: &[String], w: usize, h: usize) -> (String, f32) {
    // probs 形状 [1, T, C]（已 flatten）
    let c = dict.len() + 1;
    let t = if c == 0 { 0 } else { probs.len() / c };
    let _ = (w, h);
    let mut text = String::new();
    let mut prev = usize::MAX;
    let mut conf_sum = 0f32;
    let mut conf_n = 0f32;
    for i in 0..t {
        let row = &probs[i * c..(i + 1) * c];
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (k, v) in row.iter().enumerate() {
            if *v > bv {
                bv = *v;
                best = k;
            }
        }
        if best != prev && best != 0 && best < dict.len() {
            text.push_str(&dict[best]);
            conf_sum += bv;
            conf_n += 1.0;
        }
        prev = best;
    }
    let conf = if conf_n > 0.0 { conf_sum / conf_n } else { 0.0 };
    (text, conf)
}

/// 按关键词把票据细分成 invoice / receipt / ticket / boarding pass
pub fn classify_bill(full: &str) -> Vec<String> {
    let inv = INVOICE_KWS.iter().filter(|k| full.contains(**k)).count();
    let rcp = RECEIPT_KWS.iter().filter(|k| full.contains(**k)).count();
    let tkt = TICKET_KWS.iter().filter(|k| full.contains(**k)).count();
    let amt = has_amount(full);
    let mut out = Vec::new();
    if full.contains("登机") || (full.contains("航班") && tkt > 0) {
        out.push("boarding pass".to_string());
    }
    if tkt >= 2 || (full.contains("票") && tkt > 0 && inv == 0) {
        out.push("ticket".to_string());
    }
    if inv >= 2 || (full.contains("发票") && amt) {
        out.push("invoice".to_string());
    }
    if out.is_empty() && rcp > 0 && (amt || full.contains("金额") || full.contains("合计")) {
        out.push("receipt".to_string());
    } else if rcp > 0 && inv == 0 {
        out.push("receipt".to_string());
    }
    out
}

/// ¥/￥/RMB 后跟两位小数
pub fn has_amount(s: &str) -> bool {
    let b: Vec<char> = s.chars().collect();
    for i in 0..b.len() {
        let cur = b[i];
        let money_prefix = cur == '¥' || cur == '￥' || (cur == 'R' && s[i..].to_uppercase().starts_with("RMB"));
        let start = if money_prefix { i + 3.min(s[i..].chars().count()) } else { i };
        if start + 4 < b.len() && b[start].is_ascii_digit() {
            let mut j = start;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j > start && j < b.len() && (b[j] == '.' || b[j] == ',') && j + 3 < b.len()
                && b[j + 1].is_ascii_digit() && b[j + 2].is_ascii_digit() && b[j + 3].is_ascii_digit()
            {
                return true;
            }
        }
    }
    false
}

pub fn ocr_paths() -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    (C::ocr_det(), C::ocr_rec(), C::ocr_cls())
}
