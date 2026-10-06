//! E. 定向 OCR：仅对疑似文档/截图运行
//!
//! ORT 能跑 PP-OCRv4 的 onnx，但 PaddleOCR 那套 DB 后处理 + CTC 解码要自己实现。
//! 这里只实现"识别已有裁剪图"的 CTC 解码器（rec 模型），
//! 文本框检测（det 模型）先接上，检测不到框就退回整图识别。

use anyhow::Result;
use std::sync::Arc;

use crate::metadata::load_image_rgb;
use crate::models::{resize_rgb, Hub, Interp};
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
        let small = resize_rgb(&rgb, w, h, rw, rh, Interp::Bilinear);
        let out = rec.run(&small, rw, rh)?;
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
///
/// 全部用 char 向量操作，绝不做字节切片。
/// 旧写法 `s[i..]`（i 是 Vec<char> 的下标）在 OCR 结果含中文时直接 panic：
/// `byte index N is not a char boundary` —— 一次就崩掉整个索引进程。
pub fn has_amount(s: &str) -> bool {
    let b: Vec<char> = s.chars().collect();
    for i in 0..b.len() {
        // 货币符号后跳 1 位；"RMB" 后跳 3 位
        let skip = match b[i] {
            '\u{a5}' | '\u{ffe5}' => 1,
            'R' if b.len() >= i + 3
                && b[i + 1].eq_ignore_ascii_case(&'M')
                && b[i + 2].eq_ignore_ascii_case(&'B') => 3,
            _ => 0,
        };
        let start = i + skip;
        if start + 3 >= b.len() || !b[start].is_ascii_digit() {
            continue;
        }
        let mut j = start;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        // 形如 1,234.56 / 1234.56：小数点后恰好两位
        if j > start
            && j + 2 < b.len()
            && (b[j] == '.' || b[j] == ',')
            && b[j + 1].is_ascii_digit()
            && b[j + 2].is_ascii_digit()
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 金额识别_含中文不panic() {
        // 回归测试：旧实现在这里 panic（byte index is not a char boundary）
        assert!(has_amount("价税合计 ￥13,568.00"));
        assert!(has_amount("技术服务费 12800.00 元"));
        assert!(has_amount("人民币壹万叁仟元整"));
        assert!(!has_amount("发票代码：011002200311"));
        assert!(!has_amount("没有任何金额"));
        assert!(!has_amount(""));
    }

    #[test]
    fn rmb前缀() {
        assert!(has_amount("RMB 2,345.00"));
        assert!(has_amount("rmb1234.56"));
        assert!(!has_amount("RMB 无金额"));
    }

    #[test]
    fn 收据不被判成发票() {
        // P0 原始 bug：收据命中通用词（金额/合计/收款）被判成 invoice
        let receipt = "收据 今收到上海市某某商贸有限公司 金额(大写）贰仟叁佰肆拾伍元整 （小写）￥2,345.00 收款人王五";
        assert_eq!(classify_bill(receipt), vec!["receipt".to_string()]);

        let invoice = "增值税专用发票 发票代码3100214130 统一社会信用代码91110101MA8RP4X72H 价税合计 ￥13,568.00";
        assert_eq!(classify_bill(invoice), vec!["invoice".to_string()]);

        let ticket = "中国铁路电子客票 车次G1234 上海虹桥 座号07车12F号 票价 ￥553.00 检票口检票口A";
        assert!(classify_bill(ticket).contains(&"ticket".to_string()));

        let boarding = "中国南方航空 登机牌 航班CZ3101 登机口A12 座位3F 电子客票号784-56839217";
        assert!(classify_bill(boarding).contains(&"boarding pass".to_string()));
    }
}

