//! PP-OCRv4 rec 模型的字符字典（6623 字），从 onnx 的 metadata_props 导出。
//! 用 tools/export_ocr_dict.py 生成，不要手改。

use once_cell::sync::Lazy;

pub static OCR_DICT: Lazy<Vec<String>> = Lazy::new(|| {
    // include_str! 编译期展开，避免运行时找文件
    const RAW: &str = include_str!("ocr_dict.json");
    serde_json::from_str(RAW).unwrap_or_default()
});

pub fn load() -> Option<Vec<String>> {
    if OCR_DICT.is_empty() {
        None
    } else {
        Some(OCR_DICT.clone())
    }
}

/// 字典 + CTC blank，总类别数（= rec 模型输出最后一维）
pub fn classes() -> usize {
    OCR_DICT.len() + 1
}
