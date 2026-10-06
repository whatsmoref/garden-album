//! 本地相册语义检索系统 —— Rust 重写版
//!
//! 分层：models/metadata/visual/face → events/ocr/quality → parser/search/albums → indexer

pub mod albums;
pub mod config;
pub mod db;
pub mod events;
pub mod exif;
pub mod face;
pub mod indexer;
pub mod metadata;
pub mod models;
pub mod ocr;
pub mod ocrdict;
pub mod parser;
pub mod quality;
pub mod search;
pub mod visual;
pub mod vocab;

/// 标签 + 文件名 → FTS 文档
pub fn tags_of_doc(tags: &[(String, f32)], stem: &str) -> String {
    let mut s: Vec<&str> = tags.iter().map(|(t, _)| t.as_str()).collect();
    s.push(stem);
    s.join(" ")
}
