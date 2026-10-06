//! A. 元数据层：EXIF / GPS / phash / 截图启发式 / 高速解码
//!
//! 注意 Pillow ≥9 起 `exif.get(36867)` 取不到 Exif-IFD 里的 DateTimeOriginal，
//! Rust 的 `image` crate 直接给 `exif_data`，等价于 Python 里的 `get_ifd(0x8769)`。

use anyhow::Result;
use image::ImageReader;
use std::path::Path;

use crate::db::PhotoMeta;
use crate::exif;

/// 高速解码：返回 RGB u8 的 HWC 缓冲（不含 stride 填充）
pub fn load_image_rgb(path: &Path, max_side: u32) -> Result<(Vec<u8>, u32, u32)> {
    let img = ImageReader::open(path)
        .with_guessed_format()?
        .decode()?;
    // image crate 的 DynamicImage 已按 EXIF 方向处理过（JPEG 的 orientation）
    let rgb = img.to_rgb8();
    let (w0, h0) = rgb.dimensions();
    if max_side > 0 && (w0 > max_side || h0 > max_side) {
        // 等比缩放到长边 = max_side（与 Python im.thumbnail 一致）
        let scale = (max_side as f64 / w0.max(h0) as f64).min(1.0);
        let nw = ((w0 as f64 * scale).round() as u32).max(1);
        let nh = ((h0 as f64 * scale).round() as u32).max(1);
        // area 采样，对齐 OpenCV INTER_AREA
        let small = image::imageops::resize(&rgb, nw, nh, image::imageops::FilterType::Triangle);
        return Ok((small.into_raw(), nw, nh));
    }
    Ok((rgb.into_raw(), w0, h0))
}

/// 原始尺寸（不解码整图，靠 header）
pub fn probe_size(path: &Path) -> Result<(u32, u32)> {
    let r = ImageReader::open(path)?.with_guessed_format()?;
    Ok(r.into_dimensions()?)
}

pub fn extract_metadata(path: &Path) -> Result<PhotoMeta> {
    let mut out = PhotoMeta {
        taken_at: None,
        gps_lat: None,
        gps_lon: None,
        device: None,
        width: 0,
        height: 0,
        is_screenshot: 0,
        phash: None,
    };
    let mut has_cam = false;
    // EXIF 缺失（JPEG 无 EXIF、WebP、HEIC）时不算错误，退到 mtime
    if let Ok(e) = exif::read(path) {
        let device = format!("{} {}", e.make.clone().unwrap_or_default(), e.model.clone().unwrap_or_default())
            .trim()
            .to_string();
        out.device = if device.is_empty() { None } else { Some(device) };
        out.gps_lat = e.gps_lat;
        out.gps_lon = e.gps_lon;
        has_cam = e.from_camera();
        out.taken_at = e.best_datetime().map(exif::normalize_dt).filter(|s| !s.is_empty());
    }
    if let Ok((w, h)) = probe_size(path) {
        out.width = w as i64;
        out.height = h as i64;
    }
    if out.taken_at.is_none() {
        out.taken_at = std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(|t| {
                let dt: chrono::DateTime<chrono::Local> = t.into();
                dt.format("%Y-%m-%d %H:%M:%S").to_string()
            });
    }
    if let Ok((rgb, w, h)) = load_image_rgb(path, 0) {
        out.phash = Some(phash(&rgb, w, h));
    }
    out.is_screenshot = is_screenshot(path, out.width, out.height, has_cam);
    Ok(out)
}

/// 截图启发式：文件名黑名单 → 相机 Make 判定 → 屏幕比例
fn is_screenshot(path: &Path, w: i64, h: i64, has_cam: bool) -> i64 {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    // 注意不能把 "img_" 当截图特征 —— Android 相机默认命名就是 IMG_20240713_093012.jpg
    let pats = [
        "screenshot", "screen_shot", "截屏", "截图", "screen recording", "wx_camera", "mmexport",
        "screencap",
    ];
    if pats.iter().any(|p| name.contains(p)) {
        return 1;
    }
    if has_cam {
        return 0;
    }
    if w > 0 && h > 0 {
        let r = w.min(h) as f64 / w.max(h) as f64;
        // 手机屏幕比例（9:19.5≈0.462、9:16=0.5625）
        if w >= 1000 && ((r - 0.462).abs() < 0.02 || (r - 0.5625).abs() < 0.01) {
            return 1;
        }
    }
    0
}

// ---------------------------------------------------------------- phash

/// 32×32 灰度 DCT-II，取左上 8×8 低频（去掉 DC），对比度归一化后压成 64 bit
pub fn phash(rgb: &[u8], w: u32, h: u32) -> String {
    const N: usize = 32;
    let mut g = vec![0f32; N * N];
    for y in 0..N {
        for x in 0..N {
            let sx = (x as f32 * w as f32 / N as f32) as usize;
            let sy = (y as f32 * h as f32 / N as f32) as usize;
            let sx = sx.min(w as usize - 1);
            let sy = sy.min(h as usize - 1);
            let i = (sy * w as usize + sx) * 3;
            g[y * N + x] = 0.299 * rgb[i] as f32 + 0.587 * rgb[i + 1] as f32 + 0.114 * rgb[i + 2] as f32;
        }
    }
    // 二维 DCT-II（预计算 cos 表）
    let mut cos = [[0f32; N]; N];
    for (x, row) in cos.iter_mut().enumerate() {
        for (u, v) in row.iter_mut().enumerate() {
            *v = (((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI) / (2.0 * N as f64)).cos() as f32;
        }
    }
    let mut coef = [[0f32; N]; N];
    for u in 0..8 {
        for v in 0..8 {
            let cu = if u == 0 { 1.0 / (2f32).sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / (2f32).sqrt() } else { 1.0 };
            let mut s = 0f32;
            for y in 0..N {
                for x in 0..N {
                    s += cu * cv * cos[u][x] * cos[v][y] * g[y * N + x];
                }
            }
            coef[u][v] = s;
        }
    }
    let vals: Vec<f32> = coef.iter().take(8).flat_map(|r| r.iter().take(8)).copied().collect();
    let mut sorted = vals.clone();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let med = (sorted[31] + sorted[32]) / 2.0;
    let mut bits: u64 = 0;
    for (i, v) in vals.iter().enumerate() {
        if *v > med {
            bits |= 1u64 << i;
        }
    }
    format!("{bits:016x}")
}

pub fn phash_hamming(a: &str, b: &str) -> u32 {
    match (u64::from_str_radix(a, 16), u64::from_str_radix(b, 16)) {
        (Ok(x), Ok(y)) => (x ^ y).count_ones(),
        _ => 64,
    }
}
