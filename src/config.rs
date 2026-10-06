// 全局配置：模型路径、阈值、常量
use std::path::PathBuf;

pub fn root() -> PathBuf {
    // 优先 ALBUM_ROOT 环境变量，其次可执行文件上溯两级（target/release/album → 项目根）
    if let Ok(v) = std::env::var("ALBUM_ROOT") {
        return PathBuf::from(v);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for up in [3usize, 2, 1] {
                let mut p = dir.to_path_buf();
                for _ in 0..up {
                    p = match p.parent() {
                        Some(x) => x.to_path_buf(),
                        None => break,
                    };
                }
                if p.join("models").is_dir() {
                    return p;
                }
            }
        }
    }
    PathBuf::from(".")
}

pub fn models_dir() -> PathBuf {
    root().join("models")
}
pub fn data_dir() -> PathBuf {
    root().join("data")
}
pub fn db_path() -> PathBuf {
    data_dir().join("album.db")
}
pub fn vec_path() -> PathBuf {
    data_dir().join("vectors.npz")
}

// ---------- 模型 ----------
pub fn mobileclip_vision() -> PathBuf {
    models_dir().join("mobileclip2-s0/vision_model.onnx")
}
pub fn mobileclip_text() -> PathBuf {
    models_dir().join("mobileclip2-s0/text_model_quantized.onnx")
}
pub fn mobileclip_tok() -> PathBuf {
    models_dir().join("mobileclip2-s0/tokenizer.json")
}
pub fn ocr_det() -> PathBuf {
    models_dir().join("rapidocr/ch_PP-OCRv4_det_mobile.onnx")
}
pub fn ocr_rec() -> PathBuf {
    models_dir().join("rapidocr/ch_PP-OCRv4_rec_mobile.onnx")
}
pub fn ocr_cls() -> PathBuf {
    models_dir().join("rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx")
}
pub fn scrfd_model() -> PathBuf {
    models_dir().join("face/scrfd_10g.onnx")
}
pub fn arcface_model() -> PathBuf {
    models_dir().join("face/arcface_w600k_r50.onnx")
}
/// NIMA 已由 tools/tflite2onnx.py 转成 ONNX，Rust 侧只需 ort 一个运行时
pub fn nima_aesthetic() -> PathBuf {
    let p = models_dir().join("nima_onnx/nima_aesthetic.onnx");
    if p.exists() {
        p
    } else {
        models_dir().join("nima/nima_aesthetic_fp16.tflite")
    }
}
pub fn nima_technical() -> PathBuf {
    let p = models_dir().join("nima_onnx/nima_technical.onnx");
    if p.exists() {
        p
    } else {
        models_dir().join("nima/nima_technical_fp16.tflite")
    }
}

// ---------- MobileCLIP2-S0 ----------
// 实测校正（12 类 × 2 张图文检索）：本机 vision_model.onnx 吃的是 RGB/[0,1] 直出，
// 不做 mean/std 归一化；按 OpenAI CLIP 的 mean/std 归一化后 Top1 从 11/24 掉到 2/24。
pub const CLIP_IMG_SIZE: usize = 256;
pub const CLIP_DIM: usize = 512;
pub const CLIP_MEAN: [f32; 3] = [0.0, 0.0, 0.0];
pub const CLIP_STD: [f32; 3] = [1.0, 1.0, 1.0];
/// 实测 cubic 与 area 的检索效果几乎一致（Top1 都是 11/24），cubic 略好
pub const CLIP_INTERP_CUBIC: bool = true;
pub const CLIP_TEXT_CTX: usize = 77;
pub const CLIP_BOS: i64 = 49406;
pub const CLIP_EOS: i64 = 49407;
/// 图文相似度分布实测 [-0.06, 0.26]，零样本标签阈值按此标定
pub const TAG_THRESH: f32 = 0.15;

// ---------- 人脸 ----------
pub const FACE_DET_THRESH: f32 = 0.5;
pub const ARCFACE_INPUT: u32 = 112;
pub const FACE_COS_ASSIGN: f32 = 0.40;
pub const FACE_COS_NEWPROTO: f32 = 0.55;
pub const PERSON_MAX_PROTOS: usize = 20;

// ---------- 事件 / 连拍 ----------
pub const EVENT_GAP_H: f64 = 8.0;
pub const EVENT_GPS_KM: f64 = 200.0;
pub const EVENT_GPS_GAP_H: f64 = 2.0;
pub const BURST_DT_S: f64 = 3.0;
pub const BURST_PHASH_HAMMING: u32 = 8;
pub const MERGE_PHASH_HAMMING: u32 = 6;

// ---------- 择优权重 ----------
pub const W_AESTHETIC: f64 = 0.35;
pub const W_TECH: f64 = 0.15;
pub const W_SHARP: f64 = 0.15;
pub const W_EXPO: f64 = 0.15;
pub const W_FACE: f64 = 0.15;
pub const W_SMILE: f64 = 0.10;

// ---------- 运行时 ----------
pub fn ort_threads() -> usize {
    std::env::var("ALBUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| num_cpus::get().saturating_sub(1).max(1))
}

pub const IMAGE_EXTS: [&str; 9] = [
    "jpg", "jpeg", "png", "webp", "bmp", "heic", "heif", "tif", "tiff",
];

pub fn is_image_ext(ext: &str) -> bool {
    let e = ext.trim_start_matches('.').to_ascii_lowercase();
    IMAGE_EXTS.contains(&e.as_str())
}
