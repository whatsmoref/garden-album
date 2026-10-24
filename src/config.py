# -*- coding: utf-8 -*-
"""全局配置：模型路径、阈值、常量"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = Path(os.environ.get("GARDEN_DATA_DIR") or (ROOT / "data")).expanduser()
if not DATA_DIR.is_absolute():
    DATA_DIR = (ROOT / DATA_DIR).resolve()
DB_PATH = DATA_DIR / "album.db"
VEC_PATH = DATA_DIR / "vectors.npz"
# ---------- 模型 ----------
MOBILECLIP_VISION = MODELS_DIR / "mobileclip2-s0" / "vision_model.onnx"
MOBILECLIP_TEXT   = MODELS_DIR / "mobileclip2-s0" / "text_model_quantized.onnx"
MOBILECLIP_TOK    = MODELS_DIR / "mobileclip2-s0" / "tokenizer.json"
OCR_DET = MODELS_DIR / "rapidocr" / "ch_PP-OCRv4_det_mobile.onnx"
OCR_REC = MODELS_DIR / "rapidocr" / "ch_PP-OCRv4_rec_mobile.onnx"
OCR_CLS = MODELS_DIR / "rapidocr" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
SCRFD_MODEL   = MODELS_DIR / "face" / "scrfd_10g.onnx"
ARCFACE_MODEL = MODELS_DIR / "face" / "arcface_w600k_r50.onnx"
NIMA_AESTHETIC = MODELS_DIR / "nima" / "nima_aesthetic_fp16.tflite"
NIMA_TECHNICAL = MODELS_DIR / "nima" / "nima_technical_fp16.tflite"
# ---------- MobileCLIP2-S0 ----------
# 实测校正（12 类 × 2 张图文检索）：本机 vision_model.onnx 吃的是 RGB/[0,1] 直出，
# 不做 mean/std 归一化；按 OpenAI CLIP 的 mean/std 归一化后 Top1 从 11/24 掉到 2/24。
# 若换模型导出，先用 scripts/calib_clip.py 复测再改这两行。
CLIP_IMG_SIZE = 256
CLIP_DIM = 512
CLIP_MEAN = (0.0, 0.0, 0.0)
CLIP_STD  = (1.0, 1.0, 1.0)
CLIP_RESIZE_INTERP = "cubic"     # 'cubic' | 'area'
CLIP_TEXT_CTX = 77
CLIP_BOS, CLIP_EOS = 49406, 49407   # OpenAI CLIP 词表特殊 token，已核对本机 tokenizer.json
# 图文相似度分布实测 [-0.06, 0.26]，零样本标签阈值按此标定。
# 二次校准：先用原始余弦做下限，再减去「全库均值」做同义/通用词抑制，
# 这样 outdoor / indoor / sunny 这类到处都高的词不会挤掉真正的具体标签。
TAG_THRESH = 0.140          # 原始余弦下限
TAG_CENTER_THRESH = 0.020   # 去均值后的下限
TAG_REL_THRESH = 0.55       # 去均值后相对 top1 的比例下限
TAG_TOPK = 8
TAG_CALIBRATE = True
TAG_MIN_LIBRARY = 20        # 库内样本少于此数不做去均值校准
TAG_PROMPT = "a photo of {tag}"
TAG_VOCAB_VERSION = "2026.10-life"
# ---------- 开放词汇检索 ----------
SEARCH_CLIP_MIN_TOP = 0.18      # 无结构化条件时，CLIP top1 余弦下限
SEARCH_CLIP_MIN_SPREAD = 0.04   # top1 与第 5 名的差距下限，过滤“什么都像”的噪声查询
SEARCH_CLIP_WINDOW = 0.04       # 只保留距离 top1 这么近的结果
SEARCH_CLIP_LIMIT = 100
# ---------- 人脸 ----------
FACE_DET_THRESH = 0.5
ARCFACE_INPUT = 112
FACE_COS_ASSIGN = 0.40      # 归入已有聚类的余弦阈值
FACE_COS_NEWPROTO = 0.55    # 低于该相似度才新增原型（多原型适应角度）
PERSON_MAX_PROTOS = 20
# ---------- 事件 / 连拍 ----------
EVENT_GAP_H = 8.0
EVENT_GPS_KM = 200.0
EVENT_GPS_GAP_H = 2.0
BURST_DT_S = 3.0
BURST_PHASH_HAMMING = 8
MERGE_PHASH_HAMMING = 6
# ---------- 择优权重 ----------
W_AESTHETIC, W_TECH, W_SHARP, W_EXPO, W_FACE, W_SMILE = .35, .15, .15, .15, .15, .10
# ---------- 缓存 ----------
CACHE_DIR = DATA_DIR / "cache"
THUMB_CACHE_DIR = CACHE_DIR / "thumbs"
TAG_MATRIX_PATH = CACHE_DIR / "tag_matrix.npz"
TEXT_EMB_CACHE_PATH = CACHE_DIR / "text_embeddings.npz"
TEXT_EMB_CACHE_MAX = 4096
THUMB_SIZES = (160, 240, 320, 480, 720, 1024, 1600, 2048)
THUMB_QUALITY = 82
THUMB_MEM_BYTES = 96 * 1024 * 1024      # 内存 LRU 上限
THUMB_PREWARM_SIZES = (320, 480, 1024)       # 启动后台预热，不阻塞首屏
THUMB_PREWARM_THREADS = 2
API_CACHE_TTL = 20                      # 结构化接口短缓存（秒）

# ---------- 运行时 ----------
ORT_THREADS = 4   # big.LITTLE：大核数量
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif", ".tif", ".tiff"}
