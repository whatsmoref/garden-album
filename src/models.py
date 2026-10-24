# -*- coding: utf-8 -*-
"""模型加载中心（懒加载单例）：ONNX Runtime / TFLite / RapidOCR / HSEmotion

除了模型本身，这里还负责两个跨进程缓存：
- CLIP 文本向量：同一句查询只编码一次，落盘到 data/cache。
- 标签矩阵由 visual.py 负责，同样落盘，避免每次启动重跑 20~40 秒。
"""
import atexit
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort
from . import config as C


def _file_fingerprint(path: Path) -> str:
    try:
        st = path.stat()
        return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return f"{path.name}:missing"


class TextEmbedCache:
    """LRU + 磁盘缓存：CLIP 文本塔很贵（~180 ms），同句必须命中。"""

    def __init__(self, path: Path, fingerprint: str, max_items: int = 4096):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.max_items = max_items
        self._lock = threading.Lock()
        self._dirty = 0
        self._items = OrderedDict()
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            z = np.load(self.path, allow_pickle=False)
            if str(z["fingerprint"]) != self.fingerprint:
                return
            texts = z["texts"].tolist()
            vecs = z["vecs"]
            for t, v in zip(texts, vecs):
                self._items[str(t)] = np.asarray(v, np.float32)
        except Exception:
            self._items.clear()

    def get(self, text: str):
        with self._lock:
            v = self._items.get(text)
            if v is None:
                return None
            self._items.move_to_end(text)
            return v

    def put(self, text: str, vec: np.ndarray):
        with self._lock:
            self._items[text] = np.asarray(vec, np.float32)
            self._items.move_to_end(text)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
            self._dirty += 1
            if self._dirty >= 32:
                self._save_locked()

    def save(self):
        with self._lock:
            self._save_locked()

    def _save_locked(self):
        if not self._items:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            texts = list(self._items.keys())
            vecs = np.stack([self._items[k] for k in texts]).astype(np.float32)
            tmp = self.path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, fingerprint=self.fingerprint,
                                texts=np.asarray(texts, dtype=np.str_), vecs=vecs)
            tmp.replace(self.path)
            self._dirty = 0
        except Exception:
            pass

    def __len__(self):
        return len(self._items)
def make_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = C.ORT_THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
def load_tflite(path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    it = Interpreter(model_path=str(path), num_threads=C.ORT_THREADS)
    it.allocate_tensors()
    return it
class ModelHub:
    _inst = None
    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst
    def __init__(self):
        self._cv = self._ct = self._tok = None
        self._tok_cfg = {}
        self._scrfd = self._arc = None
        self._hse = None
        self._ocr = None
        self._text_cache = None

    @property
    def text_cache(self):
        if self._text_cache is None:
            self._text_cache = TextEmbedCache(
                C.TEXT_EMB_CACHE_PATH,
                _file_fingerprint(C.MOBILECLIP_TEXT),
                C.TEXT_EMB_CACHE_MAX)
            atexit.register(self._text_cache.save)
        return self._text_cache
    @property
    def clip_vision(self):
        if self._cv is None: self._cv = make_session(C.MOBILECLIP_VISION)
        return self._cv
    @property
    def clip_text(self):
        if self._ct is None: self._ct = make_session(C.MOBILECLIP_TEXT)
        return self._ct
    @property
    def scrfd(self):
        if self._scrfd is None: self._scrfd = make_session(C.SCRFD_MODEL)
        return self._scrfd
    @property
    def arcface(self):
        if self._arc is None: self._arc = make_session(C.ARCFACE_MODEL)
        return self._arc
    # ---------- tokenizer ----------
    @property
    def tokenizer(self):
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(C.MOBILECLIP_TOK))
            ids = self._tok.encode("test").ids
            vs = self._tok.get_vocab_size()
            # 探测该 tokenizer 是否已自动加 CLIP BOS/EOS
            self._tok_cfg = {
                "auto": bool(ids) and ids[0] == C.CLIP_BOS and ids[-1] == C.CLIP_EOS,
                "can_manual": vs > C.CLIP_EOS,
            }
        return self._tok
    def _tokens(self, text):
        self.tokenizer
        ids = self.tokenizer.encode(text).ids
        if self._tok_cfg["can_manual"] and not self._tok_cfg["auto"]:
            ids = [C.CLIP_BOS] + ids[: C.CLIP_TEXT_CTX - 2] + [C.CLIP_EOS]
        else:
            ids = ids[: C.CLIP_TEXT_CTX]
        ids += [0] * (C.CLIP_TEXT_CTX - len(ids))
        return np.asarray([ids], np.int64)
    # ---------- CLIP 编码 ----------
    def embed_image(self, img_rgb):
        interp = cv2.INTER_CUBIC if C.CLIP_RESIZE_INTERP == "cubic" else cv2.INTER_AREA
        x = cv2.resize(img_rgb, (C.CLIP_IMG_SIZE, C.CLIP_IMG_SIZE), interpolation=interp)
        x = x.astype(np.float32) / 255.0
        if C.CLIP_MEAN != (0.0, 0.0, 0.0):        # mean/std 全 0/1 时跳过，省一次运算
            x = (x - np.asarray(C.CLIP_MEAN, np.float32)) / np.asarray(C.CLIP_STD, np.float32)
        x = x.transpose(2, 0, 1)[None]
        s = self.clip_vision
        v = np.asarray(s.run(None, {s.get_inputs()[0].name: x})[0]).reshape(-1)[: C.CLIP_DIM]
        return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
    def embed_text(self, text):
        key = " ".join(str(text).split())
        hit = self.text_cache.get(key)
        if hit is not None:
            return hit
        ids = self._tokens(key)
        s = self.clip_text
        inp = s.get_inputs()[0]
        dtype = np.int64 if inp.type == "tensor(int64)" else np.int32
        out = np.asarray(s.run(None, {inp.name: ids.astype(dtype)})[0])
        if out.ndim == 3:                       # (1,77,D)：取 EOS 位置池化
            seq = ids[0].tolist()
            eos = len(seq) - 1
            while eos > 0 and seq[eos] != C.CLIP_EOS:      # 找真正的 EOS，不是 argmax
                eos -= 1
            v = out[0, eos]
        else:
            v = out.reshape(-1)[: C.CLIP_DIM]
        v = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
        self.text_cache.put(key, v)
        return v

    def embed_texts(self, texts, progress=None):
        out = []
        for i, t in enumerate(texts):
            out.append(self.embed_text(t))
            if progress is not None and (i + 1) % 16 == 0:
                progress(i + 1, len(texts))
        return np.stack(out) if out else np.zeros((0, C.CLIP_DIM), np.float32)

    @property
    def text_fingerprint(self):
        return _file_fingerprint(C.MOBILECLIP_TEXT)
    # ---------- 表情 ----------
    @property
    def hse(self):
        if self._hse is None:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            self._hse = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
        return self._hse
    # ---------- OCR ----------
    @property
    def ocr(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            try:
                self._ocr = RapidOCR(det_model_path=str(C.OCR_DET),
                                     rec_model_path=str(C.OCR_REC), cls_model_path=str(C.OCR_CLS))
            except TypeError:
                try:
                    self._ocr = RapidOCR({"det_model_path": str(C.OCR_DET),
                                          "rec_model_path": str(C.OCR_REC), "cls_model_path": str(C.OCR_CLS)})
                except TypeError:
                    self._ocr = RapidOCR()      # 老版本 API 兜底
        return self._ocr
