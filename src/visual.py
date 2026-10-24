# -*- coding: utf-8 -*-
"""B. 视觉层：MobileCLIP2 图文向量 + 零样本标签 + NIMA 质量分

标签矩阵（~270 个生活化标签）首次构建后落盘缓存，后续启动直接 mmap 载入；
标签选择做了「同义归并 + 全库去均值校准 + 相对阈值」，避免 outdoor/sunny
这类通用词挤掉具体标签。
"""
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import cv2

from . import config as C
from .models import ModelHub, load_tflite, _file_fingerprint
from .vocab import TAG_VOCAB, TAG_GROUPS, TAG_GENERIC


class NIMA:
    def __init__(self, path):
        self.it = load_tflite(path)
        d = self.it.get_input_details()[0]
        self.iidx, self.ishape, self.idtype = d["index"], tuple(d["shape"]), d["dtype"]
        self.oidx = self.it.get_output_details()[0]["index"]

    def score(self, img_rgb):        # → 1~10
        x = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
        if len(self.ishape) == 4 and self.ishape[1] == 3:   # NCHW 导出
            x = x.transpose(0, 3, 1, 2)
        self.it.set_tensor(self.iidx, x.astype(self.idtype))
        self.it.invoke()
        p = np.asarray(self.it.get_tensor(self.oidx))[0].astype(np.float64)
        p = p / p.sum()
        return float(np.dot(p, np.arange(1, len(p) + 1)))


def sharpness_exposure(img_rgb):    # → (清晰度0~100, 曝光0~100)
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    if max(h, w) > 640:
        s = 640 / max(h, w)
        g = cv2.resize(g, (int(w * s), int(h * s)))
    var = cv2.Laplacian(g, cv2.CV_64F).var()
    sharp = min(100.0, 100.0 * np.log10(1 + var) / np.log10(3001.0))
    over, under, m = float((g >= 250).mean()), float((g <= 5).mean()), float(g.mean())
    expo = max(0.0, min(100.0, 100.0 * (1 - over * 2 - under * 1.5 - abs(m - 118) / 118 * 0.6)))
    return float(sharp), float(expo)


def tag_matrix_fingerprint(tags):
    payload = json.dumps({
        "version": C.TAG_VOCAB_VERSION,
        "prompt": C.TAG_PROMPT,
        "tags": list(tags),
        "text_model": _file_fingerprint(C.MOBILECLIP_TEXT),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_tag_matrix(hub, progress=None):
    """→ (names, matrix)。磁盘缓存按词表版本/模型指纹失效。"""
    names = list(dict.fromkeys(TAG_VOCAB))
    fp = tag_matrix_fingerprint(names)
    path = Path(C.TAG_MATRIX_PATH)
    if path.exists():
        try:
            z = np.load(path, allow_pickle=False)
            if str(z["fingerprint"]) == fp and z["mat"].shape[0] == len(names):
                cached = z["tags"].tolist()
                if cached == names:
                    return names, z["mat"].astype(np.float32)
        except Exception:
            pass
    t0 = time.time()
    mat = hub.embed_texts([C.TAG_PROMPT.format(tag=t) for t in names], progress=progress)
    mat = np.asarray(mat, np.float32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, fingerprint=fp,
                            tags=np.asarray(names, dtype=np.str_), mat=mat)
        tmp.replace(path)
    except Exception:
        pass
    print(f"[tags] 标签矩阵已构建：{len(names)} 个标签，{time.time() - t0:.1f}s", flush=True)
    return names, mat


class VisualAnalyzer:
    def __init__(self, hub=None):
        self.hub = hub or ModelHub.get()
        self.nima_a = NIMA(C.NIMA_AESTHETIC)
        self.nima_t = NIMA(C.NIMA_TECHNICAL)
        self.tag_names, self.tag_mat = load_tag_matrix(self.hub)
        self.tag_index = {t: i for i, t in enumerate(self.tag_names)}
        self._groups = []
        for _, members in TAG_GROUPS:
            idx = [self.tag_index[m] for m in members if m in self.tag_index]
            if len(idx) > 1:
                self._groups.append(idx)
        self.tag_mean = None
        self.library_size = 0

    # ---------- 库级校准 ----------
    def calibrate(self, vectors):
        """用当前库里的图像向量计算每个标签的平均相似度，作为通用词基线。"""
        if not C.TAG_CALIBRATE or vectors is None or len(vectors) < C.TAG_MIN_LIBRARY:
            self.tag_mean = None
            self.library_size = len(vectors) if vectors is not None else 0
            return
        V = np.stack([np.asarray(v, np.float32) for v in vectors])
        S = self.tag_mat @ V.T
        self.tag_mean = np.asarray(S.mean(1), np.float32)
        self.library_size = len(vectors)

    def calibrate_db(self, db):
        self.calibrate(getattr(db.vectors, "vecs", []))

    # ---------- 标签 ----------
    def tag_scores(self, vec):
        """→ (raw, adjusted)，同义组内只保留最高分成员。"""
        raw = np.asarray(self.tag_mat @ np.asarray(vec, np.float32), np.float32)
        for idx in self._groups:
            best = idx[int(np.argmax(raw[idx]))]
            for i in idx:
                if i != best:
                    raw[i] = -1.0
        if self.tag_mean is not None:
            adj = raw - self.tag_mean
        else:
            adj = raw.copy()
        return raw, adj

    def top_tags(self, vec, k=None, thresh=None, calibrate=True):
        k = C.TAG_TOPK if k is None else k
        thresh = C.TAG_THRESH if thresh is None else thresh
        raw, adj = self.tag_scores(vec)
        calibrated = calibrate and self.tag_mean is not None
        score = adj if calibrated else raw

        cand = [i for i in range(len(raw)) if raw[i] >= thresh]
        if calibrated:
            cand = [i for i in cand if adj[i] >= C.TAG_CENTER_THRESH]
        if not cand:
            # 全是被压制的通用词时，保留 1~2 个高分通用词，保证仍可被搜到。
            fallback = np.argsort(-raw)[:2]
            return [(self.tag_names[i], round(float(raw[i]), 3))
                    for i in fallback if raw[i] >= thresh + 0.02]

        cand.sort(key=lambda i: -score[i])
        top_score = float(score[cand[0]])
        cut = C.TAG_CENTER_THRESH if not calibrated else max(
            C.TAG_CENTER_THRESH, top_score * C.TAG_REL_THRESH)
        picked = []
        for i in cand:
            if calibrated and score[i] < cut:
                break
            picked.append(i)
            if len(picked) >= k:
                break

        # 展示顺序：具体标签在前，通用气氛词在后；分数仍是原始余弦。
        picked.sort(key=lambda i: (self.tag_names[i] in TAG_GENERIC, -float(score[i])))
        return [(self.tag_names[i], round(float(raw[i]), 3)) for i in picked]

    def display_tags(self, tags):
        """给界面用：英文标签 → 中文，最多 6 个。"""
        from .vocab import TAG_ZH_DISPLAY
        return [TAG_ZH_DISPLAY.get(t, t) for t, _ in tags[:6]]

    # ---------- 图像 ----------
    def embed(self, img_rgb):
        return self.hub.embed_image(img_rgb)

    def quality(self, img_rgb):
        return (round(self.nima_a.score(img_rgb), 2), round(self.nima_t.score(img_rgb), 2),
                *map(lambda x: round(x, 1), sharpness_exposure(img_rgb)))
