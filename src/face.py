# -*- coding: utf-8 -*-
"""C. 人脸层：SCRFD 检测 + ArcFace 512d + HSEmotion 微笑 + 增量多原型聚类"""
import re
import numpy as np
import cv2
from . import config as C
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
def _flat(a):
    """ONNX 输出可能带 batch 维 (1,N,C) 也可能没有 (N,C)，统一压成 (N,C)"""
    a = np.asarray(a)
    return a[0] if a.ndim == 3 else a
def _nms(boxes, scores, thr=0.4):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep
class SCRFD:
    """insightface SCRFD-10g：9 输出 = scores[8/16/32] + bbox + kps（ltrb 已按 stride 放大）"""
    def __init__(self, session):
        self.sess = session
        self.iname = session.get_inputs()[0].name
        self.strides = (8, 16, 32); self.fmc = 3
        self._cache = {}
    def _centers(self, stride, na):
        key = (stride, na)
        if key not in self._cache:
            h = w = 640 // stride
            ys, xs = np.mgrid[0:h, 0:w]
            c = np.stack([xs, ys], -1).astype(np.float32).reshape(-1, 2)
            self._cache[key] = np.repeat(c, na, 0) * stride
        return self._cache[key]
    def detect(self, img_rgb, thresh=None):
        thresh = C.FACE_DET_THRESH if thresh is None else thresh
        H, W = img_rgb.shape[:2]
        ratio = H / W
        if ratio > 1: nh, nw = 640, max(1, round(640 / ratio))
        else:         nw, nh = 640, max(1, round(640 * ratio))
        scale = nh / H
        canvas = np.zeros((640, 640, 3), np.float32)
        canvas[:nh, :nw] = cv2.resize(img_rgb, (nw, nh))
        blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.iname: blob})
        S, B, K = [], [], []
        for i, stride in enumerate(self.strides):
            # 本机 scrfd_10g.onnx 输出无 batch 维：scores (N,1) / bbox (N,4) / kps (N,10)
            sc = _flat(outs[i]); n = sc.shape[0]
            # 实测：bbox/kps 导出时已除以 stride，此处乘回原图尺度
            bb = _flat(outs[i + self.fmc]) * stride
            kp = _flat(outs[i + self.fmc * 2]) * stride
            na = max(1, n // (640 // stride) ** 2)
            centers = self._centers(stride, na)
            pos = np.where(sc[:, 0] >= thresh)[0]
            if not len(pos): continue
            d, c = bb[pos], centers[pos]
            boxes = np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1],
                              c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], 1)
            kps = kp[pos].copy()
            for j in range(5):
                kps[:, 2 * j] += c[:, 0]; kps[:, 2 * j + 1] += c[:, 1]
            S.append(sc[pos, 0]); B.append(boxes); K.append(kps)
        if not B: return []
        boxes = np.concatenate(B) / scale
        kps = np.concatenate(K) / scale
        scores = np.concatenate(S)
        out = []
        for i in _nms(boxes, scores, 0.4):
            b = np.clip(boxes[i], 0, [W - 1, H - 1, W - 1, H - 1])
            if b[2] - b[0] < 24 or b[3] - b[1] < 24: continue
            out.append({"bbox": b, "kps": kps[i].reshape(5, 2), "score": float(scores[i])})
        return out
def align_face(img_rgb, kps):
    M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, np.float32), ARCFACE_DST, method=cv2.LMEDS)
    if M is None: return None
    return cv2.warpAffine(img_rgb, M, (C.ARCFACE_INPUT, C.ARCFACE_INPUT))
def arcface_embed(session, aligned):
    x = (aligned.astype(np.float32) - 127.5) / 127.5
    x = x.transpose(2, 0, 1)[None]
    v = np.asarray(session.run(None, {session.get_inputs()[0].name: x})[0])[0]
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
def eyes_open_estimate(img_rgb, kps):
    """启发式：眼部区域拉普拉斯方差（闭眼=平滑皮肤→低方差）"""
    d = float(np.linalg.norm(kps[0] - kps[1]))
    if d < 4: return True
    r = max(4, int(d * 0.30))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    flags = []
    for (ex, ey) in (kps[0], kps[1]):
        x0, x1 = max(0, int(ex - r)), min(W, int(ex + r))
        y0, y1 = max(0, int(ey - int(r * 0.6))), min(H, int(ey + int(r * 0.8)))
        if (x1 - x0) < 6 or (y1 - y0) < 6: continue
        v = cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var()
        flags.append(v > max(20.0, d * 0.8))
    return all(flags) if flags else True
class SmileDetector:
    """HSEmotion → Happiness 连续微笑分"""
    def __init__(self, hub):
        self.hub = hub
    def smiles(self, img_bgr, boxes):
        if not boxes: return []
        try:
            res = self.hub.hse.recognize_faces(img_bgr, face_boxes=[list(map(int, b)) for b in boxes])
        except Exception:
            return [0.0] * len(boxes)
        out = [float(item[2].get("Happiness", 0.0)) for item in res]
        while len(out) < len(boxes): out.append(0.0)
        return out[: len(boxes)]
class PersonStore:
    """增量聚类：新脸与已有原型比余弦；命名一次全簇生效"""
    def __init__(self, db):
        self.db = db
        self.protos = {}      # pid → (n,512) 原型矩阵
        self.named = {}       # pid → name（用户命名过的）
        for r in db.persons():
            if not re.match(r"^人物\d+$", r["name"]):
                self.named[r["id"]] = r["name"]
        for _fid, pid, vec in db.all_face_vecs():
            self._add_proto(pid, vec)
    def _add_proto(self, pid, vec):
        P = self.protos.get(pid)
        if P is None or len(P) == 0:
            self.protos[pid] = vec[None]; return
        if float((P @ vec).max()) < C.FACE_COS_NEWPROTO and len(P) < C.PERSON_MAX_PROTOS:
            self.protos[pid] = np.vstack([P, vec])
    def assign(self, vec):
        best_pid, best = None, -1.0
        for pid, P in self.protos.items():
            s = float((P @ vec).max())
            if s > best: best, best_pid = s, pid
        if best_pid is not None and best >= C.FACE_COS_ASSIGN:
            return best_pid, best
        n = self.db.one("SELECT COUNT(*) c FROM persons")["c"] + 1
        pid = self.db.add_person(f"人物{n}")
        self.protos[pid] = vec[None]
        return pid, 1.0
    def rename(self, pid, name):
        self.db.rename_person(pid, name); self.named[pid] = name
    def person_names(self):
        return list(self.named.values())
class FacePipeline:
    def __init__(self, hub, persons, db):
        self.hub = hub; self.persons = persons; self.db = db
        self.scrfd = SCRFD(hub.scrfd)
        self.smile = SmileDetector(hub)
    def analyze(self, photo_id, img_rgb):
        dets = self.scrfd.detect(img_rgb)
        self.db.update_photo(photo_id, known_face_count=0, unknown_face_count=0,
                             avg_smile=None, has_closed_eyes=0, best_face_area=None)
        if not dets: return []
        img_bgr = np.ascontiguousarray(img_rgb[..., ::-1])   # HSE 按其 README 传 BGR
        smiles = self.smile.smiles(img_bgr, [d["bbox"] for d in dets])
        H, W = img_rgb.shape[:2]
        recs = []
        for i, d in enumerate(dets):
            kps = np.asarray(d["kps"], np.float32)
            aligned = align_face(img_rgb, kps)
            if aligned is None: continue
            vec = arcface_embed(self.hub.arcface, aligned)
            pid, _sim = self.persons.assign(vec)
            smile = smiles[i] if i < len(smiles) else 0.0
            eyes = eyes_open_estimate(img_rgb, kps)
            x1, y1, x2, y2 = d["bbox"]
            area = max(0.0, (x2 - x1) * (y2 - y1)) / (W * H)
            self.db.insert_face(photo_id, pid, d["bbox"].round(1).tolist(),
                                kps.round(1).tolist(), vec, round(smile, 3), eyes, round(area, 4))
            recs.append({"person_id": pid, "smile": smile, "eyes_open": eyes, "area": area})
        if not recs: return []
        known = sum(1 for r in recs if r["person_id"] in self.persons.named)
        self.db.update_photo(
            photo_id,
            known_face_count=known, unknown_face_count=len(recs) - known,
            avg_smile=round(float(np.mean([r["smile"] for r in recs])), 3),
            has_closed_eyes=int(any(not r["eyes_open"] for r in recs)),
            best_face_area=round(max(r["area"] for r in recs), 4))
        return recs
