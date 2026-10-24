# -*- coding: utf-8 -*-
"""缩略图/头像缓存。

原实现每次请求都从原图重新解码 + LANCZOS，一张 25~40 ms，400 张首屏就是十几秒；
而且缓存只有 400 条且满了直接 clear，滚动一次就全部重新生成。
这里改成：
- 磁盘缓存（路径 + mtime + size 做 key），跨进程、跨启动复用；
- 内存 LRU（按字节限额），避免热图反复读盘；
- 同 key 生成互斥，多线程请求只解码一次；
- JPEG draft + reducing_gap，小图比 LANCZOS 全解码快一个量级。
"""
import hashlib
import io
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from . import config as C


_PLACEHOLDER = None
_placeholder_lock = threading.Lock()


def _placeholder_jpeg() -> bytes:
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        with _placeholder_lock:
            if _PLACEHOLDER is None:
                im = Image.new("RGB", (32, 32), (28, 28, 32))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=70)
                _PLACEHOLDER = buf.getvalue()
    return _PLACEHOLDER


def canonical_size(size: int) -> int:
    size = max(64, min(int(size or 320), 2048))
    for s in C.THUMB_SIZES:
        if size <= s:
            return s
    return C.THUMB_SIZES[-1]


def _key(path: Path, size: int, tag: str = "") -> str:
    try:
        st = path.stat()
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        stamp = "missing"
    raw = f"{path.resolve()}|{stamp}|{size}|{tag}".encode("utf-8", "ignore")
    return hashlib.sha1(raw).hexdigest()


class ThumbCache:
    def __init__(self):
        self.root = Path(C.THUMB_CACHE_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        self._mem = OrderedDict()
        self._mem_bytes = 0
        self._mem_lock = threading.Lock()
        self._gen_locks = {}
        self._gen_lock = threading.Lock()

    # ---------- memory LRU ----------
    def _mem_get(self, key):
        with self._mem_lock:
            data = self._mem.get(key)
            if data is not None:
                self._mem.move_to_end(key)
            return data

    def _mem_put(self, key, data):
        with self._mem_lock:
            if key in self._mem:
                self._mem_bytes -= len(self._mem[key])
            self._mem[key] = data
            self._mem_bytes += len(data)
            self._mem.move_to_end(key)
            while self._mem_bytes > C.THUMB_MEM_BYTES and self._mem:
                _, old = self._mem.popitem(last=False)
                self._mem_bytes -= len(old)

    # ---------- disk ----------
    def _disk_path(self, key, size):
        d = self.root / str(size) / key[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.jpg"

    def _lock_for(self, key):
        with self._gen_lock:
            lk = self._gen_locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self._gen_locks[key] = lk
            return lk

    def _write_atomic(self, dst: Path, data: bytes):
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)

    # ---------- public ----------
    def get(self, path, size, maker) -> bytes:
        size = canonical_size(size)
        p = Path(path)
        key = _key(p, size)
        hit = self._mem_get(key)
        if hit is not None:
            return hit
        dst = self._disk_path(key, size)
        if dst.exists():
            try:
                data = dst.read_bytes()
                self._mem_put(key, data)
                return data
            except OSError:
                pass
        lk = self._lock_for(key)
        with lk:
            # 双检：等锁期间别的线程可能已经生成
            hit = self._mem_get(key)
            if hit is not None:
                return hit
            if dst.exists():
                try:
                    data = dst.read_bytes()
                    self._mem_put(key, data)
                    return data
                except OSError:
                    pass
            try:
                data = maker(p, size)
            except Exception:
                data = _placeholder_jpeg()
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                self._write_atomic(dst, data)
            except OSError:
                pass
            self._mem_put(key, data)
            if len(self._gen_locks) > 4096:
                self._gen_locks.clear()
            return data

    def thumb(self, path, size) -> bytes:
        return self.get(path, size, _make_thumb)


def _make_thumb(path: Path, size: int) -> bytes:
    with Image.open(path) as im:
        try:
            im.draft("RGB", (size, size))
        except Exception:
            pass
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        resample = Image.LANCZOS if size >= 480 else Image.BICUBIC
        im.thumbnail((size, size), resample, reducing_gap=2.0)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=C.THUMB_QUALITY, optimize=True, progressive=True)
    return buf.getvalue()


def _make_avatar(path: Path, size: int, bbox) -> bytes:
    from .metadata import load_image_rgb
    img = load_image_rgb(path, max_side=1024)
    h, w = img.shape[:2]
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        x1, y1, x2, y2 = 0, 0, w, h
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(bw, bh) * 1.55
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, w, h
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return _placeholder_jpeg()
    im = Image.fromarray(crop)
    im.thumbnail((size, size), Image.LANCZOS, reducing_gap=2.0)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=86, optimize=True, progressive=True)
    return buf.getvalue()


CACHE = ThumbCache()


def thumb_bytes(path, size=320) -> bytes:
    return CACHE.thumb(path, size)


def avatar_bytes(path, bbox, size=160) -> bytes:
    size = canonical_size(size)
    p = Path(path)
    tag = "avatar:" + ",".join(str(round(float(v), 1)) for v in bbox)
    key = _key(p, size, tag)
    hit = CACHE._mem_get(key)
    if hit is not None:
        return hit
    dst = CACHE._disk_path(key, size)
    if dst.exists():
        try:
            data = dst.read_bytes()
            CACHE._mem_put(key, data)
            return data
        except OSError:
            pass
    lk = CACHE._lock_for(key)
    with lk:
        try:
            data = _make_avatar(p, size, bbox)
        except Exception:
            data = _placeholder_jpeg()
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            CACHE._write_atomic(dst, data)
        except OSError:
            pass
        CACHE._mem_put(key, data)
        return data


def warm_library(db, sizes=None, threads=None):
    """后台预热缩略图。只做磁盘缓存，不占内存队列。"""
    sizes = sizes or C.THUMB_PREWARM_SIZES
    threads = max(1, threads or C.THUMB_PREWARM_THREADS)
    try:
        rows = db.query("SELECT path FROM photos")
    except Exception:
        return
    if not rows:
        return

    def one(path):
        for s in sizes:
            try:
                CACHE.thumb(path, s)
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(one, [r["path"] for r in rows]))
