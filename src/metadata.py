# -*- coding: utf-8 -*-
"""A. 元数据层：EXIF / GPS / phash / 截图启发式 / 快速解码（~2ms/张）"""
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageOps
try:
    import ImageHash as imagehash   # PyPI wheel 在 Linux 上装的是小写目录
except ImportError:
    import imagehash
try:
    import pyvips
except ImportError:
    pyvips = None
def load_image_rgb(path, max_side=1280):
    """高速解码（pyvips 优先，PIL 兜底），返回 RGB uint8 HWC"""
    if pyvips is not None:
        try:
            im = pyvips.thumbnail(str(path), max_side)     # 自动按 EXIF 转正
            arr = np.frombuffer(im.write_to_memory(), dtype=np.uint8) \
                    .reshape(im.height, im.width, im.bands)
            if im.bands >= 3:
                return np.ascontiguousarray(arr[:, :, :3])
            return np.ascontiguousarray(np.stack([arr[:, :, 0]] * 3, axis=2))
        except Exception:
            pass
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_side, max_side))
        return np.asarray(im.convert("RGB"), dtype=np.uint8)
def extract_metadata(path):
    path = Path(path)
    out = {"taken_at": None, "gps_lat": None, "gps_lon": None, "device": None,
           "width": 0, "height": 0, "is_screenshot": 0, "phash": None}
    dt, has_cam = None, False
    try:
        with Image.open(path) as im:
            out["width"], out["height"] = im.size
            ex = im.getexif()
            # 注意：Pillow>=9 起 DateTimeOriginal 等 Exif-IFD 标签无法用 ex.get(36867) 取到，
            # 必须显式取 0x8769 子 IFD，否则会静默退化成文件 mtime。
            sub = ex.get_ifd(0x8769)
            dt = sub.get(36867) or sub.get(36868) or ex.get(36867) or ex.get(306)
            if isinstance(dt, bytes):
                dt = dt.decode("ascii", "ignore")
            make, model = str(ex.get(271) or ""), str(ex.get(272) or "")
            out["device"] = (make + " " + model).strip() or None
            try:
                g = ex.get_ifd(0x8825)
                out["gps_lat"] = _dms(g.get(2), g.get(1))
                out["gps_lon"] = _dms(g.get(4), g.get(3))
            except Exception:
                pass
            try:
                out["phash"] = str(imagehash.phash(im.convert("RGB")))
            except Exception:
                pass
            has_cam = bool(dt and make)      # 要求 Make：扫描件/导出图通常没有 Make
    except Exception:
        pass
    if dt:
        try:
            out["taken_at"] = datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S") \
                              .strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    if not out["taken_at"]:
        out["taken_at"] = datetime.fromtimestamp(path.stat().st_mtime) \
                          .strftime("%Y-%m-%d %H:%M:%S")
    out["is_screenshot"] = int(_is_screenshot(path, out, has_cam))
    return out
def _dms(val, ref):
    if not val: return None
    d = float(val[0]) + float(val[1]) / 60 + float(val[2]) / 3600
    if ref in ("S", "W"): d = -d
    return round(d, 6)
def _is_screenshot(path, meta, has_cam):
    name = path.name.lower()
    pats = ("screenshot", "screen_shot", "截屏", "截图", "screen recording",
            "wx_camera", "mmexport", "screencap")
    if any(p in name for p in pats):
        return True
    if has_cam:                       # 有相机 Make+时间的必定是实拍，尺寸比例不可信
        return False
    w, h = meta["width"], meta["height"]
    if w and h:
        r = min(w, h) / max(w, h)
        # 手机屏幕比例（9:19.5≈0.462、9:16=0.5625），且分辨率大
        if w >= 1000 and (abs(r - 0.462) < 0.02 or abs(r - 0.5625) < 0.01):
            return True
    return False
