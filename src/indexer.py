# -*- coding: utf-8 -*-
"""流水线编排：扫描/监听 → 元数据 → 视觉 → 人脸 → OCR → 相册（批处理、仅照片）"""
import time
from pathlib import Path
from . import config as C
from .models import ModelHub
from .db import DB
from .metadata import load_image_rgb, extract_metadata
from .visual import VisualAnalyzer
from .face import FacePipeline, PersonStore
from .ocr import TargetedOCR, refine_tags
from .albums import AlbumEngine
from . import events, quality
class Indexer:
    def __init__(self, db=None):
        self.db = db or DB()
        self.hub = ModelHub.get()
        self.visual = VisualAnalyzer(self.hub)
        self.visual.calibrate_db(self.db)
        self.persons = PersonStore(self.db)
        self.faces = FacePipeline(self.hub, self.persons, self.db)
        self.ocr = TargetedOCR()
        self.albums = AlbumEngine(self.db, self.hub)

    def _db_path(self, path):
        """统一路径口径：项目内用相对路径，项目外用绝对路径，避免重复入库。"""
        p = Path(path)
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p.absolute()
        try:
            return str(resolved.relative_to(C.ROOT.resolve()))
        except ValueError:
            return str(resolved)

    def scan(self, root):
        root = Path(root)
        return [p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in C.IMAGE_EXTS
                and not p.name.startswith(".")
                and self.db.photo_id(self._db_path(p)) is None]

    def index_folder(self, root, watch=False, progress=None):
        while True:
            paths = self.scan(root)
            print(f"[index] 发现 {len(paths)} 张新照片", flush=True)
            if progress:
                progress(0, len(paths))
            for i, p in enumerate(paths):
                try:
                    pid = self.index_photo(p)
                    if pid:
                        print(f"[{i+1}/{len(paths)}] ✓ {p.name}", flush=True)
                        if (i + 1) % 50 == 0:       # 分批落盘，中途退出不丢已建索引
                            self.db.commit()
                except Exception as e:
                    print(f"[{i+1}/{len(paths)}] ✗ {p}: {e}", flush=True)
                if progress:
                    progress(i + 1, len(paths))
            if paths:
                self.visual.calibrate_db(self.db)
                events.recompute(self.db)
                quality.recompute_burst_best(self.db)
                self.db.set_meta("tag_vocab_version", C.TAG_VOCAB_VERSION)
                self.db.commit()                   # 必须：否则退出时 SQLite 全量回滚
            live = [r["id"] for r in self.db.query("SELECT id FROM photos")]
            dropped = len(self.db.vectors.ids) - self.db.vectors.prune(live)
            if dropped:
                self.db.commit()
                print(f"[index] 清理 {dropped} 个孤儿向量")
            self.db.vectors.save(C.VEC_PATH)
            if paths:
                print("[index] 事件 / 连拍 / 向量已更新", flush=True)
            if not watch: break
            time.sleep(60)     # 轮询监听；可换 watchdog

    # ---------- 标签迁移（不重新解码图片，直接用库里的 512d 向量） ----------
    def retag_all(self, progress=None):
        rows = self.db.query("SELECT id, filename, ocr_text FROM photos")
        vecs = dict(zip(self.db.vectors.ids, self.db.vectors.vecs))
        self.visual.calibrate_db(self.db)
        changed = 0
        done = 0
        for r in rows:
            vec = vecs.get(r["id"])
            if vec is None:
                continue
            tags = refine_tags(self.visual.top_tags(vec), r["ocr_text"])
            old = self.db.tags_of([r["id"]]).get(r["id"], [])
            if [t for t, _ in old] != [t for t, _ in tags]:
                final = self.db.set_tags(r["id"], tags)
                fts = " ".join(t for t, _ in final) + " " + (r["filename"] or "")
                if r["ocr_text"]:
                    fts += " " + r["ocr_text"][:5000]
                self.db.fts_set(r["id"], fts)
                changed += 1
            done += 1
            if done % 100 == 0:
                self.db.commit()
                if progress:
                    progress(done, len(rows))
        self.db.set_meta("tag_vocab_version", C.TAG_VOCAB_VERSION)
        self.db.commit()
        events.recompute(self.db)
        Albums = AlbumEngine(self.db, self.hub)
        added = Albums.rebuild_all()
        self.db.commit()
        if progress:
            progress(len(rows), len(rows))
        return {"photos": done, "changed": changed, "album_entries": added}
    def index_photo(self, path):
        path = Path(path)
        db_path = self._db_path(path)
        if self.db.photo_id(db_path) is not None: return None
        # A 元数据
        meta = extract_metadata(path)
        img = load_image_rgb(path, max_side=1024)
        # B 视觉层
        vec = self.visual.embed(img)
        tags = self.visual.top_tags(vec)
        aes, tec, shp, exp = self.visual.quality(img)
        pid = self.db.insert_photo(
            path=db_path, filename=path.name, taken_at=meta["taken_at"],
            gps_lat=meta["gps_lat"], gps_lon=meta["gps_lon"], device=meta["device"],
            width=meta["width"], height=meta["height"], is_screenshot=meta["is_screenshot"],
            phash=meta["phash"], aesthetic=aes, technical=tec, sharpness=shp, exposure=exp)
        tags = self.db.set_tags(pid, tags)
        # C 人脸层（检测 + 识别 + 表情 + 增量聚类 + 去规范化统计）
        self.faces.analyze(pid, img)
        # FTS：标签 + 文件名
        self.db.fts_set(pid, " ".join(t for t, _ in tags) + " " + path.stem)
        # E 定向 OCR（仅疑似票据/截图）
        row = self.db.one("SELECT * FROM photos WHERE id=?", (pid,))
        if self.ocr.need(row, tags):
            full, extra = self.ocr.run(path)
            if full:
                tags = refine_tags(tags + [(t, 1.0) for t in extra], full)
                tags = self.db.set_tags(pid, tags)
                self.db.update_photo(pid, ocr_text=full[:5000])
                self.db.fts_set(pid, " ".join(t for t, _ in tags) + " " + path.stem + " " + full[:5000])
        # 向量 + F 语义相册（每相册一次点积，微秒级）
        self.db.vectors.add(pid, vec)
        self.albums.add_photo(pid, self.db.one("SELECT * FROM photos WHERE id=?", (pid,)), tags, vec)
        return pid
