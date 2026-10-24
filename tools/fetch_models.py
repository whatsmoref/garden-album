#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Hugging Face 下载 Garden 相册所需模型（打包与本地开发共用）。

用法::

    pip install huggingface_hub
    python tools/fetch_models.py                 # 下载到 ./models
    python tools/fetch_models.py --dest /tmp/models

只下载 5 个公开仓库里的 7 个文件（约 280MB），逐个做 SHA256 校验；
OCR 模型直接复用 `rapidocr-onnxruntime` 包内自带的 PP-OCRv4 模型（无需下载），
仅重命名成后端约定的文件名。

国内网络可设置镜像：``HF_ENDPOINT=https://hf-mirror.com python tools/fetch_models.py``
"""
import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

# (HF 仓库, 仓库内文件名, 本地相对路径, SHA256)
HF_FILES = [
    ("felixhrdyn/mobileclip2-s0-onnx", "vision_model.onnx",
     "mobileclip2-s0/vision_model.onnx",
     "13d20ebfa8a8f63890eb2727fe4dc63009ff970f43e0f7d9d2ed999659f70c8a"),
    ("felixhrdyn/mobileclip2-s0-onnx", "text_model_quantized.onnx",
     "mobileclip2-s0/text_model_quantized.onnx",
     "4eac03c0dde9e1ed782b91ce9e50f2819e1f062eed0f19b9d4ca72154026cfee"),
    ("felixhrdyn/mobileclip2-s0-onnx", "tokenizer.json",
     "mobileclip2-s0/tokenizer.json",
     "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842"),
    ("globalnebula/insightface-buffalo-l-onnx", "scrfd_10g.onnx",
     "face/scrfd_10g.onnx",
     "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
    ("globalnebula/insightface-buffalo-l-onnx", "arcface_w600k_r50.onnx",
     "face/arcface_w600k_r50.onnx",
     "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
    ("litert-community/NIMA-LiteRT", "nima_aesthetic_fp16.tflite",
     "nima/nima_aesthetic_fp16.tflite",
     "a5051a0fcced735682735e3e0fd58ee54c83ed664282a003f52235b3dbcb9320"),
    ("litert-community/NIMA-LiteRT", "nima_technical_fp16.tflite",
     "nima/nima_technical_fp16.tflite",
     "34fe245d9b6ae14f48cc1b0e276b93d18b905083dafe24f502e862dd8b81f0fa"),
]

# rapidocr-onnxruntime 包内模型 → 后端约定路径（内容一致，仅文件名不同）
OCR_FILES = [
    ("ch_PP-OCRv4_det_infer.onnx", "rapidocr/ch_PP-OCRv4_det_mobile.onnx",
     "d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9"),
    ("ch_PP-OCRv4_rec_infer.onnx", "rapidocr/ch_PP-OCRv4_rec_mobile.onnx",
     "48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b"),
    ("ch_ppocr_mobile_v2.0_cls_infer.onnx", "rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
     "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(path, digest):
    p = Path(path)
    if not p.is_file():
        return False
    if p.stat().st_size == 0:
        return False
    return sha256(p) == digest


def source_hint():
    print("\n提示：下载模型需要  pip install huggingface_hub；"
          "OCR 模型需要  pip install --no-deps rapidocr-onnxruntime", file=sys.stderr)


# ── Release 资产来源（自托管，CI 优先用）────────────────────────────
# 模型也可以托管在本仓库的 models-v1 Release 里：资产名 = 相对路径里的 / 换成 _。
# 这样 CI 不依赖第三方仓库（HF）是否还在，且下载走 GitHub CDN。
RELEASE_TAG = os.environ.get("GARDEN_MODELS_RELEASE_TAG", "models-v1")
RELEASE_REPO = os.environ.get("GARDEN_MODELS_RELEASE_REPO", "whatsmoref/garden-album")
RELEASE_CACHE = Path(os.environ.get(
    "GARDEN_MODELS_CACHE", Path.home() / ".cache" / "garden-models"))


def release_asset_name(rel):
    return rel.replace("/", "_")


def fetch_from_release(rel, digest):
    """从 models-v1 Release 资产取模型；返回 True 表示已就位。"""
    asset = release_asset_name(rel)
    cache_dir = RELEASE_CACHE / RELEASE_TAG
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / asset
    if not check_file(cached, digest):
        if shutil.which("gh") is None:
            return False
        cmd = ["gh", "release", "download", RELEASE_TAG, "--repo", RELEASE_REPO,
               "--pattern", asset, "--dir", str(cache_dir), "--clobber"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[release] 拉取 {asset} 失败：{(r.stderr or r.stdout).strip()[:160]}")
            return False
    if not check_file(cached, digest):
        print(f"[release] {asset} SHA256 不匹配")
        return False
    out = Path(rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, out)
    return True


def main():
    ap = argparse.ArgumentParser(description="下载 Garden 相册模型并校验 SHA256")
    ap.add_argument("--dest", default="models", help="模型输出目录（默认 ./models）")
    ap.add_argument("--check-only", action="store_true", help="只校验，不下载")
    ap.add_argument("--source", default=os.environ.get("GARDEN_MODELS_SOURCE", "auto"),
                    choices=["auto", "release", "hf"],
                    help="auto=先试 Release 资产再回退 Hugging Face；release=只用 Release；hf=只用 HF")
    args = ap.parse_args()
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    failed = []
    for repo, remote, rel, digest in HF_FILES:
        out = dest / rel
        if check_file(out, digest):
            print(f"[skip] {rel}")
            continue
        if args.check_only:
            failed.append(rel)
            continue

        if args.source in ("auto", "release"):
            print(f"[fetch] release:{RELEASE_TAG}/{release_asset_name(rel)} -> {rel}")
            if fetch_from_release(rel, digest):
                print(f"[ok]   {rel} ({out.stat().st_size / 1e6:.1f} MB)")
                continue
            if args.source == "release":
                failed.append(rel)
                print(f"[FAIL] Release 资产不可用：{rel}", file=sys.stderr)
                continue

        from huggingface_hub import hf_hub_download

        print(f"[fetch] {repo}/{remote} -> {rel}")
        out.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo_id=repo, filename=remote)
        shutil.copyfile(cached, out)
        if not check_file(out, digest):
            failed.append(rel)
            print(f"[FAIL] SHA256 不匹配：{rel}", file=sys.stderr)
        else:
            print(f"[ok]   {rel} ({out.stat().st_size / 1e6:.1f} MB)")

    # OCR：从 rapidocr-onnxruntime 包内复制（与线上 PP-OCRv4 模型一致）
    spec = importlib.util.find_spec("rapidocr_onnxruntime")
    if spec is None or not spec.origin:
        print("[warn] 没找到 rapidocr-onnxruntime，跳过 OCR 模型；"
              "打包前请先 pip install --no-deps rapidocr-onnxruntime", file=sys.stderr)
    else:
        pkg_models = Path(spec.origin).parent / "models"
        for remote, rel, digest in OCR_FILES:
            out = dest / rel
            if check_file(out, digest):
                print(f"[skip] {rel}")
                continue
            src = pkg_models / remote
            if args.check_only:
                failed.append(rel)
                continue
            if not src.is_file():
                failed.append(rel)
                print(f"[FAIL] rapidocr 包内缺少 {remote}", file=sys.stderr)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            if not check_file(out, digest):
                failed.append(rel)
                print(f"[FAIL] SHA256 不匹配：{rel}", file=sys.stderr)
            else:
                print(f"[ok]   {rel} ({out.stat().st_size / 1e6:.1f} MB)")

    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"\n模型目录：{dest}  合计 {total:.0f} MB")
    if failed:
        print("缺失或校验失败：" + ", ".join(failed), file=sys.stderr)
        source_hint()
        return 1
    print("全部模型校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
