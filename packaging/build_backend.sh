#!/usr/bin/env bash
# 把 Python 后端打成「可移植、免安装 Python」的自包含目录，供 Tauri 安装包内置：
#
#   pkg/backend/
#   ├── python/          # python-build-standalone 3.11（自带全部依赖）
#   ├── src/             # 后端代码
#   └── models/          # 模型（tools/fetch_models.py 下载并校验）
#
# 用法：
#   packaging/build_backend.sh [--out DIR] [--models-from DIR]
#
#   --out DIR          输出目录，默认 album-gui/src-tauri/pkg/backend
#   --models-from DIR  不从 HF 下载，直接复制本地模型目录（离线/本地测试用）
#
# 依赖：curl/tar（必须）、huggingface_hub（仅当没有 --models-from 时需要）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/album-gui/src-tauri/pkg/backend"
MODELS_FROM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --models-from) MODELS_FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

# ── 固定版本的 python-build-standalone（astral-sh 官方可移植 CPython）──
PBS_TAG="20261021"
PBS_VERSION="3.11.17"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/garden-album-packaging"
mkdir -p "$CACHE"

UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"
case "$UNAME_S/$UNAME_M" in
  Linux/x86_64)
    PBS_ASSET="cpython-${PBS_VERSION}+${PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"
    PBS_SHA="7cc8d9dce9ea23c31267e7f244005c8979da04794b7737a3a30e62b03abeb120" ;;
  Darwin/arm64)
    PBS_ASSET="cpython-${PBS_VERSION}+${PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz"
    PBS_SHA="b5199d66966141fb6d5447512d2b2fe0398d417da37b1949f58db3792d314eac" ;;
  *)
    echo "不支持的打包平台：$UNAME_S/$UNAME_M（仅 Linux x86_64 与 macOS arm64）" >&2
    exit 1 ;;
esac

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

say "准备可移植 Python ${PBS_VERSION}（${UNAME_S}/${UNAME_M}）"
TARBALL="$CACHE/$PBS_ASSET"
if [[ ! -f "$TARBALL" ]] || [[ "$(sha256_file "$TARBALL")" != "$PBS_SHA" ]]; then
  curl -fL --retry 3 -o "$TARBALL.tmp" \
    "https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PBS_ASSET//+/%2B}"
  actual="$(sha256_file "$TARBALL.tmp")"
  if [[ "$actual" != "$PBS_SHA" ]]; then
    echo "Python 运行时 SHA256 校验失败：期望 $PBS_SHA，实际 $actual" >&2
    rm -f "$TARBALL.tmp"
    exit 1
  fi
  mv "$TARBALL.tmp" "$TARBALL"
fi

say "展开到 $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
tar -xzf "$TARBALL" -C "$OUT"        # install_only 包内是 python/ 目录
PY="$OUT/python/bin/python3"
[[ -x "$PY" ]] || { echo "解压后找不到 $PY" >&2; exit 1; }

say "安装依赖（wheel 缓存复用，CI 上会命中 actions/cache）"
"$PY" -m pip install --disable-pip-version-check -q --upgrade pip setuptools wheel
"$PY" -m pip install --disable-pip-version-check -q -r "$ROOT/requirements.txt"
# rapidocr 必须 --no-deps：它声明的 opencv-python 会覆盖我们用的 headless 版
"$PY" -m pip install --disable-pip-version-check -q --no-deps rapidocr-onnxruntime

say "准备模型"
if [[ -n "$MODELS_FROM" ]]; then
  echo "从本地复制：$MODELS_FROM"
  rm -rf "$OUT/models"
  cp -a "$MODELS_FROM" "$OUT/models"
  "$PY" "$ROOT/tools/fetch_models.py" --dest "$OUT/models" --check-only
else
  "$PY" "$ROOT/tools/fetch_models.py" --dest "$OUT/models"
fi

say "拷贝后端代码"
cp -a "$ROOT/src" "$OUT/src"
cp -a "$ROOT/LICENSE" "$OUT/LICENSE"

say "瘦身（去掉测试/缓存/无用的标准库组件）"
find "$OUT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -name '*.pyc' -delete 2>/dev/null || true
PYLIB="$OUT/python/lib/python3.11"
rm -rf "$PYLIB/test" "$PYLIB/ensurepip" "$PYLIB/idlelib" "$PYLIB/tkinter" \
       "$PYLIB/turtledemo" "$PYLIB/lib2to3" "$PYLIB/venv/scripts" 2>/dev/null || true
# 开发头文件 / terminfo / 纯构建期工具（安装与取模型阶段已经用完了）
rm -rf "$PYLIB/site-packages/pip" "$PYLIB/site-packages/wheel" \
       "$PYLIB/site-packages/huggingface_hub" "$PYLIB/site-packages/hf_xet" \
       "$PYLIB/site-packages/tqdm" \
       "$PYLIB/site-packages"/wheel-*.dist-info "$PYLIB/site-packages"/pip-*.dist-info \
       "$PYLIB/site-packages"/huggingface_hub-*.dist-info "$PYLIB/site-packages"/hf_xet-*.dist-info \
       "$PYLIB/site-packages"/tqdm-*.dist-info \
       "$PYLIB/site-packages/sympy" "$PYLIB/site-packages"/sympy-*.dist-info \
       "$PYLIB/site-packages/ml_dtypes" "$PYLIB/site-packages"/ml_dtypes-*.dist-info 2>/dev/null || true
# jieba 只用分词主干；posseg/analyse/lac_small 是没人调用的可选子模块（~30MB）
rm -rf "$PYLIB/site-packages/jieba/analyse" "$PYLIB/site-packages/jieba/posseg" \
       "$PYLIB/site-packages/jieba/lac_small" 2>/dev/null || true
rm -rf "$OUT/python/share" "$OUT/python/include" "$OUT/python/lib/pkgconfig" 2>/dev/null || true
# python/bin 只留解释器本体（pip/hf/2to3 等脚本运行时用不到）
if [[ -d "$OUT/python/bin" ]]; then
  find "$OUT/python/bin" -maxdepth 1 -type f ! -name 'python3.11' ! -name 'python3.11-config' -delete 2>/dev/null || true
  find "$OUT/python/bin" -maxdepth 1 -type l ! -name 'python3' ! -name 'python3-config' -delete 2>/dev/null || true
fi
# 各依赖包里的 tests/ 对运行无用（numpy/scipy 的测试目录最占地方）
find "$OUT/python/lib/python3.11/site-packages" -type d \
     \( -name 'tests' -o -name 'test' \) -prune -exec rm -rf {} + 2>/dev/null || true

say "自检：解释器 / 依赖 / 模型"
(
cd "$OUT"
"$PY" - <<'PYCHECK'
import pathlib, sys
import numpy, cv2, onnxruntime as ort, tokenizers, jieba, imagehash
from ai_edge_litert.interpreter import Interpreter
from rapidocr_onnxruntime import RapidOCR
root = pathlib.Path.cwd()
assert (root / "src/server.py").is_file(), "缺少 src/server.py"
for rel in ["mobileclip2-s0/vision_model.onnx", "mobileclip2-s0/text_model_quantized.onnx",
            "mobileclip2-s0/tokenizer.json", "face/scrfd_10g.onnx", "face/arcface_w600k_r50.onnx",
            "nima/nima_aesthetic_fp16.tflite", "nima/nima_technical_fp16.tflite",
            "rapidocr/ch_PP-OCRv4_det_mobile.onnx", "rapidocr/ch_PP-OCRv4_rec_mobile.onnx",
            "rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx"]:
    p = root / "models" / rel
    assert p.is_file() and p.stat().st_size > 0, f"模型缺失: {rel}"
for p in ["models/mobileclip2-s0/vision_model.onnx", "models/face/scrfd_10g.onnx"]:
    ort.InferenceSession(str(root / p), providers=["CPUExecutionProvider"]).get_inputs()
it = Interpreter(model_path=str(root / "models/nima/nima_aesthetic_fp16.tflite"))
it.allocate_tensors()
print("python:", sys.version.split()[0], "| numpy", numpy.__version__, "| onnxruntime", ort.__version__)
print("backend 自检通过")
PYCHECK
)

DUP="$(du -sh "$OUT" | cut -f1)"
say "完成：$OUT（$DUP）"
