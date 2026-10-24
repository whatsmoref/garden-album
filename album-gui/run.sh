#!/usr/bin/env bash
# 启动 Garden 相册
#
# Tauri 会把 frontendDist（ui/）里的资源编译进二进制，
# 所以改完前端必须重新 cargo build，否则跑的还是旧版界面。
# 这里自动比对时间戳，ui/ 或 Rust 侧有改动就顺手重编一次。
set -euo pipefail

cd "$(dirname "$0")"

BIN=src-tauri/target/debug/garden-album-gui

if [[ ! -x $BIN ]]; then
  echo "首次构建（Tauri 依赖较多，约几分钟）…" >&2
  cargo build --manifest-path src-tauri/Cargo.toml
elif [[ -n "$(find ui src-tauri/src src-tauri/tauri.conf.json -newer "$BIN" -print -quit 2>/dev/null)" ]]; then
  echo "前端/Rust 有改动，重新编译…" >&2
  cargo build --manifest-path src-tauri/Cargo.toml
fi

exec "$BIN"