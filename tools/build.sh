#!/usr/bin/env bash
# 编译 + 拉取：本地机器太慢，交给 GitHub Actions 编，然后 gh run download 拉回来。
#
#   ./build.sh              # 提交改动 + 触发 CI + 等完成 + 下载产物
#   ./build.sh --no-push    # 只重新跑 CI（不提交）
#   ./build.sh --check      # 只看最近一次 CI 状态
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$REPO_DIR/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"

cd "$REPO_DIR"

MODE=build
[ "${1:-}" = "--check" ] && MODE=check
[ "${1:-}" = "--no-push" ] && MODE=nopush

latest_run() {
  gh run list --workflow build.yml --limit 1 --json databaseId,headSha \
    --jq '.[0] | "\(.databaseId) \(.headSha)"'
}

if [ "$MODE" = "check" ]; then
  read -r id sha <<<"$(latest_run)"
  echo "run=$id  conclusion=$(gh run view "$id" --json conclusion -q .conclusion)"
  gh run view "$id" --json jobs -q '.jobs[] | "  \(.name): \(.status)/\(.conclusion)"'
  exit 0
fi

if [ "$MODE" = "build" ]; then
  if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -m "${1:-$(git log -1 --pretty=%s)}"
  fi
  git push -q origin HEAD
fi

echo "→ 等待 CI 启动…"
for _ in $(seq 1 30); do
  read -r id sha <<<"$(latest_run)"
  [ "$sha" = "$(git rev-parse HEAD)" ] && break
  sleep 3
done
[ -z "${id:-}" ] && { echo "拿不到 run id"; exit 1; }
echo "→ run $id (sha ${sha:0:7})"
gh run watch "$id" --exit-status --interval 15 || {
  echo "!! CI 失败，日志尾部："
  gh run view "$id" --log-failed | tail -40
  exit 1
}

echo "→ 下载产物"
DEST="$ROOT/bin"
mkdir -p "$DEST"
gh run download "$id" -n album-linux-x86_64 -D /tmp/album-dist
cp -P /tmp/album-dist/album /tmp/album-dist/libonnxruntime.so* "$DEST/"
chmod +x "$DEST/album"

echo "→ 冒烟测试"
LD_LIBRARY_PATH="$DEST" "$DEST/album" --version || true
echo "✓ 完成：$DEST/album"
echo "  运行：LD_LIBRARY_PATH=$DEST $DEST/album <命令>"
