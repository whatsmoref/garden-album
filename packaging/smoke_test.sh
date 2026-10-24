#!/usr/bin/env bash
# 冒烟测试「安装包里那份后端」：起服务 → /api/setup 索引两张合成图 → 校验状态。
# 目的是在打包后立刻抓出「模型缺失 / 依赖缺失 / 模块导入错误」这类运行时问题
# （例如 _index_worker 里漏 import AlbumEngine 这种语法检查抓不到的 bug）。
#
# 用法：packaging/smoke_test.sh [backend_dir]
#   backend_dir 默认 album-gui/src-tauri/pkg/backend
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${1:-$ROOT/album-gui/src-tauri/pkg/backend}"
WORK="$(mktemp -d)"
PORT="$(python3 - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()
PY
)"

cleanup() {
  [[ -n "${PID:-}" ]] && kill "$PID" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

python3 - "$WORK" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageDraw
out = Path(sys.argv[1]) / "photos"
out.mkdir(parents=True, exist_ok=True)
for i, bg in enumerate([(220, 120, 90), (90, 150, 200)]):
    im = Image.new("RGB", (640, 480), bg)
    d = ImageDraw.Draw(im)
    d.ellipse((180, 110, 460, 390), fill=(240, 220, 180))
    d.rectangle((60, 40, 160, 90), fill=(30, 40, 60))
    im.save(out / f"smoke{i}.jpg", quality=92)
print(f"生成测试图片：{out}")
PY

echo "启动打包后端：$BACKEND（端口 $PORT）"
(
  cd "$BACKEND"
  GARDEN_DATA_DIR="$WORK/data" exec "$BACKEND/python/bin/python3" -u -m src.server --port "$PORT"
) > "$WORK/backend.log" 2>&1 &
PID=$!

for _ in $(seq 1 120); do
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then break; fi
  sleep 0.5
done
if ! curl -sf --max-time 5 "http://127.0.0.1:$PORT/api/health" >/dev/null; then
  echo "后端没有起来，日志：" >&2
  tail -40 "$WORK/backend.log" >&2
  exit 1
fi

curl -sf --max-time 30 -X POST "http://127.0.0.1:$PORT/api/setup" \
  -H "Content-Type: application/json" \
  -d "{\"dir\": \"$WORK/photos\"}" >/dev/null

echo "等待索引完成（模型首次加载 + 标签矩阵，约 1~3 分钟）..."
STATUS=""
for _ in $(seq 1 360); do
  sleep 2
  STATUS="$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/api/index/status" || true)"
  echo "$STATUS" | grep -q '"running": false' && break
done

echo "最终状态：$STATUS"
python3 - "$STATUS" <<'PY'
import json, sys
st = json.loads(sys.argv[1])
assert st.get("running") is False, f"索引未结束：{st}"
assert st.get("phase") == "done", f"索引阶段异常：{st}"
assert not st.get("error"), f"索引报错：{st['error']}"
print("索引阶段 OK：done / total =", st.get("total"))
PY

STATS="$(curl -sf --max-time 30 "http://127.0.0.1:$PORT/api/stats")"
echo "stats: $STATS"
python3 - "$STATS" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
assert d.get("photos") == 2, f"照片数不对：{d.get('photos')}"
print("后端冒烟测试通过：photos=2, tags=%s, faces=%s, albums=%s"
      % (d.get("tags"), d.get("faces"), d.get("albums")))
PY
