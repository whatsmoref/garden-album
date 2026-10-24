# Garden 相册 · Tauri 界面

Python 负责全部业务（索引 / 检索 / OCR / 人脸），Tauri 只做窗口和进程托管。

```
album-gui/
├── run.sh                  启动脚本
├── ui/                     前端（原生 HTML/CSS/JS，无构建步骤）
│   ├── index.html
│   ├── style.css
│   └── app.js
└── src-tauri/              Tauri 壳（Rust）
    ├── src/main.rs         拉起 Python、等端口、退出时收进程
    ├── tauri.conf.json
    └── Cargo.toml
```

## 跑起来

```bash
./run.sh
```

改了 Rust 侧先编译：

```bash
cargo build --manifest-path src-tauri/Cargo.toml
```

## 架构

启动时 `main.rs` 做三件事：

1. 从可执行文件位置向上找同时含 `src/server.py` 和 `models/` 的目录（工程根）。
2. 用工程根的 `.venv/bin/python -m src.server` 拉起后端，读它 stdout 第一行 `PORT <n>` 拿到端口。
3. 把端口通过 `api_base` 命令交给前端；窗口关闭时 `kill` 掉 Python，不留孤儿进程占端口。

前端所有数据走 `http://127.0.0.1:<port>/api/*`。

## 单独调前端

Tauri 之外也能跑，方便改样式时秒刷：

```bash
../.venv/bin/python -u -m src.server --port 8765   # 终端 A
# 浏览器打开：ui/index.html?api=http://127.0.0.1:8765   终端 B
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 存活探测，含模型是否已加载 |
| GET | `/api/stats` | 各表行数 |
| GET | `/api/search?q=&n=` | 语义检索，返回 chips / DSL / 结果 |
| GET | `/api/photos?event=&person=&ocr=&limit=&offset=` | 照片列表 |
| GET | `/api/photo/<id>` | 单张详情，含人脸与 OCR 文本 |
| GET | `/api/thumb?id=&size=` | JPEG 缩略图，size 上限 2048 |
| GET | `/api/events?limit=` | 事件时间线 |
| GET | `/api/bursts` | 连拍组 |
| GET | `/api/persons` | 人物列表 |
| POST | `/api/person/<id>` | 改名，body `{"name":"爸爸"}` |
| GET | `/api/albums` | 语义相册 |
| GET | `/api/firsts?person=` | 某人第一次系列 |
| GET | `/api/cleanup` | 清理建议（不删文件） |
| POST | `/api/index/start` | 后台重建索引 |
| GET | `/api/index/status` | 索引进度 |

## 还没接的

- `/api/bursts`、`/api/firsts` 后端已实现，界面还没入口。
- 没有打包配置之外的自动更新、安装包签名。
- `tauri build` 需要 `patchelf`（当前缺），只影响 AppImage 打包；deb 目标不受影响。