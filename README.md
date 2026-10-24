# Garden 本地相册

本地优先的照片库：Python 后端（`src/`）负责索引、标签、人脸、OCR 与语义检索，
Tauri 桌面前端（`album-gui/`，纯 HTML/CSS/JS，无构建步骤）提供 macOS「照片」式界面。
照片与索引数据全部留在本机，不上传任何云端。

后端 15 个模块已在 211 张真实照片上跑通全链路，下面是**实跑校准结果、界面交互说明与已知问题**。

## 仓库内容

本仓库只包含源码、文档和回归脚本，以下本地资源**不入库**（体积 / 第三方许可原因）：

| 路径 | 内容 | 说明 |
|---|---|---|
| `models/` | 模型权重（约 365MB） | MobileCLIP2-S0、SCRFD/ArcFace、NIMA、RapidOCR/PP-OCR；**安装包内置**，源码仓库不含，用 `tools/fetch_models.py` 下载（SHA256 校验） |
| `testphotos/`、`testphotos_raw/` | 开发测试图（约 70MB） | pixnio 抓取的第三方图片 + 4 张真实票据，不随仓库分发 |
| `data/` | 开发库快照 | `album.db` / `vectors.npz`；应用默认使用系统数据目录 |
| `bin/` | 旧 Rust 版二进制与 ONNX Runtime 动态库 | 与当前 Python 流程无关，未入库 |
| `.venv/`、`album-gui/src-tauri/target/`、`gen/` | 本地环境与构建产物 | — |

把模型按 `src/config.py` 的路径放回 `models/` 后可建立索引与语义检索；
只浏览已建好的库不需要模型（缺模型时语义检索/重打标不可用，启动会打印异常，界面仍可用）。

## 环境

```bash
python3 -m venv .venv                      # 3.10+ 均可；安装包统一用 3.11
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install --no-deps rapidocr-onnxruntime   # 避免拉全量 opencv
python3 -m pip install huggingface_hub     # 仅下载模型需要
python3 tools/fetch_models.py              # 取模型 → ./models（约 280MB，SHA256 校验）
```

实测依赖：`numpy==1.26.4`（tflite-runtime 按 1.x ABI 构建）、`onnxruntime`、
`tflite-runtime 2.x`、`opencv-python-headless`、`ImageHash`、`jieba`、`tokenizers`、
`rapidocr-onnxruntime`（建议 `--no-deps` 装，否则会拖进全量 `opencv-python` 5.x 与 headless 冲突）。

未装：`hsemotion-onnx`（微笑分恒为 0）、`pyvips`（走 PIL 兜底解码）。
Linux 原生文件夹选择器使用 `zenity`；缺失时前端会回退到路径输入。

## 运行

```bash
# 后端（开发模式，可指定独立数据目录）
GARDEN_DATA_DIR=/path/to/library ./.venv/bin/python -m src.server --port 8899

# 桌面应用（Tauri 会自动拉起内置后端；Linux 需 libwebkit2gtk-4.1-dev / libgtk-3-dev / librsvg2-dev）
cargo run --release --manifest-path album-gui/src-tauri/Cargo.toml
```

## 下载安装包（开箱即用）

GitHub Releases 提供两个平台的自包含安装包（无需装 Python、无需手动下载模型，首次启动即可用）：

| 平台 | 文件 | 说明 |
|---|---|---|
| Linux x86_64 | `garden-album_*_amd64.deb` | Ubuntu 22.04+ / Debian 12+，`sudo apt install ./xxx.deb` |
| macOS Apple Silicon | `garden-album_*_arm64.dmg` | macOS 11+，arm64（M 系列）；未做 Apple 公证 |

安装包内置可移植 CPython 3.11、全部 Python 依赖和模型，安装后约 1.1GB。后端随应用启动，
数据仍只写本地（Linux `~/.local/share/dev.garden.album/library/`，macOS `~/Library/Application Support/dev.garden.album/library/`）。

macOS 首次打开会被 Gatekeeper 拦（没有付费 Apple 开发者证书，只做了 ad-hoc 签名）：
在 Finder 里 **右键 → 打开**，或执行
`xattr -dr com.apple.quarantine "/Applications/Garden Album.app"`。

模型版权与许可归各自上游（见 `tools/fetch_models.py` 里的 HF 仓库清单），
打包内置仅为个人/研究场景的方便；商用请自行确认许可或替换模型。

## 从源码构建安装包

```bash
python3 tools/fetch_models.py          # 1) 取模型（约 280MB；国内可 HF_ENDPOINT=https://hf-mirror.com）
bash packaging/build_backend.sh        # 2) 可移植 Python + 依赖 + src + models → album-gui/src-tauri/pkg/backend
cd album-gui
npm ci
npm run build:deb                      # Linux → src-tauri/target/release/bundle/deb/*.deb
npm run build:dmg                      # macOS arm64 → .../bundle/dmg/*.dmg
```

安装包把后端整体放进 Tauri resources（macOS `Contents/Resources/backend`、Linux `/usr/lib/Garden Album/backend`）；
`main.rs` 优先用 `resource_dir()/backend` 里的自带 Python，找不到才回退开发工程的 `.venv`。
CI 定义在 `.github/workflows/build-release.yml`：Actions 里手动 Run workflow，或推 `v*` tag 自动构建并上传 Release。

## 首次使用（面向用户）

首次启动会进入引导页，选择一个存放照片的文件夹，应用只在本地建立索引，不上传照片。
索引数据保存在系统应用数据目录（Linux 默认 `~/.local/share/dev.garden.album/library/`），
不再读取项目里的 `testphotos/` 或 `data/` 示例库。之后可通过右上角菜单「添加照片文件夹」
继续添加照片；「从文件夹导入」则把指定文件夹作为相簿导入。

调试时可用 `GARDEN_DATA_DIR=/path/to/data` 指定独立数据目录。

## 实测性能（2026-10 界面/缓存优化后，211 张库）

| 场景 | 优化前 | 现在 |
|---|---|---|
| 后端就绪 | — | **204 ms**（进程启动到 `/api/health`） |
| 首次 `/api/stats` | 1.7 s | **250 ms 冷 / 1.8 ms 热** |
| 图库首屏 JSON（211 张） | 21.9 ms（旧接口） | **20.8 ms / 150 KB** |
| 结构化检索 | 0.3 ~ 2.1 ms | **2 ~ 6 ms**（含中文解析） |
| CLIP 开放词汇首次查询 | 331 ms | **~300 ms**（文本缓存未命中） |
| CLIP 重复查询 | 226 ms | **~11 ms**（LRU + 磁盘向量缓存） |
| 缩略图 | 25 ~ 44 ms/张，内存满后整体清空 | **首次 25 ~ 78 ms，热 1.5 ~ 2.3 ms**（磁盘 immutable + 内存 LRU） |
| 211 张全量重打标 | 1.9 s/张 ≈ 6.7 min | **约 1.7 ~ 2.4 s**（复用已有向量，不解码图片） |
| 332 标签矩阵 | 每次启动 44 s / 138 标签 | **首次 80 s，之后 ~0.1 s 读缓存** |
| 视图返回 | 重新请求全部数据 | **0 请求**（3 个视图 DOM 缓存 + 图片 HTTP 缓存） |

> 索引吞吐本身仍是 ~1.9 s/张（SCRFD + NIMA×2 + OCR 的模型推理），优化掉的是**重复识别**：
> 启动不再重算标签矩阵、重打标不再解码图片、重复查询不再跑文本塔。
> DB 规模：211 照片 / 211 向量 / 240 个命中标签（1187 条标注）/ 47 人脸 / 24 事件 / 10 段回忆 / 8 个相簿。

## 界面与交互

- macOS「照片」风格：左侧导航（图库 / 回忆 / 人物 / 地点 / 清理 / 重复 / 截屏 / 文字 / 连拍 / 标签 / 相簿），
  顶部工具栏（全部·年·月·日、缩略图缩放、搜索、刷新），日期分组网格。
- **防闪屏**：稳定 DOM + 视图缓存；图片不做透明度淡入；移除全屏 `backdrop-filter`；
  只有点击刷新或 `⌘R` 才重新取数。后台索引/重打标完成只弹 toast，不自动重绘。
- **灯箱**：预加载下一张后替换，键盘 ← → 切换、`I` 信息、`Esc` 关闭；信息面板含尺寸、
  设备、事件、地点、人物、中文标签、OCR 文本与文件路径；标签可直接增删，照片可加入/移出手动相簿；
  「查看原图」按需加载原图。缩放支持滚轮 / 双击 / `+` `-` `0`（重置）/ 工具栏按钮，
  百分比显示在底部，`zoom === 1` 时显示「适应」；打开、切图、关闭都会复位缩放与平移。
- **右键菜单**：缩略图右键打开照片菜单，相簿、人物、相簿空白区右键打开通用菜单；
  支持 `Shift+F10` / 菜单键，菜单会自动避让视口边缘，左键点别处或 `Esc` 关闭。
- **标签批量入相簿**：标签页勾选多个 chip 后「加入已有相簿」把标签下照片并集加入所选手动相簿；
  「合并为相簿」新建相簿。chip 是 `role="checkbox"` 的按钮，带 `aria-checked`，支持键盘空格/回车。
- **人物改名**：人物页卡片右键或详情页 hero「重命名」修改姓名，写回后端并刷新当前视图。
- **手动整理**：侧栏「我的相簿」的 `+` 或「新建相簿」新建；手动相簿支持重命名/删除，空相簿也保留；
  灯箱里可把照片加入/移出手动相簿，也可手动添加标签（回车确认）或移除单张照片的标签。
  工具栏勾选按钮进入批量模式，点缩略图多选（Shift 连选），底部操作条可一次加入相簿。
- **从文件夹导入**：侧栏「从文件夹导入」调起本地文件夹选择器；已有照片直接加入新相簿，
  新照片在后台扫描/索引，完成后点刷新查看，不会自动重绘当前网格。
- **标签浏览/合并**：左侧「标签」页点标签进入该标签的照片合集；勾选多个标签后可合并成一个新相簿
  （照片取并集）。标签页仍支持筛选和全库删除，删除后会进入屏蔽表，重打标不再出现。
- **检索**：结构化标签/人物/时间优先；开放词汇走 CLIP，低置信或没有区分度的查询直接返回空，
  不再把整库当成结果倒出来。整句正好命中库内标签时优先按标签过滤，手动标签也能直接搜到。
- **标签体系**：332 个候选生活标签 + 同义组合并 + 全库相似度校准 + 通用词降权；
  实际命中 240 个标签；重打标 211 张约 2 s，且会保留手动标签；票据/证件再用 OCR 分类校正视觉标签
  （登机牌不再串出护照/信用卡/发票）。
  旧内置相册（所有发票 / 风景美图）已迁移为
  `所有票据 / 风景旅行 / 美食时刻 / 毛孩子 / 家人朋友 / 高分精选 / 笑脸精选 / 宝宝第一次`。
- **加载**：缩略图 320/480/1024 后台预热，网格按需 `loading=lazy`；CLIP 文本向量与
  标签矩阵均落盘，重启复用。

## 6 个"上线前必须核验点"的实测结论

| # | 项 | 实测结论 |
|---|---|---|
| 1 | **SCRFD 输出** | 9 个输出**没有 batch 维**：`scores (12800/3200/800, 1)`、`bbox (N,4)`、`kps (N,10)`。文档里的 `outs[i][0]` 会崩，已加 `_flat()` 兼容两种导出。**ltrb 确实需要 `* stride`**（不乘时人脸框只有 7×11 px，乘后 124×182 px 合理）。输出顺序与文档假设一致：scores[0:3] / bbox[3:6] / kps[6:9] |
| 2 | **CLIP tokenizer** | 词表 49408，`RobertaProcessing` 自动加 BOS/EOS（49406/49407），`auto=True` 探测正确，**不需要手工拼 BOS/EOS** |
| 3 | **CLIP 文本输出** | 本机是 `(batch, 512)`，不走 EOS 池化分支。仍把 `argmax(ids)` 改成找真 EOS 位置（原来的 `argmax` 取的是最大 token id 的位置，不是 EOS） |
| 4 | **NIMA TFLite** | NHWC `float32 [1,224,224,3]` → `[1,10]`。fp16 权重但 IO 是 fp32，`/255` 归一化正确，美学分均值 4.57（1~10 合理区间） |
| 5 | **HSEmotion** | 未安装，`SmileDetector` 静默兜底返回 0。**后果：`avg_smile` 恒为 0，「笑得自然/大笑/笑脸精选」相册永远为空** |
| 6 | **RapidOCR** | 三层兜底走通，参数落到本地模型路径。真实票据识别质量很好（见下） |

### 额外发现的重大问题：CLIP 图像预处理

文档假设 `vision_model.onnx` 吃 OpenAI CLIP 的 mean/std 归一化。**实测不成立**：

| 预处理 | 图文检索 Top1（12 类 × 2 张） | cos(mean_img, mean_txt) |
|---|---|---|
| CLIP mean/std + INTER_AREA | 2/24 | +0.03 |
| ImageNet mean/std | 2/24 | +0.03 |
| [-1,1] | 1/24 | +0.04 |
| **RGB/[0,1] 不归一化** | **11/24** | **+0.175** |

已改 `config.py`：`CLIP_MEAN=(0,0,0)`、`CLIP_STD=(1,1,1)`、`CLIP_RESIZE_INTERP="cubic"`，
`TAG_THRESH` 按实测分布（[-0.06, 0.26]）从 0.20 降到 0.15。
换模型导出后请先用同类脚本复测这两行。

## 代码层面修掉的 bug（文档原样跑会出错）

1. **`index_folder` 从不 `commit()`** → 202 张全部索引成功、`vectors.npz` 也写盘了，但 `photos` 表 0 行，进程退出时 SQLite 全量回滚。已加 `DB.commit()`（每 50 张 + 每轮结束）。
2. **EXIF `DateTimeOriginal` 读不到** → Pillow ≥9 起 `ex.get(36867)` 返回 `None`，必须 `ex.get_ifd(0x8769)`。原代码会**静默退化成文件 mtime**，时间查询、事件切分、连拍全失效。
3. **SCRFD 崩溃** → 见上表第 1 项。
4. **收据被判成发票** → `INVOICE_KWS` 把发票专属词（发票/价税合计/税号）与收据通用词（收据/收款/合计/金额）混在一个计数器里，`elif "收据" in full` 永远不可达。已拆成 `INVOICE_KWS` / `RECEIPT_KWS` / `TICKET_KWS` 并新增 `classify_bill()`（能区分 invoice / receipt / ticket / boarding pass）。
5. **`img_` 前缀被误判成截图** → Android 相机默认命名就是 `IMG_20240713_093012.jpg`，原启发式会把绝大多数真实照片打成截图。已改为：无 `Make` 才看文件名模式，文件名黑名单收紧为 `screenshot/截图/screencap/m mexport` 等。
6. **`last_meeting` 返回结构与普通查询不同** → `cmd_search` 里 `res["chips"]` KeyError。已加分支。
7. **删照片后向量残留** → `VectorStore` 无 `prune`，5 个孤儿向量会继续参与召回。已加 `prune()` 并在每轮索引末尾执行。
8. **`tags_of` 无排序** → 搜索结果里标签顺序随机（CLI 只打前 4 个）。已加 `ORDER BY score DESC`。
9. **`ImageHash` 导入名** → 4.3.2 在 Linux 装的是小写 `imagehash` 目录，`import ImageHash` 直接 ModuleNotFoundError。已加 try/except。
10. **`events.py` 硬编码阈值** → 8.0h / 200km / 3.0s / 6 / 8 全是字面量，改 config 无效。已改为读 `config.EVENT_GAP_H` 等。
11. **CLIP tag 矩阵初始化 44s** → 138 个标签各跑一次文本塔。已加 `data/cache/tag_matrix.npz` 磁盘缓存（按模型 + 标签集指纹失效）；332 标签首次构建 ~80 s，之后启动 ~0.1 s 读取。

## 端到端验证结果

真实票据 OCR（4 张，含用户提供的真图）：

| 样本 | OCR 抽取（节选） | 判定 |
|---|---|---|
| 增值税专用发票 | `发票代码：3100214130` `发票号码：34567890` `91310101MA8RP4X72H` `上海智联办公设备有限公司` `价税合计` | `invoice` |
| 南航登机牌 | `张伟/ ZHANGWEI` `CZ3101` `广州CAN→北京PEK` `登机口 A12` `座位 3F` `07:50` | `boarding pass` |
| 微信聊天截屏 | `周末爬山小队` `去紫云山徒步` `集合点还是地铁2号线东门站` `我带急救包` | `chat screenshot` |
| 火车票（Edge 截屏） | `银川站 K1178 北京站` `14车015号上铺` `￥280.5元` `新空调硬卧` | `ticket` |

其他：
- **事件切分**：24 个事件，跨零点正确合并（`2024-12-31 ~ 2025-01-01`），GPS→城市离线逆地理（上海/广州/成都/青岛/北京/杭州/三亚）
- **语义相册**：`风景旅行` 66 / `美食时刻` 26 / `家人朋友` 43 / `毛孩子` 17 / `所有票据` 9 / `高分精选` 1 ✓
- **清理建议**：51 条，精确重复置信度 0.95 ✓（含跨目录同图）
- **人物聚类**：47 张脸 → 33 个簇。同一人的多原型生效（最大簇 6 张脸），但 pixnio 数据里大部分是互不相干的人，碎片化属预期
- **时间文法**：最近N天 / 去年 / 绝对年份 / 上下月 / 本周 / 季节 / 节日 / 上半年 / X月X日 全部解析正确
- **连拍**：13 组检出，择优打分正常

## UI 回归（headless Firefox）

真实浏览器回归不依赖 Tauri：后端用可丢弃的数据目录启动，前端静态托管后在浏览器里以
`?api=<后端地址>` 找后端（`resolveApi()` 的浏览器回退分支），geckodriver 驱动 headless Firefox。

```bash
GARDEN_DATA_DIR=$PWD/.uitest/library .venv/bin/python -m src.server --port 8901 &
python3 -m http.server 8902 --directory album-gui/ui &
geckodriver --port 4444 &
python3 tools/ui_regression.py     # 56 项：启动/灯箱缩放/右键菜单/标签批量/人物改名/a11y
```

脚本会自建并清理临时手动相簿，可重复运行。当前结果：**56/56 通过**。

## 已知遗留与已改善项

1. **零样本标签质量**：已从 138 扩到 332 个生活标签，加入同义组合并、全库校准和通用词降权；票据/截图/风景/食物/宠物等明显改善。但 pixnio 测试图与文件夹名不对应，**不能拿目录名当标签 ground truth**；真实手机相册仍需抽样核验。
2. **事件特殊 emoji**：已改为组内占比 ≥25% 才打主题，不再因单张图误标圣诞/婚礼。
3. **`firsts` 的 `skip` 名单没用**。`第一次鹿`、`第一次狗` 这类标签来自 CLIP 噪声（实际是动物园背景），不是宝宝成长里程碑。
4. **「第一次」被当成 BM25 关键词**。`search "第一次去海边"` 里 `第一次` 进了 `keywords`，会污染 FTS MATCH，应进 STOP 或做成修饰词。
5. **`firsts` 查询 `firsts <person>` 未走 tag 兜底**，人物未命名时静默返回空列表，没提示。
6. **HSEmotion 缺失导致微笑相关功能整体失效**（见上表第 5 项）。
7. **`pyvips` 未装**，解码走 PIL，比 pyvips 慢（索引 1.9 s/张里解码占了一部分）。
8. **`Photo.width/height` 存的是原始 EXIF 尺寸**，`load_image_rgb(max_side=1024)` 缩过，两者不一致，`best_face_area` 用的是缩后图的相对面积（相对值所以比例仍正确），但 `cleanup_suggestions` 里"同 phash 同尺寸"用的是原图尺寸，与视觉实际不一致。

## 目录

```
Garden/
├── src/                  # Python 后端：索引 / 标签 / 人脸 / OCR / 检索 / HTTP API / 缩略图
├── album-gui/
│   ├── src-tauri/        # Tauri 壳（Rust）：拉起内置后端、原生文件夹选择器
│   └── ui/               # 前端：index.html / style.css / app.js（无构建步骤）
├── tools/                # ui_regression.py + fetch_models.py（从 HF 下载并校验模型）
├── packaging/            # build_backend.sh：组装安装包内置的可移植 Python 后端
├── .github/workflows/    # build-release.yml：CI 构建 deb / dmg
├── models/               # 模型权重（未入库，见上）
├── testphotos/           # 开发测试库（未入库）
├── testphotos_raw/       # 开发素材原图（未入库）
├── data/                 # 开发库快照（未入库）
├── requirements.txt
└── LICENSE               # MIT
```

`testphotos/` 是开发期用一次性脚本生成的：把 pixnio 散图按 9 次行程分配、注入 EXIF
（拍摄时间 / GPS / 多设备）、裁剪出 13 组连拍、复制 6 个完全重复文件，再加 4 张真实票据；
该脚本不随仓库提供。

## 许可证

MIT，见 [LICENSE](LICENSE)。
