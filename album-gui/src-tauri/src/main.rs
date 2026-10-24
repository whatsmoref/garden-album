// Garden 相册 —— Tauri 壳
//
// 职责很窄：把 Python 后端拉起来、等它就绪、把地址交给前端、退出时收干净。
// 所有业务逻辑（检索 / 索引 / OCR / 人脸）都在 Python 侧，这里不碰模型。

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Manager, State};

/// Python 后端进程 + 它实际监听的端口。
struct Backend {
    child: Mutex<Option<Child>>,
    port: Mutex<u16>,
}

/// 前端要用的 API 根地址，例如 `http://127.0.0.1:53421`。
#[tauri::command]
fn api_base(state: State<'_, Backend>) -> String {
    let port = *state.port.lock().unwrap();
    format!("http://127.0.0.1:{port}")
}

/// 原生文件夹选择器；取消时返回 None。
#[tauri::command]
fn pick_folder() -> Result<Option<String>, String> {
    let out = Command::new("zenity")
        .args(["--file-selection", "--directory", "--title=选择照片文件夹"])
        .output()
        .map_err(|e| format!("找不到 zenity 文件夹选择器：{e}"))?;
    if !out.status.success() {
        return Ok(None);
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    Ok(if path.is_empty() { None } else { Some(path) })
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let root = app.path().app_data_dir()?;
            let data_dir = root.join("library");
            let _ = std::fs::create_dir_all(&data_dir);

            // 从可执行文件所在位置往上找工程根（含 src/server.py 的那一层）
            let project = find_project_root()
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

            let python = project.join(".venv/bin/python");
            let python = if python.exists() { python } else { "python3".into() };

            let mut child = Command::new(&python)
                .current_dir(&project)
                .env("GARDEN_DATA_DIR", &data_dir)
                .args(["-u", "-m", "src.server"])
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|e| format!("拉起 Python 后端失败（{}）: {e}", python.display()))?;

            // 后端第一行 stdout 是 `PORT <n>`
            let stdout = child.stdout.take().ok_or("拿不到后端 stdout")?;
            let mut reader = std::io::BufReader::new(stdout);
            let mut port = 0u16;
            let mut line = String::new();
            for _ in 0..40 {
                line.clear();
                if std::io::BufRead::read_line(&mut reader, &mut line).unwrap_or(0) == 0 {
                    break;
                }
                if let Some(rest) = line.trim().strip_prefix("PORT ") {
                    port = rest.trim().parse().unwrap_or(0);
                    break;
                }
            }
            if port == 0 {
                let _ = child.kill();
                return Err(format!("后端没有报告端口，读到的首行是：{line:?}").into());
            }

            // 读掉剩余 stdout，避免管道写满把后端卡死
            std::thread::spawn(move || {
                use std::io::Read;
                let mut sink = Vec::new();
                let _ = reader.read_to_end(&mut sink);
            });

            app.manage(Backend {
                child: Mutex::new(Some(child)),
                port: Mutex::new(port),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![api_base, pick_folder])
        .build(tauri::generate_context!())
        .expect("构建 Tauri 应用失败")
        .run(|app, event| {
            // 关窗时确保 Python 不会变成孤儿进程占着端口
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(backend) = app.try_state::<Backend>() {
                    if let Ok(mut guard) = backend.child.lock() {
                        if let Some(c) = guard.as_mut() {
                            let _ = c.kill();
                            let _ = c.wait();
                        }
                    }
                }
            }
        });
}

/// 向上找同时含 `src/server.py` 和 `models/` 的目录。
fn find_project_root() -> Option<std::path::PathBuf> {
    let mut dir = std::env::current_exe().ok()?;
    for _ in 0..8 {
        if dir.join("src/server.py").exists() && dir.join("models").is_dir() {
            return Some(dir);
        }
        dir = dir.parent()?.to_path_buf();
    }
    None
}