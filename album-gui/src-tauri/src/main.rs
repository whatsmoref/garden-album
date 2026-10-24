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
    pick_folder_impl()
}

#[cfg(target_os = "macos")]
fn pick_folder_impl() -> Result<Option<String>, String> {
    // macOS 没有 zenity，用系统自带的 osascript 弹原生选择框
    let script = r#"POSIX path of (choose folder with prompt "选择照片文件夹")"#;
    let out = Command::new("osascript")
        .args(["-e", script])
        .output()
        .map_err(|e| format!("调用 osascript 失败：{e}"))?;
    if !out.status.success() {
        return Ok(None); // 用户取消时 osascript 返回非 0
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    Ok(if path.is_empty() { None } else { Some(path) })
}

#[cfg(not(target_os = "macos"))]
fn pick_folder_impl() -> Result<Option<String>, String> {
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

/// 后端所在目录：
/// - 安装包模式：`resource_dir()/backend`（packaging/build_backend.sh 生成的自包含后端）
/// - 开发模式：向上找含 `src/server.py` 的工程根
fn backend_root(app: &tauri::App) -> Result<std::path::PathBuf, String> {
    if let Ok(res) = app.path().resource_dir() {
        let cand = res.join("backend");
        if cand.join("src/server.py").is_file() {
            return Ok(cand);
        }
    }
    find_project_root()
        .ok_or_else(|| "找不到后端：安装包内没有 backend 资源，也不在开发工程树内".to_string())
}

/// 后端解释器：包内自带 Python → 开发用 .venv → 系统 python3
fn backend_python(root: &std::path::Path) -> std::path::PathBuf {
    let bundled = root.join("python/bin/python3");
    if bundled.is_file() {
        return bundled;
    }
    let venv = root.join(".venv/bin/python");
    if venv.is_file() {
        return venv;
    }
    "python3".into()
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let root = app.path().app_data_dir()?;
            let data_dir = root.join("library");
            let _ = std::fs::create_dir_all(&data_dir);

            let project = backend_root(app)?;
            let python = backend_python(&project);

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

/// 向上找含 `src/server.py` 的目录；优先返回同时含 `models/` 的那一层。
fn find_project_root() -> Option<std::path::PathBuf> {
    let mut dir = std::env::current_exe().ok()?;
    let mut fallback = None;
    for _ in 0..8 {
        if dir.join("src/server.py").exists() {
            if dir.join("models").is_dir() {
                return Some(dir);
            }
            if fallback.is_none() {
                fallback = Some(dir.clone());
            }
        }
        dir = dir.parent()?.to_path_buf();
    }
    fallback
}