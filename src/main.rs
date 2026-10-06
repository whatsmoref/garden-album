//! CLI 入口：album <命令>
use anyhow::Result;
use clap::Parser;

use album::{albums, db::DB, indexer, models::init_ort, quality, search::Answer, search::SearchEngine};

#[derive(Parser)]
#[command(name = "album", about = "本地相册语义检索系统", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Subcommand)]
enum Cmd {
    /// 离线索引（首次全量，之后增量）
    Index {
        dir: String,
        #[arg(long)]
        watch: bool,
        /// 并行度（ORT 内部也用线程，默认 2 较稳）
        #[arg(long, default_value_t = 2)]
        jobs: usize,
    },
    /// 语义检索
    Search {
        query: String,
        #[arg(short = 'n', default_value_t = 30)]
        n: usize,
        /// 输出 JSON（给 GUI 用）
        #[arg(long)]
        json: bool,
    },
    /// 人物：list | <id> <名字>
    Person {
        op: String,
        name: Option<String>,
    },
    /// 事件时间线
    Events {
        #[arg(long, default_value_t = 100)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// 语义相册
    Albums {
        #[arg(long)]
        json: bool,
    },
    /// 某人第一次…
    Firsts {
        person: String,
        #[arg(long)]
        json: bool,
    },
    /// 清理建议（只给建议，不删文件）
    Cleanup {
        #[arg(short = 'n', default_value_t = 50)]
        n: usize,
        #[arg(long)]
        json: bool,
    },
    /// 库状态统计
    Stats {
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    init_ort();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Index { dir, watch, jobs } => {
            set_jobs(jobs);
            let db = DB::open()?;
            let ix = indexer::Indexer::new(db)?;
            ix.index_folder(std::path::Path::new(&dir), watch)
        }
        Cmd::Search { query, n, json } => {
            let db = DB::open()?;
            let hub = std::sync::Arc::new(album::models::Hub::new());
            let se = SearchEngine::new(db, hub)?;
            let ans = se.search(&query, n)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&ans)?);
                return Ok(());
            }
            match ans {
                Answer::Error { message } => println!("{message}"),
                Answer::LastMeeting { person, last_time, total_photos, photo } => {
                    println!("和「{person}」最近一次同框：{last_time}（共 {total_photos} 张）");
                    println!("  {}", photo.path);
                }
                Answer::Search(r) => {
                    let emoji = |k: &str| match k {
                        "time" => "⏱", "tag" => "🏷", "clip" => "👗",
                        "person" => "👤", "mod" => "😊", _ => "🔍",
                    };
                    let chips: Vec<String> = r
                        .chips
                        .iter()
                        .map(|c| format!("{}{}", emoji(&c.kind), c.text))
                        .collect();
                    println!("解析 chips: {}", if chips.is_empty() { "(无)".into() } else { chips.join(" ") });
                    println!("DSL: {}", serde_json::to_string(&r.dsl)?);
                    println!("共 {} 条：", r.results.len());
                    for (i, p) in r.results.iter().enumerate() {
                        let tags: Vec<String> = p.tags.iter().take(4).cloned().collect();
                        println!("{:3}. {}  {}  [{}]", i + 1, p.taken_at.clone().unwrap_or_default(), p.path, tags.join(","));
                    }
                }
            }
            Ok(())
        }
        Cmd::Person { op, name } => {
            let db = DB::open()?;
            if op == "list" {
                let names = db.persons()?;
                for (pid, cnt) in db.face_counts()? {
                    let n = names.iter().find(|p| p.id == pid).map(|p| p.name.clone()).unwrap_or_default();
                    println!("  #{pid:<4}{n:<10}{cnt} 张脸");
                }
                println!("命名：album person <id> <名字>（建议用：我/爸爸/妈妈/宝宝/老婆…）");
            } else if let Some(name) = name {
                let pid: i64 = op.parse()?;
                db.rename_person(pid, &name)?;
                db.commit()?;
                println!("已命名 #{pid} → {name}（全簇生效）");
            } else {
                println!("用法: album person list | album person <id> <名字>");
            }
            Ok(())
        }
        Cmd::Events { limit, json } => {
            let db = DB::open()?;
            let evs = db.events(limit)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&evs)?);
                return Ok(());
            }
            for e in evs {
                println!("{} ~ {}  {}  ({}张/{}设备)", &e.start[..10], &e.end[..10], e.title, e.photo_count, e.device_count);
            }
            Ok(())
        }
        Cmd::Albums { json } => {
            let db = DB::open()?;
            let hub = std::sync::Arc::new(album::models::Hub::new());
            let ae = albums::AlbumEngine::new(db, hub)?;
            if json {
                let out: Vec<_> = ae
                    .albums
                    .iter()
                    .map(|a| {
                        let n = ae.db().album_count(a.id).unwrap_or(0);
                        serde_json::json!({ "id": a.id, "name": a.name, "count": n, "dsl": a })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&out)?);
                return Ok(());
            }
            for a in &ae.albums {
                let n = ae.db().album_count(a.id).unwrap_or(0);
                println!("《{}》 {n}+ 张", a.name);
                for p in ae.album_photos(a.id, 10)? {
                    println!("   {}  {}", p.taken_at.clone().unwrap_or_default(), p.path);
                }
            }
            Ok(())
        }
        Cmd::Firsts { person, json } => {
            let db = DB::open()?;
            let hub = std::sync::Arc::new(album::models::Hub::new());
            let ae = albums::AlbumEngine::new(db, hub)?;
            let items = ae.firsts(&person)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&items)?);
                return Ok(());
            }
            if items.is_empty() {
                println!("没有找到人物「{person}」的照片（请先 album person <id> {person} 命名）");
            }
            for it in items {
                println!("{}: {}  {}", it.title, it.taken_at, it.path);
            }
            Ok(())
        }
        Cmd::Cleanup { n, json } => {
            let db = DB::open()?;
            let sug = quality::cleanup_suggestions(&db)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&sug)?);
                return Ok(());
            }
            println!("共 {} 条清理建议（仅供参考，不会删除任何文件）：", sug.len());
            for s in sug.iter().take(n) {
                println!("[{:.2}] {:<15} {}", s.confidence, s.kind, s.photo.path);
                println!("        ← {}", s.reason);
            }
            Ok(())
        }
        Cmd::Stats { json } => {
            let db = DB::open()?;
            let hub = std::sync::Arc::new(album::models::Hub::new());
            let ix = indexer::Indexer::new(db)?;
            let st = ix.stats()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&st)?);
            } else {
                println!("照片 {} | 标签 {} | 人脸 {} | 人物 {}", st.photos, st.tags, st.faces, st.persons);
                println!("事件 {} | 连拍组 {} | 向量 {}", st.events, st.bursts, st.vectors);
                println!("OCR 文本 {} 张 | 截图 {} 张", st.with_ocr, st.screenshots);
            }
            Ok(())
        }
    }
}

/// rayon 线程池（ORT 自己也会开线程，默认 2 比较稳）
fn set_jobs(n: usize) {
    if n > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }
}
