# -*- coding: utf-8 -*-
"""CLI 入口：python -m src.main <command>"""
import json
import argparse
from .db import DB
from . import quality
CHIP_EMOJI = {"time": "⏱", "tag": "🏷", "clip": "👗", "person": "👤", "mod": "😊", "kw": "🔍"}
def cmd_index(args):
    from .indexer import Indexer
    Indexer().index_folder(args.dir, watch=args.watch)
def cmd_search(args):
    from .search import SearchEngine
    res = SearchEngine(DB()).search(args.query, topk=args.n)
    if "error" in res:
        print(res["error"]); return
    if "last_time" in res:                       # 「上次见XX」型问句，返回结构不同
        p = res["photo"]
        print(f"和「{res['person']}」最近一次同框：{res['last_time']}  "
              f"（共 {res['total_photos']} 张）")
        print(f"  {p['path']}")
        return
    print("解析 chips:", " ".join(f"{CHIP_EMOJI.get(k, '')}{v}" for k, v in res["chips"]) or "(无)")
    print("DSL:", json.dumps(res["dsl"], ensure_ascii=False))
    print(f"共 {len(res['results'])} 条：")
    for i, r in enumerate(res["results"], 1):
        print(f"{i:3}. {r['taken_at']}  {r['path']}  [{','.join(r['tags'][:4])}]")
def cmd_person(args):
    db = DB()
    if args.op == "list":
        for r in db.query("SELECT p.id id, p.name name, COUNT(f.id) c FROM persons p "
                          "LEFT JOIN faces f ON f.person_id=p.id GROUP BY p.id ORDER BY c DESC"):
            print(f"  #{r['id']:<4}{r['name']:<10}{r['c']} 张脸")
        print("命名：python -m src.main person <id> <名字>（建议用：我/爸爸/妈妈/宝宝/老婆…）")
    elif args.name:
        db.rename_person(int(args.op), args.name)
        db.commit()
        print(f"已命名 #{args.op} → {args.name}（全簇生效）")
    else:
        print("用法: person list | person <id> <名字>")
def cmd_events(args):
    db = DB()
    for r in db.query("SELECT * FROM events ORDER BY start DESC LIMIT 100"):
        print(f"{r['start'][:10]} ~ {r['end'][:10]}  {r['title']}  ({r['photo_count']}张/{r['device_count']}设备)")
def cmd_albums(args):
    db = DB()
    for a in db.albums():
        rows = db.query("SELECT p.* FROM album_photos ap JOIN photos p ON p.id=ap.photo_id "
                        "WHERE ap.album_id=? ORDER BY p.taken_at DESC LIMIT 50", (a["id"],))
        print(f"《{a['name']}》 {len(rows)}+ 张")
        for r in rows[:10]:
            print(f"   {r['taken_at']}  {r['path']}")
def cmd_firsts(args):
    from .albums import AlbumEngine
    from .models import ModelHub
    for it in AlbumEngine(DB(), ModelHub.get()).firsts(args.person):
        print(f"{it['title']}: {it['photo']['taken_at']}  {it['photo']['path']}")
def cmd_cleanup(args):
    sug = quality.cleanup_suggestions(DB())
    print(f"共 {len(sug)} 条清理建议（仅供参考，不会删除任何文件）：")
    for s in sug[:args.n]:
        print(f"[{s['confidence']:.2f}] {s['type']:<15} {s['photo']['path']}")
        print(f"        ← {s['reason']}")
def main():
    ap = argparse.ArgumentParser(prog="album")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index"); p.add_argument("dir"); p.add_argument("--watch", action="store_true"); p.set_defaults(fn=cmd_index)
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("-n", type=int, default=30); p.set_defaults(fn=cmd_search)
    p = sub.add_parser("person"); p.add_argument("op"); p.add_argument("name", nargs="?"); p.set_defaults(fn=cmd_person)
    p = sub.add_parser("events"); p.set_defaults(fn=cmd_events)
    p = sub.add_parser("albums"); p.set_defaults(fn=cmd_albums)
    p = sub.add_parser("firsts"); p.add_argument("person", nargs="?", default="宝宝"); p.set_defaults(fn=cmd_firsts)
    p = sub.add_parser("cleanup"); p.add_argument("-n", type=int, default=50); p.set_defaults(fn=cmd_cleanup)
    args = ap.parse_args()
    args.fn(args)
if __name__ == "__main__":
    main()
