#!/usr/bin/env python3
"""Garden 相册 UI 回归（headless Firefox + geckodriver，无 selenium 依赖）。

用法（需要 :4444 上的 geckodriver、:8901 后端、:8902 静态 UI）：

    # 1) 用可丢弃的数据目录起后端
    GARDEN_DATA_DIR=$PWD/.uitest/library .venv/bin/python -m src.server --port 8901 &
    # 2) 起静态 UI（浏览器里没有 Tauri，前端会读 ?api= 参数）
    python3 -m http.server 8902 --directory album-gui/ui &
    # 3) 起 geckodriver
    geckodriver --port 4444 &
    # 4) 跑回归
    python3 tools/ui_regression.py

覆盖：启动渲染、灯箱缩放（按钮/双击/复位）、照片与通用右键菜单
（照片 / 非照片区 / 人物 / Shift+F10 / 点外关闭）、标签 chip 的
role=checkbox + aria-checked + 键盘空格、标签批量加入已有相簿、
人物改名、modal 无淡入动画、运行期无未捕获 JS 错误。
脚本会自建并清理临时手动相簿，可重复运行。
"""
import json
import os
import sys
import time
import urllib.request

GECKO = os.environ.get("GECKO_URL", "http://127.0.0.1:4444")
API = os.environ.get("GARDEN_API", "http://127.0.0.1:8901")
UI = os.environ.get("GARDEN_UI", "http://127.0.0.1:8902")
APP = UI + "/index.html?api=" + API
EKEY = "element-6066-11e4-a52e-4f735466cecf"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (("   <- " + str(detail)) if (detail and not ok) else ""))
    return bool(ok)


def http(method, url, payload=None, timeout=90):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body) if body else {}


def api(path):
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return json.loads(r.read())


def api_post(path, payload):
    return http("POST", API + path, payload)


class WD:
    def __init__(self):
        r = http("POST", GECKO + "/session", {"capabilities": {"alwaysMatch": {
            "browserName": "firefox",
            "moz:firefoxOptions": {"args": ["-headless", "-width=1440", "-height=920"]}}}})
        self.sid = r["value"]["sessionId"]
        self.e = GECKO + "/session/" + self.sid

    def close(self):
        try:
            http("DELETE", self.e)
        except Exception:
            pass

    def get(self, url):
        http("POST", self.e + "/url", {"url": url})

    def ex(self, script, *args):
        return http("POST", self.e + "/execute/sync", {"script": script, "args": list(args)})["value"]

    def js(self, expr, *args):
        """执行单个表达式，自动补 return（WebDriver 不会返回裸表达式值）。"""
        s = expr.strip()
        if not s.startswith("return"):
            s = "return (" + s + ")"
        return self.ex(s, *args)

    def find(self, css):
        v = http("POST", self.e + "/element", {"using": "css selector", "value": css})["value"]
        return v[EKEY]

    def find_all(self, css):
        return [x[EKEY] for x in http("POST", self.e + "/elements",
                                      {"using": "css selector", "value": css})["value"]]

    def click(self, eid):
        http("POST", self.e + "/element/" + eid + "/click", {})

    def click_css(self, css):
        self.click(self.find(css))

    def text(self, eid):
        return http("GET", self.e + "/element/" + eid + "/text")["value"]

    def attr(self, eid, name):
        return http("GET", self.e + "/element/" + eid + "/attribute/" + name)["value"]

    def keys(self, eid, text):
        http("POST", self.e + "/element/" + eid + "/value", {"text": text, "value": list(text)})

    def right_click(self, eid, x=0, y=0):
        http("POST", self.e + "/actions", {"actions": [{
            "type": "pointer", "id": "m1", "parameters": {"pointerType": "mouse"},
            "actions": [
                {"type": "pointerMove", "origin": {EKEY: eid}, "x": x, "y": y},
                {"type": "pointerDown", "button": 2},
                {"type": "pointerUp", "button": 2},
            ]}]})
        http("DELETE", self.e + "/actions")

    KEYS = {"Escape": "\ue00c", "Enter": "\ue007", "Shift": "\ue008", "Control": "\ue009",
            "Alt": "\ue00a", "Space": "\ue00d", "Tab": "\ue004", "ArrowLeft": "\ue012",
            "ArrowRight": "\ue014", "ArrowUp": "\ue013", "ArrowDown": "\ue015",
            "F10": "\ue03a"}

    def key_combo(self, *keys):
        keys = [self.KEYS.get(k, k) for k in keys]
        acts = [{"type": "keyDown", "value": k} for k in keys]
        acts += [{"type": "keyUp", "value": k} for k in reversed(keys)]
        http("POST", self.e + "/actions", {"actions": [{"type": "key", "id": "kb", "actions": acts}]})
        http("DELETE", self.e + "/actions")

    def wait(self, expr, timeout=15, interval=0.2):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.js(expr):
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False


SHIFT = "\ue008"
F10 = "\ue03a"


def main():
    # 清理历史回归相簿，再新建一个干净的临时手动相簿，保证脚本可重复运行
    for a in api("/api/albums")["albums"]:
        if a["kind"] == "manual" and a["name"].startswith("回归批量相簿"):
            api_post("/api/album/%s" % a["id"], {"action": "delete"})
    alb_name = "回归批量相簿%d" % int(time.time())
    created = api_post("/api/albums", {"name": alb_name})
    check("后端：新建临时手动相簿", "id" in created, created)
    manual_id = created.get("id")
    before_count = 0

    before_persons = api("/api/persons")["persons"]
    check("后端：存在人物数据", len(before_persons) > 0, len(before_persons))
    status = api("/api/index/status")
    check("后端：索引任务 running=false", status.get("running") is False, status)

    wd = WD()
    try:
        wd.get(APP)
        ok = wd.wait("document.readyState==='complete' && document.querySelectorAll('.tile').length>0", 25)
        check("启动：图库缩略图渲染", ok, wd.js("document.querySelectorAll('.tile').length"))
        wd.ex("window.__errs=[];"
              "window.addEventListener('error',e=>window.__errs.push(String(e.message)));"
              "window.addEventListener('unhandledrejection',e=>window.__errs.push('rej:'+String(e.reason)));")
        check("启动：引导页已跳过", wd.js("document.querySelector('#welcome') && document.querySelector('#welcome').hidden") is True)

        # ── 灯箱缩放 ─────────────────────────────────────────────
        wd.click_css(".tile")
        ok = wd.wait("!document.querySelector('#lightbox').hidden", 10)
        check("灯箱：点击缩略图打开", ok)
        check("灯箱：初始缩放显示「适应」", wd.js("document.querySelector('#lbZoomLabel').textContent.trim()") == "适应",
              wd.js("document.querySelector('#lbZoomLabel').textContent"))
        wd.click_css("#lbZoomIn")
        time.sleep(0.3)
        lbl = wd.js("document.querySelector('#lbZoomLabel').textContent.trim()")
        check("灯箱：放大后显示百分比", lbl == "125%", lbl)
        zoomed = wd.js("document.querySelector('#lbStage').classList.contains('zoomed')")
        check("灯箱：放大后 stage.zoomed 生效", zoomed)
        wd.click_css("#lbFitBtn")
        time.sleep(0.3)
        check("灯箱：重置回「适应」", wd.js("document.querySelector('#lbZoomLabel').textContent.trim()") == "适应")
        check("灯箱：重置后 transform 清空",
              wd.js("document.querySelector('#lbImg').style.transform") == "",
              wd.js("document.querySelector('#lbImg').style.transform"))
        wd.js("document.querySelector('#lbStage').dispatchEvent(new MouseEvent('dblclick',{bubbles:true,cancelable:true}))")
        time.sleep(0.3)
        lbl = wd.js("document.querySelector('#lbZoomLabel').textContent.trim()")
        check("灯箱：双击放大到 250%", lbl == "250%", lbl)
        wd.js("document.querySelector('#lbStage').dispatchEvent(new MouseEvent('dblclick',{bubbles:true,cancelable:true}))")
        time.sleep(0.3)
        check("灯箱：再次双击回到「适应」", wd.js("document.querySelector('#lbZoomLabel').textContent.trim()") == "适应")
        wd.key_combo("Escape")
        ok = wd.wait("document.querySelector('#lightbox').hidden", 5)
        check("灯箱：Esc 关闭", ok)
        wd.click_css(".tile")
        wd.wait("!document.querySelector('#lightbox').hidden", 10)
        check("灯箱：重新打开后缩放复位为「适应」",
              wd.js("document.querySelector('#lbZoomLabel').textContent.trim()") == "适应")
        wd.key_combo("Escape")
        wd.wait("document.querySelector('#lightbox').hidden", 5)

        # ── 照片右键菜单 ─────────────────────────────────────────
        tile = wd.find(".tile")
        wd.right_click(tile)
        ok = wd.wait("!document.querySelector('#photoMenu').hidden", 5)
        check("右键：缩略图弹出照片菜单", ok)
        pos = wd.js("(()=>{const m=document.querySelector('#photoMenu');const r=m.getBoundingClientRect();"
                    "return {l:r.left,t:r.top,r:r.right,b:r.bottom,w:innerWidth,h:innerHeight}})()")
        check("右键：菜单定位在视口内",
              pos["l"] >= 0 and pos["t"] >= 0 and pos["r"] <= pos["w"] and pos["b"] <= pos["h"], pos)
        wd.js("document.querySelector('#scroller').dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}))")
        check("右键：左键点击别处关闭照片菜单", wd.js("document.querySelector('#photoMenu').hidden") is True)
        wd.js("document.querySelectorAll('.tile')[0].focus()")
        wd.key_combo(SHIFT, F10)
        ok = wd.wait("!document.querySelector('#photoMenu').hidden", 5)
        check("键盘：Shift+F10 打开照片菜单", ok)
        check("键盘：菜单首项获得焦点",
              wd.js("document.activeElement && document.activeElement.closest('#photoMenu')!==null") is True)
        wd.key_combo("Escape")
        check("键盘：Esc 关闭照片菜单", wd.js("document.querySelector('#photoMenu').hidden") is True)

        # ── 标签视图：chip 无障碍 + 批量加入相簿 ──────────────────
        wd.click_css('.nav-item[data-view="tags"]')
        ok = wd.wait("document.querySelectorAll('.tag-admin').length>0", 15)
        check("标签：视图渲染标签行", ok, wd.js("document.querySelectorAll('.tag-admin').length"))
        first_chip = wd.find_all(".tag-check")[0]
        check("标签：chip 是 button 且 role=checkbox",
              wd.js("(()=>{const c=document.querySelector('.tag-check');return c.tagName==='BUTTON'&&c.getAttribute('role')==='checkbox'})()") is True)
        check("标签：初始 aria-checked=false", wd.attr(first_chip, "aria-checked") == "false", wd.attr(first_chip, "aria-checked"))
        wd.click(first_chip)
        time.sleep(0.6)
        check("标签：点击后 aria-checked=true", wd.attr(first_chip, "aria-checked") == "true", wd.attr(first_chip, "aria-checked"))
        check("标签：选中行加 .picked", wd.js("document.querySelectorAll('.tag-admin')[0].classList.contains('picked')") is True)
        check("标签：计数文案更新", wd.js("document.querySelector('#tagPickedCount').textContent.trim()").startswith("1"),
              wd.js("document.querySelector('#tagPickedCount').textContent"))
        bg = wd.js("getComputedStyle(document.querySelector('.tag-admin.picked .tag-check')).backgroundColor")
        check("标签：选中 chip 有高亮底色", bg not in ("rgba(0, 0, 0, 0)", "transparent"), bg)
        # 键盘：空格切换第二个 chip（验证 scroller keydown 不再拦截 button）
        second = wd.find_all(".tag-check")[1]
        wd.js("document.querySelectorAll('.tag-check')[1].focus()")
        wd.keys(second, " ")
        time.sleep(0.3)
        check("标签：键盘空格切换 chip 的 aria-checked（不再误触发行跳转）",
              wd.attr(second, "aria-checked") == "true" and wd.js("location.hash.includes('v=tags')") is True,
              {"aria": wd.attr(second, "aria-checked"), "hash": wd.js("location.hash")})
        check("标签：两个批量按钮均已启用",
              wd.js("(()=>{const a=document.querySelector('[data-action=\"tagToAlbum\"]');const m=document.querySelector('[data-action=\"tagMerge\"]');return !!(a&&!a.disabled&&m&&!m.disabled)})()") is True)
        # 点击「加入已有相簿」
        wd.js("document.querySelector('[data-action=\"tagToAlbum\"]').click()")
        ok = wd.wait("!document.querySelector('#modal').hidden && document.querySelectorAll('.picker-item').length>0", 8)
        check("标签：加入已有相簿 -> 弹出相簿选择", ok,
              wd.js("document.querySelectorAll('.picker-item').length"))
        anim = wd.js("getComputedStyle(document.querySelector('.modal-card')).animationName")
        check("Modal：不播放淡入动画", anim in ("none", ""), anim)
        wd.js("[...document.querySelectorAll('.picker-item')].find(b=>b.textContent.includes('%s')).click()" % alb_name)
        ok = wd.wait("document.querySelector('#modal').hidden", 15)
        check("标签：选择相簿后弹窗关闭", ok)
        after = {a["id"]: a for a in api("/api/albums")["albums"]}
        now_count = (after.get(manual_id) or {}).get("photo_count", 0)
        check("标签：照片已批量加入相簿（后端计数增加）",
              now_count > before_count,
              {"album": alb_name, "before": before_count, "after": now_count})
        toast = wd.js("document.querySelector('#toast').textContent")
        check("标签：成功 toast 提示", "加入" in toast, toast)
        wd.wait("document.querySelectorAll('.tag-admin.picked').length===0", 5)
        check("标签：成功后清空已选（aria 复位）",
              wd.js("document.querySelectorAll('.tag-admin.picked').length") == 0,
              wd.js("document.querySelectorAll('.tag-admin.picked').length"))
        check("标签：chip aria-checked 全部复位",
              wd.js("[...document.querySelectorAll('.tag-check')].every(c=>c.getAttribute('aria-checked')==='false')") is True)

        # ── 非照片区右键（手动相簿页）────────────────────────────
        wd.click_css('.album-item[data-album="%s"]' % manual_id)
        ok = wd.wait("location.hash.includes('album=%s')" % manual_id, 10)
        check("相簿：侧栏打开手动相簿", ok, wd.js("location.hash"))
        wd.js("document.querySelector('#scroller').dispatchEvent(new MouseEvent('contextmenu',"
              "{bubbles:true,cancelable:true,clientX:420,clientY:320}))")
        ok = wd.wait("!document.querySelector('#ctxMenu').hidden", 5)
        check("非照片区右键：弹出通用菜单", ok)
        labels = wd.js("[...document.querySelectorAll('#ctxMenu button')].map(b=>b.textContent.trim())")
        check("非照片区右键：包含相簿操作项",
              any("新建相簿" in x for x in labels) and any("重命名" in x for x in labels), labels)
        # 左键点别处关闭
        wd.js("document.body.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}))")
        check("非照片区右键：左键点击别处关闭通用菜单",
              wd.js("document.querySelector('#ctxMenu').hidden") is True)
        # 点重命名项 -> modal -> Esc
        wd.js("document.querySelector('#scroller').dispatchEvent(new MouseEvent('contextmenu',"
              "{bubbles:true,cancelable:true,clientX:420,clientY:320}))")
        wd.wait("!document.querySelector('#ctxMenu').hidden", 5)
        wd.js("[...document.querySelectorAll('#ctxMenu button')].find(b=>b.textContent.includes('重命名')).click()")
        ok = wd.wait("!document.querySelector('#modal').hidden", 5)
        check("非照片区右键：重命名项打开弹窗", ok, wd.js("document.querySelector('#modalTitle').textContent"))
        check("非照片区右键：菜单点击后关闭", wd.js("document.querySelector('#ctxMenu').hidden") is True)
        wd.key_combo("Escape")
        wd.wait("document.querySelector('#modal').hidden", 5)

        # ── 侧栏相簿：键盘右键 ────────────────────────────────────
        sel = '.album-item[data-album="%s"]' % manual_id
        wd.js("document.querySelector('%s').focus()" % sel)
        wd.key_combo(SHIFT, F10)
        ok = wd.wait("!document.querySelector('#ctxMenu').hidden", 5)
        check("侧栏相簿：键盘 Shift+F10 打开菜单", ok)
        geom = wd.js("(()=>{const b=document.querySelector('%s');const r=b.getBoundingClientRect();"
                     "const m=document.querySelector('#ctxMenu');const mr=m.getBoundingClientRect();"
                     "return {l:mr.left,t:mr.top,r:mr.right,b:mr.bottom,cl:r.left,ct:r.top,"
                     "w:innerWidth,h:innerHeight}})()" % sel)
        check("侧栏相簿：键盘菜单在视口内且横向贴合按钮",
              geom["l"] >= 8 and geom["t"] >= 8 and geom["r"] <= geom["w"] and geom["b"] <= geom["h"]
              and abs(geom["l"] - (geom["cl"] + 24)) < 2, geom)
        wd.key_combo("Escape")
        check("侧栏相簿：Esc 关闭菜单", wd.js("document.querySelector('#ctxMenu').hidden") is True)

        # ── 人物：右键重命名 + 键盘菜单定位 ────────────────────────
        wd.click_css('.nav-item[data-view="people"]')
        ok = wd.wait("document.querySelectorAll('.person').length>0", 20)
        check("人物：视图渲染人物卡", ok, wd.js("document.querySelectorAll('.person').length"))
        person = wd.find(".person")
        pid = wd.attr(person, "data-id")
        old_name = wd.attr(person, "data-name")
        wd.right_click(person)
        ok = wd.wait("!document.querySelector('#ctxMenu').hidden", 5)
        check("人物：右键弹出通用菜单", ok)
        labels = wd.js("[...document.querySelectorAll('#ctxMenu button')].map(b=>b.textContent.trim())")
        check("人物：菜单包含重命名", any("重命名" in x for x in labels), labels)
        wd.js("[...document.querySelectorAll('#ctxMenu button')].find(b=>b.textContent.includes('重命名')).click()")
        ok = wd.wait("!document.querySelector('#modal').hidden && !!document.querySelector('#personNameInput')", 5)
        check("人物：改名弹窗打开且带输入框", ok)
        new_name = "回归测试人物"
        wd.js("(()=>{const i=document.querySelector('#personNameInput');i.value='%s';i.dispatchEvent(new Event('input',{bubbles:true}));})()" % new_name)
        wd.js("document.querySelector('#modalActions .primary').click()")
        ok = wd.wait("document.querySelector('#modal').hidden", 15)
        check("人物：保存后弹窗关闭", ok)
        names = {str(p["id"]): p["name"] for p in api("/api/persons")["persons"]}
        check("人物：后端已更新姓名（pid=%s, %s -> %s）" % (pid, old_name, new_name),
              names.get(str(pid)) == new_name, names.get(str(pid)))
        # 键盘 Shift+F10 定位
        wd.js("document.querySelectorAll('.person')[0].focus()")
        wd.key_combo(SHIFT, F10)
        ok = wd.wait("!document.querySelector('#ctxMenu').hidden", 5)
        check("人物：键盘 Shift+F10 打开菜单", ok)
        geom = wd.js("(()=>{const c=document.querySelectorAll('.person')[0];const r=c.getBoundingClientRect();"
                     "const m=document.querySelector('#ctxMenu');const l=parseFloat(m.style.left);const t=parseFloat(m.style.top);"
                     "return {l,t,cl:r.left,ct:r.top}})()")
        check("人物：键盘菜单定位在卡片附近（非 0,0 左上角）",
              abs(geom["l"] - (geom["cl"] + 24)) < 2 and abs(geom["t"] - (geom["ct"] + 24)) < 2, geom)
        wd.key_combo("Escape")
        check("人物：Esc 关闭通用菜单", wd.js("document.querySelector('#ctxMenu').hidden") is True)

        # ── 收尾：运行时错误 ──────────────────────────────────────
        errs = wd.js("window.__errs||[]")
        check("运行时：交互过程中无未捕获 JS 错误", len(errs) == 0, errs)
    finally:
        wd.close()
        try:
            api_post("/api/album/%s" % manual_id, {"action": "delete"})
        except Exception:
            pass

    print("\n===== 汇总 =====")
    bad = [r for r in results if not r[1]]
    print(f"{len(results) - len(bad)}/{len(results)} 通过")
    for name, _, detail in bad:
        print("FAILED:", name, detail)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
