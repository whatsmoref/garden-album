# -*- coding: utf-8 -*-
"""E. 定向 OCR：仅对疑似文档/截图（约5%）运行 RapidOCR"""
import re
import numpy as np
from .vocab import DOC_TAGS
from .metadata import load_image_rgb
# 发票专属词：命中 ≥2 个才判 invoice（收据上不会出现这些词）
INVOICE_KWS = ("发票", "价税合计", "税号", "增值税", "统一社会信用代码", "纳税人识别号")
# 收据特征词：与上面的分开计数，否则"金额/合计/收据/收款"会让所有收据都命中发票分支
RECEIPT_KWS = ("收据", "收款", "收款人", "今收到", "已收到")
TICKET_KWS = ("票号", "检票口", "座位", "登机口", "航班号", "车次", "座席", "行程单",
               "中国铁路", "电子客票", "限乘当日当次车", "硬卧", "软卧", "二等座", "一等座", "12306")
# 通用词：不参与判定，只用来确认"这是一张有金额的单据"
AMOUNT_KWS = ("金额", "合计", "小写", "大写")
AMOUNT_RE = re.compile(r"(?:¥|￥|RMB)?\s*\d+[,.]\d{2}")
class TargetedOCR:
    def need(self, row, tags):
        if row.get("is_screenshot"): return True
        return bool(DOC_TAGS & {t for t, _ in tags})
    def run(self, path):
        """→ (全文, 追加标签)"""
        from .models import ModelHub
        img = load_image_rgb(path, max_side=1800)
        bgr = np_bgr(img)
        try:
            res = ModelHub.get().ocr(bgr)
        except Exception:
            return "", []
        if isinstance(res, tuple): res = res[0]
        if not res: return "", []
        full = " ".join(item[1] for item in res if len(item) > 1)
        return full, classify_bill(full)
def classify_bill(full):
    """按关键词把票据细分成 invoice / receipt / ticket / boarding pass"""
    inv = sum(1 for k in INVOICE_KWS if k in full)
    rcp = sum(1 for k in RECEIPT_KWS if k in full)
    tkt = sum(1 for k in TICKET_KWS if k in full)
    amt = bool(AMOUNT_RE.search(full))
    out = []
    if "登机" in full or ("航班" in full and tkt):
        out.append("boarding pass")
    if tkt >= 2 or ("票" in full and tkt and not inv):
        out.append("ticket")
    if inv >= 2 or ("发票" in full and amt):
        out.append("invoice")
    if not out and rcp and (amt or any(k in full for k in AMOUNT_KWS)):
        out.append("receipt")
    elif rcp and not inv:
        out.append("receipt")
    return out
BILL_TAGS = {"invoice", "receipt", "ticket", "boarding pass"}
# 票据/证件类图片的视觉标签很容易互相串（登机牌被标成护照/信用卡/发票）。
# 一旦 OCR 能确定票据类型，就只保留该类票据，并去掉明显冲突的证件类标签。
DOC_NOISE = {"id card", "passport", "credit card", "business card",
             "banknote", "coin", "menu", "poster", "whiteboard",
             "blackboard", "map", "handwritten note", "spreadsheet",
             "code editor", "web page"}


def refine_tags(tags, ocr_text):
    """用 OCR 分类结果校正视觉标签；无 OCR 文本时原样返回。"""
    if not ocr_text:
        return tags
    bills = set(classify_bill(ocr_text))
    if not bills:
        return tags
    drop = (BILL_TAGS - bills) | DOC_NOISE
    out = [(t, s) for t, s in tags if t not in drop]
    have = {t for t, _ in out}
    for t in bills:
        if t not in have:
            out.append((t, 1.0))
    return out


def np_bgr(img):
    return np.ascontiguousarray(img[..., ::-1])
