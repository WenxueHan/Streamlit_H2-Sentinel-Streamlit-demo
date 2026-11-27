# -*- coding: utf-8 -*-
"""Draw result figures with Pillow (no matplotlib in this environment)."""
import json
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"

NAVY = "#1F3864"
BLUE = "#4472C4"
LBLUE = "#8FAADC"
ORANGE = "#ED7D31"
RED = "#C00000"
GREEN = "#2E7D32"
PURPLE = "#7030A0"
GRAY = "#7F7F7F"
LGRAY = "#D9D9D9"


class Chart:
    def __init__(self, w, h, scale=2):
        self.w, self.h, self.s = w, h, scale
        self.img = Image.new("RGB", (w * scale, h * scale), "white")
        self.d = ImageDraw.Draw(self.img)

    def font(self, size):
        return ImageFont.truetype(FONT_PATH, int(size * self.s))

    def _p(self, pt):
        return (int(pt[0] * self.s), int(pt[1] * self.s))

    def line(self, pts, color, width=2):
        self.d.line([self._p(p) for p in pts], fill=color, width=int(width * self.s), joint="curve")

    def rect(self, box, fill=None, outline=None, width=1):
        self.d.rectangle([self._p((box[0], box[1])), self._p((box[2], box[3]))],
                         fill=fill, outline=outline, width=int(width * self.s))

    def text(self, xy, s, size=16, color="black", anchor="la", bold=False):
        f = self.font(size)
        self.d.text(self._p(xy), s, font=f, fill=color, anchor=anchor)

    def dashed_v(self, x, color, width=2, dash=8, gap=6, y0=0, y1=None):
        y1 = self.h if y1 is None else y1
        y = y0
        while y < y1:
            self.line([(x, y), (x, min(y + dash, y1))], color, width)
            y += dash + gap

    def dashed_h(self, y, color, width=2, dash=8, gap=6, x0=0, x1=None):
        x1 = self.w if x1 is None else x1
        x = x0
        while x < x1:
            self.line([(x, y), (min(x + dash, x1), y)], color, width)
            x += dash + gap

    def save(self, name):
        path = os.path.join(FIG, name)
        self.img.resize((self.w, self.h), Image.LANCZOS).save(path, dpi=(200, 200))
        print("saved", path)
        return path


def mapv(v, vmin, vmax, y0, y1):
    return y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)


def mapt(t, t0, t1, x0, x1):
    return x0 + (t - t0) / (t1 - t0) * (x1 - x0)


def fault_cn(x):
    return {"feed_sulfur": "原料硫阶跃", "amine": "胺洗失效", "deactivation": "催化剂失活",
            "furnace": "加热炉异常", "h2_pressure": "氢分压下降",
            "instrument": "仪表故障"}.get(x, x)


# --------------------------------------------------------------------------
def fig_case():
    d = pd.read_csv(os.path.join(BASE, "artifacts", "case_R05_E01.csv"), parse_dates=["time"])
    d = d[(d.time >= "2025-10-29 23:30") & (d.time <= "2025-10-30 01:15")].reset_index(drop=True)
    c = Chart(1600, 900)
    L, R, T, B = 110, 60, 120, 190
    x0, x1, y0, y1 = L, 1600 - R, 110, 900 - B
    t0, t1 = d.time.min(), d.time.max()
    vmin, vmax = 2800, 5300

    c.text((40, 34), "R05-E01 原料硫阶跃：质量预警 00:50（提前 3 min），DCS 报警 01:03（滞后 10 min）", 23, NAVY)
    c.text((40, 72), "test 运行（模型未见过）；阈值 4,500 ppm；橙线为 t+5 预测", 15, GRAY)

    # grid
    for v in range(3000, 5500, 500):
        y = mapv(v, vmin, vmax, y0, y1)
        c.line([(x0, y), (x1, y)], LGRAY, 1)
        c.text((x0 - 10, y), str(v), 14, GRAY, anchor="rm")
    ticks = pd.date_range("2025-10-29 23:30", "2025-10-30 01:15", freq="10min")
    for t in ticks:
        x = mapt(t, t0, t1, x0, x1)
        c.line([(x, y1), (x, y1 + 6)], GRAY, 1)
        c.text((x, y1 + 10), t.strftime("%H:%M"), 13, GRAY, anchor="ma")
    c.line([(x0, y0), (x0, y1)], NAVY, 2)
    c.line([(x0, y1), (x1, y1)], NAVY, 2)
    c.text((x0, y1 + 38), "时间（2025-10-30）", 15, NAVY)
    c.text((x0 - 10, y0 - 8), "出口硫 (ppm)", 14, NAVY, anchor="rm")

    # threshold
    yth = mapv(4500, vmin, vmax, y0, y1)
    c.dashed_h(yth, RED, 2, 10, 7, x0, x1)
    c.text((x1 - 8, yth - 30), "质量限值 4,500 ppm", 14, RED, anchor="ra")

    # highlight the 3-minute prediction lead (drawn before the curves)
    xa = mapt(pd.Timestamp("2025-10-30 00:50"), t0, t1, x0, x1)
    xb = mapt(pd.Timestamp("2025-10-30 00:53"), t0, t1, x0, x1)
    c.rect((xa, y0 + 4, xb, y1 - 4), fill="#FBE5D6")

    # lines
    c.line([(mapt(r.time, t0, t1, x0, x1), mapv(r.sulfur_true, vmin, vmax, y0, y1)) for r in d.itertuples()], NAVY, 3)
    c.line([(mapt(r.time, t0, t1, x0, x1), mapv(r.sulfur_analyzer, vmin, vmax, y0, y1)) for r in d.itertuples()], LBLUE, 2)
    c.line([(mapt(r.time, t0, t1, x0, x1), mapv(r.pred_5, vmin, vmax, y0, y1)) for r in d.itertuples()], ORANGE, 2)

    # markers
    marks = [
        ("2025-10-30 00:00", "故障注入", "", GRAY),
        ("2025-10-30 00:50", "质量预警", "提前 3 min", ORANGE),
        ("2025-10-30 00:53", "真值越限", "", RED),
        ("2025-10-30 01:03", "DCS 质量报警", "滞后 10 min", "#B45F06"),
    ]
    ty = 900 - 55
    for i, (ts, lab, lead, col) in enumerate(marks):
        t = pd.Timestamp(ts)
        x = mapt(t, t0, t1, x0, x1)
        c.dashed_v(x, col, 2, 6, 5, y0, y1)
        c.line([(x, ty - 10), (x - 6, ty - 20), (x + 6, ty - 20), (x, ty - 10)], col, 1)
        second = t.strftime("%H:%M") + (("  " + lead) if lead else "")
        if i % 2 == 0:
            c.text((x, ty + 4), lab, 12, col, anchor="ma")
            c.text((x, ty + 20), second, 12, col, anchor="ma")
        else:
            c.text((x, ty - 48), lab, 12, col, anchor="ma")
            c.text((x, ty - 32), second, 12, col, anchor="ma")

    # legend
    lx, ly = x0 + 20, y0 + 16
    for i, (lab, col, w) in enumerate([("真值", NAVY, 3), ("分析仪", LBLUE, 2), ("t+5 预测", ORANGE, 2)]):
        yy = ly + i * 26
        c.line([(lx, yy), (lx + 34, yy)], col, w)
        c.text((lx + 44, yy), lab, 14, "#333333", anchor="lm")
    c.save("fig1_case_timeline.png")


# --------------------------------------------------------------------------
def fig_monitor():
    d = pd.read_csv(os.path.join(BASE, "artifacts", "case_R05_E01.csv"), parse_dates=["time"])
    d = d[(d.time >= "2025-10-29 23:30") & (d.time <= "2025-10-30 01:15")].reset_index(drop=True)
    c = Chart(1600, 720)
    L, R, T, B = 110, 60, 110, 110
    x0, x1, y0, y1 = L, 1600 - R, T, 720 - B
    t0, t1 = d.time.min(), d.time.max()
    ymin, ymax = -2.0, 3.4  # log10

    c.text((40, 34), "质量相关监测：T² / SPE（归一化到 99% 阈值，纵轴 log10）", 24, NAVY)
    c.text((40, 70), "00:02 为趋势关注；质量预警由 3 步预测在 00:50 触发（提前 3 min）", 14, GRAY)

    def ly(v):
        return mapv(np.log10(max(v, 10 ** ymin)), ymin, ymax, y0, y1)

    for e in range(-2, 4):
        y = mapv(e, ymin, ymax, y0, y1)
        c.line([(x0, y), (x1, y)], LGRAY, 1)
        c.text((x0 - 10, y), "1e%d" % e, 13, GRAY, anchor="rm")
    for t in pd.date_range("2025-10-29 23:30", "2025-10-30 01:15", freq="10min"):
        x = mapt(t, t0, t1, x0, x1)
        c.line([(x, y1), (x, y1 + 6)], GRAY, 1)
        c.text((x, y1 + 10), t.strftime("%H:%M"), 13, GRAY, anchor="ma")
    c.line([(x0, y0), (x0, y1)], NAVY, 2)
    c.line([(x0, y1), (x1, y1)], NAVY, 2)
    yth = mapv(0, ymin, ymax, y0, y1)
    c.dashed_h(yth, RED, 2, 10, 7, x0, x1)
    c.text((x1 - 8, yth - 30), "阈值（归一化=1）", 14, RED, anchor="ra")

    c.line([(mapt(r.time, t0, t1, x0, x1), ly(r.T2_norm)) for r in d.itertuples()], NAVY, 3)
    c.line([(mapt(r.time, t0, t1, x0, x1), ly(r.SPE_norm)) for r in d.itertuples()], ORANGE, 3)
    for ts, lab, col, dy in [("2025-10-30 00:00", "故障注入", GRAY, 44),
                             ("2025-10-30 00:02", "趋势关注", GREEN, 8),
                             ("2025-10-30 00:50", "质量预警", ORANGE, 44),
                             ("2025-10-30 00:53", "真值越限", RED, 8)]:
        x = mapt(pd.Timestamp(ts), t0, t1, x0, x1)
        c.dashed_v(x, col, 2, 6, 5, y0, y1)
        c.text((x + 6, y0 + dy), lab, 12, col)

    lx, ly0 = x0 + 20, y0 + 16
    for i, (lab, col) in enumerate([("T²（质量相关）", NAVY), ("SPE（质量相关）", ORANGE)]):
        yy = ly0 + i * 26
        c.line([(lx, yy), (lx + 34, yy)], col, 3)
        c.text((lx + 44, yy), lab, 14, "#333333", anchor="lm")
    c.save("fig2_monitor.png")


# --------------------------------------------------------------------------
def fig_shap():
    with open(os.path.join(BASE, "artifacts", "rootcause_details.json"), encoding="utf-8") as f:
        d = json.load(f)["R05-E01"]
    items = d["shap"][:8]
    c = Chart(1500, 760)
    L, R, T, B = 300, 90, 130, 90
    x0, x1, y0, y1 = L, 1500 - R, 165, 760 - B
    vmax = max(abs(i["shap"]) for i in items) * 1.15
    zero = (x0 + x1) / 2
    step = (y1 - y0) / len(items)

    c.text((40, 34), "R05-E01 质量预警时刻（00:50）的 SHAP 归因（8 个辅助变量，256 联盟精确解）", 22, NAVY)
    c.text((40, 72), "SHAP 前四为 t_in、循环氢 H2S、dt_ad、原料硫；Agent 用原料硫优先 + 互斥规则判定为原料硫（0.794）", 14, GRAY)

    c.line([(zero, y0), (zero, y1)], GRAY, 1)
    for i, it in enumerate(items):
        y = y0 + step * (i + 0.55)
        v = it["shap"]
        w = abs(v) / vmax * (x1 - x0) / 2
        lab = it.get("label", "NEUTRAL")
        col = {"ROOT": GREEN, "COMP": ORANGE, "INT": BLUE}.get(lab, "#A6A6A6")
        if v >= 0:
            c.rect((zero, y - 12, zero + w, y + 12), fill=col)
            c.text((zero + w + 8, y), "%+.1f" % v, 14, "#333333", anchor="lm")
        else:
            c.rect((zero - w, y - 12, zero, y + 12), fill=col)
            c.text((zero - w - 8, y), "%+.1f" % v, 14, "#333333", anchor="rm")
        c.text((x0 - 12, y), "%s [%s]" % (it["feature"], lab), 14, "#333333", anchor="rm")

    c.text((x1 - 10, y1 - 10), "SHAP 值 (ppm)", 13, GRAY, anchor="ra")
    lx, ly = x0 + 10, y0 - 35
    for i, (lab, col) in enumerate([("ROOT 根因", GREEN), ("COMP 补偿", ORANGE), ("INT 传递", BLUE), ("NEUTRAL", "#A6A6A6")]):
        xx = lx + i * 160
        c.rect((xx, ly - 8, xx + 22, ly + 8), fill=col)
        c.text((xx + 30, ly), lab, 13, "#333333", anchor="lm")
    c.save("fig3_shap.png")


# --------------------------------------------------------------------------
def fig_lead():
    ev = pd.read_csv(os.path.join(BASE, "data", "events.csv"))
    me = pd.read_csv(os.path.join(BASE, "artifacts", "monitor_events.csv"))
    me = me[me.method == "Q1p_quality_predict_persist"][["event_id", "lead_min", "detected"]]
    me = me.rename(columns={"detected": "q1_detected"})
    m = ev.merge(me, on="event_id", how="left")
    m = m[(m.quality_limit_reached == 1) & (m.is_single_fault == 1) &
          (m.q1_detected == 1)].copy()
    m = m.sort_values("lead_min", ascending=False).head(12).reset_index(drop=True)

    c = Chart(1500, 780)
    L, R, T, B = 360, 80, 150, 90
    x0, x1, y0, y1 = L, 1500 - R, 170, 780 - B
    vmax = 5
    step = (y1 - y0) / len(m)
    c.text((40, 34), "质量预警提前量（Q1p，1-5 min 口径）", 24, NAVY)
    c.text((40, 72), "已检出的 8 个单故障事件全部 ≤4 min，中位数 3 min；慢变/低幅/仪表冻结不在该类别", 14, GRAY)
    for v in range(0, 6):
        x = x0 + v / vmax * (x1 - x0)
        c.line([(x, y0), (x, y1)], LGRAY, 1)
        c.text((x, y1 + 10), str(v), 13, GRAY, anchor="ma")
    c.text((x1 + 10, y1 + 10), "min", 13, GRAY)
    c.line([(x0, y0), (x0, y1)], NAVY, 2)
    c.line([(x0, y1), (x1, y1)], NAVY, 2)

    for i, r in m.iterrows():
        y = y0 + step * (i + 0.55)
        lab = "%s %s" % (r.event_id, fault_cn(r.fault_type))
        c.text((x0 - 12, y), lab, 13, "#333333", anchor="rm")
        v = 0 if pd.isna(r.lead_min) else float(r.lead_min)
        col = GREEN if 1 <= v <= 5 else ORANGE
        w = max(v, 0) / vmax * (x1 - x0)
        c.rect((x0, y - 12, x0 + max(w, 2), y + 12), fill=col)
        c.text((x0 + w + 6, y), "%.0f" % v, 13, col, anchor="lm")

    lx, ly = x0 + 10, y0 - 35
    for i, (lab, col) in enumerate([("1-5 min（达标）", GREEN), ("0 min（偏晚）", ORANGE)]):
        xx = lx + i * 200
        c.rect((xx, ly - 8, xx + 24, ly + 8), fill=col)
        c.text((xx + 32, ly), lab, 13, "#333333", anchor="lm")
    c.save("fig4_leadtime.png")


# --------------------------------------------------------------------------
def fig_accuracy():
    with open(os.path.join(BASE, "artifacts", "rootcause_metrics.json"), encoding="utf-8") as f:
        r = json.load(f)
    with open(os.path.join(BASE, "artifacts", "agent_metrics.json"), encoding="utf-8") as f:
        a = json.load(f)
    items = [
        ("SHAP\ntop-1", r["top1_accuracy"] * 100, "#A6A6A6"),
        ("SHAP\ntop-3", r["top3_accuracy"] * 100, "#A6A6A6"),
        ("Agent\n总体", a["top1_accuracy"] * 100, BLUE),
        ("Agent\n单故障", a["single_fault_accuracy"] * 100, GREEN),
        ("Agent\n复合故障", a["compound_accuracy"] * 100, ORANGE),
    ]
    c = Chart(1300, 700)
    L, R, T, B = 110, 60, 130, 110
    x0, x1, y0, y1 = L, 1300 - R, 120, 700 - B
    c.text((40, 34), "质量预警时刻的根因准确率：SHAP 原始 vs 知识 Agent", 24, NAVY)
    c.text((40, 72), "33 个质量事件；在 Q1p 质量预警时刻解释；单故障 27 个、复合 6 个", 14, GRAY)
    for v in range(0, 101, 20):
        y = mapv(v, 0, 100, y0, y1)
        c.line([(x0, y), (x1, y)], LGRAY, 1)
        c.text((x0 - 10, y), str(v) + "%", 13, GRAY, anchor="rm")
    c.line([(x0, y0), (x0, y1)], NAVY, 2)
    c.line([(x0, y1), (x1, y1)], NAVY, 2)
    step = (x1 - x0) / len(items)
    for i, (lab, v, col) in enumerate(items):
        cx = x0 + step * (i + 0.5)
        w = 70
        yv = mapv(v, 0, 100, y0, y1)
        c.rect((cx - w / 2, yv, cx + w / 2, y1), fill=col)
        c.text((cx, yv - 8), "%.1f%%" % v, 15, "#333333", anchor="ma")
        c.text((cx, y1 + 12), lab, 13, "#333333", anchor="ma")
    c.save("fig5_accuracy.png")


# --------------------------------------------------------------------------
def fig_far():
    m = pd.read_csv(os.path.join(BASE, "artifacts", "monitor_metrics.csv"))
    picks = ["B1p_aux_persist", "B2p_quality_persist", "B4_forecast_persist", "B5p_quality_residual_persist"]
    labels = {"B1p_aux_persist": "B1p\n辅助+持续", "B2p_quality_persist": "B2p\n加硫+持续",
              "B4_forecast_persist": "B4\n预测+持续", "B5p_quality_residual_persist": "B5p\n质量残差"}
    vals = []
    for p in picks:
        row = m[m.method == p]
        vals.append(float(row.episodes_per_day.iloc[0]) if len(row) else 0)
    c = Chart(1200, 640)
    L, R, T, B = 110, 60, 130, 110
    x0, x1, y0, y1 = L, 1200 - R, 120, 640 - B
    vmax = 2.2
    c.text((40, 34), "正常段误报率（次/天）：质量增广并未自动降低误报", 22, NAVY)
    c.text((40, 70), "17 个质量事件均 17/17 检出；误报差异来自测量噪声与统计量设计", 14, GRAY)
    for v in np.arange(0, 2.01, 0.5):
        y = mapv(v, 0, vmax, y0, y1)
        c.line([(x0, y), (x1, y)], LGRAY, 1)
        c.text((x0 - 10, y), "%.1f" % v, 13, GRAY, anchor="rm")
    c.line([(x0, y0), (x0, y1)], NAVY, 2)
    c.line([(x0, y1), (x1, y1)], NAVY, 2)
    step = (x1 - x0) / len(picks)
    for i, (p, v) in enumerate(zip(picks, vals)):
        cx = x0 + step * (i + 0.5)
        yv = mapv(v, 0, vmax, y0, y1)
        col = GREEN if v < 1.0 else (ORANGE if v < 1.6 else RED)
        c.rect((cx - 55, yv, cx + 55, y1), fill=col)
        c.text((cx, yv - 8), "%.2f" % v, 16, "#333333", anchor="ma")
        c.text((cx, y1 + 12), labels[p], 13, "#333333", anchor="ma")
    c.save("fig6_far.png")


if __name__ == "__main__":
    fig_case()
    fig_monitor()
    fig_shap()
    fig_lead()
    fig_accuracy()
    fig_far()
