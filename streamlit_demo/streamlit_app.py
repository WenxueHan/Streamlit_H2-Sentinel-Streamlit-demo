# -*- coding: utf-8 -*-
"""加氢哨兵 · Streamlit 工控演示界面.

Run: streamlit run app/streamlit_app.py
Playback uses st.fragment(run_every=0.5) instead of a st.rerun loop.
"""
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
for p in (str(APP), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import agent_bridge
import deepseek_agent

DATA = APP / "data" / "replay_test_events.csv"
FAULT_CN = {
    "feed_sulfur": "原料硫阶跃", "amine": "胺洗失效", "furnace": "加热炉异常",
    "h2_pressure": "氢分压下降", "deactivation": "催化剂失活", "instrument": "仪表异常",
}
AUX_VARS = [
    ("feed_sulfur_analyzer", "原料硫", "ppm"),
    ("feed_flow", "进料负荷", "相对值"),
    ("t_in", "入口温度", "°C"),
    ("t_bed", "床层温度", "°C"),
    ("dt_ad", "床层温升", "°C"),
    ("p_h2", "氢分压", "bar"),
    ("recycle_h2s", "循环氢 H2S", "vol%"),
    ("fuel_valve", "燃料气阀", "%"),
]
FEATURES = [
    "feed_sulfur_analyzer", "feed_flow", "t_in", "t_bed", "dt_ad",
    "p_h2", "recycle_h2s", "fuel_valve", "sulfur_analyzer",
]
FAULT_TEMPLATES = APP / "data" / "fault_templates.json"


@st.cache_data
def load_fault_templates():
    with open(FAULT_TEMPLATES, encoding="utf-8") as f:
        return json.load(f)


def fault_scores(row):
    model = load_fault_templates()
    z = np.array([float(row["z_" + f]) for f in FEATURES])
    names, vals = [], []
    for fault, t in model["templates"].items():
        idx = [FEATURES.index(x) for x in t["features"]]
        m = np.asarray(t["mean"])
        sc = np.asarray(t["scale"])
        wt = np.asarray(t["weights"])
        d = float((((z[idx] - m) / sc) ** 2 * wt).sum() / wt.sum())
        names.append(fault)
        vals.append(-d)
    vals = np.asarray(vals)
    probs = np.exp((vals - vals.max()) / 2.0)
    probs = probs / probs.sum()
    order = np.argsort(-probs)
    return [{"fault": names[i], "label": FAULT_CN.get(names[i], names[i]),
             "score": float(probs[i])} for i in order]

st.set_page_config(
    page_title="加氢哨兵 · H2 SENTINEL",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root { --cyan:#00E5FF; --blue:#2E7BFF; --red:#FF4D6D; --amber:#FFB020; }
.stApp {
  background:
    radial-gradient(circle at 12% 0%, rgba(0,229,255,.13), transparent 30%),
    radial-gradient(circle at 88% 8%, rgba(46,123,255,.12), transparent 32%),
    linear-gradient(180deg, #070B14 0%, #0A1120 55%, #070B14 100%);
  color:#E6F1FF;
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0A1222 0%, #070B14 100%);
  border-right: 1px solid rgba(0,229,255,.18);
}
.hero {
  border:1px solid rgba(0,229,255,.28); border-radius:16px; padding:18px 22px;
  background: linear-gradient(120deg, rgba(0,229,255,.10), rgba(46,123,255,.06) 45%, rgba(7,11,20,.2));
  box-shadow: 0 0 32px rgba(0,229,255,.10), inset 0 0 30px rgba(0,229,255,.04);
  margin-bottom:14px;
}
.kicker { color:#00E5FF; font-size:12px; letter-spacing:3px; }
.hero-title { font-size:34px; font-weight:800; margin:4px 0 2px; color:#F2FAFF; }
.hero-sub { color:#8FB3D9; font-size:14px; }
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px;
  border:1px solid rgba(0,229,255,.45); color:#00E5FF; background:rgba(0,229,255,.08); margin-right:6px; }
.badge.red { border-color:rgba(255,77,109,.55); color:#FF4D6D; background:rgba(255,77,109,.08); }
.badge.amber { border-color:rgba(255,176,32,.55); color:#FFB020; background:rgba(255,176,32,.08); }
.card {
  border:1px solid rgba(0,229,255,.18); border-radius:14px; padding:14px 16px;
  background: linear-gradient(160deg, rgba(16,28,48,.75), rgba(8,14,26,.75));
  box-shadow: inset 0 0 24px rgba(0,229,255,.03); margin-bottom:10px;
}
.metric-label { color:#7FA6CC; font-size:12px; letter-spacing:1px; }
.metric-value { color:#F2FAFF; font-size:30px; font-weight:800; line-height:1.2; }
.metric-unit { color:#7FA6CC; font-size:12px; margin-left:4px; }
.metric-delta { font-size:12px; color:#8FB3D9; }
.ok { color:#37E6A5; } .warn { color:#FFB020; } .bad { color:#FF4D6D; }
.small { color:#7FA6CC; font-size:12px; }
.stButton>button {
  border-radius:10px; border:1px solid rgba(0,229,255,.35);
  background:linear-gradient(180deg, rgba(0,229,255,.16), rgba(0,229,255,.05));
  color:#E6F1FF; font-weight:600;
}
.stButton>button:hover { border-color:#00E5FF; color:#FFFFFF; box-shadow:0 0 18px rgba(0,229,255,.25); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df["fault_time"] = pd.to_datetime(df["fault_time"], format="mixed")
    df["limit_time"] = pd.to_datetime(df["limit_time"], format="mixed")
    return df


def _env_or_secret(name, default=""):
    val = os.getenv(name, default)
    try:
        val = st.secrets.get(name, val)
    except Exception:
        pass
    return val


def metric(label, value, unit, delta="", cls=""):
    st.markdown(
        '<div class="card"><div class="metric-label">%s</div>'
        '<div class="metric-value %s">%s<span class="metric-unit">%s</span></div>'
        '<div class="metric-delta">%s</div></div>' % (label, cls, value, unit, delta),
        unsafe_allow_html=True,
    )


def trend_chart(view, limit, now, warn_time):
    melt = view.melt(
        id_vars=["time"],
        value_vars=["sulfur_true", "sulfur_analyzer", "pred_1", "pred_5"],
        var_name="series", value_name="ppm",
    )
    melt["series"] = melt["series"].map({
        "sulfur_true": "出口硫真值",
        "sulfur_analyzer": "分析仪",
        "pred_1": "t+1 预测",
        "pred_5": "t+5 预测",
    })
    base = alt.Chart(melt).mark_line(strokeWidth=2.2).encode(
        x=alt.X("time:T", title=None,
                axis=alt.Axis(labelColor="#7FA6CC", gridColor="rgba(127,166,204,.08)")),
        y=alt.Y("ppm:Q", title="出口硫 (ppm)", scale=alt.Scale(domain=[2800, 4900]),
                axis=alt.Axis(labelColor="#7FA6CC", gridColor="rgba(127,166,204,.08)")),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(
                domain=["出口硫真值", "分析仪", "t+1 预测", "t+5 预测"],
                range=["#00E5FF", "#7C9CFF", "#FFB020", "#FF4D6D"]),
            legend=alt.Legend(orient="top", labelColor="#C9E2FF", title=None)),
        tooltip=["time:T", "series:N", alt.Tooltip("ppm:Q", format=".1f")],
    )
    rule = alt.Chart(pd.DataFrame({"y": [limit]})).mark_rule(
        color="#FF4D6D", strokeDash=[6, 4], strokeWidth=1.6).encode(y="y:Q")
    vline = alt.Chart(pd.DataFrame({"time": [now]})).mark_rule(
        color="#37E6A5", strokeWidth=1.2, opacity=.75).encode(x="time:T")
    layers = base + rule + vline
    if pd.notna(warn_time) and warn_time <= now:
        band = alt.Chart(pd.DataFrame({"start": [warn_time], "end": [now]})).mark_rect(
            color="#FFB020", opacity=.08).encode(x="start:T", x2="end:T")
        layers = band + layers
    return layers.properties(height=390, background="transparent").configure_view(strokeOpacity=0)


def monitor_chart(view):
    melt = view.melt(
        id_vars=["time"], value_vars=["t2_norm", "spe_norm", "ci_norm"],
        var_name="series", value_name="v",
    )
    melt["series"] = melt["series"].map({
        "t2_norm": "T² / 阈值", "spe_norm": "SPE / 阈值", "ci_norm": "CI 综合指标 / 阈值"})
    base = alt.Chart(melt).mark_line(strokeWidth=2).encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(labelColor="#7FA6CC", grid=False)),
        y=alt.Y("v:Q", title="归一化",
                axis=alt.Axis(labelColor="#7FA6CC", gridColor="rgba(127,166,204,.08)")),
        color=alt.Color("series:N", scale=alt.Scale(range=["#00E5FF", "#FFB020", "#37E6A5"]),
                        legend=alt.Legend(orient="top", labelColor="#C9E2FF", title=None)),
    )
    rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
        color="#FF4D6D", strokeDash=[5, 4]).encode(y="y:Q")
    return (base + rule).properties(height=190, background="transparent").configure_view(strokeOpacity=0)


def shap_chart(evidence):
    rows = evidence.get("shap", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["label"] = df["label"].fillna("NEUTRAL")
    return alt.Chart(df).mark_bar(cornerRadius=3).encode(
        x=alt.X("shap:Q", title="SHAP (ppm)", axis=alt.Axis(labelColor="#7FA6CC")),
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelColor="#C9E2FF")),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=["ROOT", "COMP", "INT", "NEUTRAL"],
                            range=["#37E6A5", "#FFB020", "#7C9CFF", "#5C7088"]),
            legend=alt.Legend(orient="top", labelColor="#C9E2FF", title=None)),
        tooltip=["feature", "shap", "label"],
    ).properties(height=210, background="transparent").configure_view(strokeOpacity=0)


# ---------------- state ----------------
df = load_data()
n = len(df)
ss = st.session_state
ss.setdefault("idx", 0)
ss.setdefault("playing", True)
ss.setdefault("steps", 50)
ss.setdefault("loop", True)
ss.setdefault("api_key", _env_or_secret("DEEPSEEK_API_KEY", ""))
ss.setdefault("base_url", _env_or_secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
ss.setdefault("model", _env_or_secret("DEEPSEEK_MODEL", "deepseek-chat"))
ss.setdefault("use_openjiuwen", False)
ss.setdefault("agent_result", None)
ss.setdefault("agent_future", None)
ss["idx"] = max(0, min(int(ss["idx"]), n - 1))


@st.cache_resource
def _agent_pool():
    # 后台执行 Agent；单 worker 避免多个诊断并发争用环境变量/模型客户端
    return ThreadPoolExecutor(max_workers=1)

# ---------------- sidebar controls ----------------
with st.sidebar:
    st.markdown('<div class="hero" style="padding:12px 14px">'
                '<div class="kicker">H2 SENTINEL</div>'
                '<div class="hero-title" style="font-size:22px">运行控制</div>'
                '<div class="hero-sub">1 min 采样 · 可调速回放</div></div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("暂停" if ss["playing"] else "播放", width="stretch"):
        ss["playing"] = not ss["playing"]
        st.rerun()
    if c2.button("单步", width="stretch"):
        ss["idx"] = min(n - 1, ss["idx"] + ss["steps"])
        ss["playing"] = False
        st.rerun()
    if st.button("停止（停在当前）", width="stretch"):
        ss["playing"] = False
        st.rerun()
    if st.button("回到开头（停止）", width="stretch"):
        ss["idx"] = 0
        ss["playing"] = False
        ss["agent_result"] = None
        st.rerun()
    cur_no = int(df.iloc[int(ss["idx"])]["event_no"])
    jc1, jc2 = st.columns(2)
    if jc1.button("上一事件", width="stretch"):
        target = max(1, cur_no - 1)
        ss["idx"] = int(df.index[df["event_no"] == target][0])
        ss["playing"] = True
        st.rerun()
    if jc2.button("下一事件", width="stretch"):
        target = min(int(df["event_no"].max()), cur_no + 1)
        ss["idx"] = int(df.index[df["event_no"] == target][0])
        ss["playing"] = True
        st.rerun()
    if st.button("跳到本事件 D1 预警", width="stretch"):
        cur_eid = df.iloc[int(ss["idx"])]["event_id"]
        seg = df[df["event_id"] == cur_eid]
        d1 = seg[seg["D1_Q1_fast"] == 1]
        if len(d1):
            ss["idx"] = int(d1.index[0])
            ss["playing"] = True
            st.rerun()
        else:
            st.warning("本事件没有 D1 预警（非质量越限类），请用当前时刻或“下一事件”。")
    ss["steps"] = st.slider("每 0.5 秒前进（步）", 1, 50, int(ss["steps"]),
                            help="1 步 = 1 min 数据；50 步约 3.3 分钟一轮")
    ss["loop"] = st.checkbox("循环播放全部测试事件", value=bool(ss["loop"]))
    st.caption("暂停=停住可继续；停止=停在当前；回到开头=复位。默认自动播放，15 个事件循环。")
    st.markdown("---")
    st.markdown("### DeepSeek API")
    ss["api_key"] = st.text_input("API Key", value=ss["api_key"], type="password",
                                  placeholder="sk-...", help="只保存在当前会话，不写入磁盘")
    ss["base_url"] = st.text_input("Base URL", value=ss["base_url"])
    ss["model"] = st.text_input("Model", value=ss["model"])
    if ss["api_key"]:
        k = str(ss["api_key"])
        st.caption("已从 secrets 预填：%s…%s（密码框，无需重复输入）" % (
            k[:5], k[-4:] if len(k) > 9 else ""))
    else:
        st.caption("未检测到 secrets；请在上方输入 Key，或配置 .streamlit/secrets.toml。")
    ss["use_openjiuwen"] = st.checkbox("使用 openJiuwen 运行时（需已安装）", value=bool(ss["use_openjiuwen"]))
    if st.button("测试连接", width="stretch"):
        ok, msg = agent_bridge.test_connection(ss["api_key"], ss["base_url"], ss["model"])
        (st.success if ok else st.error)(msg)
    st.caption("不填 Key 时使用本地规则诊断；填入后由 DeepSeek 解释证据。")
    st.markdown("---")
    st.markdown("### 测试事件")
    st.markdown('<span class="badge">15 个事件</span><span class="badge amber">R05 / R07 / R09</span>',
                unsafe_allow_html=True)
    st.caption("全部 test split 事件；按事件窗口循环播放")


# ---------------- live fragment (0.5s tick) ----------------
@st.fragment(run_every=0.5)
def live():
    if ss["playing"]:
        nxt = int(ss["idx"]) + int(ss["steps"])
        if nxt >= n:
            if ss["loop"]:
                ss["idx"] = 0
            else:
                ss["idx"] = n - 1
                ss["playing"] = False
        else:
            ss["idx"] = nxt

    idx = int(ss["idx"])
    row = df.iloc[idx]
    event_id = row["event_id"]
    event_no = int(row["event_no"])
    seg = df[df["event_id"] == event_id]
    seg_start = int(seg.index[0])
    seg_end = int(seg.index[-1])
    total_events = int(df["event_no"].max())
    now = row["time"]
    limit_time = row["limit_time"]
    fault_time = row["fault_time"]
    has_limit = bool(int(row["has_limit"]))
    d1_rows = seg[seg["D1_Q1_fast"] == 1]
    warn_time = d1_rows["time"].iloc[0] if len(d1_rows) else pd.NaT
    lead_warn = (limit_time - warn_time).total_seconds() / 60 if (has_limit and pd.notna(warn_time)) else None
    lead_now = (limit_time - now).total_seconds() / 60 if has_limit else None
    is_warned = pd.notna(warn_time) and now >= warn_time

    if not has_limit:
        status, status_cls = "无质量越限", ""
    elif now >= limit_time:
        status, status_cls = "已越限", "red"
    elif is_warned:
        status, status_cls = "预警中", "amber"
    else:
        status, status_cls = "正常", ""
    st.markdown(
        '<div class="hero"><div class="kicker">INDUSTRIAL ALARM INTELLIGENCE · LIVE REPLAY</div>'
        '<div class="hero-title">加氢哨兵 <span style="font-size:16px;color:#8FB3D9">/ H2 SENTINEL</span></div>'
        '<div class="hero-sub">蜡油加氢预处理 · 五步预测 · 质量监测 · 根因诊断 · DeepSeek Agent</div>'
        '<div style="margin-top:10px"><span class="badge">REPLAY</span>'
        '<span class="badge">%s · %s</span><span class="badge">%s</span>'
        '<span class="badge %s">%s</span>'
        '<span class="badge">事件 %d / %d</span></div></div>' % (
            event_id, FAULT_CN.get(str(row["fault_type"]), str(row["fault_type"])),
            now.strftime("%Y-%m-%d %H:%M"), status_cls, status, event_no, total_events),
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    with cols[0]:
        metric("出口硫真值", "%.0f" % row["sulfur_true"], "ppm",
               "分析仪 %.0f ppm" % row["sulfur_analyzer"],
               "bad" if row["sulfur_true"] >= row["limit"] else "")
    with cols[1]:
        metric("t+5 预测", "%.0f" % row["pred_5"], "ppm",
               "较当前 %+.0f ppm" % (row["pred_5"] - row["sulfur_true"]),
               "warn" if row["pred_5"] >= row["limit"] else "")
    with cols[2]:
        metric("质量限值", "%.0f" % row["limit"], "ppm", "出口硫不超过 4,500 ppm")
    with cols[3]:
        if not has_limit:
            metric("质量越限", "—", "", "本事件未越限", "ok")
        elif is_warned and lead_warn is not None:
            metric("预警提前量", "%.0f" % lead_warn, "min", "质量预警（D1）", "ok")
        else:
            metric("距越限", "%.0f" % lead_now, "min", "本事件真值越限",
                   "warn" if lead_now is not None and lead_now <= 5 else "")
    with cols[4]:
        metric("T² / 阈值", "%.2f" % row["t2_norm"], "倍",
               "SPE %.2f 倍" % row["spe_norm"],
               "bad" if row["t2_norm"] > 1 else "ok")
    with cols[5]:
        metric("CI 综合指标", "%.2f" % row["ci_norm"], "×阈值",
               "T²ⁿ + SPEⁿ", "bad" if row["ci_norm"] > 1 else "ok")

    left, right = st.columns([2.15, 1.0], gap="large")
    with left:
        st.markdown('<div class="metric-label">出口硫趋势 / 五步预测</div>', unsafe_allow_html=True)
        view = df.iloc[seg_start:idx + 1]
        if len(view) > 800:
            view = view.iloc[::max(1, len(view) // 800)]
        st.altair_chart(trend_chart(view, row["limit"], now, warn_time), width="stretch")
        st.markdown('<div class="metric-label" style="margin-top:10px">质量相关监测 T² / SPE（归一化）</div>',
                    unsafe_allow_html=True)
        st.altair_chart(monitor_chart(view), width="stretch")
        st.markdown('<div class="metric-label" style="margin:8px 0 4px">五步预测（ppm）</div>',
                    unsafe_allow_html=True)
        pcols = st.columns(5)
        for k in range(5):
            v = row["pred_%d" % (k + 1)]
            with pcols[k]:
                st.markdown(
                    '<div class="card" style="padding:10px 8px;text-align:center">'
                    '<div class="metric-label">t+%d</div>'
                    '<div style="font-size:22px;font-weight:800;color:%s">%.0f</div>'
                    '<div class="small">%+.0f</div></div>' % (
                        k + 1, "#FF4D6D" if v >= row["limit"] else "#00E5FF", v, v - row["sulfur_true"]),
                    unsafe_allow_html=True)

        st.markdown('<div class="metric-label" style="margin:8px 0 4px">辅助变量实时值（8 个）</div>',
                    unsafe_allow_html=True)
        for rr in range(2):
            acols = st.columns(4)
            for cc in range(4):
                key, label, unit = AUX_VARS[rr * 4 + cc]
                v = float(row[key])
                z = float(row["z_" + key])
                vtxt = "%.0f" % v if abs(v) >= 1000 else "%.2f" % v
                cls = "bad" if abs(z) > 3 else "warn" if abs(z) > 2 else "ok"
                with acols[cc]:
                    st.markdown(
                        '<div class="card" style="padding:9px 10px">'
                        '<div class="metric-label">%s</div>'
                        '<div style="font-size:20px;font-weight:800;color:#F2FAFF">%s'
                        '<span class="metric-unit">%s</span></div>'
                        '<div class="small %s">z = %+.1f</div></div>' % (
                            label, vtxt, unit, cls, z),
                        unsafe_allow_html=True)

        st.markdown('<div class="metric-label" style="margin:8px 0 4px">辅助变量 z-score 趋势</div>',
                    unsafe_allow_html=True)
        labels = [x[1] for x in AUX_VARS]
        sel = st.multiselect("选择辅助变量", labels,
                             default=["原料硫", "循环氢 H2S", "入口温度", "床层温升"],
                             key="aux_sel")
        if sel:
            zcols = ["z_" + x[0] for x in AUX_VARS if x[1] in sel]
            mv = view.melt(id_vars=["time"], value_vars=zcols, var_name="zvar", value_name="z")
            name_map = {"z_" + x[0]: x[1] for x in AUX_VARS}
            mv["变量"] = mv["zvar"].map(name_map)
            zbase = alt.Chart(mv).mark_line(strokeWidth=1.8).encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(labelColor="#7FA6CC", grid=False)),
                y=alt.Y("z:Q", title="z-score", axis=alt.Axis(labelColor="#7FA6CC")),
                color=alt.Color("变量:N", legend=alt.Legend(orient="top", labelColor="#C9E2FF", title=None)),
                tooltip=["time:T", "变量:N", alt.Tooltip("z:Q", format=".1f")],
            )
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#5C7088").encode(y="y:Q")
            band = alt.Chart(pd.DataFrame({"y": [-3, 3]})).mark_rule(
                color="#FF4D6D", strokeDash=[4, 4], opacity=.5).encode(y="y:Q")
            st.altair_chart((zbase + zero + band).properties(height=220, background="transparent")
                            .configure_view(strokeOpacity=0), width="stretch")

        with st.expander("辅助变量明细"):
            aux_df = pd.DataFrame([{
                "变量": label,
                "当前值": float(row[key]),
                "单位": unit,
                "z-score": float(row["z_" + key]),
                "状态": "显著偏离" if abs(float(row["z_" + key])) > 3 else
                        "偏离" if abs(float(row["z_" + key])) > 2 else "正常",
            } for key, label, unit in AUX_VARS])
            st.dataframe(aux_df, width="stretch", hide_index=True)

        with st.expander("报警 / 事件时间线"):
            rows = [
                {"时刻": fault_time, "事件": "故障注入：%s" % FAULT_CN.get(str(row["fault_type"]), row["fault_type"]), "状态": "已发生"},
                {"时刻": warn_time, "事件": "质量预警（D1，3 步预测越限）", "状态": "已触发" if is_warned else "未触发"},
            ]
            if has_limit:
                rows += [
                    {"时刻": limit_time, "事件": "出口硫真值越限 4,500 ppm", "状态": "已越限" if now >= limit_time else "未越限"},
                    {"时刻": limit_time + pd.Timedelta(minutes=10), "事件": "DCS 质量 HH 报警（典型滞后）", "状态": "滞后约 10 min"},
                ]
            else:
                rows.append({"时刻": pd.NaT, "事件": "本事件未触发质量越限", "状态": "—"})
            ev = pd.DataFrame(rows)
            ev["时刻"] = pd.to_datetime(ev["时刻"])
            st.dataframe(ev, width="stretch", hide_index=True)

    with right:
        st.markdown('<div class="card"><div class="metric-label">检测器</div>', unsafe_allow_html=True)
        for key, name in [("D1_Q1_fast", "D1 质量预警"), ("D2_Q1_low", "D2 低幅趋势"),
                          ("D3_instrument", "D3 仪表健康"), ("D4_deactivation", "D4 失活趋势")]:
            on = bool(row[key])
            st.markdown('<div style="display:flex;justify-content:space-between;margin:6px 0">'
                        '<span class="small">%s</span><span class="%s">%s</span></div>' % (
                            name, "ok" if on else "small", "已触发" if on else "未触发"),
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="metric-label">故障专项监测（每类独立模板）</div>',
                    unsafe_allow_html=True)
        for item in fault_scores(row)[:3]:
            st.progress(min(max(item["score"], 0.0), 1.0),
                        text="%s · %.1f%%" % (item["label"], 100 * item["score"]))
        st.caption("每类故障独立 z 模板；Agent 将其作为候选先验融合。")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="metric-label">Agent 诊断（当前事件）</div>', unsafe_allow_html=True)
        st.caption("当前事件：%s · %s · %s" % (
            row["event_id"], FAULT_CN.get(str(row["fault_type"]), row["fault_type"]),
            row["time"].strftime("%H:%M")))
        if now < fault_time:
            st.info("当前仍在故障注入前，证据只有正常基线；建议先“跳到本事件 D1 预警”。")
        elif has_limit and not is_warned:
            st.info("当前尚未触发 D1，故障特征较弱；建议先“跳到本事件 D1 预警”。")
        st.caption("本地证据 + DeepSeek 解释；无 Key 时走规则兜底。")
        # Agent 放到后台线程，回放 fragment 不阻塞
        fut = ss.get("agent_future")
        if fut is not None and fut.done():
            try:
                ss["agent_result"] = fut.result()
            except Exception as exc:
                ss["agent_result"] = {
                    "mode": "error",
                    "answer": {"conclusion": "Agent 运行失败：%s" % exc},
                    "evidence": {},
                }
            ss["agent_future"] = None
        pending = ss.get("agent_future") is not None
        if st.button("诊断中…" if pending else "运行 Agent 诊断",
                     width="stretch", disabled=pending):
            q = "分析当前事件 %s，给出根因和处置建议。" % row["event_id"]
            ss["agent_result"] = None
            ss["agent_future"] = _agent_pool().submit(
                agent_bridge.diagnose,
                api_key=ss["api_key"], base_url=ss["base_url"], model=ss["model"],
                event_id=row["event_id"], query=q,
                use_openjiuwen=bool(ss["use_openjiuwen"]),
                run_id=row["run_id"], time=str(row["time"]),
                fault_type=str(row["fault_type"]),
            )
        r = ss["agent_result"]
        if r:
            old_eid = (r.get("evidence") or {}).get("event_id")
            if old_eid and old_eid != row["event_id"]:
                st.warning("当前显示的是事件 %s 的结果；请点“运行 Agent 诊断”重新分析 %s。" % (
                    old_eid, row["event_id"]))
            a = r.get("answer", {})
            mode = r.get("mode", "local_rule")
            badge = {"deepseek": "DeepSeek", "openjiuwen": "OPENJIUWEN",
                     "local_rule": "LOCAL RULE"}.get(mode, "ERROR")
            st.markdown('<span class="badge %s">%s</span>' % ("amber" if mode in ("deepseek", "openjiuwen") else "", badge),
                        unsafe_allow_html=True)
            st.markdown("**结论**：%s" % a.get("conclusion", "-"))
            st.markdown("**根因**：`%s`　**置信度**：%.3f" % (
                a.get("root_cause", "-"), float(a.get("confidence", 0) or 0)))
            for i, act in enumerate(a.get("actions", [])[:5], 1):
                if isinstance(act, dict):
                    st.markdown("%d. %s（%s）" % (i, act.get("step", ""), act.get("role", "")))
                else:
                    st.markdown("%d. %s" % (i, act))
            if a.get("refuse"):
                st.warning("证据不足，Agent 建议转人工确认。")
            if r.get("openjiuwen_note"):
                st.caption(r["openjiuwen_note"])
            sc = shap_chart(r.get("evidence", {}))
            if sc is not None:
                st.altair_chart(sc, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    event_pos = idx - seg_start + 1
    event_len = seg_end - seg_start + 1
    st.progress(event_pos / max(event_len, 1),
                text="当前事件 %d / %d · %s · %d / %d" % (
                    event_no, total_events, event_id, event_pos, event_len))
    st.caption("全部测试集 %d 行 · 15 个事件循环" % n)


live()

st.markdown(
    '<div style="margin-top:18px;color:#5C7088;font-size:11px;'
    'border-top:1px solid rgba(0,229,255,.12);padding-top:10px">'
    'H2 SENTINEL · 数据回放为仿真数据（1 min/步）· 预测/监测为确定性模型 · Agent 仅做解释，不写 DCS'
    '</div>',
    unsafe_allow_html=True,
)
