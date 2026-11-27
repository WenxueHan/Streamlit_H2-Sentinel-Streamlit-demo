# -*- coding: utf-8 -*-
"""diagnose_root: fuse SHAP/SCA, directions and knowledge into top-3 causes."""
import math

import numpy as np
import pandas as pd

from tools.data.get_window import get_window
from tools.diagnosis.shap_explain import shap_explain
from tools.paths import ARTIFACTS, FEATURES, baseline, knowledge, load_frame

SPEC = {
    "id": "diagnose_root",
    "name": "diagnose_root",
    "description": "输入事件与证据包，输出 top-3 根因、置信度与处置建议。",
    "input_params": {
        "type": "object",
        "properties": {
            "event_id": {"type": "string"},
            "evidence": {"type": "object", "description": "可选，含 time/shap"},
        },
        "required": ["event_id"],
    },
}


def _softmax(scores, T=2.0):
    m = max(scores.values())
    ex = {k: math.exp((v - m) / T) for k, v in scores.items()}
    s = sum(ex.values()) or 1.0
    return {k: v / s for k, v in ex.items()}


def _event_time(event_id):
    ev = pd.read_csv(ARTIFACTS.parent / "data" / "events.csv")
    row = ev[ev["event_id"] == event_id]
    if row.empty:
        raise ValueError("DIAG_001: unknown event_id")
    r = row.iloc[0]
    if pd.notna(r.get("quality_limit_time_true")):
        return str(pd.Timestamp(r["quality_limit_time_true"]) - pd.Timedelta(minutes=3))
    return str(pd.Timestamp(r["start_time"]) + pd.Timedelta(minutes=30))


def diagnose_root(event_id: str, evidence: dict = None) -> dict:
    evidence = evidence or {}
    rid = event_id.split("-")[0]
    t = evidence.get("time") or _event_time(event_id)
    w = get_window(rid, t, 60)
    cur = np.asarray(w["X"][-1], dtype=float)
    b = baseline()
    mu = np.asarray(b["z_mu"], dtype=float)
    sd = np.asarray(b["z_sd"], dtype=float)
    z = {f: float((cur[i] - mu[i]) / sd[i]) for i, f in enumerate(FEATURES)}
    shap = evidence.get("shap") or shap_explain(rid, t)["shap"]
    top_feats = [s["feature"] for s in shap[:5]]
    kb = knowledge()

    scores = {}
    for name, k in kb.items():
        if name == "instrument":
            continue
        s = 0.0
        for feat, exp in k["expected"].items():
            zi = z.get(feat, 0.0)
            wgt = k["weights"].get(feat, 0.3)
            if exp == 0:
                if abs(zi) > 2.0:
                    s -= 0.4 * wgt
                continue
            match = 1 if zi * exp > 0 else 0
            strength = min(abs(zi), 3.0) / 3.0
            s += wgt * (match * (0.6 + 0.4 * strength) - (1 - match) * 0.9)
        for rank, feat in enumerate(top_feats):
            if feat in k["expected"] and k["expected"][feat] != 0:
                s += (1.2 if rank == 0 else 0.7 if rank < 3 else 0.3) * k["weights"].get(feat, 0.3)
        for item in shap[:5]:
            if item["feature"] in k["key"] and item.get("label") == "COMP":
                s -= 1.5
        zfeed = z.get("feed_sulfur_analyzer", 0.0)
        zh2s = z.get("recycle_h2s", 0.0)
        if name == "amine" and zfeed > 2.0:
            s -= 3.0
        if name == "feed_sulfur" and zh2s > 2.0 and zfeed < 1.0:
            s -= 2.0
        if name == "deactivation" and zfeed > 2.0:
            s -= 2.0
        if name == "h2_pressure" and zh2s > 2.0:
            s -= 2.0
        scores[name] = s

    if w.get("quality_flag") != "ok":
        process_max = max(scores.values(), default=0.0)
        scores["instrument"] = 2.0 + (3.0 if process_max < 2.0 else 0.0)

    conf = _softmax(scores)
    ranked = sorted(conf.items(), key=lambda kv: -kv[1])
    top, p = ranked[0]
    k = kb[top]
    actions = [{"step": s, "role": "内操/工艺", "risk": "中"} for s in k["sop"][:4]]
    return {
        "event_id": event_id,
        "time": w["time"],
        "top1": top,
        "name": k["name"],
        "confidence": round(float(p), 3),
        "top3": [{"fault": n, "score": round(float(v), 3)} for n, v in ranked[:3]],
        "evidence": shap[:5],
        "actions": actions,
        "refuse": bool(p < 0.6),
    }
