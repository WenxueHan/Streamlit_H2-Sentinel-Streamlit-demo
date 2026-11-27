# -*- coding: utf-8 -*-
"""parameter_sandbox: replay a threshold sweep on the R05-E01 case."""
import numpy as np
import pandas as pd

from tools.paths import ARTIFACTS

SPEC = {
    "id": "parameter_sandbox",
    "name": "parameter_sandbox",
    "description": "对历史事件做阈值/余量回放，返回误报与提前量的 Pareto 候选。",
    "input_params": {
        "type": "object",
        "properties": {
            "thresholds": {"type": "array", "items": {"type": "number"}},
            "margin": {"type": "number", "default": 0},
            "case": {"type": "string", "default": "R05-E01"},
        },
        "required": [],
    },
}


def simulate(thresholds=None, margin: float = 0.0, case: str = "R05-E01") -> dict:
    df = pd.read_csv(ARTIFACTS / "case_R05_E01.csv", parse_dates=["time"])
    true_limit = df.loc[df["sulfur_true"] >= 4500, "time"]
    if true_limit.empty:
        raise ValueError("SANDBOX_001: case never crosses limit")
    t_limit = true_limit.iloc[0]
    thresholds = thresholds or [4400, 4450, 4500, 4550, 4600]
    points = []
    for th in thresholds:
        hit = df.loc[df["pred_5"] >= float(th) + float(margin), "time"]
        if hit.empty:
            points.append({"threshold": th, "lead_min": None, "alarm_time": None})
            continue
        lead = (t_limit - hit.iloc[0]).total_seconds() / 60.0
        points.append({
            "threshold": th,
            "lead_min": round(float(lead), 1),
            "alarm_time": str(hit.iloc[0]),
        })
    valid = [p for p in points if p["lead_min"] is not None and 1 <= p["lead_min"] <= 5]
    return {
        "case": case,
        "true_limit": str(t_limit),
        "margin": margin,
        "points": points,
        "recommended": valid[0] if valid else None,
        "note": "基于单事件回放；全量回放需接 monitor_events/历史库。",
    }
