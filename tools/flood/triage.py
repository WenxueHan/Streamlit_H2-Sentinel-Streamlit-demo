# -*- coding: utf-8 -*-
"""triage_alarms: group alarms, find first-out, mark invalid/deduplicated."""
import pandas as pd

SPEC = {
    "id": "triage_alarms",
    "name": "triage_alarms",
    "description": "对报警流做 10 分钟分组、首出识别与无效报警判别。",
    "input_params": {
        "type": "object",
        "properties": {
            "alarms": {"type": "array", "description": "报警列表"},
            "window": {"type": "integer", "default": 10},
        },
        "required": ["alarms"],
    },
}


def triage_alarms(alarms: list, window: int = 10) -> dict:
    if not alarms:
        return {"groups": [], "first_out": None, "invalid": [], "compression": 0.0}
    df = pd.DataFrame(alarms).copy()
    if "time" not in df.columns:
        raise ValueError("TRIAGE_001: alarms need time")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["bucket"] = (df["time"] - df["time"].min()).dt.total_seconds() // (window * 60)
    groups = []
    for b, g in df.groupby("bucket"):
        first = g.iloc[0]
        invalid = g[g.get("sensor_status", "ok") != "ok"] if "sensor_status" in g else g.iloc[0:0]
        groups.append({
            "bucket": int(b),
            "start": str(g["time"].min()),
            "end": str(g["time"].max()),
            "count": int(len(g)),
            "first_out": str(first.get("variable", "")),
            "variables": sorted(set(str(v) for v in g.get("variable", []))),
            "invalid_count": int(len(invalid)),
        })
    first_out = df.iloc[0].to_dict()
    invalid = df[df.get("sensor_status", "ok") != "ok"].to_dict("records") if "sensor_status" in df else []
    return {
        "groups": groups,
        "first_out": first_out,
        "invalid": invalid,
        "total": int(len(df)),
        "compression": round(1.0 - len(groups) / len(df), 3),
    }
