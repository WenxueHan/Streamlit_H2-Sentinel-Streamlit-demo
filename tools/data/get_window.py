# -*- coding: utf-8 -*-
"""get_window: read a 60-minute online window for one run/time."""
import numpy as np
import pandas as pd

from tools.paths import FEATURES, load_frame

SPEC = {
    "id": "get_window",
    "name": "get_window",
    "description": "读取某运行、某时刻之前的 60 分钟在线窗口（9 个变量），用于预测。",
    "input_params": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "运行编号，如 R05"},
            "time": {"type": "string", "description": "ISO 时间，如 2025-10-30T00:50:00"},
            "length": {"type": "integer", "default": 60, "minimum": 20, "maximum": 180},
        },
        "required": ["run_id", "time"],
    },
}


def get_window(run_id: str, time: str, length: int = 60) -> dict:
    """Return the last `length` rows up to `time`."""
    df = load_frame()
    g = df[df["run_id"] == run_id].sort_values("time").reset_index(drop=True)
    if g.empty:
        raise ValueError("DATA_001: unknown run_id %s" % run_id)
    t = pd.Timestamp(time)
    pos = int(g["time"].searchsorted(t, side="right") - 1)
    if pos < 0:
        raise ValueError("DATA_002: time before run start")
    end = pos + 1
    start = end - int(length)
    if start < 0:
        raise ValueError("DATA_003: not enough history")
    w = g.iloc[start:end]
    X = w[FEATURES].to_numpy(np.float32)
    if len(X) != int(length):
        raise ValueError("DATA_004: window length mismatch")
    return {
        "run_id": run_id,
        "time": str(w["time"].iloc[-1]),
        "features": FEATURES,
        "X": X.tolist(),
        "quality_flag": "ok" if "sensor_status" not in w or str(w["sensor_status"].iloc[-1]) == "ok" else "sensor_fault",
    }
