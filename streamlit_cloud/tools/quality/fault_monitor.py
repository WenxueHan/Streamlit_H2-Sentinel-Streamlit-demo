# -*- coding: utf-8 -*-
"""Fault-specific monitor tool: each fault has its own z-template."""
import json
from pathlib import Path

import numpy as np

from tools.data.get_window import get_window
from tools.paths import ARTIFACTS, FEATURES, baseline

SPEC = {
    "id": "detect_fault_type",
    "name": "detect_fault_type",
    "description": "用每类故障独立的 z 模板，给出当前时刻的故障类型匹配度。",
    "input_params": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "time": {"type": "string"},
        },
        "required": ["run_id", "time"],
    },
}

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        p = Path(ARTIFACTS) / "fault_templates.json"
        _MODEL = json.loads(p.read_text(encoding="utf-8"))
    return _MODEL


def _softmax(x, T=2.0):
    x = np.asarray(x, dtype=float)
    e = np.exp((x - x.max()) / T)
    return e / e.sum()


def detect_fault_type(run_id: str, time: str) -> dict:
    w = get_window(run_id, time, 60)
    cur = np.asarray(w["X"][-1], dtype=float)
    b = baseline()
    mu = np.asarray(b["z_mu"], dtype=float)
    sd = np.asarray(b["z_sd"], dtype=float)
    z = (cur - mu) / sd
    model = _model()
    names, vals = [], []
    for fault, t in model["templates"].items():
        idx = [FEATURES.index(x) for x in t["features"]]
        m = np.asarray(t["mean"])
        sc = np.asarray(t["scale"])
        wt = np.asarray(t["weights"])
        d = float((((z[idx] - m) / sc) ** 2 * wt).sum() / wt.sum())
        names.append(fault)
        vals.append(-d)
    probs = _softmax(vals)
    order = np.argsort(-probs)
    return {
        "run_id": run_id,
        "time": w["time"],
        "top1": names[order[0]],
        "confidence": round(float(probs[order[0]]), 3),
        "top3": [{"fault": names[i], "score": round(float(probs[i]), 3)} for i in order[:3]],
        "scores": {names[i]: round(float(probs[i]), 3) for i in order},
    }
