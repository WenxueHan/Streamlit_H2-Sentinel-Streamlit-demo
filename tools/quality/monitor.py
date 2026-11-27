# -*- coding: utf-8 -*-
"""monitor_quality: T2/SPE on the predicted vector plus D1-D4 detectors."""
import numpy as np

from tools.paths import FEATURES, SULFUR_IDX, baseline

SPEC = {
    "id": "monitor_quality",
    "name": "monitor_quality",
    "description": "用 5 步预测向量与当前值计算 T2/SPE，并给出 D1-D4 检测器状态。",
    "input_params": {
        "type": "object",
        "properties": {
            "pred": {"type": "array", "description": "5x9 预测"},
            "current": {"type": "array", "description": "当前 9 维"},
        },
        "required": ["pred", "current"],
    },
}


def _t2_spe(current, b):
    z = (np.asarray(current, dtype=float) - np.asarray(b["mu"])) / np.asarray(b["sd"])
    P = np.asarray(b["P"], dtype=float)
    lam = np.asarray(b["lam"], dtype=float)
    t2 = float((((z @ P) ** 2) / lam).sum())
    E = z - (z @ P) @ P.T
    spe = float((E ** 2).sum())
    return t2, spe


def monitor_quality(pred, current) -> dict:
    p = np.asarray(pred, dtype=float)
    c = np.asarray(current, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(FEATURES):
        raise ValueError("MONITOR_001: pred must be Hx9")
    if c.shape[0] != len(FEATURES):
        raise ValueError("MONITOR_002: current must be 9")
    b = baseline()
    t2, spe = _t2_spe(c, b)
    t2n = t2 / float(b["thr_t2"])
    spen = spe / float(b["thr_spe"])
    ci_rms = float(np.sqrt(t2n ** 2 + spen ** 2))
    ci_lit = float(t2n + spen)
    thr_ci_rms = float(b.get("thr_ci_rms", 1.2))
    thr_ci_lit = float(b.get("thr_ci_lit", 1.5))
    thr_ci = float(b.get("thr_ci", thr_ci_lit))
    ci = t2n + spen
    ci_norm = ci / thr_ci
    # D1: 3-step quality crossing; D2: 5-step low-severity trend
    d1 = bool(p[2, SULFUR_IDX] >= 4490.0 and (p[2, SULFUR_IDX] - p[0, SULFUR_IDX]) >= 15.0)
    d2 = bool(p[4, SULFUR_IDX] >= 4470.0 and (p[4, SULFUR_IDX] - p[0, SULFUR_IDX]) >= 8.0)
    # D3: analyzer vs soft sensor residual
    sigma = float(b.get("sulfur_sigma", 60.0))
    d3 = bool(abs(c[SULFUR_IDX] - p[0, SULFUR_IDX]) > 3.0 * sigma)
    # D4: bed temperature rise far below normal
    mu = np.asarray(b["z_mu"], dtype=float)
    sd = np.asarray(b["z_sd"], dtype=float)
    d4 = bool((c[4] - mu[4]) / sd[4] < -3.0)
    return {
        "t2": round(t2, 3),
        "spe": round(spe, 3),
        "t2_norm": round(t2n, 3),
        "spe_norm": round(spen, 3),
        "ci": round(ci, 3),
        "ci_norm": round(ci_norm, 3),
        "ci_rms": round(ci_rms, 3),
        "ci_rms_norm": round(ci_rms / thr_ci_rms, 3),
        "ci_lit": round(ci_lit, 3),
        "ci_lit_norm": round(ci_lit / thr_ci_lit, 3),
        "detectors": {
            "D1_Q1_fast": d1,
            "D2_Q1_low": d2,
            "D3_instrument": d3,
            "D4_deactivation": d4,
        },
        "thresholds": {"t2": b["thr_t2"], "spe": b["thr_spe"],
                       "ci": thr_ci, "ci_rms": thr_ci_rms, "ci_lit": thr_ci_lit},
    }
