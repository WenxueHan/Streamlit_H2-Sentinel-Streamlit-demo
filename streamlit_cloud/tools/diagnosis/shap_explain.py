# -*- coding: utf-8 -*-
"""Exact grouped SHAP for one run/time (8 auxiliary variables, 256 coalitions)."""
import numpy as np
import pandas as pd

from tools.paths import ARTIFACTS, FEATURES, SULFUR_IDX, load_frame

SPEC = {
    "id": "shap_explain",
    "name": "shap_explain",
    "description": "对某运行/时刻计算 8 个辅助变量的精确分组 SHAP（256 联盟）。",
    "input_params": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "time": {"type": "string"},
            "top_k": {"type": "integer", "default": 8},
        },
        "required": ["run_id", "time"],
    },
}

_BUNDLE = None


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        from pipeline.rootcause import load_bundle
        _BUNDLE = load_bundle(str(ARTIFACTS / "forecast.npz"))
    return _BUNDLE


def shap_explain(run_id: str, time: str, top_k: int = 8) -> dict:
    from pipeline.common import windows_for_run
    from pipeline.rootcause import AUX, AUX_NAMES, ROLES, exact_shap
    from pipeline.ssm import predict_readout

    b = _bundle()
    df = load_frame()
    g = df[df["run_id"] == run_id].sort_values("time").reset_index(drop=True)
    if g.empty:
        raise ValueError("SHAP_001: unknown run_id")
    X, Y, meta = windows_for_run(g, b["L"], b["H"], 1)
    if not len(X):
        raise ValueError("SHAP_002: no window")
    times = meta["time"].to_numpy()
    t = pd.Timestamp(time)
    idx = int(np.argmin(np.abs(times - np.datetime64(t))))
    Xw = X[idx:idx + 1]
    Xs = b["sx"].transform(Xw)
    n = len(AUX)
    masks = np.array([[(m >> j) & 1 for j in range(n)] for m in range(1 << n)], np.float32)
    Xm = np.repeat(Xs, len(masks), axis=0)
    for j, f in enumerate(AUX):
        Xm[:, :, f] = Xm[:, :, f] * masks[:, j][:, None]
    F = b["ssm"].features(Xm)
    P = predict_readout(F, b["W"], b["Fm"], b["Fs"]).reshape(len(masks), b["H"], len(FEATURES))
    last = b["sx"].inverse(Xm[:, -1, :])
    pred = b["sy"].inverse(P) + last[:, None, :]
    h = b["H"] - 1
    fvals = pred[:, h, SULFUR_IDX]
    phi = exact_shap(fvals, n)
    order = np.argsort(-np.abs(phi))
    f_all = float(fvals[-1])
    delta = np.array([f_all - fvals[(1 << n) - 1 - (1 << i)] for i in range(n)])
    med = np.median(np.abs(delta)) + 1e-9
    max_exo = max([abs(phi[i]) for i in range(n) if ROLES[AUX_NAMES[i]] == "EXO"] or [0])
    rows = []
    for i in order:
        role = ROLES[AUX_NAMES[i]]
        ratio = abs(delta[i]) / (abs(phi[i]) + 1e-6)
        if role in ("MV", "PV") and ratio < 0.5:
            label = "COMP"
        elif role == "EXO" and abs(phi[i]) >= med:
            label = "ROOT"
        elif role == "PV" and max_exo > abs(phi[i]):
            label = "INT"
        else:
            label = "NEUTRAL"
        rows.append({
            "feature": AUX_NAMES[i],
            "shap": round(float(phi[i]), 2),
            "delta": round(float(delta[i]), 2),
            "role": role,
            "label": label,
        })
    return {
        "run_id": run_id,
        "time": str(times[idx]),
        "base_value": round(f_all, 2),
        "shap": rows[:int(top_k)],
        "n_coalitions": int(1 << n),
    }
