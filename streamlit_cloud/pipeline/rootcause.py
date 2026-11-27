# -*- coding: utf-8 -*-
"""Exact grouped SHAP over auxiliary variables + structural counterfactual
attribution (SCA) for root-cause localisation.

The quality variable (sulfur) is kept as the prediction target's history but is
NOT a SHAP group: we want to explain which auxiliary variable drives the
forecast, not re-discover that sulfur predicts itself.
"""
import argparse
import itertools
import json
import math
import os
import numpy as np
import pandas as pd

from .common import (FEATURES, SULFUR_IDX, Scaler, load_df, windows_for_run)
from .ssm import MambaSSM, predict_readout

AUX = [i for i in range(len(FEATURES)) if i != SULFUR_IDX]
AUX_NAMES = [FEATURES[i] for i in AUX]
ROLES = {
    "feed_sulfur_analyzer": "EXO",
    "feed_flow": "EXO",
    "p_h2": "EXO",
    "recycle_h2s": "INT",
    "t_in": "MV",
    "fuel_valve": "MV",
    "t_bed": "PV",
    "dt_ad": "PV",
}
ACCEPT = {
    "feed_sulfur": ["feed_sulfur_analyzer"],
    "h2_pressure": ["p_h2"],
    "amine": ["recycle_h2s"],
    "furnace": ["fuel_valve", "t_in"],
    "deactivation": ["dt_ad", "t_bed"],
}


def load_bundle(path):
    z = np.load(path, allow_pickle=True)
    d_in = len(z["features"])
    ssm = MambaSSM(d_in=d_in, d_model=int(z["d_model"]),
                   d_state=int(z["d_state"]), seed=int(z["seed"]))
    return {
        "ssm": ssm,
        "sx": Scaler(z["Xm"], z["Xs"]),
        "sy": Scaler(z["Ym"], z["Ys"]),
        "W": z["W"], "Fm": z["Fm"], "Fs": z["Fs"],
        "L": int(z["L"]), "H": int(z["H"]),
    }


def shapley_weights(n):
    return np.array([math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                     for s in range(n)])


def exact_shap(fvals, n):
    """fvals indexed by bitmask (2^n)."""
    w = shapley_weights(n)
    phi = np.zeros(n)
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            s = bin(mask).count("1")
            phi[i] += w[s] * (fvals[mask | (1 << i)] - fvals[mask])
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--artifact", default="artifacts/forecast.npz")
    ap.add_argument("--monitor", default="artifacts/monitor_events.csv")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    b = load_bundle(args.artifact)
    df = load_df(args.data)
    me = pd.read_csv(args.monitor)
    me = me[me.method == "Q1p_quality_predict_persist"].copy()
    me["alarm_time"] = pd.to_datetime(me["alarm_time"])

    ev = pd.read_csv(os.path.join(args.data, "events.csv"))
    ev = ev[ev.quality_limit_reached == 1].copy()
    ev["start_time"] = pd.to_datetime(ev["start_time"])
    ev["quality_alarm_time"] = pd.to_datetime(ev["quality_alarm_time"])
    ev["quality_limit_time_true"] = pd.to_datetime(ev["quality_limit_time_true"])
    ev = ev.merge(me[["event_id", "alarm_time"]], on="event_id", how="left")

    n = len(AUX)
    masks = np.array([[(m >> j) & 1 for j in range(n)] for m in range(1 << n)], np.float32)
    rows, details = [], {}

    for _, e in ev.iterrows():
        if e.run_id not in set(df.run_id):
            continue
        g = df[df.run_id == e.run_id].reset_index(drop=True)
        X, Y, meta = windows_for_run(g, b["L"], b["H"], 1)
        if not len(X):
            continue
        times = meta["time"].to_numpy()
        # diagnose at the 1-5 min quality-predictive alarm; fall back to the
        # DCS quality alarm / limit if the predictive alarm was missed.
        if pd.notna(e.alarm_time):
            target_t = e.alarm_time
        elif pd.notna(e.quality_alarm_time):
            target_t = e.quality_alarm_time
        elif pd.notna(e.quality_limit_time_true):
            target_t = e.quality_limit_time_true - pd.Timedelta(minutes=5)
        else:
            target_t = e.start_time + pd.Timedelta(minutes=30)
        idx = int(np.argmin(np.abs(times - np.datetime64(target_t))))
        Xw = X[idx:idx + 1]
        Xs = b["sx"].transform(Xw)
        Xm = np.repeat(Xs, len(masks), axis=0)
        for j, f in enumerate(AUX):
            Xm[:, :, f] = Xm[:, :, f] * masks[:, j][:, None]
        F = b["ssm"].features(Xm)
        P = predict_readout(F, b["W"], b["Fm"], b["Fs"]).reshape(len(masks), b["H"], len(FEATURES))
        last_raw = b["sx"].inverse(Xm[:, -1, :])
        pred = b["sy"].inverse(P) + last_raw[:, None, :]
        h = b["H"] - 1
        fvals = pred[:, h, SULFUR_IDX]
        phi = exact_shap(fvals, n)
        order = np.argsort(-np.abs(phi))
        top = [(AUX_NAMES[i], float(phi[i])) for i in order[:5]]
        f_all = float(fvals[-1])
        delta = np.array([f_all - fvals[(1 << n) - 1 - (1 << i)] for i in range(n)])
        med = np.median(np.abs(delta)) + 1e-9
        labels = []
        max_exo = max([abs(phi[i]) for i in range(n) if ROLES[AUX_NAMES[i]] == "EXO"] or [0])
        for i in range(n):
            role = ROLES[AUX_NAMES[i]]
            ratio = abs(delta[i]) / (abs(phi[i]) + 1e-6)
            if role in ("MV", "PV") and ratio < 0.5:
                labels.append("COMP")
            elif role == "EXO" and abs(phi[i]) >= med:
                labels.append("ROOT")
            elif role == "PV" and max_exo > abs(phi[i]):
                labels.append("INT")
            else:
                labels.append("NEUTRAL")
        top1 = top[0][0]
        comps = str(e.fault_type).split("|")
        accepted = set()
        for c in comps:
            accepted.update(ACCEPT.get(c, []))
        c1 = int(top1 in accepted) if accepted else None
        c3 = int(any(t[0] in accepted for t in top[:3])) if accepted else None
        rows.append({
            "event_id": e.event_id, "run_id": e.run_id, "fault_type": e.fault_type,
            "alarm_time": target_t, "top1": top1, "top1_shap": round(top[0][1], 2),
            "top3": "|".join(t[0] for t in top[:3]),
            "top1_correct": c1, "top3_correct": c3,
            "sca": "|".join("%s:%s" % (AUX_NAMES[i], labels[i]) for i in order[:5]),
        })
        details[e.event_id] = {
            "fault_type": e.fault_type,
            "alarm_time": str(target_t),
            "shap": [{"feature": AUX_NAMES[i], "shap": round(float(phi[i]), 2),
                      "delta": round(float(delta[i]), 2),
                      "role": ROLES[AUX_NAMES[i]], "label": labels[i]} for i in order],
        }

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, "rootcause_events.csv"), index=False)
    with open(os.path.join(args.out, "rootcause_details.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    valid = out[out.top1_correct.notna()]
    metrics = {
        "events": int(len(out)),
        "evaluated": int(len(valid)),
        "top1_accuracy": round(float(valid.top1_correct.mean()), 3),
        "top3_accuracy": round(float(valid.top3_correct.mean()), 3),
    }
    with open(os.path.join(args.out, "rootcause_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(out[["event_id", "fault_type", "top1", "top3", "top1_correct", "top3_correct"]]
          .to_string(index=False))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
