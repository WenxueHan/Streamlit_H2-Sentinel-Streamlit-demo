# -*- coding: utf-8 -*-
"""Fault-specific monitor templates.

For each fault type, learn a z-score template from training events, then score
new samples by negative Mahalanobis-like distance. This is a lightweight
"different monitor per fault" layer; the agent can use it as extra evidence.

Usage:
    python -m pipeline.fault_monitors --data data --out artifacts
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .common import FEATURES, RUN_SPLIT, load_df

FAULT_FEATURES = {fault: FEATURES for fault in
                 ["feed_sulfur", "amine", "furnace", "h2_pressure", "deactivation", "instrument"]}
FAULT_WEIGHTS = {}

def _z(row, mu, sd):
    return (row[FEATURES].to_numpy(float) - mu) / sd


def _event_time(e):
    if pd.notna(e.get("quality_limit_time_true")):
        return pd.Timestamp(e["quality_limit_time_true"]) - pd.Timedelta(minutes=3)
    if pd.notna(e.get("end_time")):
        return pd.Timestamp(e["end_time"]) - pd.Timedelta(minutes=3)
    return pd.Timestamp(e["start_time"]) + pd.Timedelta(minutes=30)


def _softmax(x, T=2.0):
    x = np.asarray(x, float)
    e = np.exp((x - x.max()) / T)
    return e / e.sum()


def fit(df, events):
    r00 = df[df.run_id == "R00"]
    mu = r00[FEATURES].mean().to_numpy(float)
    sd = (r00[FEATURES].std() + 1e-9).to_numpy(float)
    templates = {}
    for _, e in events.iterrows():
        if e["run_id"] not in RUN_SPLIT["train"]:
            continue
        g = df[df.run_id == e["run_id"]].sort_values("time")
        t = _event_time(e)
        sub = g[g["time"] <= t]
        if sub.empty:
            continue
        z = _z(sub.iloc[-1], mu, sd)
        fault = str(e["fault_type"]).split("|")[0]
        feats = FAULT_FEATURES.get(fault, FEATURES)
        idx = [FEATURES.index(x) for x in feats]
        templates.setdefault(fault, []).append(z[idx])
    out = {}
    for fault, arr in templates.items():
        arr = np.asarray(arr)
        out[fault] = {
            "features": FAULT_FEATURES.get(fault, FEATURES),
            "weights": [FAULT_WEIGHTS.get(fault, {}).get(x, 1.0) for x in FAULT_FEATURES.get(fault, FEATURES)],
            "mean": arr.mean(0).round(3).tolist(),
            "scale": (arr.std(0) + 0.5).round(3).tolist(),
            "n": int(len(arr)),
        }
    # normal thresholds per template
    r00z = ((r00[FEATURES].to_numpy(float) - mu) / sd)
    for fault, t in out.items():
        idx = [FEATURES.index(x) for x in t["features"]]
        m = np.asarray(t["mean"])
        sc = np.asarray(t["scale"])
        w = np.asarray(t["weights"])
        d = ((((r00z[:, idx] - m) / sc) ** 2) * w).sum(1) / w.sum()
        t["thr"] = float(np.quantile(d, 0.99))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "templates": out}


def score(z, model):
    names, scores = [], []
    for fault, t in model["templates"].items():
        idx = [FEATURES.index(x) for x in t["features"]]
        m = np.asarray(t["mean"])
        sc = np.asarray(t["scale"])
        w = np.asarray(t["weights"])
        d = float((((z[idx] - m) / sc) ** 2 * w).sum() / w.sum())
        names.append(fault)
        scores.append(-d)
    probs = _softmax(scores)
    order = np.argsort(-probs)
    return {
        "top1": names[order[0]],
        "confidence": round(float(probs[order[0]]), 3),
        "top3": [{"fault": names[i], "score": round(float(probs[i]), 3)} for i in order[:3]],
        "scores": {names[i]: round(float(probs[i]), 3) for i in order},
    }


def evaluate(df, events, model):
    rows = []
    for _, e in events.iterrows():
        if e["run_id"] not in RUN_SPLIT["test"]:
            continue
        g = df[df.run_id == e["run_id"]].sort_values("time")
        t = _event_time(e)
        sub = g[g["time"] <= t]
        if sub.empty:
            continue
        z = _z(sub.iloc[-1], np.asarray(model["mu"]), np.asarray(model["sd"]))
        r = score(z, model)
        gold = str(e["fault_type"]).split("|")[0]
        rows.append({"event_id": e["event_id"], "gold": gold, "top1": r["top1"],
                     "confidence": r["confidence"], "correct": int(gold == r["top1"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    df = load_df(args.data)
    ev = pd.read_csv(os.path.join(args.data, "events.csv"))
    model = fit(df, ev)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fault_templates.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    rows = evaluate(df, ev, model)
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / max(n, 1)
    report = {"n": n, "top1_accuracy": round(acc, 3), "rows": rows}
    with open(os.path.join(args.out, "fault_monitor_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"n": n, "top1_accuracy": round(acc, 3),
                      "per_fault": pd.DataFrame(rows).groupby("gold")["correct"].agg(["count", "mean"]).round(2).to_dict() if n else {}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
