# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")
with open(os.path.join(BASE, "artifacts", "rootcause_details.json"), encoding="utf-8") as f:
    det = json.load(f)

df = pd.read_csv(os.path.join(D, "timeseries.csv"), parse_dates=["time"])
r00 = df[df.run_id == "R00"]
feats = ["feed_sulfur_analyzer", "feed_flow", "t_in", "t_bed", "dt_ad",
         "p_h2", "recycle_h2s", "fuel_valve", "sulfur_analyzer"]
mu, sd = r00[feats].mean(), r00[feats].std() + 1e-9

for eid in ["R01-E01", "R01-E04", "R06-E01", "R05-E01"]:
    d = det[eid]
    t = pd.Timestamp(d["alarm_time"])
    g = df[df.run_id == eid.split("-")[0]].sort_values("time")
    sub = g[g.time <= t]
    row = sub.iloc[-1]
    base = sub[sub.time <= t - pd.Timedelta(minutes=30)].iloc[-1]
    base6 = sub[sub.time <= t - pd.Timedelta(minutes=360)].iloc[-1]
    z = (row[feats] - mu) / sd
    dz30 = (row[feats] - base[feats]) / sd
    dz6 = (row[feats] - base6[feats]) / sd
    print("=" * 80)
    print(eid, d["fault_type"], "alarm", t)
    print("z     :", {k: round(v, 2) for k, v in z.items()})
    print("dz30  :", {k: round(v, 2) for k, v in dz30.items()})
    print("dz6h  :", {k: round(v, 2) for k, v in dz6.items()})
    print("shap  :", [(s["feature"], s["shap"], s["label"]) for s in d["shap"][:5]])
