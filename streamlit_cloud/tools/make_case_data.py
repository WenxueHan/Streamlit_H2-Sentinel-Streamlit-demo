# -*- coding: utf-8 -*-
"""Build the per-minute case table for R05-E01 (feed sulfur step)."""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from pipeline.common import load_df, windows_for_run, SULFUR_IDX
from pipeline.monitor import load_model

predict, L, H = load_model(os.path.join(BASE, "artifacts", "forecast.npz"))
df = load_df(os.path.join(BASE, "data"))
g = df[df.run_id == "R05"].reset_index(drop=True)

start = pd.Timestamp("2025-10-30 00:00:00")
limit = pd.Timestamp("2025-10-30 00:53:00")
t0 = start - pd.Timedelta(minutes=90)
t1 = limit + pd.Timedelta(minutes=30)
g2 = g[(g["time"] >= t0) & (g["time"] <= t1)].reset_index(drop=True)
X, Y, meta = windows_for_run(g2, L, H, 1)
P = predict(X)

base = g2.set_index("time")
out = pd.DataFrame({
    "time": pd.to_datetime(meta["time"]),
    "sulfur_true": base.loc[meta["time"], "sulfur_true"].to_numpy(),
    "sulfur_analyzer": base.loc[meta["time"], "sulfur_analyzer"].to_numpy(),
    "pred_1": P[:, 0, SULFUR_IDX],
    "pred_5": P[:, H - 1, SULFUR_IDX],
})

al = pd.read_csv(os.path.join(BASE, "artifacts", "monitor_alarms.csv"),
                 usecols=["run_id", "time", "method", "T2", "SPE"],
                 parse_dates=["time"])
r00 = al[(al.run_id == "R00") & (al.method == "B3_forecast")]
thr_t2 = float(r00["T2"].quantile(0.99))
thr_spe = float(r00["SPE"].quantile(0.99))
mon = al[(al.run_id == "R05") & (al.method == "B3_forecast")][["time", "T2", "SPE"]]
mon = mon.rename(columns={"T2": "T2_B3", "SPE": "SPE_B3"})
out = out.merge(mon, on="time", how="left")
out["T2_norm"] = out["T2_B3"] / thr_t2
out["SPE_norm"] = out["SPE_B3"] / thr_spe
out.to_csv(os.path.join(BASE, "artifacts", "case_R05_E01.csv"), index=False, float_format="%.3f")
print(out.head(3).to_string(index=False))
print("thr_t2", round(thr_t2, 2), "thr_spe", round(thr_spe, 2))
print("rows", len(out), "range", out.time.min(), out.time.max())
print("max pred_5", round(out.pred_5.max(), 1), "max true", round(out.sulfur_true.max(), 1))
