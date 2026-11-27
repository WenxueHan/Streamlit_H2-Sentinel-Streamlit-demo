# -*- coding: utf-8 -*-
"""DET-style sweep: detection rate at a controlled false-alarm rate."""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(BASE, "artifacts")
D = os.path.join(BASE, "data")

al = pd.read_csv(os.path.join(A, "monitor_alarms.csv"), parse_dates=["time"])
ev = pd.read_csv(os.path.join(D, "events.csv"))
ev = ev[ev.quality_limit_reached == 1].copy()
ev["start_time"] = pd.to_datetime(ev["start_time"])
ev["quality_limit_time_true"] = pd.to_datetime(ev["quality_limit_time_true"])
ev = ev[ev.run_id.isin(["R04", "R05", "R07", "R08", "R09"])]

r00 = al[al.run_id == "R00"]
rows = []
for method in sorted(al.method.unique()):
    g0 = r00[r00.method == method]
    for q in [0.95, 0.975, 0.99, 0.995, 0.999]:
        t2q = float(g0["T2"].quantile(q))
        spq = float(g0["SPE"].quantile(q))
        t2q = max(t2q, 1e-9)
        spq = max(spq, 1e-9)
        # false alarms on normal
        g = r00[r00.method == method].sort_values("time")
        s = np.maximum(g["T2"].to_numpy() / t2q, g["SPE"].to_numpy() / spq)
        alr = s > 1.0
        alr = alr & np.r_[False, alr[:-1]]
        eps = int(np.sum(alr & ~np.r_[False, alr[:-1]]))
        far = eps / 14.0
        det, leads = 0, []
        for _, e in ev.iterrows():
            gg = al[(al.run_id == e.run_id) & (al.method == method)].sort_values("time")
            tt = pd.to_datetime(gg["time"]).to_numpy()
            ss = np.maximum(gg["T2"].to_numpy() / t2q, gg["SPE"].to_numpy() / spq)
            aa = ss > 1.0
            aa = aa & np.r_[False, aa[:-1]]
            m = (tt >= e.start_time.to_datetime64()) & (tt <= e.quality_limit_time_true.to_datetime64())
            idx = np.where(m & aa)[0]
            if len(idx):
                det += 1
                leads.append((e.quality_limit_time_true - pd.Timestamp(tt[idx[0]])).total_seconds() / 60.0)
        rows.append({"method": method, "q": q, "far_per_day": round(far, 2),
                     "detected": det, "events": len(ev),
                     "det_rate": round(det / len(ev), 3),
                     "lead_median": round(float(np.median(leads)), 1) if leads else np.nan})

out = pd.DataFrame(rows)
# pick, per method, the lowest-FAR point with detection >= 90%
best = out[out.det_rate >= 0.9].sort_values("far_per_day").groupby("method").head(1)
print("=== operating points at detection >= 90% ===")
print(best.to_string(index=False))
out.to_csv(os.path.join(A, "monitor_det.csv"), index=False)
