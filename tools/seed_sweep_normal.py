# -*- coding: utf-8 -*-
"""Seed sweep of the normal scenario: is '0 alarms / 0 over-limit' robust?"""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from generator.config import Params, Scenario, ALARM_SPECS
from generator.model import simulate
from generator.measure import build_measurements
from generator.alarms import detect_alarms

p = Params()
rows = []
for seed in range(1, 11):
    sim = simulate(p, Scenario("normal", 14, []), seed)
    n = len(sim["time_min"])
    meas = build_measurements(sim, p, [], seed + 9999)
    df = pd.DataFrame({
        "mode": sim["mode"],
        "sensor_status": meas["sensor_status"],
        "event_id": sim["event_id"],
        "sulfur_analyzer": meas["sulfur_analyzer"],
        "feed_sulfur_analyzer": meas["feed_sulfur_analyzer"],
        "p_h2": meas["p_h2"],
        "recycle_h2s": meas["recycle_h2s"],
        "t_bed": meas["t_bed"],
        "dt_ad": meas["dt_ad"],
        "fuel_valve": meas["fuel_valve"],
        "t_in": meas["t_in"],
    })
    al = detect_alarms(df, ALARM_SPECS)
    rows.append({
        "seed": seed,
        "mean": round(float(np.mean(meas["sulfur_analyzer"])), 2),
        "std": round(float(np.std(meas["sulfur_analyzer"], ddof=1)), 2),
        "max_true": round(float(np.max(sim["sulfur_true"])), 2),
        "alarms": len(al),
        "over_limit_min": int(np.sum(sim["sulfur_true"] > p.sulfur_limit)),
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
print("\nmean range:", out["mean"].min(), "-", out["mean"].max())
print("std range:", out["std"].min(), "-", out["std"].max())
print("max_true range:", out["max_true"].min(), "-", out["max_true"].max())
print("alarms total:", int(out["alarms"].sum()),
      "| over_limit total:", int(out["over_limit_min"].sum()))
