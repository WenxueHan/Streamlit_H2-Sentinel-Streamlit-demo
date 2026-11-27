# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1] if len(sys.argv) > 1 else "data_quick"
if not os.path.isabs(D):
    D = os.path.join(BASE, D)

ts = pd.read_csv(os.path.join(D, "timeseries.csv"), parse_dates=["time"])
lab = pd.read_csv(os.path.join(D, "labels.csv"), parse_dates=["time"])
lat = pd.read_csv(os.path.join(D, "latent.csv"), parse_dates=["time"])
ev = pd.read_csv(os.path.join(D, "events.csv"))
al = pd.read_csv(os.path.join(D, "alarms.csv"))

print("=== shapes ===")
print("timeseries", ts.shape, "labels", lab.shape, "latent", lat.shape,
      "events", ev.shape, "alarms", al.shape)

print("\n=== normal run ===")
n = ts[ts.run_id == "R00"]
ln = lab[lab.run_id == "R00"]
print(n[["sulfur_analyzer", "t_bed", "t_in", "dt_ad", "p_h2",
         "recycle_h2s", "fuel_valve"]].describe().T[["mean", "std", "min", "max"]])
print("normal over_limit minutes:", int(ln.over_limit.sum()),
      "alarms:", int((al.run_id == "R00").sum()))

print("\n=== fast run R01, first 30 days window ===")
f = ts[ts.run_id == "R01"]
lf = lab[lab.run_id == "R01"]
print(f[["sulfur_analyzer", "t_bed", "fuel_valve", "recycle_h2s", "p_h2"]]
      .describe().T[["mean", "std", "min", "max"]])
print("fault minutes:", int(lf.is_fault.sum()), "over_limit:", int(lf.over_limit.sum()))

print("\n=== events ===")
cols = ["event_id", "fault_type", "severity", "is_single_fault", "detected",
        "root_alarm_valid", "quality_alarm_valid", "quality_limit_reached",
        "lead_root_min", "dcs_lag_min"]
print(ev[cols].to_string(index=False))

print("\n=== alarms by tag/kind ===")
print(al.groupby(["tag", "kind"]).size().sort_values(ascending=False).to_string())
print("\nnuisance reasons:")
print(al[al.is_nuisance == 1].groupby("nuisance_reason").size().to_string())

print("\n=== data quality ===")
num = ts.select_dtypes(include=[np.number])
print("NaN count:", int(num.isna().sum().sum()))
print("negative sulfur:", int((ts.sulfur_analyzer < 0).sum()))
print("sulfur range:", ts.sulfur_analyzer.min(), ts.sulfur_analyzer.max())
print("sensor_status:", ts.sensor_status.value_counts().to_dict())
print("mode:", ts["mode"].value_counts().to_dict())
