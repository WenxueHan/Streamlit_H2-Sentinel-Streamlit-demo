# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")
ts = pd.read_csv(os.path.join(D, "timeseries.csv"), usecols=["run_id", "split", "mode", "sulfur_analyzer"])
lab = pd.read_csv(os.path.join(D, "labels.csv"), usecols=["run_id", "over_limit", "is_fault"])
ev = pd.read_csv(os.path.join(D, "events.csv"))
al = pd.read_csv(os.path.join(D, "alarms.csv"))
mp = ts.drop_duplicates("run_id").set_index("run_id")["split"]
ev["split"] = ev["run_id"].map(mp)

print("== rows by split ==")
print(ts.groupby("split").size().to_string())
print("total", len(ts))

print("\n== events by split/type ==")
print(ev.groupby(["split", "fault_type"]).size().to_string())
print("total", len(ev))

print("\n== event detection by type ==")
print(ev.groupby("fault_type")[["detected", "root_alarm_valid",
                                "quality_alarm_valid",
                                "quality_limit_reached"]].mean().round(3).to_string())

print("\n== lead root (min) ==")
print(ev["lead_root_min"].describe().round(1).to_string())
print("positive share", round((ev["lead_root_min"] > 0).mean(), 3))

print("\n== dcs lag (min) ==")
print(ev["dcs_lag_min"].describe().round(1).to_string())
print("abs median", round(ev["dcs_lag_min"].abs().median(), 1))

print("\n== alarms by tag ==")
print(al.groupby("tag").size().sort_values(ascending=False).to_string())
al["split"] = al["run_id"].map(mp)
print("\n== alarms by split ==")
print(al.groupby("split").size().to_string())
print("nuisance rate", round(al["is_nuisance"].mean(), 3))
print("invalid rate", round(al["is_invalid"].mean(), 3))

print("\n== over-limit share ==")
print(round(lab["over_limit"].mean(), 4), int(lab["over_limit"].sum()))
print("fault share", round(lab["is_fault"].mean(), 4))

print("\n== normal baseline ==")
n = ts[ts.run_id == "R00"]
print(n["sulfur_analyzer"].describe().round(2).to_string())

print("\n== alarm kinds ==")
print(al.groupby(["tag", "kind"]).size().to_string())
