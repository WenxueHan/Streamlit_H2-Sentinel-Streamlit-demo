# -*- coding: utf-8 -*-
"""Verify the R00 normal-run statistics reported in the docs."""
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")

ts = pd.read_csv(os.path.join(D, "timeseries.csv"), parse_dates=["time"])
lab = pd.read_csv(os.path.join(D, "labels.csv"), parse_dates=["time"])
lat = pd.read_csv(os.path.join(D, "latent.csv"), parse_dates=["time"])
al = pd.read_csv(os.path.join(D, "alarms.csv"))
ev = pd.read_csv(os.path.join(D, "events.csv"))

r00 = ts[ts.run_id == "R00"]
l00 = lab[lab.run_id == "R00"]
t00 = lat[lat.run_id == "R00"]

print("run_id=R00 rows:", len(r00))
print("time range:", r00.time.min(), "->", r00.time.max())
print("scenario:", r00.scenario.unique().tolist(), "split:", r00.split.unique().tolist())
print("mode counts:", r00["mode"].value_counts().to_dict())

print("\n[measured analyzer sulfur_analyzer]")
print(r00.sulfur_analyzer.describe().round(2).to_string())
print("p50 / p95 / p99:",
      round(r00.sulfur_analyzer.quantile(0.5), 2),
      round(r00.sulfur_analyzer.quantile(0.95), 2),
      round(r00.sulfur_analyzer.quantile(0.99), 2))

print("\n[true sulfur_true]")
print(t00.sulfur_true.describe().round(2).to_string())
print("max true:", round(t00.sulfur_true.max(), 2),
      "| limit:", 4500)

print("\nalarms for R00:", len(al[al.run_id == "R00"]))
print("over_limit minutes:", int(l00.over_limit.sum()))
print("is_fault minutes:", int(l00.is_fault.sum()))
print("events for R00:", len(ev[ev.run_id == "R00"]))
print("sensor_status:", r00.sensor_status.value_counts().to_dict())
