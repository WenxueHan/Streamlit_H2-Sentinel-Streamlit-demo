# -*- coding: utf-8 -*-
"""Shared data utilities: loading, windowing, scaling, splits."""
import os
import numpy as np
import pandas as pd

FEATURES = [
    "feed_sulfur_analyzer", "feed_flow", "t_in", "t_bed", "dt_ad",
    "p_h2", "recycle_h2s", "fuel_valve", "sulfur_analyzer",
]
TARGETS = FEATURES
SULFUR_IDX = FEATURES.index("sulfur_analyzer")

RUN_SPLIT = {
    "train": ["R00", "R01", "R02", "R03", "R06"],
    "val": ["R04", "R08"],
    "test": ["R05", "R07", "R09"],
}


def load_df(data_dir: str) -> pd.DataFrame:
    ts = pd.read_csv(os.path.join(data_dir, "timeseries.csv"))
    lat = pd.read_csv(os.path.join(data_dir, "latent.csv"),
                      usecols=["run_id", "time", "sulfur_true"])
    lab = pd.read_csv(os.path.join(data_dir, "labels.csv"),
                      usecols=["run_id", "time", "over_limit"])
    df = ts.merge(lat, on=["run_id", "time"], how="left")
    df = df.merge(lab, on=["run_id", "time"], how="left")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values(["run_id", "time"]).reset_index(drop=True)


def runs_of(df: pd.DataFrame, split: str):
    return [r for r in RUN_SPLIT[split] if r in set(df["run_id"])]


def windows_for_run(g: pd.DataFrame, L: int, H: int, stride: int):
    v = g[FEATURES].to_numpy(np.float32)
    # inputs use online measurements; the quality target is the true sulfur
    # (the "lab" value), not the 3-min-delayed analyzer.
    ymat = v.copy()
    ymat[:, SULFUR_IDX] = g["sulfur_true"].to_numpy(np.float32)
    n = len(v)
    idx = np.arange(L - 1, n - H, stride)
    if len(idx) == 0:
        return (np.empty((0, L, len(FEATURES)), np.float32),
                np.empty((0, H, len(FEATURES)), np.float32),
                pd.DataFrame(columns=["run_id", "time", "sulfur_true_h"]))
    X = np.stack([v[i - L + 1:i + 1] for i in idx]).astype(np.float32)
    Y = np.stack([ymat[i + 1:i + 1 + H] for i in idx]).astype(np.float32)
    true_h = np.stack([g["sulfur_true"].to_numpy(np.float32)[i + 1:i + 1 + H]
                       for i in idx])
    meta = pd.DataFrame({
        "run_id": g["run_id"].iloc[0],
        "time": g["time"].iloc[idx].to_numpy(),
        "sulfur_true_h": list(true_h),
    })
    return X, Y, meta


def build_windows(df: pd.DataFrame, split: str, L: int, H: int, stride: int):
    xs, ys, ms = [], [], []
    for rid in runs_of(df, split):
        g = df[df["run_id"] == rid]
        X, Y, meta = windows_for_run(g, L, H, stride)
        if len(X):
            xs.append(X)
            ys.append(Y)
            ms.append(meta)
    if not xs:
        return (np.empty((0, L, len(FEATURES)), np.float32),
                np.empty((0, H, len(FEATURES)), np.float32),
                pd.DataFrame())
    return np.concatenate(xs), np.concatenate(ys), pd.concat(ms, ignore_index=True)


class Scaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, X):
        flat = X.reshape(-1, X.shape[-1])
        return cls(flat.mean(0), flat.std(0) + 1e-6)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse(self, X):
        return X * self.std + self.mean


def save_json(obj, path):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
