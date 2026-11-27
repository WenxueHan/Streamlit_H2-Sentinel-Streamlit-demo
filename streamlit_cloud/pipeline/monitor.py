# -*- coding: utf-8 -*-
"""Quality-augmented process monitoring: PCA T2 / SPE on 5-step forecasts.

Methods compared
----------------
B1  aux-only PCA on current auxiliary variables (8 dims)
B2  quality-augmented PCA on current aux + sulfur (9 dims)
B3  quality-augmented PCA on the 5-step forecast (per-horizon, max over k)
B4  B3 + 2-sample persistence
"""
import argparse
import json
import os
import numpy as np
import pandas as pd

from .common import (FEATURES, SULFUR_IDX, RUN_SPLIT, Scaler, load_df,
                     windows_for_run)
from .ssm import MambaSSM, predict_readout

AUX_IDX = [i for i in range(len(FEATURES)) if i != SULFUR_IDX]


def load_model(artifact_path):
    z = np.load(artifact_path, allow_pickle=True)
    d_in = len(z["features"])
    ssm = MambaSSM(d_in=d_in, d_model=int(z["d_model"]),
                   d_state=int(z["d_state"]), seed=int(z["seed"]))
    sx = Scaler(z["Xm"], z["Xs"])
    sy = Scaler(z["Ym"], z["Ys"])
    W, Fm, Fs = z["W"], z["Fm"], z["Fs"]
    L, H = int(z["L"]), int(z["H"])

    def predict(X):
        F = ssm.features(sx.transform(X))
        P = predict_readout(F, W, Fm, Fs).reshape(len(X), H, d_in)
        return sy.inverse(P) + X[:, -1:, :]

    return predict, L, H


def fit_pca(Z, var=0.95, q=0.99):
    mu = Z.mean(0)
    sd = Z.std(0) + 1e-9
    Zc = (Z - mu) / sd
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    ev = S ** 2 / (len(Z) - 1)
    cum = np.cumsum(ev) / ev.sum()
    k = int(np.searchsorted(cum, var) + 1)
    P = Vt[:k].T
    lam = ev[:k]
    T2 = ((Zc @ P) ** 2 / lam).sum(1)
    E = Zc - (Zc @ P) @ P.T
    SPE = (E ** 2).sum(1)
    return {"mu": mu, "sd": sd, "P": P, "lam": lam, "k": k,
            "thr_t2": float(np.quantile(T2, q)),
            "thr_spe": float(np.quantile(SPE, q))}


def stats(Z, m):
    Zc = (Z - m["mu"]) / m["sd"]
    T2 = ((Zc @ m["P"]) ** 2 / m["lam"]).sum(1)
    E = Zc - (Zc @ m["P"]) @ m["P"].T
    SPE = (E ** 2).sum(1)
    return T2, SPE


def persist(raw):
    out = raw.copy()
    out[1:] = raw[1:] & raw[:-1]
    out[0] = False
    return out


def episodes(alarm, times):
    alarm = np.asarray(alarm, dtype=bool)
    times = np.asarray(times)
    idx = np.where(alarm & ~np.r_[False, alarm[:-1]])[0]
    return times[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--artifact", default="artifacts/forecast.npz")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_df(args.data)
    predict, L, H = load_model(args.artifact)

    # ---- per-run forecasts and current vectors ----
    cache = {}
    for rid in sorted(df["run_id"].unique()):
        g = df[df["run_id"] == rid].reset_index(drop=True)
        X, Y, meta = windows_for_run(g, L, H, 1)
        P = predict(X)
        current = g[FEATURES].to_numpy(np.float32)[L - 1:L - 1 + len(P)]
        cache[rid] = {"g": g, "X": X, "P": P, "current": current,
                      "time": meta["time"].to_numpy()}

    # ---- fit PCA on normal reference R00 ----
    r00 = cache["R00"]
    m_b1 = fit_pca(r00["current"][:, AUX_IDX])
    m_b2 = fit_pca(r00["current"])
    m_b3 = [fit_pca(r00["P"][:, k, :]) for k in range(H)]
    T2_r00 = np.max(np.stack([stats(r00["P"][:, k, :], m_b3[k])[0]
                              for k in range(H)], 1), 1)
    SPE_r00 = np.max(np.stack([stats(r00["P"][:, k, :], m_b3[k])[1]
                               for k in range(H)], 1), 1)
    thr_b3_t2 = float(np.quantile(T2_r00, 0.99))
    thr_b3_spe = float(np.quantile(SPE_r00, 0.99))

    # quality-residual model: sulfur predicted from current auxiliaries (normal fit)
    aux_r = r00["current"][:, AUX_IDX]
    y_r = r00["current"][:, SULFUR_IDX]
    A_r = np.c_[aux_r, np.ones(len(aux_r))]
    beta = np.linalg.lstsq(A_r, y_r, rcond=None)[0]
    res_r = y_r - A_r @ beta
    sigma_res = float(res_r.std() + 1e-9)

    # ---- score every run ----
    alarms = []
    for rid, c in cache.items():
        T2_b1, SPE_b1 = stats(c["current"][:, AUX_IDX], m_b1)
        T2_b2, SPE_b2 = stats(c["current"], m_b2)
        T2s, SPEs = [], []
        for k in range(H):
            t2, spe = stats(c["P"][:, k, :], m_b3[k])
            T2s.append(t2)
            SPEs.append(spe)
        T2_b3 = np.max(np.stack(T2s, 1), 1)
        SPE_b3 = np.max(np.stack(SPEs, 1), 1)

        aux_c = c["current"][:, AUX_IDX]
        y_c = c["current"][:, SULFUR_IDX]
        e_res = y_c - (np.c_[aux_c, np.ones(len(aux_c))] @ beta)
        z_res = e_res / sigma_res
        raw_res = np.abs(z_res) > 2.576

        raw_b1 = None
        for name, T2, SPE, mt2, mspe in [
            ("B1_aux_only", T2_b1, SPE_b1, m_b1["thr_t2"], m_b1["thr_spe"]),
            ("B2_quality_aug", T2_b2, SPE_b2, m_b2["thr_t2"], m_b2["thr_spe"]),
            ("B3_forecast", T2_b3, SPE_b3, thr_b3_t2, thr_b3_spe),
        ]:
            raw = (T2 > mt2) | (SPE > mspe)
            if name == "B1_aux_only":
                raw_b1 = raw
            alarms.append(pd.DataFrame({
                "run_id": rid, "time": c["time"], "method": name,
                "T2": T2, "SPE": SPE, "alarm": raw.astype(int),
            }))
            pname = {"B1_aux_only": "B1p_aux_persist",
                     "B2_quality_aug": "B2p_quality_persist",
                     "B3_forecast": "B4_forecast_persist"}[name]
            alarms.append(pd.DataFrame({
                "run_id": rid, "time": c["time"], "method": pname,
                "T2": T2, "SPE": SPE, "alarm": persist(raw).astype(int),
            }))
        alarms.append(pd.DataFrame({
            "run_id": rid, "time": c["time"], "method": "B5_quality_residual",
            "T2": z_res ** 2, "SPE": np.zeros_like(z_res),
            "alarm": raw_res.astype(int),
        }))
        alarms.append(pd.DataFrame({
            "run_id": rid, "time": c["time"], "method": "B5p_quality_residual_persist",
            "T2": z_res ** 2, "SPE": np.zeros_like(z_res),
            "alarm": persist(raw_res).astype(int),
        }))
        alarms.append(pd.DataFrame({
            "run_id": rid, "time": c["time"], "method": "B6_combined_persist",
            "T2": T2_b1, "SPE": SPE_b1,
            "alarm": (persist(raw_b1) | persist(raw_res)).astype(int),
        }))

        # Q1: quality-limit predictive alarm at a 3-min horizon.
        # Using the 3-step (not the 5-step) forecast keeps the warning inside
        # the required 1-5 min window; the 5-step model is still used for the
        # full trajectory. A minimum 2-min rise filters out slow deactivation.
        pred_s = c["P"][:, :, SULFUR_IDX]
        pred_1 = pred_s[:, 0]
        pred_3 = pred_s[:, 2]
        rise2 = pred_3 - pred_1
        raw_q = (pred_3 >= 4490.0) & (rise2 >= 15.0)
        alarms.append(pd.DataFrame({
            "run_id": rid, "time": c["time"], "method": "Q1_quality_predict",
            "T2": T2_b3, "SPE": SPE_b3, "alarm": raw_q.astype(int),
        }))
        alarms.append(pd.DataFrame({
            "run_id": rid, "time": c["time"], "method": "Q1p_quality_predict_persist",
            "T2": T2_b3, "SPE": SPE_b3, "alarm": persist(raw_q).astype(int),
        }))
    alarms = pd.concat(alarms, ignore_index=True)

    # ---- normal false-alarm rate ----
    days = {}
    for rid in cache:
        days[rid] = len(cache[rid]["g"]) / 1440.0
    normal = alarms[alarms.run_id == "R00"]
    far_rows = []
    for name, g in normal.groupby("method"):
        eps = episodes(g["alarm"].to_numpy(bool), pd.to_datetime(g["time"]))
        far_rows.append({"method": name, "normal_run": "R00",
                         "episodes": len(eps), "days": round(days["R00"], 1),
                         "episodes_per_day": round(len(eps) / days["R00"], 3)})
    far = pd.DataFrame(far_rows)

    # ---- event detection / lead time (val + test) ----
    ev = pd.read_csv(os.path.join(args.data, "events.csv"))
    ev = ev[ev.quality_limit_reached == 1].copy()
    split_map = {}
    for s, runs in RUN_SPLIT.items():
        for r in runs:
            split_map[r] = s
    ev["split"] = ev["run_id"].map(split_map)
    ev = ev[ev.split.isin(["val", "test"])].copy()
    ev["start_time"] = pd.to_datetime(ev["start_time"])
    ev["quality_limit_time_true"] = pd.to_datetime(ev["quality_limit_time_true"])

    det_rows = []
    for _, e in ev.iterrows():
        if e.run_id not in cache:
            continue
        a = alarms[(alarms.run_id == e.run_id)]
        for name, g in a.groupby("method"):
            g = g.sort_values("time")
            t = pd.to_datetime(g["time"]).to_numpy()
            al = g["alarm"].to_numpy(bool)
            mask = (t >= e.start_time.to_datetime64()) & (t <= e.quality_limit_time_true.to_datetime64())
            idx = np.where(mask & al)[0]
            if len(idx):
                alarm_time = pd.Timestamp(t[idx[0]])
                lead = (e.quality_limit_time_true - alarm_time).total_seconds() / 60.0
                det_rows.append({"event_id": e.event_id, "fault_type": e.fault_type,
                                 "method": name, "detected": 1,
                                 "alarm_time": alarm_time, "lead_min": round(lead, 1)})
            else:
                det_rows.append({"event_id": e.event_id, "fault_type": e.fault_type,
                                 "method": name, "detected": 0,
                                 "alarm_time": pd.NaT, "lead_min": np.nan})
    det = pd.DataFrame(det_rows)

    summary = det.groupby("method").agg(
        events=("event_id", "nunique"),
        detected=("detected", "sum"),
        detection_rate=("detected", "mean"),
        lead_median=("lead_min", "median"),
        lead_mean=("lead_min", "mean"),
    ).reset_index()
    summary = summary.merge(far[["method", "episodes", "episodes_per_day"]], on="method", how="left")
    summary["detection_rate"] = summary["detection_rate"].round(3)
    summary["lead_median"] = summary["lead_median"].round(1)
    summary["lead_mean"] = summary["lead_mean"].round(1)

    alarms.to_csv(os.path.join(args.out, "monitor_alarms.csv"), index=False)
    det.to_csv(os.path.join(args.out, "monitor_events.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "monitor_metrics.csv"), index=False)
    print(summary.to_string(index=False))
    with open(os.path.join(args.out, "monitor_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
