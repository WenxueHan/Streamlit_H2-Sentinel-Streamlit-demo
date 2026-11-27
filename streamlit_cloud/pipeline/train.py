# -*- coding: utf-8 -*-
"""Train and evaluate the NumPy Mamba-style multi-step forecaster."""
import argparse
import json
import os
import numpy as np
import pandas as pd

from .common import (FEATURES, SULFUR_IDX, Scaler, build_windows, load_df)
from .ssm import MambaSSM, fit_readout, predict_readout


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def r2(a, b):
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def evaluate(Y, P, name):
    H = Y.shape[1]
    rows = []
    for k in range(H):
        y = Y[:, k, SULFUR_IDX]
        p = P[:, k, SULFUR_IDX]
        rows.append({"split": name, "horizon": k + 1,
                     "rmse": round(rmse(y, p), 3),
                     "mae": round(mae(y, p), 3),
                     "r2": round(r2(y, p), 4)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=12)
    ap.add_argument("--d-state", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_df(args.data)
    L, H = args.window, args.horizon
    print("building windows ...")
    Xtr, Ytr, _ = build_windows(df, "train", L, H, args.stride)
    Xva, Yva, _ = build_windows(df, "val", L, H, args.stride)
    Xte, Yte, _ = build_windows(df, "test", L, H, max(2, args.stride))
    print("windows:", Xtr.shape, Xva.shape, Xte.shape)

    sx = Scaler.fit(Xtr)
    # learn the deviation from the last observed value (persistence residual)
    Ytr_d = Ytr - Xtr[:, -1:, :]
    sy = Scaler.fit(Ytr_d)
    Xtr_s, Xva_s, Xte_s = sx.transform(Xtr), sx.transform(Xva), sx.transform(Xte)
    Ytr_s = sy.transform(Ytr_d)

    ssm = MambaSSM(d_in=len(FEATURES), d_model=args.d_model,
                   d_state=args.d_state, seed=args.seed)
    print("extracting SSM features ...")
    Ftr = ssm.features(Xtr_s)
    Fva = ssm.features(Xva_s)
    Fte = ssm.features(Xte_s)
    print("feature dim:", Ftr.shape[1])

    Ytr_flat = Ytr_s.reshape(len(Ytr_s), -1)
    best = None
    for lam in [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0]:
        W, Fm, Fs = fit_readout(Ftr, Ytr_flat, lam)
        Pva = predict_readout(Fva, W, Fm, Fs).reshape(len(Fva), H, -1)
        Pva = sy.inverse(Pva) + Xva[:, -1:, :]
        score = rmse(Yva[:, :, SULFUR_IDX], Pva[:, :, SULFUR_IDX])
        print("lam=%-6s val sulfur RMSE=%.4f" % (lam, score))
        if best is None or score < best[0]:
            best = (score, lam, W, Fm, Fs)

    _, lam, W, Fm, Fs = best
    print("best lambda:", lam)
    path = os.path.join(args.out, "forecast.npz")
    np.savez(path, W=W, Fm=Fm, Fs=Fs,
             Xm=sx.mean, Xs=sx.std, Ym=sy.mean, Ys=sy.std,
             L=L, H=H, d_model=args.d_model, d_state=args.d_state,
             seed=args.seed, lam=lam, features=np.array(FEATURES), delta=1)

    Pva = sy.inverse(predict_readout(Fva, W, Fm, Fs).reshape(len(Fva), H, -1)) + Xva[:, -1:, :]
    Pte = sy.inverse(predict_readout(Fte, W, Fm, Fs).reshape(len(Fte), H, -1)) + Xte[:, -1:, :]

    rows = evaluate(Yva, Pva, "val") + evaluate(Yte, Pte, "test")
    # persistence baseline
    Pva_p = np.repeat(Xva[:, -1:, :], H, axis=1)
    Pte_p = np.repeat(Xte[:, -1:, :], H, axis=1)
    base = evaluate(Yva, Pva_p, "val_persistence") + evaluate(Yte, Pte_p, "test_persistence")

    out = pd.DataFrame(rows + base)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(args.out, "forecast_metrics.csv"), index=False)
    summary = {
        "window": L, "horizon": H, "stride": args.stride,
        "lambda": lam, "feature_dim": int(Ftr.shape[1]),
        "train_windows": int(len(Xtr)), "val_windows": int(len(Xva)),
        "test_windows": int(len(Xte)),
        "sulfur_test_rmse_h1": float(out[(out.split == "test") & (out.horizon == 1)].rmse.iloc[0]),
        "sulfur_test_rmse_h5": float(out[(out.split == "test") & (out.horizon == 5)].rmse.iloc[0]),
        "sulfur_test_r2_h5": float(out[(out.split == "test") & (out.horizon == 5)].r2.iloc[0]),
    }
    with open(os.path.join(args.out, "forecast_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
