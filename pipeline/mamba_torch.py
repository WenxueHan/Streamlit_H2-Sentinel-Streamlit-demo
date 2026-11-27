# -*- coding: utf-8 -*-
"""Reference PyTorch Mamba forecaster (requires torch).

This is a readable S6 implementation, not the fused CUDA kernel. It is provided
for GPU environments; the NumPy fallback in ``pipeline/ssm.py`` is used when
torch is not installed.

    python -m pipeline.mamba_torch --data data --epochs 20 --device cuda
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import FEATURES, SULFUR_IDX, Scaler, build_windows, load_df


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt, Bm, Cm = torch.split(self.x_proj(x_conv),
                                 [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :, None] * A)
            dB = dt[:, t, :, None] * Bm[:, t, None, :]
            h = dA * h + dB * x_conv[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1) + self.D * x_conv[:, t, :])
        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class Residual(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = MambaBlock(d_model)

    def forward(self, x):
        return x + self.block(self.norm(x))


class MambaForecaster(nn.Module):
    def __init__(self, d_in, d_out, horizon=5, d_model=64, n_layers=2):
        super().__init__()
        self.horizon = horizon
        self.d_out = d_out
        self.inp = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([Residual(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon * d_out)

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x).mean(1)
        return self.head(x).view(-1, self.horizon, self.d_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_df(args.data)
    Xtr, Ytr, _ = build_windows(df, "train", args.window, args.horizon, args.stride)
    Xva, Yva, _ = build_windows(df, "val", args.window, args.horizon, args.stride)
    sx, sy = Scaler.fit(Xtr), Scaler.fit(Ytr - Xtr[:, -1:, :])
    Xtr_s = torch.tensor(sx.transform(Xtr), dtype=torch.float32)
    Xva_s = torch.tensor(sx.transform(Xva), dtype=torch.float32)
    Ytr_d = torch.tensor(sy.transform(Ytr - Xtr[:, -1:, :]), dtype=torch.float32)
    Yva_d = torch.tensor(sy.transform(Yva - Xva[:, -1:, :]), dtype=torch.float32)
    last_tr = torch.tensor(Xtr[:, -1, :], dtype=torch.float32)
    last_va = torch.tensor(Xva[:, -1, :], dtype=torch.float32)
    Yva_raw = torch.tensor(Yva, dtype=torch.float32)

    model = MambaForecaster(len(FEATURES), len(FEATURES), args.horizon).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(Xtr_s, Ytr_d), batch_size=args.batch, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        print("epoch %d loss %.5f" % (epoch, total / len(loader.dataset)))

    model.eval()
    with torch.no_grad():
        pred = model(Xva_s.to(args.device)).cpu().numpy()
        pred = sy.inverse(pred) + last_va.numpy()[:, None, :]
    rmse = np.sqrt(np.mean((pred[:, :, SULFUR_IDX] - Yva_raw.numpy()[:, :, SULFUR_IDX]) ** 2, axis=0))
    print("val sulfur RMSE by horizon:", np.round(rmse, 2))
    torch.save({"state_dict": model.state_dict(), "Xm": sx.mean, "Xs": sx.std,
                "Ym": sy.mean, "Ys": sy.std, "L": args.window, "H": args.horizon,
                "d_model": 64, "n_layers": 2}, os.path.join(args.out, "mamba_torch.pt"))


if __name__ == "__main__":
    main()
