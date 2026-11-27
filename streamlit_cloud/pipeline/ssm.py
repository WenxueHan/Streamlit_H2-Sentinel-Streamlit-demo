# -*- coding: utf-8 -*-
"""NumPy selective state-space (Mamba-style) feature extractor + ridge readout.

This is the fallback used when PyTorch is not installed. The recurrence is a
diagonal selective SSM:

    delta_t = softplus(x_t W_delta + b)
    A_bar_t = exp(delta_t * A),  A < 0
    h_t     = A_bar_t * h_{t-1} + delta_t * (x_t W_B)
    y_t     = <h_t, x_t W_C> + x_t D
    y_t     = y_t * sigmoid(x_t W_g)

The SSM weights are fixed (seeded); a ridge regression readout is fitted on
pooled state features for the 5-step multi-output forecast. The full trainable
Mamba is provided in ``mamba_torch.py`` for environments with PyTorch.
"""
import numpy as np


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


class MambaSSM:
    def __init__(self, d_in: int, d_model: int = 12, d_state: int = 6, seed: int = 0):
        self.d_in = int(d_in)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_in)
        self.W_delta = rng.normal(0, s, (d_in, d_model)).astype(np.float32)
        self.b_delta = rng.normal(0, 0.1, d_model).astype(np.float32)
        self.A_log = rng.uniform(-1.5, 1.0, d_model).astype(np.float32)
        self.W_B = rng.normal(0, s, (d_in, d_state)).astype(np.float32)
        self.W_C = rng.normal(0, s, (d_in, d_state)).astype(np.float32)
        self.D = rng.normal(0, 0.1, (d_in, d_model)).astype(np.float32)
        self.W_g = rng.normal(0, s, (d_in, d_model)).astype(np.float32)

    def features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n, L, d = X.shape
        h = np.zeros((n, self.d_model, self.d_state), np.float32)
        h_sum = np.zeros_like(h)
        y_sum = np.zeros((n, self.d_model), np.float32)
        A = -np.exp(self.A_log)
        for t in range(L):
            x = X[:, t, :]
            pre = x @ self.W_delta + self.b_delta
            delta = _softplus(pre)
            a_bar = np.exp(delta * A)
            b = x @ self.W_B
            c = x @ self.W_C
            h = a_bar[:, :, None] * h + delta[:, :, None] * b[:, None, :]
            y = (h * c[:, None, :]).sum(-1) + x @ self.D
            y = y * (1.0 / (1.0 + np.exp(-(x @ self.W_g))))
            h_sum += h
            y_sum += y
        h_T = h.reshape(n, -1)
        h_mean = (h_sum / L).reshape(n, -1)
        y_mean = y_sum / L
        last5 = X[:, -5:, :].reshape(n, -1)
        last = X[:, -1, :]
        return np.concatenate([h_T, h_mean, y_mean, last5, last], axis=1)


def fit_readout(F: np.ndarray, Y: np.ndarray, lam: float = 1.0):
    """Ridge regression from SSM features to flattened multi-step targets."""
    Fm = F.mean(0)
    Fs = F.std(0) + 1e-6
    Fc = (F - Fm) / Fs
    Fc = np.concatenate([Fc, np.ones((len(Fc), 1), np.float32)], axis=1)
    A = Fc.T @ Fc + lam * np.eye(Fc.shape[1], dtype=np.float32)
    A[-1, -1] -= lam
    B = Fc.T @ Y
    W = np.linalg.solve(A, B)
    return W, Fm, Fs


def predict_readout(F: np.ndarray, W: np.ndarray, Fm: np.ndarray, Fs: np.ndarray):
    Fc = (F - Fm) / Fs
    Fc = np.concatenate([Fc, np.ones((len(Fc), 1), np.float32)], axis=1)
    return Fc @ W
