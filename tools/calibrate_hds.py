# -*- coding: utf-8 -*-
"""Steady-state check for the reduced-order HDS data generator.

Rate law (LHHW form with H2S inhibition):
    r = a * k_ref * exp(Ea/R * (1/T_ref - 1/T)) * (p_H2/p_ref)^0.45
        * C_S / (1 + K_H2S * p_H2S)^2

Reactor (CSTR lump):
    dC_S/dt   = (C_S,in - C_S)/tau - r
    dp_H2S/dt = beta * r - p_H2S/tau_g
"""
import numpy as np

EA_R = 80000.0 / 8.314
N_H2 = 0.45
KH = 2.0
TREF = 623.15

SETS = {
    "v1.0_current": dict(Cin=1000.0, tau=60.0, kref=4.0, pref=40.0,
                         beta=0.002, taug=4.0, limit=10.0),
    "v1.1_vgo":     dict(Cin=20000.0, tau=120.0, kref=0.0926, pref=80.0,
                         beta=2.8e-4, taug=5.0, limit=4500.0),
}


def steady(Cin, T, P, act, tau, kref, pref, beta, taug):
    S = Cin * 0.1
    h2s = 0.0
    for _ in range(500):
        rate = (Cin - S) / tau
        h2s = beta * taug * rate
        k = kref * act * np.exp(EA_R * (1.0 / TREF - 1.0 / T)) * (P / pref) ** N_H2
        S_new = Cin / (1.0 + tau * k / (1.0 + KH * h2s) ** 2)
        S = 0.5 * S + 0.5 * S_new
    return S, h2s


def report(name):
    p = SETS[name]
    base, base_h2s = steady(p["Cin"], TREF, p["pref"], 1.0,
                            p["tau"], p["kref"], p["pref"], p["beta"], p["taug"])
    print("=" * 78)
    print(f"{name}: normal S={base:,.1f} ppm, p_H2S={base_h2s:.3f} bar, "
          f"limit={p['limit']:,.0f} ppm, conversion={100*(1-base/p['Cin']):.2f}%")
    cases = [
        ("feed +25%", dict(Cin=p["Cin"] * 1.25)),
        ("feed +35%", dict(Cin=p["Cin"] * 1.35)),
        ("feed +50%", dict(Cin=p["Cin"] * 1.50)),
        ("H2 -20%", dict(P=p["pref"] * 0.80)),
        ("H2 -30%", dict(P=p["pref"] * 0.70)),
        ("T -12K", dict(T=TREF - 12)),
        ("T -18K", dict(T=TREF - 18)),
        ("activity 0.65", dict(act=0.65)),
        ("activity 0.35", dict(act=0.35)),
        ("amine tau_g x5", dict(taug=p["taug"] * 5)),
        ("amine tau_g x10", dict(taug=p["taug"] * 10)),
    ]
    print(f"{'case':<18}{'S_ss(ppm)':>12}{'p_H2S(bar)':>12}{'over limit':>12}")
    for label, ov in cases:
        args = dict(Cin=p["Cin"], T=TREF, P=p["pref"], act=1.0, tau=p["tau"],
                    kref=p["kref"], pref=p["pref"], beta=p["beta"], taug=p["taug"])
        args.update(ov)
        s, h = steady(**args)
        print(f"{label:<18}{s:>12,.1f}{h:>12.3f}{'YES' if s > p['limit'] else 'no':>12}")


if __name__ == "__main__":
    report("v1.0_current")
    report("v1.1_vgo")
