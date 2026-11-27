# -*- coding: utf-8 -*-
"""Knowledge-grounded diagnosis agent.

Inputs
------
- artifacts/rootcause_details.json : SHAP values + SCA labels per event
- data/timeseries.csv              : current values at the alarm time
- knowledge/faults.json            : mechanism, expected directions, SOP

The agent scores each fault hypothesis by combining
  (1) direction consistency of the standardized auxiliary variables,
  (2) SHAP rank of the hypothesis' key variables,
  (3) SCA label (a compensation variable cannot be the root cause),
and emits a structured diagnosis with evidence, contradictions and actions.
"""
import argparse
import json
import math
import os
import numpy as np
import pandas as pd

from .common import FEATURES, load_df
from .fault_monitors import score as fm_score


def softmax(scores, T=2.0):
    m = max(scores.values())
    ex = {k: math.exp((v - m) / T) for k, v in scores.items()}
    s = sum(ex.values()) or 1.0
    return {k: v / s for k, v in ex.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--rootcause", default="artifacts/rootcause_details.json")
    ap.add_argument("--knowledge", default="knowledge/faults.json")
    ap.add_argument("--fault-monitors", default="artifacts/fault_templates.json")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.knowledge, encoding="utf-8") as f:
        kb = json.load(f)
    with open(args.rootcause, encoding="utf-8") as f:
        details = json.load(f)
    fm = None
    if os.path.exists(args.fault_monitors):
        with open(args.fault_monitors, encoding="utf-8") as f:
            fm = json.load(f)

    df = load_df(args.data)
    r00 = df[df.run_id == "R00"]
    mu = r00[FEATURES].mean()
    sd = r00[FEATURES].std() + 1e-9

    ev = pd.read_csv(os.path.join(args.data, "events.csv"))
    single_map = dict(zip(ev["event_id"], ev["is_single_fault"]))

    records, rows = [], []
    for eid, d in details.items():
        rid = eid.split("-")[0]
        g = df[df.run_id == rid].sort_values("time")
        t = pd.Timestamp(d["alarm_time"])
        sub = g[g["time"] <= t]
        if len(sub) == 0:
            continue
        row = sub.iloc[-1]
        z = {f: float((row[f] - mu[f]) / sd[f]) for f in FEATURES}
        sensor = str(row.get("sensor_status", "ok"))
        shap = d["shap"]
        top_feats = [s["feature"] for s in shap[:5]]
        fm_res = fm_score(np.array([z[f] for f in FEATURES]), fm) if fm else None

        scores = {}
        for name, k in kb.items():
            if name == "instrument":
                continue
            s = 0.0
            for feat, exp in k["expected"].items():
                zi = z.get(feat, 0.0)
                w = k["weights"].get(feat, 0.3)
                if exp == 0:
                    if abs(zi) > 2.0:
                        s -= 0.4 * w
                    continue
                match = 1 if zi * exp > 0 else 0
                strength = min(abs(zi), 3.0) / 3.0
                s += w * (match * (0.6 + 0.4 * strength) - (1 - match) * 0.9)
            for rank, feat in enumerate(top_feats):
                if feat in k["expected"] and k["expected"][feat] != 0:
                    s += (1.2 if rank == 0 else 0.7 if rank < 3 else 0.3) * k["weights"].get(feat, 0.3)
            for item in shap[:5]:
                if item["feature"] in k["key"] and item.get("label") == "COMP":
                    s -= 1.5
            # mutual exclusion: a feed-sulfur step also raises recycle H2S, so
            # H2S alone must not be read as amine failure; and vice versa.
            zfeed = z.get("feed_sulfur_analyzer", 0.0)
            zh2s = z.get("recycle_h2s", 0.0)
            if name == "amine" and zfeed > 2.0:
                s -= 3.0
            if name == "feed_sulfur" and zh2s > 2.0 and zfeed < 1.0:
                s -= 2.0
            if name == "deactivation" and zfeed > 2.0:
                s -= 2.0
            if name == "h2_pressure" and zh2s > 2.0:
                s -= 2.0
            if fm_res:
                s += 1.5 * fm_res["scores"].get(name, 0.0)
            scores[name] = s

        if sensor != "ok":
            # an abnormal analyzer may be the root cause, or may mask a process
            # fault. Keep it as a candidate; only dominate when process variables
            # show no coherent fault signature.
            process_max = max(scores.values(), default=0.0)
            scores["instrument"] = 2.0 + (3.0 if process_max < 2.0 else 0.0)

        conf = softmax(scores)
        ranked = sorted(conf.items(), key=lambda kv: -kv[1])
        top, p = ranked[0]
        k = kb[top]

        evidence = []
        for item in shap[:5]:
            f = item["feature"]
            evidence.append({
                "variable": f,
                "shap": round(item["shap"], 1),
                "z": round(z.get(f, 0.0), 2),
                "direction": "↑" if z.get(f, 0.0) > 0 else "↓",
                "sca": item.get("label", ""),
            })

        contradictions = []
        for feat, exp in k["expected"].items():
            if exp != 0 and z.get(feat, 0.0) * exp < 0 and abs(z.get(feat, 0.0)) > 1.5:
                contradictions.append("%s 方向与机理预期相反（z=%+.2f）" % (feat, z.get(feat, 0.0)))
        for c in k.get("contradictions", []):
            contradictions.append(c)

        actions = k.get("sop", [])
        report = (
            "事件 %s：判定为「%s」，置信度 %.0f%%。\n"
            "主要证据：%s。\n"
            "机理：%s\n"
            "排除/矛盾：%s\n"
            "建议处置：%s"
            % (eid, k["name"], p * 100,
               "；".join("%s %s（SHAP %.1f，%s）" % (e["variable"], e["direction"],
                                                     e["shap"], e["sca"]) for e in evidence[:4]),
               k["mechanism"],
               "；".join(contradictions) if contradictions else "未发现显著矛盾",
               "；".join(actions))
        )

        records.append({
            "event_id": eid,
            "fault_type": d["fault_type"],
            "top1": top,
            "name": k["name"],
            "confidence": round(p, 3),
            "alternatives": [{"fault": a, "confidence": round(b, 3)} for a, b in ranked[1:3]],
            "evidence": evidence,
            "contradictions": contradictions,
            "actions": actions,
            "report": report,
            "sensor_status": sensor,
        })
        comps = str(d["fault_type"]).split("|")
        rows.append({"event_id": eid, "fault_type": d["fault_type"], "top1": top,
                     "confidence": round(p, 3), "correct": int(top in comps),
                     "is_single_fault": int(single_map.get(eid, 0))})

    with open(os.path.join(args.out, "diagnosis.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    md = ["# 智能体诊断结果", ""]
    for r in records:
        md.append("## %s（%s）" % (r["event_id"], r["fault_type"]))
        md.append("")
        md.append(r["report"])
        md.append("")
    with open(os.path.join(args.out, "diagnosis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    res = pd.DataFrame(rows)
    metrics = {
        "events": int(len(res)),
        "top1_accuracy": round(float(res.correct.mean()), 3),
        "single_fault_events": int(res.is_single_fault.sum()),
        "single_fault_accuracy": round(float(res[res.is_single_fault == 1].correct.mean()), 3),
        "compound_events": int((1 - res.is_single_fault).sum()),
        "compound_accuracy": round(float(res[res.is_single_fault == 0].correct.mean()), 3),
        "mean_confidence": round(float(res.confidence.mean()), 3),
        "by_type": res.groupby("fault_type").correct.mean().round(3).to_dict(),
    }
    with open(os.path.join(args.out, "agent_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(res.to_string(index=False))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
