# -*- coding: utf-8 -*-
"""Bridge between the Streamlit demo and the openJiuwen agent.

- Build a *dynamic* evidence packet for the event currently being replayed.
- If openJiuwen is enabled, run the ReActAgent with that event/time.
- Otherwise ask DeepSeek to explain the same evidence, or use the local rule.
"""
import asyncio
import json
import os
import re

import deepseek_agent


def _parse_json(text):
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _normalize(result):
    if isinstance(result, dict):
        out = result.get("output") or result.get("result") or result
        if isinstance(out, str):
            out = _parse_json(out) or out
        if isinstance(out, dict) and "answer" in out:
            out = out["answer"]
        if isinstance(out, dict):
            return {
                "conclusion": out.get("conclusion") or out.get("answer") or str(out),
                "root_cause": out.get("root_cause") or out.get("top1") or "unknown",
                "confidence": float(out.get("confidence", 0) or 0),
                "actions": out.get("actions", []),
                "refuse": bool(out.get("refuse", False)),
            }
    return {"conclusion": str(result), "root_cause": "unknown", "confidence": 0,
            "actions": [], "refuse": True}


def build_context(event_id, run_id, time, fault_type=""):
    """Compute prediction/monitoring/SHAP/diagnosis for the current replay row."""
    try:
        from tools.data.get_window import get_window
        from tools.diagnosis.diagnose import diagnose_root
        from tools.diagnosis.shap_explain import shap_explain
        from tools.quality.fault_monitor import detect_fault_type
        from tools.quality.monitor import monitor_quality
        from tools.quality.predict import predict_quality

        w = get_window(run_id, time, 60)
        cur = w["X"][-1]
        p = predict_quality(run_id, time, 5)
        m = monitor_quality(p["pred"], cur)
        fm = detect_fault_type(run_id, time)
        sh = shap_explain(run_id, time, top_k=8)
        local = diagnose_root(event_id, {"time": time, "shap": sh["shap"]})
        return {
            "event_id": event_id,
            "run_id": run_id,
            "time": str(time),
            "fault_type": fault_type,
            "current": {f: round(float(v), 3) for f, v in zip(w["features"], cur)},
            "pred": p["pred"],
            "monitor": m,
            "fault_monitor": fm,
            "shap": sh["shap"],
            "local_rule": {
                "top1": local.get("top1"),
                "confidence": local.get("confidence"),
                "top3": local.get("top3"),
                "actions": local.get("actions", []),
                "refuse": local.get("refuse", False),
            },
        }
    except Exception as exc:
        ev = deepseek_agent.evidence_packet(event_id)
        ev["context_error"] = str(exc)
        return ev


def diagnose(api_key="", base_url="https://api.deepseek.com/v1", model="deepseek-chat",
             event_id="R05-E01", query="为什么报警？给出根因和处置建议。",
             use_openjiuwen=False, run_id=None, time=None, fault_type=""):
    ctx = build_context(event_id, run_id, time, fault_type) if (run_id and time) else deepseek_agent.evidence_packet(event_id)
    note = ""
    if use_openjiuwen:
        try:
            from agent.agent_config import HAS_OPENJIUWEN, run_once
            if HAS_OPENJIUWEN:
                os.environ["API_BASE"] = base_url
                os.environ["API_KEY"] = api_key
                os.environ["MODEL_NAME"] = model
                os.environ.setdefault("MODEL_PROVIDER", "openai")
                q = ("当前回放事件 event_id=%s，run_id=%s，time=%s，fault_type=%s。%s"
                     "请基于当前时刻的预测、T2/SPE/CI、辅助变量、SHAP/SCA 与故障模板，"
                     "给出当前根因和处置建议。" % (event_id, run_id, time, fault_type, query))
                raw = asyncio.run(run_once(q))
                return {"mode": "openjiuwen", "answer": _normalize(raw), "evidence": ctx}
            note = "openJiuwen 未安装，已回退到 DeepSeek/本地规则"
        except Exception as exc:
            note = "openJiuwen 运行失败，已回退：%s" % exc
    res = deepseek_agent.diagnose_with_evidence(
        ctx, api_key=api_key, base_url=base_url, model=model, query=query)
    if note:
        res["openjiuwen_note"] = note
    return res


def test_connection(api_key, base_url, model):
    return deepseek_agent.test_connection(api_key, base_url, model)
