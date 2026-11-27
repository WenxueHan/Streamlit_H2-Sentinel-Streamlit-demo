# -*- coding: utf-8 -*-
"""DeepSeek-driven explanation layer for the Streamlit demo.

The numeric work is done by local deterministic tools; DeepSeek only explains the
evidence packet. If no API key is provided, the app falls back to the local rule
diagnosis so the demo never blocks.
"""
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETAILS = ROOT / "artifacts" / "rootcause_details.json"
DIAG = ROOT / "artifacts" / "diagnosis.jsonl"

SYSTEM = """你是蜡油加氢预处理装置的工艺报警专家。只依据给出的证据包回答，不要编造测点或数值。
要求输出 JSON：conclusion（一句话）、root_cause（故障类型）、confidence（0-1）、
evidence_ids（数组）、actions（数组，每项 step/role/risk）、refuse（bool）。
原料硫上升会带来循环氢 H2S 上升，不能仅凭 H2S 判胺洗失效。"""


def _load_local(event_id="R05-E01"):
    detail = {}
    if DETAILS.exists():
        try:
            detail = json.loads(DETAILS.read_text(encoding="utf-8")).get(event_id, {})
        except Exception:
            detail = {}
    local = {}
    if DIAG.exists():
        for line in DIAG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("event_id") == event_id:
                local = d
                break
    return detail, local


def evidence_packet(event_id="R05-E01"):
    detail, local = _load_local(event_id)
    rel = {k: v for k, v in local.items() if k in ("top1", "confidence", "top3", "actions", "refuse")}
    return {
        "event_id": event_id,
        "alarm_time": detail.get("alarm_time", ""),
        "fault_type": detail.get("fault_type", ""),
        "shap": detail.get("shap", [])[:6],
        "local_rule": rel,
    }


def _post(url, payload, api_key, timeout=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def test_connection(api_key, base_url="https://api.deepseek.com/v1", model="deepseek-chat"):
    if not api_key:
        return False, "未填写 API Key"
    try:
        data = _post(base_url.rstrip("/") + "/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 4,
            "temperature": 0,
        }, api_key, timeout=30)
        return True, "连接成功：%s" % data.get("model", model)
    except urllib.error.HTTPError as e:
        return False, "HTTP %s：%s" % (e.code, e.read().decode("utf-8", "ignore")[:200])
    except Exception as e:
        return False, str(e)


def _extract_json(text):
    text = text.strip()
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


def diagnose_with_evidence(ev, api_key="", base_url="https://api.deepseek.com/v1",
                           model="deepseek-chat", query="为什么报警？给出根因和处置建议。"):
    local = ev.get("local_rule", {})
    if not api_key:
        return {
            "mode": "local_rule",
            "answer": {
                "conclusion": "本地规则诊断：%s" % local.get("top1", "unknown"),
                "root_cause": local.get("top1"),
                "confidence": local.get("confidence", 0.0),
                "actions": local.get("actions", []),
                "refuse": local.get("refuse", False),
            },
            "evidence": ev,
        }
    user = "当前事件证据包：\n%s\n\n问题：%s" % (json.dumps(ev, ensure_ascii=False), query)
    data = _post(base_url.rstrip("/") + "/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }, api_key)
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json(content) or {"conclusion": content, "root_cause": local.get("top1"),
                                        "confidence": local.get("confidence", 0.0),
                                        "actions": local.get("actions", []), "refuse": False}
    return {"mode": "deepseek", "model": data.get("model", model),
            "answer": parsed, "evidence": ev}


def diagnose(api_key="", base_url="https://api.deepseek.com/v1", model="deepseek-chat",
             event_id="R05-E01", query="为什么报警？给出根因和处置建议。"):
    return diagnose_with_evidence(evidence_packet(event_id), api_key, base_url, model, query)
