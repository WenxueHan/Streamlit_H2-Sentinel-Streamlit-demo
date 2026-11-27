# -*- coding: utf-8 -*-
"""build_report: turn an evidence packet into a Markdown/JSON report."""
from datetime import datetime

SPEC = {
    "id": "build_report",
    "name": "build_report",
    "description": "把事件与证据包生成 Markdown 报告和结构化 JSON。",
    "input_params": {
        "type": "object",
        "properties": {
            "event_id": {"type": "string"},
            "evidence": {"type": "object"},
        },
        "required": ["event_id"],
    },
}


def build_report(event_id: str, evidence: dict = None) -> dict:
    e = evidence or {}
    top1 = e.get("top1", "unknown")
    conf = e.get("confidence", 0.0)
    actions = e.get("actions", [])
    lines = [
        "# 加氢哨兵诊断报告",
        "",
        "- 事件：%s" % event_id,
        "- 时间：%s" % e.get("time", ""),
        "- 结论：%s（置信度 %.3f）" % (top1, float(conf)),
        "",
        "## 证据",
    ]
    for item in e.get("evidence", [])[:5]:
        lines.append("- %s：SHAP %.2f，SCA %s" % (
            item.get("feature"), float(item.get("shap", 0)), item.get("label", "")))
    lines += ["", "## 处置建议"]
    for a in actions:
        lines.append("- [%s] %s（风险：%s）" % (a.get("role", ""), a.get("step", ""), a.get("risk", "")))
    lines += ["", "---", "生成时间：%s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    return {"event_id": event_id, "markdown": "\n".join(lines), "json": e}
