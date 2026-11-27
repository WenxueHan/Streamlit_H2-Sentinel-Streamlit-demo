# -*- coding: utf-8 -*-
"""openJiuwen tool registry: ToolCard + LocalFunction for all 11 tools."""
from tools.action.parameter_sandbox import SPEC as SANDBOX_SPEC, simulate
from tools.action.sop import SPEC as SOP_SPEC, get_sop
from tools.data.get_window import SPEC as WINDOW_SPEC, get_window
from tools.diagnosis.diagnose import SPEC as DIAG_SPEC, diagnose_root
from tools.diagnosis.sca import SPEC as SCA_SPEC, sca_label
from tools.diagnosis.shap_explain import SPEC as SHAP_SPEC, shap_explain
from tools.flood.triage import SPEC as TRIAGE_SPEC, triage_alarms
from tools.quality.ensemble import SPEC as ENSEMBLE_SPEC, route_detector
from tools.quality.fault_monitor import SPEC as FAULT_SPEC, detect_fault_type
from tools.quality.monitor import SPEC as MONITOR_SPEC, monitor_quality
from tools.quality.predict import SPEC as PREDICT_SPEC, predict_quality
from tools.report.report import SPEC as REPORT_SPEC, build_report

BINDINGS = [
    (WINDOW_SPEC, get_window),
    (PREDICT_SPEC, predict_quality),
    (MONITOR_SPEC, monitor_quality),
    (FAULT_SPEC, detect_fault_type),
    (ENSEMBLE_SPEC, route_detector),
    (SHAP_SPEC, shap_explain),
    (SCA_SPEC, sca_label),
    (DIAG_SPEC, diagnose_root),
    (SOP_SPEC, get_sop),
    (SANDBOX_SPEC, simulate),
    (TRIAGE_SPEC, triage_alarms),
    (REPORT_SPEC, build_report),
]


def build_tools():
    """Build real openJiuwen LocalFunction tools (requires openjiuwen installed)."""
    try:
        from openjiuwen.core.foundation.tool import LocalFunction, ToolCard
    except Exception as exc:
        raise RuntimeError(
            "openjiuwen not installed; run `pip install -U openjiuwen` first"
        ) from exc
    tools = []
    for spec, func in BINDINGS:
        card = ToolCard(**spec)
        tools.append(LocalFunction(card=card, func=func))
    return tools


def register(agent):
    """Register all tools into Runner and expose them to an agent."""
    from openjiuwen.core.runner import Runner
    tools = build_tools()
    for t in tools:
        Runner.resource_mgr.add_tool(t)
        agent.ability_manager.add(t.card)
    return tools
