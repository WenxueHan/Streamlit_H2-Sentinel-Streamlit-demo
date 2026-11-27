# -*- coding: utf-8 -*-
"""All ToolCard specifications in one place (for docs and contract tests)."""
from tools.action.parameter_sandbox import SPEC as SANDBOX
from tools.action.sop import SPEC as SOP
from tools.data.get_window import SPEC as WINDOW
from tools.diagnosis.diagnose import SPEC as DIAGNOSE
from tools.diagnosis.sca import SPEC as SCA
from tools.diagnosis.shap_explain import SPEC as SHAP
from tools.flood.triage import SPEC as TRIAGE
from tools.quality.ensemble import SPEC as ENSEMBLE
from tools.quality.fault_monitor import SPEC as FAULT
from tools.quality.monitor import SPEC as MONITOR
from tools.quality.predict import SPEC as PREDICT
from tools.report.report import SPEC as REPORT

ALL_SPECS = [WINDOW, PREDICT, MONITOR, FAULT, ENSEMBLE, SHAP, SCA, DIAGNOSE,
             SOP, SANDBOX, TRIAGE, REPORT]
