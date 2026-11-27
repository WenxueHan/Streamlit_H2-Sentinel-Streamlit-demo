# -*- coding: utf-8 -*-
"""get_sop: knowledge-grounded disposal steps."""
from tools.paths import knowledge

SPEC = {
    "id": "get_sop",
    "name": "get_sop",
    "description": "按故障类型返回机理、处置步骤与矛盾证据。",
    "input_params": {
        "type": "object",
        "properties": {
            "fault_type": {"type": "string"},
            "context": {"type": "object"},
        },
        "required": ["fault_type"],
    },
}


def get_sop(fault_type: str, context: dict = None) -> dict:
    kb = knowledge()
    if fault_type not in kb:
        raise ValueError("SOP_001: unknown fault_type %s" % fault_type)
    k = kb[fault_type]
    return {
        "fault_type": fault_type,
        "name": k["name"],
        "mechanism": k["mechanism"],
        "steps": k["sop"],
        "contradictions": k.get("contradictions", []),
        "context": context or {},
    }
