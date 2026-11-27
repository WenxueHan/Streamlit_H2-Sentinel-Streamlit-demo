# -*- coding: utf-8 -*-
"""Multi-detector routing: D3 instrument > D1 quality > D4 deactivation > D2 low."""
SPEC = {
    "id": "route_detector",
    "name": "route_detector",
    "description": "把 D1-D4 检测器结果按优先级融合为单一告警通道。",
    "input_params": {
        "type": "object",
        "properties": {
            "flags": {"type": "object", "description": "D1-D4 布尔值"},
        },
        "required": ["flags"],
    },
}

DETECTORS = {
    "D1_Q1_fast": {"name": "质量预警", "role": "1-5 min 质量预警（计入指标）"},
    "D2_Q1_low": {"name": "低幅趋势", "role": "趋势通道（不计入 1-5 min）"},
    "D3_instrument": {"name": "仪表健康", "role": "仪表通道（不计入 1-5 min）"},
    "D4_deactivation": {"name": "失活趋势", "role": "趋势通道（不计入 1-5 min）"},
}
PRIORITY = ["D3_instrument", "D1_Q1_fast", "D4_deactivation", "D2_Q1_low"]


def fuse(flags: dict):
    for d in PRIORITY:
        if flags.get(d):
            return d
    return None


def route_detector(flags: dict) -> dict:
    d = fuse(flags)
    return {
        "detector": d,
        "name": DETECTORS[d]["name"] if d else None,
        "role": DETECTORS[d]["role"] if d else None,
        "fired": [k for k in PRIORITY if flags.get(k)],
    }
