# -*- coding: utf-8 -*-
"""Multi-detector routing layer used by the diagnosis agent.

D1  Q1 fast quality warning      : 3-step quality forecast crosses the limit
D2  Q1 low / slow quality trend  : 5-step forecast + slope, low-severity ramp
D3  instrument health            : analyzer stuck / drift / residual
D4  deactivation trend           : dT_ad / T_in slow trend

The detectors are intentionally complementary:
- D1 is the only channel counted in the 1-5 min quality-warning metric;
- D2/D3/D4 extend coverage to slow, low-amplitude and instrument faults,
  and are reported separately so they do not pollute the 1-5 min metric.
"""
DETECTORS = {
    "D1_Q1_fast": {
        "name": "质量预警",
        "object": "快变质量越限",
        "trigger": "3 步预测硫 >= 4490 且 2 min 上升 >= 15 ppm",
        "role": "1-5 min 质量预警（计入指标）",
    },
    "D2_Q1_low": {
        "name": "低幅/慢斜坡趋势",
        "object": "低幅原料硫、慢斜坡",
        "trigger": "5 步预测硫 >= 4470 且 5 min 上升 >= 8 ppm",
        "role": "趋势通道（不计入 1-5 min）",
    },
    "D3_instrument": {
        "name": "仪表健康",
        "object": "分析仪冻结/漂移/尖峰",
        "trigger": "滚动方差塌缩或软测量残差 > 3σ",
        "role": "仪表通道（不计入 1-5 min）",
    },
    "D4_deactivation": {
        "name": "失活趋势",
        "object": "催化剂缓慢失活",
        "trigger": "ΔT 30 min 斜率 < -0.02 K/min 且 T_in 斜率 > 0",
        "role": "趋势通道（不计入 1-5 min）",
    },
}

# priority: instrument first (measurement validity), then the fast quality
# warning, then slow process trends.
PRIORITY = ["D3_instrument", "D1_Q1_fast", "D4_deactivation", "D2_Q1_low"]


def fuse(flags):
    """Return the highest-priority detector that fired, or None."""
    for d in PRIORITY:
        if flags.get(d):
            return d
    return None


def evidence(event_id, detector, alarm_time, detail=None):
    """Build the detector evidence block consumed by the agent."""
    info = DETECTORS.get(detector, {})
    return {
        "event_id": event_id,
        "detector": detector,
        "detector_name": info.get("name", ""),
        "detector_role": info.get("role", ""),
        "alarm_time": str(alarm_time),
        "detail": detail or {},
    }
