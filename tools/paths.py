# -*- coding: utf-8 -*-
"""Shared paths, feature list and cached data access for the tool package."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
KNOWLEDGE = ROOT / "knowledge"
CONFIG = ROOT / "config"

FEATURES = [
    "feed_sulfur_analyzer", "feed_flow", "t_in", "t_bed", "dt_ad",
    "p_h2", "recycle_h2s", "fuel_valve", "sulfur_analyzer",
]
SULFUR_IDX = FEATURES.index("sulfur_analyzer")

_df = None
_baseline = None


def load_frame():
    global _df
    if _df is None:
        from pipeline.common import load_df
        _df = load_df(str(DATA))
    return _df


def baseline():
    global _baseline
    if _baseline is None:
        with open(ARTIFACTS / "monitor_baseline.json", encoding="utf-8") as f:
            _baseline = json.load(f)
    return _baseline


def knowledge():
    with open(KNOWLEDGE / "faults.json", encoding="utf-8") as f:
        return json.load(f)
