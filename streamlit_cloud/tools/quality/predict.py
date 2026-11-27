# -*- coding: utf-8 -*-
"""predict_quality: five-step forecast on a 60-minute window."""
import numpy as np

from tools.data.get_window import get_window
from tools.paths import ARTIFACTS

SPEC = {
    "id": "predict_quality",
    "name": "predict_quality",
    "description": "对给定运行/时刻，预测未来 5 分钟的出口硫与 8 个辅助变量。",
    "input_params": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "time": {"type": "string"},
            "horizon": {"type": "integer", "default": 5, "minimum": 1, "maximum": 5},
        },
        "required": ["run_id", "time"],
    },
}

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from pipeline.monitor import load_model
        _MODEL = load_model(str(ARTIFACTS / "forecast.npz"))
    return _MODEL


def predict_quality(run_id: str, time: str, horizon: int = 5) -> dict:
    w = get_window(run_id, time, 60)
    predict, L, H = _model()
    X = np.asarray([w["X"]], dtype=np.float32)
    P = predict(X)[0]
    h = max(1, min(int(horizon), H))
    return {
        "run_id": run_id,
        "time": w["time"],
        "pred": P[:h].tolist(),
        "horizon": h,
        "model_version": "ssm-ridge-v1.0",
        "quality_flag": w["quality_flag"],
    }
