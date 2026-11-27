# -*- coding: utf-8 -*-
"""SCA labels: root / intermediate / compensation / instrument."""
SPEC = {
    "id": "sca_label",
    "name": "sca_label",
    "description": "根据 SHAP、反事实 delta 与结构角色，给出 R/I/C/D 标签。",
    "input_params": {
        "type": "object",
        "properties": {
            "shap": {"type": "array", "description": "shap_explain 的输出"},
            "roles": {"type": "object", "description": "变量到 EXO/INT/MV/PV 的映射"},
        },
        "required": ["shap"],
    },
}

DEFAULT_ROLES = {
    "feed_sulfur_analyzer": "EXO",
    "feed_flow": "EXO",
    "p_h2": "EXO",
    "recycle_h2s": "INT",
    "t_in": "MV",
    "fuel_valve": "MV",
    "t_bed": "PV",
    "dt_ad": "PV",
}


def sca_label(shap: list, roles: dict = None) -> dict:
    roles = roles or DEFAULT_ROLES
    out = []
    for item in shap:
        f = item["feature"]
        role = roles.get(f, item.get("role", "PV"))
        phi = abs(float(item.get("shap", 0.0)))
        delta = abs(float(item.get("delta", 0.0)))
        ratio = delta / (phi + 1e-6)
        if role in ("MV", "PV") and ratio < 0.5:
            label = "COMP"
        elif role == "EXO":
            label = "ROOT" if phi > 0 else "NEUTRAL"
        elif role == "PV":
            label = "INT"
        else:
            label = "NEUTRAL"
        out.append({"feature": f, "role": role, "label": label,
                    "phi": round(phi, 2), "delta": round(delta, 2)})
    return {"sca": out}
