# 工具包 Toolkit（openJiuwen）

12 个工具按“取数—预测—监测—诊断—处置—推演—报告”分层，全部是 `ToolCard + LocalFunction` 可注册的普通函数。

| 工具 | 模块 | 用途 |
|---|---|---|
| get_window | `tools/data/get_window.py` | 取 60 min 窗口 |
| predict_quality | `tools/quality/predict.py` | 5 步预测 |
| monitor_quality | `tools/quality/monitor.py` | T²/SPE + CI + D1–D4 |
| detect_fault_type | `tools/quality/fault_monitor.py` | 每类故障独立模板匹配 |
| route_detector | `tools/quality/ensemble.py` | 多检测器优先级 |
| shap_explain | `tools/diagnosis/shap_explain.py` | 256 联盟精确 SHAP |
| sca_label | `tools/diagnosis/sca.py` | R/I/C/D 标签 |
| diagnose_root | `tools/diagnosis/diagnose.py` | top-3 根因 |
| get_sop | `tools/action/sop.py` | 处置步骤 |
| parameter_sandbox | `tools/action/parameter_sandbox.py` | 阈值回放 |
| triage_alarms | `tools/flood/triage.py` | 洪泛分诊 |
| build_report | `tools/report/report.py` | 证据包/报告 |

## 直接调用

```python
from tools.data.get_window import get_window
from tools.quality.predict import predict_quality
from tools.quality.monitor import monitor_quality

w = get_window("R05", "2025-10-30T00:50:00")
p = predict_quality("R05", "2025-10-30T00:50:00")
m = monitor_quality(p["pred"], p["pred"][0])
```

## 注册到 openJiuwen

```python
from tools.registry import build_tools, register
from agent.agent_config import build_diagnose_agent

agent = build_diagnose_agent()
register(agent)
```

`build_tools()` 需要 `pip install -U openjiuwen`；当前环境未安装时只做静态检查，不执行注册。

## MCP

```bash
python -m tools.mcp_server
```

`config/mcp_server.json` 是服务端工具清单，`config/mcp_client.json` 是客户端配置。
