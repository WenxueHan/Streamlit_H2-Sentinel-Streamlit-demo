# Agent 配置（openJiuwen）

- `agent/prompts.py`：系统提示词与拒答规则。
- `agent/agent_config.py`：`build_diagnose_agent()`（ReActAgent）、`build_orchestrator()`（WorkflowAgent）、`run_once()`。
- `config/agent.json`：模型、工具白名单、门禁。
- `config/workflow.json`：取数→预测→监测→分诊→诊断→报告。

## 当前环境

- openJiuwen **0.1.21** 已安装到此项目 Python 环境，`HAS_OPENJIUWEN=True`；
- 12 个 ToolCard 已注册；
- 已用 DeepSeek（OpenAI 兼容，`MODEL_PROVIDER=openai`）跑通 ReActAgent：
  - 模式 `openjiuwen`；
  - 根因 `feed_sulfur`，置信度 0.794；
  - 6 条处置建议；
  - 单次约 30 s（含 12 工具选择与多轮调用）。
- 尚未在 910B/昇腾 MindIE 上跑；Ascend 环境把 `MODEL_PROVIDER=ascend`、`API_BASE` 换成 MindIE/vLLM 地址即可。

## 运行（DeepSeek API）

```bash
export MODEL_PROVIDER=openai
export API_BASE=https://api.deepseek.com/v1
export API_KEY=sk-...
export MODEL_NAME=deepseek-chat
python -c "import asyncio; from agent.agent_config import run_once; print(asyncio.run(run_once('R05-E01 为什么报警？')))"
```

## 运行（昇腾本地模型）

```bash
export MODEL_PROVIDER=ascend
export API_BASE=http://<mindie-host>:8000/v1
export API_KEY=EMPTY
export MODEL_NAME=qwen2.5-14b-instruct
```

`run_once` 是异步函数，命令行用 `asyncio.run(...)`；Streamlit 里由 `app/agent_bridge.py` 调用。
