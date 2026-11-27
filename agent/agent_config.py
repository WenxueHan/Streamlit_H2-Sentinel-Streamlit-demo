# -*- coding: utf-8 -*-
"""openJiuwen Agent configuration (requires openjiuwen installed)."""
import os

from agent.prompts import SYSTEM_PROMPT

try:
    from openjiuwen.core.application.workflow_agent import WorkflowAgent
    from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig
    from openjiuwen.core.runner import Runner
    from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig
    from openjiuwen.core.single_agent.legacy import WorkflowAgentConfig
    HAS_OPENJIUWEN = True
except Exception:
    HAS_OPENJIUWEN = False


def model_configs():
    if not HAS_OPENJIUWEN:
        raise RuntimeError("openjiuwen not installed")
    client = ModelClientConfig(
        client_provider=os.getenv("MODEL_PROVIDER", "ascend"),
        api_base=os.getenv("API_BASE", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("API_KEY", "EMPTY"),
        verify_ssl=False,
    )
    req = ModelRequestConfig(
        model=os.getenv("MODEL_NAME", "qwen2.5-14b-instruct"),
        temperature=0.2,
        top_p=0.8,
    )
    return client, req


def build_diagnose_agent():
    """ReActAgent: multi-turn diagnosis with tools."""
    if not HAS_OPENJIUWEN:
        raise RuntimeError("openjiuwen not installed")
    client, req = model_configs()
    agent = ReActAgent(
        card=AgentCard(id="h2_diagnose_agent", description="加氢报警根因诊断")
    ).configure(ReActAgentConfig(
        model_client_config=client,
        model_config_obj=req,
        prompt_template=[{"role": "system", "content": SYSTEM_PROMPT}],
    ))
    from tools.registry import register
    register(agent)
    return agent


def build_orchestrator():
    """WorkflowAgent: deterministic pipeline entry (workflow registered separately)."""
    if not HAS_OPENJIUWEN:
        raise RuntimeError("openjiuwen not installed")
    return WorkflowAgent(WorkflowAgentConfig(
        id="h2_orchestrator",
        version="1.0",
        description="加氢报警编排：取数-预测-监测-诊断-报告",
    ))


def run_once(query: str, conversation_id: str = "demo"):
    agent = build_diagnose_agent()
    return Runner.run_agent(agent=agent, inputs={"query": query, "conversation_id": conversation_id})
