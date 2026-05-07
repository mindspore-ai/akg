# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TypedDict

import pytest
from langgraph.graph import END, StateGraph

from akg_agents.core_v2.langgraph_base import BaseWorkflow
from akg_agents.core_v2.langgraph_base.checkpointing import (
    FileCheckpointSaver,
    build_invoke_config,
    get_existing_debug_state,
)
from akg_agents.core_v2.tools.tool_executor import ToolExecutor


class CounterState(TypedDict, total=False):
    x: int


async def _inc_node(state: CounterState) -> dict:
    return {"x": state.get("x", 0) + 1}


def _build_counter_graph(checkpoint_file: str):
    graph = StateGraph(CounterState)
    graph.add_node("inc", _inc_node)
    graph.set_entry_point("inc")
    graph.add_edge("inc", END)
    return graph.compile(checkpointer=FileCheckpointSaver(checkpoint_file))


@pytest.mark.asyncio
async def test_file_checkpoint_saver_persists_and_resumes(tmp_path):
    checkpoint_file = str(tmp_path / "debug_checkpoint.pkl")
    cfg = {
        "debug": {
            "enabled": True,
            "session_id": "unit-thread",
            "checkpoint_file": checkpoint_file,
        }
    }
    invoke_config = build_invoke_config(cfg, recursion_limit=25)

    app = _build_counter_graph(checkpoint_file)
    first = await app.ainvoke({"x": 1}, invoke_config)
    assert first["x"] == 2
    assert (tmp_path / "debug_checkpoint.pkl").is_file()

    app_reloaded = _build_counter_graph(checkpoint_file)
    saved_state = await get_existing_debug_state(app_reloaded, invoke_config)
    assert saved_state["x"] == 2

    resumed = await app_reloaded.ainvoke(None, invoke_config)
    assert resumed["x"] == 2


class _CounterWorkflow(BaseWorkflow[CounterState]):
    def __init__(self, agents, config, trace=None, **kwargs):
        super().__init__(config=config, trace=trace)
        self.agents = agents

    def build_graph(self):
        graph = StateGraph(CounterState)
        graph.add_node("inc", _inc_node)
        graph.set_entry_point("inc")
        graph.add_edge("inc", END)
        return graph

    @classmethod
    def build_initial_state(cls, arguments, agent_context):
        return {"x": arguments.get("x", 0), "max_iterations": 1}

    def format_result(self, state):
        return {"status": "success", "output": state, "error_information": ""}


@pytest.mark.asyncio
async def test_tool_executor_workflow_uses_debug_invoke_config(tmp_path):
    checkpoint_file = str(tmp_path / "tool_executor_checkpoint.pkl")
    cfg = {
        "debug": {
            "enabled": True,
            "session_id": "workflow-thread",
            "checkpoint_file": checkpoint_file,
        }
    }
    workflow_registry = {
        "counter_tool": {
            "workflow_class": _CounterWorkflow,
            "workflow_name": "counter_workflow",
        }
    }
    agent_context = {
        "get_workflow_resources": lambda: {
            "agents": {"dummy": object()},
            "config": dict(cfg),
        }
    }
    executor = ToolExecutor(
        workflow_registry=workflow_registry,
        agent_context=agent_context,
    )

    first = await executor.execute("counter_tool", {"x": 2})
    assert first["output"]["x"] == 3
    assert (tmp_path / "tool_executor_checkpoint.pkl").is_file()

    resume_cfg = {
        "debug": {
            "enabled": True,
            "resume": True,
            "session_id": "workflow-thread",
            "checkpoint_file": checkpoint_file,
        }
    }
    resume_context = {
        "get_workflow_resources": lambda: {
            "agents": {"dummy": object()},
            "config": dict(resume_cfg),
        }
    }
    resume_executor = ToolExecutor(
        workflow_registry=workflow_registry,
        agent_context=resume_context,
    )

    resumed = await resume_executor.execute("counter_tool", {"x": 99})
    assert resumed["output"]["x"] == 3


def test_debug_checkpoint_load_failure_is_not_suppressed(tmp_path):
    checkpoint_file = tmp_path / "corrupted.pkl"
    checkpoint_file.write_text("not a pickle", encoding="utf-8")
    workflow = _CounterWorkflow(
        agents={"dummy": object()},
        config={
            "debug": {
                "enabled": True,
                "checkpoint_file": str(checkpoint_file),
            }
        },
    )

    with pytest.raises(RuntimeError, match="Failed to load LangGraph checkpoint"):
        workflow.compile()
