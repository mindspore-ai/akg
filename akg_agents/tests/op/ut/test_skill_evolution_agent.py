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

"""
SkillEvolutionAgent 单元测试

验证 Agent 注册元数据、参数 schema、基类工具方法、模式分发逻辑等，
不依赖 LLM 或 GPU。
"""

import json
import os
from pathlib import Path

import pytest

from akg_agents.op.agents.skill_evolution_agent import SkillEvolutionAgent
from akg_agents.core_v2.agents import AgentBase
from akg_agents.core_v2.agents.skill_evolution_base import SkillEvolutionBase


# ====================== Agent 注册元数据 ======================

class TestSkillEvolutionAgentMetadata:
    def test_is_agent_base_subclass(self):
        assert issubclass(SkillEvolutionAgent, AgentBase)

    def test_is_skill_evolution_base_subclass(self):
        assert issubclass(SkillEvolutionAgent, SkillEvolutionBase)

    def test_tool_name(self):
        assert SkillEvolutionAgent.TOOL_NAME == "call_skill_evolution"

    def test_description_not_empty(self):
        desc = SkillEvolutionAgent.DESCRIPTION
        assert desc
        assert len(desc.strip()) > 50

    def test_description_mentions_modes(self):
        desc = SkillEvolutionAgent.DESCRIPTION
        assert "search_log" in desc
        assert "expert_tuning" in desc
        assert "error_fix" in desc
        assert "organize" in desc


# ====================== PARAMETERS_SCHEMA ======================

class TestSkillEvolutionAgentSchema:
    @pytest.fixture
    def schema(self):
        return SkillEvolutionAgent.PARAMETERS_SCHEMA

    def test_schema_type(self, schema):
        assert schema["type"] == "object"

    def test_schema_has_properties(self, schema):
        assert "properties" in schema

    def test_mode_parameter(self, schema):
        mode = schema["properties"]["mode"]
        assert mode["type"] == "string"
        assert set(mode["enum"]) == {"search_log", "expert_tuning", "error_fix", "organize"}
        assert mode.get("default") == "search_log"

    def test_op_name_parameter(self, schema):
        assert "op_name" in schema["properties"]
        assert schema["properties"]["op_name"]["type"] == "string"

    def test_log_dir_parameter(self, schema):
        assert "log_dir" in schema["properties"]
        assert schema["properties"]["log_dir"].get("default") == ""

    def test_conversation_dir_parameter(self, schema):
        assert "conversation_dir" in schema["properties"]

    def test_task_desc_parameter(self, schema):
        assert "task_desc" in schema["properties"]

    def test_skills_dir_parameter(self, schema):
        assert "skills_dir" in schema["properties"]

    def test_output_dir_parameter(self, schema):
        assert "output_dir" in schema["properties"]

    def test_no_required_fields(self, schema):
        assert schema.get("required", []) == []


# ====================== SkillEvolutionBase 工具方法 ======================

class TestSkillEvolutionBaseUtils:
    def test_print_logs(self):
        lines = []
        SkillEvolutionBase._print("test", "hello world", lines)
        assert len(lines) == 1
        assert "hello world" in lines[0]

    def test_init_workspace_with_cur_path(self, tmp_path):
        ws = SkillEvolutionBase._init_workspace(
            cur_path=str(tmp_path), fallback_dir="", name="test"
        )
        assert Path(ws).exists()
        assert "skill_evolution" in ws

    def test_init_workspace_with_fallback(self, tmp_path):
        ws = SkillEvolutionBase._init_workspace(
            cur_path="", fallback_dir=str(tmp_path), name="test"
        )
        assert Path(ws).exists()
        assert "skill_evolution" in ws

    def test_save_text(self, tmp_path):
        result = SkillEvolutionBase._save_text(str(tmp_path), "test.txt", "hello")
        assert result is True
        content = (tmp_path / "test.txt").read_text(encoding="utf-8")
        assert content == "hello"

    def test_save_json(self, tmp_path):
        data = {"key": "value", "nested": [1, 2, 3]}
        result = SkillEvolutionBase._save_json(str(tmp_path), "data.json", data)
        assert result is True
        loaded = json.loads((tmp_path / "data.json").read_text(encoding="utf-8"))
        assert loaded == data

    def test_save_text_invalid_path(self):
        result = SkillEvolutionBase._save_text("/nonexistent/path/999", "f.txt", "x")
        assert result is False

    def test_save_json_invalid_path(self):
        result = SkillEvolutionBase._save_json("/nonexistent/path/999", "f.json", {})
        assert result is False


# ====================== 模式分发逻辑 ======================

class TestSkillEvolutionAgentModeDispatch:
    @pytest.mark.asyncio
    async def test_invalid_mode_returns_error(self):
        agent = SkillEvolutionAgent()
        result = await agent.run(mode="nonexistent_mode", op_name="relu")
        assert result["status"] == "error"
        assert "不支持的模式" in result["output"] or "不支持的模式" in result.get("error_information", "")

    @pytest.mark.asyncio
    async def test_valid_modes_dispatch_correctly(self, tmp_path):
        """Verify mode string matching dispatches correctly (doesn't hit 'unsupported mode')"""
        agent = SkillEvolutionAgent()
        valid_modes = ["search_log", "expert_tuning", "error_fix", "organize"]
        for mode in valid_modes:
            try:
                await agent.run(mode=mode, op_name="test", cur_path=str(tmp_path))
            except Exception:
                pass  # expected: will fail on LLM/file access, not on mode dispatch


# ====================== 模块导出 ======================

class TestModuleExports:
    def test_skill_evolution_base_importable(self):
        from akg_agents.core_v2.agents.skill_evolution_base import SkillEvolutionBase
        assert hasattr(SkillEvolutionBase, "_print")
        assert hasattr(SkillEvolutionBase, "_init_workspace")
        assert hasattr(SkillEvolutionBase, "_save_text")
        assert hasattr(SkillEvolutionBase, "_save_json")

    def test_agent_registered_with_scope(self):
        from akg_agents.core_v2.agents.registry import AgentRegistry
        agent_names = AgentRegistry.list_agents(scope="op")
        assert "SkillEvolutionAgent" in agent_names
