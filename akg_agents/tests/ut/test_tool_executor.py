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

"""ToolExecutor 单元测试

重点覆盖 required 参数校验功能（_validate_required_params / _get_parameters_schema），
防止 LLM 遗漏必需参数时静默传递空值导致下游不可预期的错误。
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from akg_agents.core_v2.tools.tool_executor import ToolExecutor


# ==================== Fixtures ====================

WORKFLOW_SCHEMA = {
    "type": "object",
    "properties": {
        "op_name": {"type": "string", "description": "算子名称"},
        "task_desc": {"type": "string", "description": "任务描述（框架代码）"},
        "dsl": {"type": "string", "description": "目标 DSL"},
        "framework": {"type": "string", "description": "框架"},
        "backend": {"type": "string", "description": "后端"},
        "arch": {"type": "string", "description": "架构"},
        "task_id": {"type": "string", "description": "任务 ID（可选）", "default": ""},
        "user_requirements": {
            "type": "string",
            "description": "用户额外需求（可选）",
            "default": "",
        },
    },
    "required": ["op_name", "task_desc", "dsl", "framework", "backend", "arch"],
}

AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "user_input": {"type": "string", "description": "用户需求描述"},
        "framework": {
            "type": "string",
            "description": "目标框架",
            "default": "torch",
        },
    },
    "required": ["user_input"],
}

NO_REQUIRED_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "description": "模式"},
    },
    "required": [],
}


def _make_registry_entry(schema):
    """构造 agent_registry / workflow_registry 中的 config 结构"""
    return {
        "config": {
            "function": {
                "name": "test_tool",
                "description": "test",
                "parameters": schema,
            }
        }
    }


@pytest.fixture
def executor_with_workflow():
    """创建包含 workflow 注册的 ToolExecutor"""
    return ToolExecutor(
        workflow_registry={
            "call_kernelgen_workflow": {
                **_make_registry_entry(WORKFLOW_SCHEMA),
                "workflow_class": MagicMock(),
                "workflow_name": "KernelGenOnlyWorkflow",
            }
        },
    )


@pytest.fixture
def executor_with_agent():
    """创建包含 agent 注册的 ToolExecutor"""
    return ToolExecutor(
        agent_registry={
            "call_op_task_builder": {
                **_make_registry_entry(AGENT_SCHEMA),
                "agent_class": MagicMock(),
            }
        },
    )


@pytest.fixture
def executor_with_no_required():
    """创建 required 为空的 ToolExecutor"""
    return ToolExecutor(
        agent_registry={
            "call_skill_evolution": {
                **_make_registry_entry(NO_REQUIRED_SCHEMA),
                "agent_class": MagicMock(),
            }
        },
    )


@pytest.fixture
def executor_empty():
    """创建空注册表的 ToolExecutor"""
    return ToolExecutor()


# ==================== _get_parameters_schema 测试 ====================


class TestGetParametersSchema:
    """测试 _get_parameters_schema 从三种注册表中正确获取 schema"""

    def test_from_workflow_registry(self, executor_with_workflow):
        schema = executor_with_workflow._get_parameters_schema("call_kernelgen_workflow")
        assert schema is not None
        assert "required" in schema
        assert "task_desc" in schema["required"]

    def test_from_agent_registry(self, executor_with_agent):
        schema = executor_with_agent._get_parameters_schema("call_op_task_builder")
        assert schema is not None
        assert "required" in schema
        assert "user_input" in schema["required"]

    @patch("akg_agents.core_v2.tools.tool_executor.ToolRegistry")
    def test_from_tool_registry(self, mock_registry, executor_empty):
        mock_tool = MagicMock()
        mock_tool.parameters = {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        }
        mock_registry.get_tool.return_value = mock_tool

        schema = executor_empty._get_parameters_schema("read_file")
        assert schema is not None
        assert "file_path" in schema["required"]

    @patch("akg_agents.core_v2.tools.tool_executor.ToolRegistry")
    def test_unknown_tool_returns_none(self, mock_registry, executor_empty):
        mock_registry.get_tool.return_value = None
        schema = executor_empty._get_parameters_schema("nonexistent_tool")
        assert schema is None

    def test_workflow_takes_priority_over_tool_registry(self, executor_with_workflow):
        """workflow_registry 优先于 ToolRegistry"""
        with patch("akg_agents.core_v2.tools.tool_executor.ToolRegistry") as mock_registry:
            mock_registry.get_tool.return_value = MagicMock(
                parameters={"required": ["something_else"]}
            )
            schema = executor_with_workflow._get_parameters_schema(
                "call_kernelgen_workflow"
            )
            assert "task_desc" in schema["required"]
            mock_registry.get_tool.assert_not_called()


# ==================== _validate_required_params 测试 ====================


class TestValidateRequiredParams:
    """测试 _validate_required_params 校验逻辑"""

    def test_all_required_present_passes(self, executor_with_workflow):
        """所有必需参数都提供时校验通过"""
        args = {
            "op_name": "relu",
            "task_desc": "import torch\nclass Model...",
            "dsl": "cpp",
            "framework": "torch",
            "backend": "cpu",
            "arch": "x86_64",
        }
        result = executor_with_workflow._validate_required_params(
            "call_kernelgen_workflow", args
        )
        assert result is None

    def test_missing_param_returns_error(self, executor_with_workflow):
        """缺少必需参数时返回错误"""
        args = {"op_name": "relu"}
        result = executor_with_workflow._validate_required_params(
            "call_kernelgen_workflow", args
        )
        assert result is not None
        assert result["status"] == "error"
        assert "task_desc" in result["error_information"]
        assert "dsl" in result["error_information"]

    def test_empty_string_param_returns_error(self, executor_with_workflow):
        """必需参数为空字符串时返回错误"""
        args = {
            "op_name": "relu",
            "task_desc": "",
            "dsl": "cpp",
            "framework": "torch",
            "backend": "cpu",
            "arch": "x86_64",
        }
        result = executor_with_workflow._validate_required_params(
            "call_kernelgen_workflow", args
        )
        assert result is not None
        assert result["status"] == "error"
        assert "task_desc" in result["error_information"]
        assert "为空字符串" in result["error_information"]

    def test_whitespace_only_param_returns_error(self, executor_with_workflow):
        """必需参数只有空格时也视为空"""
        args = {
            "op_name": "relu",
            "task_desc": "   \n  ",
            "dsl": "cpp",
            "framework": "torch",
            "backend": "cpu",
            "arch": "x86_64",
        }
        result = executor_with_workflow._validate_required_params(
            "call_kernelgen_workflow", args
        )
        assert result is not None
        assert result["status"] == "error"
        assert "task_desc" in result["error_information"]

    def test_none_param_returns_error(self, executor_with_workflow):
        """必需参数为 None 时返回错误"""
        args = {
            "op_name": "relu",
            "task_desc": None,
            "dsl": "cpp",
            "framework": "torch",
            "backend": "cpu",
            "arch": "x86_64",
        }
        result = executor_with_workflow._validate_required_params(
            "call_kernelgen_workflow", args
        )
        assert result is not None
        assert result["status"] == "error"
        assert "task_desc" in result["error_information"]
        assert "未提供" in result["error_information"]

    def test_optional_param_with_default_allows_empty(self, executor_with_workflow):
        """有 default 值的可选参数允许为空（不在 required 中的参数不校验）"""
        args = {
            "op_name": "relu",
            "task_desc": "import torch\nclass Model...",
            "dsl": "cpp",
            "framework": "torch",
            "backend": "cpu",
            "arch": "x86_64",
            "task_id": "",
            "user_requirements": "",
        }
        result = executor_with_workflow._validate_required_params(
            "call_kernelgen_workflow", args
        )
        assert result is None

    def test_no_required_field_passes(self, executor_with_no_required):
        """required 列表为空时直接通过"""
        result = executor_with_no_required._validate_required_params(
            "call_skill_evolution", {}
        )
        assert result is None

    def test_unknown_tool_passes(self, executor_empty):
        """未注册的工具直接通过（无 schema 可校验）"""
        with patch("akg_agents.core_v2.tools.tool_executor.ToolRegistry") as mock_reg:
            mock_reg.get_tool.return_value = None
            result = executor_empty._validate_required_params("unknown_tool", {})
            assert result is None

    def test_multiple_missing_params_all_reported(self, executor_with_workflow):
        """多个必需参数缺失时全部报告"""
        args = {}
        result = executor_with_workflow._validate_required_params(
            "call_kernelgen_workflow", args
        )
        assert result is not None
        error_info = result["error_information"]
        for param in ["op_name", "task_desc", "dsl", "framework", "backend", "arch"]:
            assert param in error_info

    def test_error_result_format(self, executor_with_workflow):
        """错误返回格式符合标准"""
        args = {"op_name": "relu"}
        result = executor_with_workflow._validate_required_params(
            "call_kernelgen_workflow", args
        )
        assert result is not None
        assert "status" in result
        assert "output" in result
        assert "error_information" in result
        assert result["status"] == "error"
        assert result["output"] == ""

    def test_agent_required_validation(self, executor_with_agent):
        """Agent 工具的 required 校验同样生效"""
        result = executor_with_agent._validate_required_params(
            "call_op_task_builder", {}
        )
        assert result is not None
        assert "user_input" in result["error_information"]

    def test_agent_with_valid_params_passes(self, executor_with_agent):
        """Agent 参数完整时校验通过"""
        result = executor_with_agent._validate_required_params(
            "call_op_task_builder", {"user_input": "生成 relu 算子"}
        )
        assert result is None


# ==================== execute 集成测试 ====================


class TestExecuteWithValidation:
    """测试 execute() 中的 required 校验集成行为"""

    @pytest.mark.asyncio
    async def test_execute_blocks_missing_required_workflow(self, executor_with_workflow):
        """execute 在 workflow 缺少 required 参数时直接返回错误，不执行 workflow"""
        result = await executor_with_workflow.execute(
            "call_kernelgen_workflow", {"op_name": "relu"}
        )
        assert result["status"] == "error"
        assert "task_desc" in result["error_information"]

    @pytest.mark.asyncio
    async def test_execute_blocks_empty_required_workflow(self, executor_with_workflow):
        """execute 在 workflow 的 required 参数为空字符串时阻止执行"""
        result = await executor_with_workflow.execute(
            "call_kernelgen_workflow",
            {
                "op_name": "relu",
                "task_desc": "",
                "dsl": "cpp",
                "framework": "torch",
                "backend": "cpu",
                "arch": "x86_64",
            },
        )
        assert result["status"] == "error"
        assert "task_desc" in result["error_information"]

    @pytest.mark.asyncio
    async def test_execute_blocks_missing_required_agent(self, executor_with_agent):
        """execute 在 agent 缺少 required 参数时直接返回错误"""
        result = await executor_with_agent.execute("call_op_task_builder", {})
        assert result["status"] == "error"
        assert "user_input" in result["error_information"]

    @pytest.mark.asyncio
    async def test_execute_proceeds_when_all_required_present(
        self, executor_with_workflow
    ):
        """所有 required 参数存在时正常进入 workflow 执行流程"""
        mock_workflow_class = executor_with_workflow.workflow_registry[
            "call_kernelgen_workflow"
        ]["workflow_class"]

        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(
            return_value={"verifier_result": True, "coder_code": "code"}
        )
        mock_workflow_instance = MagicMock()
        mock_workflow_instance.compile.return_value = mock_app
        mock_workflow_instance.format_result.return_value = {
            "status": "success",
            "code": "code",
        }
        mock_workflow_class.return_value = mock_workflow_instance
        mock_workflow_class.build_initial_state = MagicMock(
            return_value={"task_desc": "code", "op_name": "relu"}
        )
        mock_workflow_class.ensure_resources = AsyncMock()
        mock_workflow_class.prepare_config = MagicMock()

        executor_with_workflow.agent_context = {
            "get_workflow_resources": lambda: {
                "agents": {"kernel_gen": MagicMock(), "verifier": MagicMock()},
                "config": {},
                "trace": MagicMock(),
                "device_pool": None,
                "private_worker": None,
                "worker_manager": MagicMock(),
                "backend": "cpu",
                "arch": "x86_64",
            }
        }

        result = await executor_with_workflow.execute(
            "call_kernelgen_workflow",
            {
                "op_name": "relu",
                "task_desc": "import torch\nclass Model(nn.Module):\n  pass",
                "dsl": "cpp",
                "framework": "torch",
                "backend": "cpu",
                "arch": "x86_64",
            },
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_resolves_expression_then_validates(
        self, executor_with_workflow
    ):
        """验证 resolve_arguments 在校验之前执行：表达式解析失败时保留原值再校验"""
        args = {
            "op_name": "relu",
            "task_desc": "read_json_file('/nonexistent/path.json')['key']",
            "dsl": "cpp",
            "framework": "torch",
            "backend": "cpu",
            "arch": "x86_64",
        }
        result = await executor_with_workflow.execute(
            "call_kernelgen_workflow", args
        )
        assert result is not None


# ==================== 回归测试：复现原始 bug 场景 ====================


class TestRegressionEmptyTaskDesc:
    """回归测试：复现 LLM 遗漏 task_desc 导致 relu_torch.py 为空的 bug"""

    @pytest.mark.asyncio
    async def test_only_op_name_provided_is_rejected(self, executor_with_workflow):
        """
        复现场景：LLM 调用 call_kernelgen_workflow 时只传了 {"op_name": "relu"}，
        遗漏了 task_desc 等所有其他必需参数。
        期望：校验立即拦截并返回清晰错误，而不是让空 task_desc 流入 verifier。
        """
        result = await executor_with_workflow.execute(
            "call_kernelgen_workflow", {"op_name": "relu"}
        )
        assert result["status"] == "error"
        assert "task_desc" in result["error_information"]
        assert "dsl" in result["error_information"]
        assert "framework" in result["error_information"]
        assert "backend" in result["error_information"]
        assert "arch" in result["error_information"]


# ==================== 错误信息可读性测试 ====================


class TestErrorMessageReadability:
    """测试错误信息的可读性，防止超长 tool_name 污染报错"""

    @pytest.mark.asyncio
    async def test_unknown_tool_error_truncates_long_name(self, executor_empty):
        """超长工具名在报错信息中被截断"""
        long_name = "ask_user` 而不是 `ask_user`）\n\n3. 当前状态" + "x" * 500
        with patch("akg_agents.core_v2.tools.tool_executor.ToolRegistry") as mock_reg:
            mock_reg.get_tool.return_value = None
            mock_reg.list_names.return_value = ["ask_user", "read_file"]
            result = await executor_empty.execute(long_name, {})
        assert result["status"] == "error"
        assert len(result["error_information"]) < 500
        assert "..." in result["error_information"]

    @pytest.mark.asyncio
    async def test_unknown_tool_error_message_is_actionable(self, executor_empty):
        """未知工具的报错信息应该包含可操作指引"""
        with patch("akg_agents.core_v2.tools.tool_executor.ToolRegistry") as mock_reg:
            mock_reg.get_tool.return_value = None
            mock_reg.list_names.return_value = ["ask_user", "read_file"]
            result = await executor_empty.execute("nonexistent", {})
        assert "请检查工具名拼写" in result["error_information"]
        assert "可用工具" in result["error_information"]

    @pytest.mark.asyncio
    async def test_short_tool_name_not_truncated(self, executor_empty):
        """短工具名不被截断"""
        with patch("akg_agents.core_v2.tools.tool_executor.ToolRegistry") as mock_reg:
            mock_reg.get_tool.return_value = None
            mock_reg.list_names.return_value = ["ask_user"]
            result = await executor_empty.execute("bad_name", {})
        assert "'bad_name'" in result["error_information"]


# ==================== 非法 tool_name 报错测试 ====================


class TestInvalidToolNameDetection:
    """测试非法 function.name 到达 ToolExecutor 后的清晰报错

    核心理念：不做偷偷修正，而是清晰报错让 LLM 自行修正。
    """

    @pytest.mark.asyncio
    async def test_polluted_name_rejected_by_tool_executor(self, executor_empty):
        """被 LLM 思考链污染的 function.name 到达 ToolExecutor 后应返回可读错误"""
        polluted = (
            'ask_user` 而不是 `ask_user`）\n\n3. **当前状态**：\n'
            '   - task_desc 已经生成成功（在 node_002）\n'
            '   - 但是上一步的 ask_user 失败了\n\n'
            '</think>我看到上一步出现了错误。<tool_call>ask_user'
        )
        with patch("akg_agents.core_v2.tools.tool_executor.ToolRegistry") as mock_reg:
            mock_reg.get_tool.return_value = None
            mock_reg.list_names.return_value = ["ask_user", "call_kernelgen_workflow"]
            result = await executor_empty.execute(polluted, {})
        assert result["status"] == "error"
        assert "..." in result["error_information"]
        assert "请检查工具名拼写" in result["error_information"]

    @pytest.mark.asyncio
    async def test_short_polluted_name_shows_full_name(self, executor_empty):
        """短的非法 tool_name（<60字符）应完整展示"""
        polluted = '<tool_call>ask_user'
        with patch("akg_agents.core_v2.tools.tool_executor.ToolRegistry") as mock_reg:
            mock_reg.get_tool.return_value = None
            mock_reg.list_names.return_value = ["ask_user"]
            result = await executor_empty.execute(polluted, {})
        assert result["status"] == "error"
        assert "请检查工具名拼写" in result["error_information"]
        assert "..." not in result["error_information"]

    @pytest.mark.asyncio
    async def test_normal_tool_name_passes_through(self, executor_empty):
        """正常工具名应正常传递，不触发报错"""
        with patch("akg_agents.core_v2.tools.tool_executor.ToolRegistry") as mock_reg:
            mock_tool = MagicMock()
            mock_tool.parameters = None
            mock_reg.get_tool.return_value = mock_tool
            mock_reg.aexecute = AsyncMock(return_value={"status": "success", "output": "ok"})
            result = await executor_empty.execute("ask_user", {"message": "hello"})
        assert result["status"] == "success"
