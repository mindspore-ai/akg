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

"""FixCodeGen 单元测试

覆盖：
- CodeMatcher: L1 精确匹配、L2 行级 trim 匹配、find_match 降级逻辑、边界情况
- DiffApplier: 单/多 Modification 应用、部分失败、空编辑检测、diff 生成
- parse_modifications: 正常 JSON、markdown 包裹、格式异常容错
- FixCodeGen 节点: Mock LLM 调用验证 State 更新
"""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from akg_agents.op.utils.diff_utils import (
    CodeMatcher,
    DiffApplier,
    DiffResult,
    Modification,
    parse_modifications,
)


# ==================== CodeMatcher L1: 精确匹配 ====================

class TestCodeMatcherExactMatch:

    def test_exact_match_found(self):
        content = "import torch\n\nclass Model:\n    pass"
        result = CodeMatcher.exact_match(content, "class Model:")
        assert result == "class Model:"

    def test_exact_match_not_found(self):
        content = "import torch\n\nclass Model:\n    pass"
        result = CodeMatcher.exact_match(content, "class NotExist:")
        assert result is None

    def test_exact_match_multiline(self):
        content = "def foo():\n    x = 1\n    return x"
        search = "    x = 1\n    return x"
        result = CodeMatcher.exact_match(content, search)
        assert result == search

    def test_exact_match_empty_search(self):
        content = "import torch"
        result = CodeMatcher.exact_match(content, "")
        assert result is None

    def test_exact_match_multiple_occurrences(self):
        content = "x = 1\nx = 1\nx = 1"
        result = CodeMatcher.exact_match(content, "x = 1")
        assert result == "x = 1"


# ==================== CodeMatcher L2: 行级 trim 匹配 ====================

class TestCodeMatcherTrimmedLineMatch:

    def test_trimmed_match_indent_difference(self):
        content = "def foo():\n    x = 1\n    return x"
        search = "def foo():\n  x = 1\n  return x"  # 2-space indent vs 4-space
        result = CodeMatcher.trimmed_line_match(content, search)
        assert result is not None
        assert result == "def foo():\n    x = 1\n    return x"

    def test_trimmed_match_trailing_whitespace(self):
        content = "x = 1\ny = 2"
        search = "x = 1   \ny = 2   "
        result = CodeMatcher.trimmed_line_match(content, search)
        assert result is not None
        assert result == "x = 1\ny = 2"

    def test_trimmed_match_tab_vs_space(self):
        content = "\tx = 1\n\ty = 2"
        search = "    x = 1\n    y = 2"
        result = CodeMatcher.trimmed_line_match(content, search)
        assert result is not None
        assert "\tx = 1" in result

    def test_trimmed_match_not_found(self):
        content = "x = 1\ny = 2\nz = 3"
        search = "a = 10\nb = 20"
        result = CodeMatcher.trimmed_line_match(content, search)
        assert result is None

    def test_trimmed_match_empty_search(self):
        result = CodeMatcher.trimmed_line_match("some code", "")
        assert result is None

    def test_trimmed_match_all_blank_search(self):
        result = CodeMatcher.trimmed_line_match("some code", "   \n   ")
        assert result is None

    def test_trimmed_match_search_longer_than_content(self):
        content = "x = 1"
        search = "x = 1\ny = 2\nz = 3"
        result = CodeMatcher.trimmed_line_match(content, search)
        assert result is None

    def test_trimmed_match_partial_code(self):
        content = "import torch\nimport os\n\ndef foo():\n    x = 1\n    return x\n\ndef bar():\n    pass"
        search = "def foo():\n  x = 1\n  return x"
        result = CodeMatcher.trimmed_line_match(content, search)
        assert result is not None
        assert "def foo():" in result
        assert "    x = 1" in result


# ==================== CodeMatcher find_match (降级逻辑) ====================

class TestCodeMatcherFindMatch:

    def test_find_match_exact_first(self):
        content = "x = 1\ny = 2"
        matched, level = CodeMatcher.find_match(content, "x = 1")
        assert matched == "x = 1"
        assert level == "exact"

    def test_find_match_fallback_to_trimmed(self):
        content = "    x = 1\n    y = 2"
        search = "x = 1\ny = 2"  # no indent
        matched, level = CodeMatcher.find_match(content, search)
        assert matched is not None
        assert level == "trimmed"
        assert "    x = 1" in matched

    def test_find_match_none(self):
        content = "x = 1"
        matched, level = CodeMatcher.find_match(content, "completely_different")
        assert matched is None
        assert level == "none"


# ==================== DiffApplier ====================

class TestDiffApplier:

    def test_single_modification(self):
        code = "import os\n\nclass Model:\n    pass"
        mods = [Modification(
            old_string="import os",
            new_string="import os\nimport torch",
            reason="add torch import"
        )]
        result = DiffApplier.apply_modifications(code, mods)
        assert result.success is True
        assert result.applied_count == 1
        assert "import torch" in result.modified_code
        assert "import os" in result.modified_code
        assert len(result.errors) == 0
        assert len(result.diff_text) > 0

    def test_multiple_modifications(self):
        code = "x = 1\ny = 2\nz = 3"
        mods = [
            Modification(old_string="x = 1", new_string="x = 10", reason="fix x"),
            Modification(old_string="z = 3", new_string="z = 30", reason="fix z"),
        ]
        result = DiffApplier.apply_modifications(code, mods)
        assert result.success is True
        assert result.applied_count == 2
        assert "x = 10" in result.modified_code
        assert "z = 30" in result.modified_code
        assert "y = 2" in result.modified_code

    def test_partial_failure(self):
        code = "x = 1\ny = 2"
        mods = [
            Modification(old_string="x = 1", new_string="x = 10", reason="fix x"),
            Modification(old_string="NOT_EXIST", new_string="abc", reason="bad mod"),
        ]
        result = DiffApplier.apply_modifications(code, mods)
        assert result.success is True  # at least 1 succeeded
        assert result.applied_count == 1
        assert len(result.errors) == 1
        assert "x = 10" in result.modified_code

    def test_all_fail(self):
        code = "x = 1"
        mods = [
            Modification(old_string="NOT_EXIST", new_string="abc", reason="bad"),
        ]
        result = DiffApplier.apply_modifications(code, mods)
        assert result.success is False
        assert result.applied_count == 0
        assert result.modified_code == code

    def test_skip_identical_old_new(self):
        code = "x = 1\ny = 2"
        mods = [
            Modification(old_string="x = 1", new_string="x = 1", reason="no-op"),
        ]
        result = DiffApplier.apply_modifications(code, mods)
        assert result.success is False
        assert result.applied_count == 0
        assert len(result.errors) == 1
        assert "相同" in result.errors[0]

    def test_original_code_preserved(self):
        code = "original code"
        mods = [Modification(old_string="original", new_string="modified", reason="test")]
        result = DiffApplier.apply_modifications(code, mods)
        assert result.original_code == "original code"
        assert "modified" in result.modified_code

    def test_diff_text_contains_changes(self):
        code = "line1\nline2\nline3"
        mods = [Modification(old_string="line2", new_string="changed2", reason="test")]
        result = DiffApplier.apply_modifications(code, mods)
        assert "-line2" in result.diff_text
        assert "+changed2" in result.diff_text

    def test_add_lines(self):
        """old_string 和 new_string 行数不同的场景"""
        code = "def foo():\n    pass"
        mods = [Modification(
            old_string="    pass",
            new_string="    x = 1\n    y = 2\n    return x + y",
            reason="add body"
        )]
        result = DiffApplier.apply_modifications(code, mods)
        assert result.success is True
        assert "    x = 1" in result.modified_code
        assert "    return x + y" in result.modified_code

    def test_delete_lines(self):
        """删除代码行的场景"""
        code = "import os\nimport sys\nimport torch\n\nclass Model:\n    pass"
        mods = [Modification(
            old_string="import os\nimport sys\n",
            new_string="",
            reason="remove unused imports"
        )]
        result = DiffApplier.apply_modifications(code, mods)
        assert result.success is True
        assert "import os" not in result.modified_code
        assert "import torch" in result.modified_code


# ==================== parse_modifications ====================

class TestParseModifications:

    def test_full_json(self):
        llm_output = json.dumps({
            "analysis": "found a bug",
            "modifications": [
                {"old_string": "x = 1", "new_string": "x = 2", "reason": "fix value"}
            ],
            "summary": "fixed"
        })
        mods = parse_modifications(llm_output)
        assert len(mods) == 1
        assert mods[0].old_string == "x = 1"
        assert mods[0].new_string == "x = 2"
        assert mods[0].reason == "fix value"

    def test_array_format(self):
        llm_output = json.dumps([
            {"old_string": "a", "new_string": "b", "reason": "r"}
        ])
        mods = parse_modifications(llm_output)
        assert len(mods) == 1

    def test_markdown_wrapped(self):
        llm_output = '```json\n{"modifications": [{"old_string": "x", "new_string": "y"}]}\n```'
        mods = parse_modifications(llm_output)
        assert len(mods) == 1
        assert mods[0].old_string == "x"
        assert mods[0].new_string == "y"
        assert mods[0].reason == ""

    def test_invalid_json(self):
        mods = parse_modifications("this is not json at all")
        assert mods == []

    def test_missing_fields(self):
        llm_output = json.dumps({
            "modifications": [
                {"old_string": "x"},  # missing new_string
                {"new_string": "y"},  # missing old_string
                {"old_string": "a", "new_string": "b"},  # valid
            ]
        })
        mods = parse_modifications(llm_output)
        assert len(mods) == 1
        assert mods[0].old_string == "a"

    def test_empty_modifications(self):
        llm_output = json.dumps({"modifications": []})
        mods = parse_modifications(llm_output)
        assert mods == []

    def test_non_dict_items_skipped(self):
        llm_output = json.dumps({
            "modifications": [
                "not a dict",
                42,
                {"old_string": "a", "new_string": "b"},
            ]
        })
        mods = parse_modifications(llm_output)
        assert len(mods) == 1


# ==================== FixCodeGen 节点 (Mock LLM) ====================

class TestFixCodeGenNode:

    @pytest.mark.asyncio
    async def test_fix_code_gen_node_success(self):
        """Mock LLM 返回正常的修改方案，验证 State 更新"""
        from akg_agents.op.langgraph_op.nodes import NodeFactory

        llm_response = json.dumps({
            "analysis": "缺少 import torch",
            "modifications": [
                {
                    "old_string": "class ModelNew(nn.Module):",
                    "new_string": "import torch\nimport torch.nn as nn\n\nclass ModelNew(nn.Module):",
                    "reason": "添加缺失的 import"
                }
            ],
            "summary": "添加了 import 语句"
        })

        mock_trace = MagicMock()
        config = {"agent_model_config": {"fix_code_gen": "standard"}}

        state = {
            "task_id": "test_1",
            "op_name": "relu",
            "session_id": "",
            "dsl": "triton_cuda",
            "backend": "cuda",
            "arch": "a100",
            "framework": "torch",
            "workflow_name": "coder_only",
            "task_desc": "ReLU operator",
            "coder_code": "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x.relu()",
            "verifier_error": "NameError: name 'torch' is not defined",
            "conductor_suggestion": "添加 import torch",
            "step_count": 2,
        }

        with patch("akg_agents.core_v2.agents.AgentBase") as MockAgentBase:
            mock_agent = MagicMock()
            mock_agent.load_template.return_value = MagicMock()
            mock_agent.run_llm = AsyncMock(
                return_value=(llm_response, "prompt_text", "reasoning_text")
            )
            MockAgentBase.return_value = mock_agent

            node_fn = NodeFactory.create_fix_code_gen_node(mock_trace, config)
            result = await node_fn(state)

        assert result["fix_code_gen_success"] is True
        assert "import torch" in result["coder_code"]
        assert "class ModelNew(nn.Module):" in result["coder_code"]
        assert result["step_count"] == 3
        assert "fix_code_gen" in result["agent_history"]

    @pytest.mark.asyncio
    async def test_fix_code_gen_node_parse_failure(self):
        """Mock LLM 返回无法解析的内容"""
        from akg_agents.op.langgraph_op.nodes import NodeFactory

        mock_trace = MagicMock()
        config = {}

        state = {
            "task_id": "test_2",
            "op_name": "relu",
            "session_id": "",
            "dsl": "triton_cuda",
            "backend": "cuda",
            "arch": "a100",
            "framework": "torch",
            "workflow_name": "",
            "task_desc": "",
            "coder_code": "x = 1",
            "verifier_error": "error",
            "conductor_suggestion": "fix it",
            "step_count": 0,
        }

        with patch("akg_agents.core_v2.agents.AgentBase") as MockAgentBase:
            mock_agent = MagicMock()
            mock_agent.load_template.return_value = MagicMock()
            mock_agent.run_llm = AsyncMock(
                return_value=("not valid json", "prompt", "reasoning")
            )
            MockAgentBase.return_value = mock_agent

            node_fn = NodeFactory.create_fix_code_gen_node(mock_trace, config)
            result = await node_fn(state)

        assert result["fix_code_gen_success"] is False
        assert "coder_code" not in result  # 不应修改代码

    @pytest.mark.asyncio
    async def test_fix_code_gen_node_no_code(self):
        """没有待修复的代码"""
        from akg_agents.op.langgraph_op.nodes import NodeFactory

        mock_trace = MagicMock()
        config = {}
        state = {"task_id": "test_3", "coder_code": "", "step_count": 0}

        node_fn = NodeFactory.create_fix_code_gen_node(mock_trace, config)
        result = await node_fn(state)

        assert result["fix_code_gen_success"] is False
        assert "没有待修复的代码" in result["fix_code_gen_message"]
