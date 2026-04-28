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
TaskConstructor agent 单元测试

验证 Agent 注册元数据、接口协议、辅助类（StepRecord / SessionLogger）等，
不依赖 LLM 或 GPU。
"""

import json
import time
from pathlib import Path
from typing import Dict

import pytest

from akg_agents.op.agents.task_constructor import (
    TaskConstructor,
    StepRecord,
    SessionLogger,
)
from akg_agents.core_v2.agents import AgentBase


# ====================== Agent 注册元数据 ======================

class TestTaskConstructorMetadata:
    def test_is_agent_base_subclass(self):
        assert issubclass(TaskConstructor, AgentBase)

    def test_tool_name(self):
        assert TaskConstructor.TOOL_NAME == "call_task_constructor"

    def test_description_not_empty(self):
        assert TaskConstructor.DESCRIPTION
        assert len(TaskConstructor.DESCRIPTION.strip()) > 20

    def test_parameter_schema_structure(self):
        schema = TaskConstructor.PARAMETERS_SCHEMA
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "user_input" in schema["properties"]
        assert "required" in schema
        assert "user_input" in schema["required"]

    def test_parameter_schema_user_input_type(self):
        props = TaskConstructor.PARAMETERS_SCHEMA["properties"]
        assert props["user_input"]["type"] == "string"

    def test_parameter_schema_source_path_optional(self):
        props = TaskConstructor.PARAMETERS_SCHEMA["properties"]
        assert "source_path" in props
        assert props["source_path"].get("default") == ""
        assert "source_path" not in TaskConstructor.PARAMETERS_SCHEMA["required"]

    def test_max_steps_positive(self):
        assert TaskConstructor.MAX_STEPS > 0

    def test_max_retries_positive(self):
        assert TaskConstructor.MAX_RETRIES_PER_STEP > 0

    def test_finish_tool_schema(self):
        ft = TaskConstructor._FINISH_TOOL
        assert ft["type"] == "function"
        func = ft["function"]
        assert func["name"] == "finish"
        params = func["parameters"]["properties"]
        assert "task_code" in params
        assert "summary" in params
        assert "op_name" in params
        assert "error" in params


# ====================== StepRecord 数据类 ======================

class TestStepRecord:
    def test_fields(self):
        sr = StepRecord(
            step=1, thought="think", action="read_file",
            arguments={"path": "/tmp/a.py"},
            result={"status": "success"},
        )
        assert sr.step == 1
        assert sr.thought == "think"
        assert sr.action == "read_file"
        assert sr.arguments == {"path": "/tmp/a.py"}
        assert sr.result == {"status": "success"}
        assert sr.raw_response == ""
        assert isinstance(sr.timestamp, float)

    def test_timestamp_auto_generated(self):
        before = time.time()
        sr = StepRecord(step=0, thought="", action="", arguments={}, result={})
        after = time.time()
        assert before <= sr.timestamp <= after


# ====================== SessionLogger ======================

class TestSessionLogger:
    def test_init_creates_directory(self, tmp_path):
        log_dir = tmp_path / "session_logs"
        logger = SessionLogger(log_dir)
        assert log_dir.exists()
        assert (log_dir / "session.log").exists()

    def test_log_step_writes_jsonl(self, tmp_path):
        logger = SessionLogger(tmp_path)
        logger.log_step(
            step=1, thought="analyze", action="read",
            args={"path": "/a.py"}, result={"status": "ok"}
        )
        msgs_file = tmp_path / "messages.jsonl"
        assert msgs_file.exists()
        with open(msgs_file, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["step"] == 1
        assert entry["action"] == "read"

    def test_log_system_prompt(self, tmp_path):
        logger = SessionLogger(tmp_path)
        logger.log_system_prompt("You are a task constructor.")
        content = (tmp_path / "system_prompt.txt").read_text(encoding="utf-8")
        assert "task constructor" in content

    def test_log_initial_message(self, tmp_path):
        logger = SessionLogger(tmp_path)
        logger.log_initial_message("Extract relu from repo")
        content = (tmp_path / "initial_message.txt").read_text(encoding="utf-8")
        assert "relu" in content

    def test_log_final_saves_result(self, tmp_path):
        logger = SessionLogger(tmp_path)
        result = {"status": "success", "task_code": "def kernel(): pass", "output": "done", "messages": []}
        logger.log_final(result)
        saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
        assert saved["status"] == "success"
        assert "messages" not in saved  # only "messages" key is excluded from result.json

    def test_log_final_saves_task_code_file(self, tmp_path):
        logger = SessionLogger(tmp_path)
        result = {"status": "success", "task_code": "import torch\ndef kernel(): pass"}
        logger.log_final(result)
        assert (tmp_path / "task_output.py").exists()
        code = (tmp_path / "task_output.py").read_text(encoding="utf-8")
        assert "import torch" in code

    def test_safe_truncate(self):
        long_str = "x" * 1000
        result = SessionLogger._safe_truncate({"key": long_str}, max_len=100)
        assert len(result["key"]) < 200
        assert "chars)" in result["key"]

    def test_safe_truncate_short_string(self):
        result = SessionLogger._safe_truncate({"key": "short"}, max_len=100)
        assert result["key"] == "short"

    def test_log_llm_call(self, tmp_path):
        logger = SessionLogger(tmp_path)
        logger.log_llm_call(step=1, msg_count=3, response="I will use read_file")
        log_content = (tmp_path / "session.log").read_text(encoding="utf-8")
        assert "LLM Call" in log_content

    def test_log_final_saves_prompt_final(self, tmp_path):
        logger = SessionLogger(tmp_path)
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        logger.log_llm_call(step=1, msg_count=2, response="ok", messages=messages)
        logger.log_final({"status": "success"})
        assert (tmp_path / "prompt_final.json").exists()
        saved = json.loads((tmp_path / "prompt_final.json").read_text(encoding="utf-8"))
        assert len(saved) == 2
        assert saved[0]["role"] == "system"


# ====================== TaskConstructor 实例化 ======================

class TestTaskConstructorInit:
    def test_instantiation(self):
        tc = TaskConstructor()
        assert tc.messages == []
        assert tc.history == []
        assert tc._llm_client is None
        assert tc.output_dir is None
        assert tc.workspace_dir is None

    def test_build_tools_for_api(self):
        tc = TaskConstructor()
        tools = tc._build_tools_for_api()
        assert isinstance(tools, list)
        assert len(tools) >= 1
        tool_names = [t["function"]["name"] for t in tools]
        assert "finish" in tool_names

    def test_init_workspace(self, tmp_path):
        tc = TaskConstructor()
        tc._init_workspace(cur_path=str(tmp_path))
        assert tc.output_dir is not None
        assert tc.output_dir.exists()
        assert tc.workspace_dir is not None
        assert tc.workspace_dir.exists()
        assert tc.session_log is not None
