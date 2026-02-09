# Copyright 2025 Huawei Technologies Co., Ltd
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
FileSystemState 单元测试
"""

import json
import pytest
import shutil
import tempfile
from pathlib import Path

from akg_agents.core_v2.filesystem import (
    FileSystemState,
    NodeState,
    ActionRecord,
    ActionHistoryFact,
    ThinkingState,
    PendingTool,
    PendingToolsState,
    NodeNotFoundError,
)


class TestFileSystemStateInitialization:
    """测试 FileSystemState 初始化"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def fs(self, temp_dir):
        """创建 FileSystemState 实例"""
        return FileSystemState("test_task_001", base_dir=temp_dir)
    
    def test_initialize_task_creates_directories(self, fs):
        """测试 initialize_task 创建目录结构"""
        fs.initialize_task()
        
        assert fs.task_dir.exists()
        assert fs.nodes_dir.exists()
        assert fs.logs_dir.exists()
        assert (fs.nodes_dir / "root").exists()
    
    def test_initialize_task_creates_root_node(self, fs):
        """测试 initialize_task 创建 root 节点"""
        fs.initialize_task()
        
        root_state = fs.load_node_state("root")
        assert root_state.node_id == "root"
        assert root_state.turn == 0
        assert root_state.status == "init"
    
    def test_initialize_task_sets_current_node(self, fs):
        """测试 initialize_task 设置当前节点"""
        fs.initialize_task()
        
        assert fs.get_current_node() == "root"
    
    def test_initialize_task_force_reinitialize(self, fs):
        """测试强制重新初始化"""
        fs.initialize_task()
        
        # 添加一些数据
        state = NodeState(node_id="node_001", turn=1, status="running")
        fs.save_node_state("node_001", state)
        
        # 强制重新初始化
        fs.initialize_task(force=True)
        
        # node_001 应该不存在了
        assert not fs.node_exists("node_001")
    
    def test_task_exists(self, fs):
        """测试 task_exists"""
        assert not fs.task_exists()
        
        fs.initialize_task()
        
        assert fs.task_exists()


class TestNodeStateManagement:
    """测试节点状态管理"""
    
    @pytest.fixture
    def fs(self):
        """创建初始化的 FileSystemState"""
        temp = tempfile.mkdtemp()
        fs = FileSystemState("test_task_002", base_dir=temp)
        fs.initialize_task()
        yield fs
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_save_load_node_state(self, fs):
        """测试保存和加载节点状态"""
        state = NodeState(
            node_id="node_001",
            turn=1,
            status="running",
            agent_info={"agent_name": "KernelAgent"},
            task_info={"task_id": "test_task_002"},
            metrics={"token_used": 1000},
        )
        
        fs.save_node_state("node_001", state)
        loaded = fs.load_node_state("node_001")
        
        assert loaded.node_id == "node_001"
        assert loaded.turn == 1
        assert loaded.status == "running"
        assert loaded.agent_info["agent_name"] == "KernelAgent"
        assert loaded.metrics["token_used"] == 1000
    
    def test_load_nonexistent_node_raises_error(self, fs):
        """测试加载不存在的节点抛出异常"""
        with pytest.raises(NodeNotFoundError):
            fs.load_node_state("nonexistent_node")
    
    def test_update_node_state(self, fs):
        """测试更新节点状态"""
        state = NodeState(node_id="node_001", turn=1, status="running")
        fs.save_node_state("node_001", state)
        
        updated = fs.update_node_state(
            "node_001",
            turn=2,
            status="completed",
            metrics={"performance": 0.85}
        )
        
        assert updated.turn == 2
        assert updated.status == "completed"
        assert updated.metrics["performance"] == 0.85
    
    def test_node_exists(self, fs):
        """测试 node_exists"""
        assert not fs.node_exists("node_001")
        
        state = NodeState(node_id="node_001", turn=1, status="running")
        fs.save_node_state("node_001", state)
        
        assert fs.node_exists("node_001")


class TestActionHistoryManagement:
    """测试动作历史管理"""
    
    @pytest.fixture
    def fs(self):
        """创建初始化的 FileSystemState"""
        temp = tempfile.mkdtemp()
        fs = FileSystemState("test_task_003", base_dir=temp)
        fs.initialize_task()
        yield fs
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_save_load_action_history_fact(self, fs):
        """测试保存和加载增量动作历史"""
        action = ActionRecord(
            action_id="action_001",
            tool_name="call_coder",
            arguments={"strategy": "optimize"},
            result={"success": True},
        )
        
        history = ActionHistoryFact(
            node_id="node_001",
            parent_node_id="root",
            turn=1,
        )
        history.add_action(action)
        
        fs.save_action_history_fact("node_001", history)
        loaded = fs.load_action_history_fact("node_001")
        
        assert loaded.node_id == "node_001"
        assert loaded.parent_node_id == "root"
        assert len(loaded.actions) == 1
        assert loaded.actions[0].tool_name == "call_coder"
    
    def test_load_nonexistent_action_history_returns_empty(self, fs):
        """测试加载不存在的动作历史返回空历史"""
        history = fs.load_action_history_fact("nonexistent_node")
        
        assert history.node_id == "nonexistent_node"
        assert len(history.actions) == 0
    
    def test_append_action(self, fs):
        """测试追加动作"""
        action1 = ActionRecord(
            action_id="action_001",
            tool_name="call_designer",
            result={"success": True},
        )
        
        fs.append_action("node_001", action1, parent_node_id="root", turn=1)
        
        action2 = ActionRecord(
            action_id="action_002",
            tool_name="call_coder",
            result={"success": True},
        )
        
        fs.append_action("node_001", action2, turn=1)
        
        history = fs.load_action_history_fact("node_001")
        
        assert len(history.actions) == 2
        assert history.actions[0].tool_name == "call_designer"
        assert history.actions[1].tool_name == "call_coder"
    
    def test_incremental_save_no_duplicate(self, fs):
        """测试增量保存不重复"""
        # 模拟 node_001 的动作
        action1 = ActionRecord(
            action_id="action_001",
            tool_name="call_designer",
            result={"success": True},
        )
        history1 = ActionHistoryFact(
            node_id="node_001",
            parent_node_id="root",
            turn=1,
        )
        history1.add_action(action1)
        fs.save_action_history_fact("node_001", history1)
        
        # 模拟 node_002 的动作（只保存 node_002 自己的）
        action2 = ActionRecord(
            action_id="action_002",
            tool_name="call_coder",
            result={"success": True},
        )
        history2 = ActionHistoryFact(
            node_id="node_002",
            parent_node_id="node_001",
            turn=2,
        )
        history2.add_action(action2)
        fs.save_action_history_fact("node_002", history2)
        
        # 验证 node_002 只有自己的动作，没有 node_001 的
        loaded2 = fs.load_action_history_fact("node_002")
        assert len(loaded2.actions) == 1
        assert loaded2.actions[0].action_id == "action_002"


class TestThinkingManagement:
    """测试 Thinking 管理"""
    
    @pytest.fixture
    def fs(self):
        """创建初始化的 FileSystemState"""
        temp = tempfile.mkdtemp()
        fs = FileSystemState("test_task_004", base_dir=temp)
        fs.initialize_task()
        yield fs
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_save_load_thinking(self, fs):
        """测试保存和加载 Thinking"""
        thinking = ThinkingState(
            node_id="node_001",
            turn=1,
            current_plan={"goal": "实现 softmax", "steps": ["设计", "编码", "验证"]},
            latest_thinking="当前需要优化内存访问...",
        )
        
        fs.save_thinking("node_001", thinking)
        loaded = fs.load_thinking("node_001")
        
        assert loaded.node_id == "node_001"
        assert loaded.current_plan["goal"] == "实现 softmax"
        assert loaded.latest_thinking == "当前需要优化内存访问..."
    
    def test_load_nonexistent_thinking_returns_none(self, fs):
        """测试加载不存在的 Thinking 返回 None"""
        thinking = fs.load_thinking("nonexistent_node")
        assert thinking is None
    
    def test_update_thinking(self, fs):
        """测试更新 Thinking"""
        # 先创建节点状态
        state = NodeState(node_id="node_001", turn=1, status="running")
        fs.save_node_state("node_001", state)
        
        # 更新 thinking
        updated = fs.update_thinking(
            "node_001",
            latest_thinking="决定使用 shared memory",
            decision="采用 shared memory 优化",
            current_plan={"goal": "优化性能"}
        )
        
        assert updated.latest_thinking == "决定使用 shared memory"
        assert len(updated.decision_history) == 1
        assert updated.current_plan["goal"] == "优化性能"


class TestPendingToolsManagement:
    """测试待执行工具管理"""
    
    @pytest.fixture
    def fs(self):
        """创建初始化的 FileSystemState"""
        temp = tempfile.mkdtemp()
        fs = FileSystemState("test_task_005", base_dir=temp)
        fs.initialize_task()
        yield fs
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_save_load_pending_tools(self, fs):
        """测试保存和加载待执行工具"""
        tool = PendingTool(
            tool_call_id="call_001",
            tool_name="file_write",
            arguments={"path": "output.cu", "content": "..."},
        )
        
        pending = PendingToolsState(node_id="node_001", turn=1)
        pending.add_pending_tool(tool)
        
        fs.save_pending_tools("node_001", pending)
        loaded = fs.load_pending_tools("node_001")
        
        assert loaded.node_id == "node_001"
        assert len(loaded.pending_tools) == 1
        assert loaded.pending_tools[0].tool_name == "file_write"
    
    def test_pending_tools_state_mark_completed_returns_bool(self, fs):
        """测试 PendingToolsState.mark_completed 返回正确的布尔值"""
        tool = PendingTool(
            tool_call_id="call_001",
            tool_name="file_write",
            arguments={},
        )
        
        pending = PendingToolsState(node_id="node_001", turn=1)
        pending.add_pending_tool(tool)
        original_timestamp = pending.last_updated
        
        # 标记存在的工具 - 应该返回 True
        result = pending.mark_completed("call_001")
        assert result is True
        assert pending.pending_tools[0].status == "completed"
        assert pending.last_updated != original_timestamp  # timestamp 应该更新
        
        # 标记不存在的工具 - 应该返回 False
        updated_timestamp = pending.last_updated
        result = pending.mark_completed("nonexistent")
        assert result is False
        assert pending.last_updated == updated_timestamp  # timestamp 不应该更新
    
    def test_add_pending_tool(self, fs):
        """测试添加待执行工具"""
        tool = PendingTool(
            tool_call_id="call_001",
            tool_name="file_write",
            arguments={"path": "output.cu"},
        )
        
        fs.add_pending_tool("node_001", tool, turn=1)
        
        loaded = fs.load_pending_tools("node_001")
        assert len(loaded.pending_tools) == 1
    
    def test_mark_tool_completed(self, fs):
        """测试标记工具完成"""
        tool = PendingTool(
            tool_call_id="call_001",
            tool_name="file_write",
            arguments={},
        )
        
        fs.add_pending_tool("node_001", tool, turn=1)
        result = fs.mark_tool_completed("node_001", "call_001")
        
        loaded = fs.load_pending_tools("node_001")
        assert loaded.pending_tools[0].status == "completed"
    
    def test_mark_tool_completed_not_found(self, fs):
        """测试标记不存在的工具完成 - 应该不更新 timestamp"""
        tool = PendingTool(
            tool_call_id="call_001",
            tool_name="file_write",
            arguments={},
        )
        
        fs.add_pending_tool("node_001", tool, turn=1)
        original = fs.load_pending_tools("node_001")
        original_timestamp = original.last_updated
        
        # 标记一个不存在的工具
        result = fs.mark_tool_completed("node_001", "nonexistent_call")
        
        # 加载并验证 timestamp 没有变化（因为文件不会被保存）
        loaded = fs.load_pending_tools("node_001")
        assert loaded.last_updated == original_timestamp
        assert loaded.pending_tools[0].status == "pending"  # 原工具状态不变
    
    def test_mark_tool_completed_raise_if_not_found(self, fs):
        """测试标记不存在的工具时抛出异常"""
        from akg_agents.core_v2.filesystem import FileSystemStateError
        
        tool = PendingTool(
            tool_call_id="call_001",
            tool_name="file_write",
            arguments={},
        )
        
        fs.add_pending_tool("node_001", tool, turn=1)
        
        # 应该抛出异常
        with pytest.raises(FileSystemStateError):
            fs.mark_tool_completed("node_001", "nonexistent_call", raise_if_not_found=True)
    
    def test_clear_pending_tools(self, fs):
        """测试清空待执行工具"""
        # 先创建节点状态
        state = NodeState(node_id="node_001", turn=1, status="running")
        fs.save_node_state("node_001", state)
        
        tool = PendingTool(
            tool_call_id="call_001",
            tool_name="file_write",
            arguments={},
        )
        fs.add_pending_tool("node_001", tool, turn=1)
        
        fs.clear_pending_tools("node_001")
        
        loaded = fs.load_pending_tools("node_001")
        assert len(loaded.pending_tools) == 0


class TestSystemPromptManagement:
    """测试系统提示词管理"""
    
    @pytest.fixture
    def fs(self):
        """创建初始化的 FileSystemState"""
        temp = tempfile.mkdtemp()
        fs = FileSystemState("test_task_006", base_dir=temp)
        fs.initialize_task()
        yield fs
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_save_load_system_prompt(self, fs):
        """测试保存和加载系统提示词"""
        prompt = "# Task\n实现 softmax kernel\n\n# Available Tools\n..."
        
        fs.save_system_prompt("node_001", turn=1, prompt=prompt)
        loaded = fs.load_system_prompt("node_001", turn=1)
        
        assert loaded == prompt
    
    def test_load_nonexistent_prompt_returns_none(self, fs):
        """测试加载不存在的提示词返回 None"""
        prompt = fs.load_system_prompt("node_001", turn=1)
        assert prompt is None
    
    def test_get_latest_system_prompt(self, fs):
        """测试获取最新提示词"""
        fs.save_system_prompt("node_001", turn=1, prompt="Prompt 1")
        fs.save_system_prompt("node_001", turn=2, prompt="Prompt 2")
        fs.save_system_prompt("node_001", turn=3, prompt="Prompt 3")
        
        latest = fs.get_latest_system_prompt("node_001")
        assert latest == "Prompt 3"


class TestCodeFileManagement:
    """测试代码文件管理"""
    
    @pytest.fixture
    def fs(self):
        """创建初始化的 FileSystemState"""
        temp = tempfile.mkdtemp()
        fs = FileSystemState("test_task_007", base_dir=temp)
        fs.initialize_task()
        # 创建节点状态
        state = NodeState(node_id="node_001", turn=1, status="running")
        fs.save_node_state("node_001", state)
        yield fs
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_save_load_code_file(self, fs):
        """测试保存和加载代码文件"""
        code = "__global__ void kernel() { ... }"
        
        fs.save_code_file("node_001", "kernel.cu", code)
        loaded = fs.load_code_file("node_001", "kernel.cu")
        
        assert loaded == code
    
    def test_save_code_file_updates_file_state(self, fs):
        """测试保存代码文件更新文件状态"""
        code = "__global__ void kernel() { ... }"
        
        fs.save_code_file("node_001", "kernel.cu", code)
        
        state = fs.load_node_state("node_001")
        assert "code/kernel.cu" in state.file_state
        assert state.file_state["code/kernel.cu"]["size"] == len(code)
    
    def test_list_code_files(self, fs):
        """测试列出代码文件"""
        fs.save_code_file("node_001", "kernel.cu", "code1")
        fs.save_code_file("node_001", "test.py", "code2")
        
        files = fs.list_code_files("node_001")
        
        assert "kernel.cu" in files
        assert "test.py" in files


class TestNodeCopy:
    """测试节点复制"""
    
    @pytest.fixture
    def fs(self):
        """创建初始化的 FileSystemState"""
        temp = tempfile.mkdtemp()
        fs = FileSystemState("test_task_008", base_dir=temp)
        fs.initialize_task()
        yield fs
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_copy_node_state(self, fs):
        """测试复制节点状态"""
        # 创建源节点
        source_state = NodeState(
            node_id="node_001",
            turn=1,
            status="running",
            metrics={"token_used": 1000},
        )
        fs.save_node_state("node_001", source_state)
        
        # 复制到新节点
        new_state = fs.copy_node_state("node_001", "node_002")
        
        assert new_state.node_id == "node_002"
        assert new_state.turn == 1
        assert new_state.metrics["token_used"] == 1000
    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
