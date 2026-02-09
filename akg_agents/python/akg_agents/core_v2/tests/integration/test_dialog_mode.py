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
对话模式集成测试

测试 TraceSystem 和 FileSystemState 在对话式交互模式下的集成：
- MainOpAgent 对话流程
- 用户交互中的节点切换
- HITL (Human-in-the-Loop) pending tools 处理
"""

import pytest
import shutil
import tempfile
from pathlib import Path

from akg_agents.core_v2.filesystem import (
    FileSystemState,
    TraceSystem,
    NodeState,
    ActionRecord,
    ThinkingState,
    PendingTool,
    PendingToolsState,
)


class TestDialogModeBasic:
    """测试对话模式基本功能"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_dialog", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_dialog_basic_flow(self, trace_system):
        """测试基本对话流程"""
        # 用户请求设计
        design_node = trace_system.add_node(
            {"type": "user_request", "intent": "design_softmax"},
            {"success": True, "response": "I'll design a softmax kernel..."}
        )
        
        # Agent 调用 Designer
        designer_result = trace_system.add_node(
            {"type": "call_designer"},
            {"success": True, "design": {"algorithm": "online_softmax"}}
        )
        
        # 用户确认设计
        confirm_node = trace_system.add_node(
            {"type": "user_confirm", "confirmed": True},
            {"success": True, "next_step": "implement"}
        )
        
        # Agent 调用 Coder
        coder_result = trace_system.add_node(
            {"type": "call_coder"},
            {"success": True, "code": "..."}
        )
        
        # 验证对话历史
        history = trace_system.get_full_action_history(coder_result)
        assert len(history) == 4
        
        # 验证可以追溯完整对话流程
        path = trace_system.get_path_to_node(coder_result)
        assert len(path) == 5  # root + 4 nodes
    
    def test_dialog_with_user_feedback(self, trace_system):
        """测试用户反馈引起的分叉"""
        # 初始设计
        design_v1 = trace_system.add_node(
            {"type": "call_designer", "version": 1},
            {"success": True, "design": {"approach": "basic"}}
        )
        
        # 用户反馈：需要更高性能
        feedback = trace_system.add_node(
            {"type": "user_feedback", "feedback": "need higher performance"},
            {"success": True, "action": "redesign"}
        )
        
        # 重新设计
        design_v2 = trace_system.add_node(
            {"type": "call_designer", "version": 2, "optimization": "performance"},
            {"success": True, "design": {"approach": "optimized"}}
        )
        
        # 验证流程记录
        history = trace_system.get_full_action_history(design_v2)
        assert len(history) == 3
        
        # 用户反馈应该在历史中
        feedback_action = history[1]
        assert feedback_action.tool_name == "user_feedback"


class TestDialogSwitchAndFork:
    """测试对话中的节点切换和分叉"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_dialog_switch", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_user_switch_and_fork(self, trace_system):
        """测试用户切换节点并创建分叉"""
        # 完整流程：设计 → 编码 → 验证
        design = trace_system.add_node(
            {"type": "call_designer"},
            {"success": True}
        )
        coder = trace_system.add_node(
            {"type": "call_coder"},
            {"success": True, "lines": 120}
        )
        node_after_coder = coder
        verify1 = trace_system.add_node(
            {"type": "verify"},
            {"success": True, "performance": 0.65}
        )
        
        # 用户不满意性能，想从 coder 阶段尝试另一种方案
        trace_system.switch_node(node_after_coder)
        
        # 创建新分叉
        coder_v2 = trace_system.add_node(
            {"type": "call_coder", "strategy": "shared_memory"},
            {"success": True, "lines": 150}
        )
        verify2 = trace_system.add_node(
            {"type": "verify"},
            {"success": True, "performance": 0.85}
        )
        
        # 对比两个方案
        comparison = trace_system.compare_nodes(verify1, verify2)
        assert comparison["fork_point"] == node_after_coder
        
        # 验证两条路径
        path1 = trace_system.get_path_to_node(verify1)
        path2 = trace_system.get_path_to_node(verify2)
        
        # 公共部分
        assert path1[:3] == path2[:3]  # root -> design -> coder
    
    def test_multiple_user_explorations(self, trace_system):
        """测试多次用户探索"""
        # 基础设计
        base = trace_system.add_node(
            {"type": "design"},
            {"success": True}
        )
        
        # 用户探索方案 A
        trace_system.add_node(
            {"type": "implement", "approach": "A"},
            {"success": True, "performance": 0.6}
        )
        explore_a = trace_system.get_current_node()
        
        # 用户回到基础，探索方案 B
        trace_system.switch_node(base)
        trace_system.add_node(
            {"type": "implement", "approach": "B"},
            {"success": True, "performance": 0.7}
        )
        explore_b = trace_system.get_current_node()
        
        # 用户再次回到基础，探索方案 C
        trace_system.switch_node(base)
        trace_system.add_node(
            {"type": "implement", "approach": "C"},
            {"success": True, "performance": 0.8}
        )
        explore_c = trace_system.get_current_node()
        
        # 验证 base 有 3 个子节点
        base_info = trace_system.get_node(base)
        assert len(base_info.children) == 3
        
        # 获取所有叶节点
        leaves = trace_system.get_all_leaf_nodes()
        assert len(leaves) == 3
        assert explore_a in leaves
        assert explore_b in leaves
        assert explore_c in leaves


class TestHITLPendingTools:
    """测试 HITL 场景下的 pending_tools 处理"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_hitl", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_pending_tools_for_user_confirmation(self, trace_system):
        """测试等待用户确认的 pending tools"""
        fs = trace_system.fs
        
        # Agent 完成设计，等待用户确认
        design_node = trace_system.add_node(
            {"type": "call_designer"},
            {"success": True, "design": {"algorithm": "softmax"}}
        )
        
        # 保存节点状态
        fs.save_node_state(design_node, NodeState(
            node_id=design_node,
            turn=1,
            status="waiting_confirmation",
        ))
        
        # 添加 pending tool - 等待用户确认
        pending = PendingToolsState(node_id=design_node, turn=1)
        pending.add_pending_tool(PendingTool(
            tool_call_id="confirm_001",
            tool_name="ask_user",
            arguments={"question": "确认使用此设计方案?"},
        ))
        fs.save_pending_tools(design_node, pending)
        
        # 模拟用户确认
        loaded_pending = fs.load_pending_tools(design_node)
        assert len(loaded_pending.pending_tools) == 1
        assert loaded_pending.pending_tools[0].status == "pending"
        
        # 用户确认后，标记完成
        fs.mark_tool_completed(design_node, "confirm_001")
        
        # 验证状态更新
        updated = fs.load_pending_tools(design_node)
        assert updated.pending_tools[0].status == "completed"
    
    def test_pending_tools_multiple_confirmations(self, trace_system):
        """测试多个需要确认的 pending tools"""
        fs = trace_system.fs
        
        # 节点需要多个用户确认
        node = trace_system.add_node(
            {"type": "multi_step_action"},
            {"success": True}
        )
        
        fs.save_node_state(node, NodeState(
            node_id=node,
            turn=1,
            status="waiting_multiple",
        ))
        
        # 添加多个 pending tools
        pending = PendingToolsState(node_id=node, turn=1)
        pending.add_pending_tool(PendingTool(
            tool_call_id="file_001",
            tool_name="file_write",
            arguments={"path": "kernel.py"},
        ))
        pending.add_pending_tool(PendingTool(
            tool_call_id="exec_001",
            tool_name="execute_code",
            arguments={"code": "run_test()"},
        ))
        fs.save_pending_tools(node, pending)
        
        # 验证有 2 个 pending
        loaded = fs.load_pending_tools(node)
        assert len(loaded.pending_tools) == 2
        
        # 逐个完成
        fs.mark_tool_completed(node, "file_001")
        loaded = fs.load_pending_tools(node)
        pending_count = sum(1 for t in loaded.pending_tools if t.status == "pending")
        completed_count = sum(1 for t in loaded.pending_tools if t.status == "completed")
        assert pending_count == 1
        assert completed_count == 1
        
        # 完成剩余的
        fs.mark_tool_completed(node, "exec_001")
        loaded = fs.load_pending_tools(node)
        all_completed = all(t.status == "completed" for t in loaded.pending_tools)
        assert all_completed
    
    def test_resume_with_pending_tools(self, temp_dir):
        """测试带 pending tools 的断点续跑"""
        # 第一次执行
        ts1 = TraceSystem("test_resume_pending", base_dir=temp_dir)
        ts1.initialize()
        fs1 = ts1.fs
        
        # 执行到中途，有待处理的工具
        node = ts1.add_node(
            {"type": "action"},
            {"success": True}
        )
        
        fs1.save_node_state(node, NodeState(
            node_id=node,
            turn=1,
            status="waiting",
        ))
        
        pending = PendingToolsState(node_id=node, turn=1)
        pending.add_pending_tool(PendingTool(
            tool_call_id="pending_001",
            tool_name="ask_user",
            arguments={"question": "Continue?"},
        ))
        fs1.save_pending_tools(node, pending)
        
        # 模拟重启
        ts2 = TraceSystem("test_resume_pending", base_dir=temp_dir)
        fs2 = ts2.fs
        
        # 验证恢复时能看到 pending tools
        loaded = fs2.load_pending_tools(node)
        assert len(loaded.pending_tools) == 1
        assert loaded.pending_tools[0].status == "pending"
        
        # 继续执行
        fs2.mark_tool_completed(node, "pending_001")
        
        ts2.add_node(
            {"type": "continue"},
            {"success": True}
        )
        
        # 验证完整历史
        history = ts2.get_full_action_history(ts2.get_current_node())
        assert len(history) == 2


class TestDialogThinkingEvolution:
    """测试对话模式下的 thinking 演进"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_dialog_thinking", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_thinking_evolution_in_dialog(self, trace_system):
        """测试对话中 thinking 的演进"""
        fs = trace_system.fs
        
        # 初始设计阶段
        n1 = trace_system.add_node(
            {"type": "design"},
            {"success": True}
        )
        fs.save_node_state(n1, NodeState(node_id=n1, turn=1, status="completed"))
        fs.save_thinking(n1, ThinkingState(
            node_id=n1,
            turn=1,
            current_plan={"goal": "implement softmax"},
            latest_thinking="Starting with basic design",
        ))
        
        # 用户反馈后更新
        n2 = trace_system.add_node(
            {"type": "user_feedback"},
            {"success": True, "feedback": "need optimization"}
        )
        fs.save_node_state(n2, NodeState(node_id=n2, turn=2, status="completed"))
        
        # 加载并更新 thinking
        thinking = fs.update_thinking(
            n2,
            latest_thinking="User wants optimization, considering shared memory",
            decision="Switch to optimized implementation",
            current_plan={"goal": "implement optimized softmax"},
        )
        
        # 继续优化
        n3 = trace_system.add_node(
            {"type": "optimize"},
            {"success": True}
        )
        fs.save_node_state(n3, NodeState(node_id=n3, turn=3, status="completed"))
        
        thinking = fs.update_thinking(
            n3,
            latest_thinking="Implementing shared memory optimization",
            decision="Use shared memory for reduction",
        )
        
        # 验证 decision_history 累积
        final_thinking = fs.load_thinking(n3)
        assert len(final_thinking.decision_history) == 2
    
    def test_thinking_in_different_branches(self, trace_system):
        """测试不同分支中的 thinking 独立性"""
        fs = trace_system.fs
        
        # 基础节点
        base = trace_system.add_node(
            {"type": "base"},
            {"success": True}
        )
        fs.save_node_state(base, NodeState(node_id=base, turn=1, status="completed"))
        fs.save_thinking(base, ThinkingState(
            node_id=base,
            turn=1,
            current_plan={"approach": "undecided"},
            latest_thinking="Considering options",
        ))
        
        # 分支 A
        branch_a = trace_system.add_node(
            {"type": "branch_a"},
            {"success": True}
        )
        fs.save_node_state(branch_a, NodeState(node_id=branch_a, turn=2, status="completed"))
        fs.save_thinking(branch_a, ThinkingState(
            node_id=branch_a,
            turn=2,
            current_plan={"approach": "A"},
            latest_thinking="Trying approach A",
        ))
        
        # 回到基础，创建分支 B
        trace_system.switch_node(base)
        
        branch_b = trace_system.add_node(
            {"type": "branch_b"},
            {"success": True}
        )
        fs.save_node_state(branch_b, NodeState(node_id=branch_b, turn=2, status="completed"))
        fs.save_thinking(branch_b, ThinkingState(
            node_id=branch_b,
            turn=2,
            current_plan={"approach": "B"},
            latest_thinking="Trying approach B",
        ))
        
        # 验证两个分支的 thinking 独立
        thinking_a = fs.load_thinking(branch_a)
        thinking_b = fs.load_thinking(branch_b)
        
        assert thinking_a.current_plan["approach"] == "A"
        assert thinking_b.current_plan["approach"] == "B"


class TestDialogTreeVisualization:
    """测试对话模式下的树可视化"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_dialog_viz", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_visualize_dialog_tree(self, trace_system):
        """测试对话树可视化"""
        # 创建对话树
        d1 = trace_system.add_node({"type": "design"}, {"success": True})
        c1 = trace_system.add_node({"type": "code"}, {"success": True})
        v1 = trace_system.add_node({"type": "verify"}, {"success": True, "perf": 0.6})
        
        # 创建分叉
        trace_system.switch_node(c1)
        c2 = trace_system.add_node({"type": "code_v2"}, {"success": True})
        v2 = trace_system.add_node({"type": "verify"}, {"success": True, "perf": 0.8})
        
        # 获取树可视化
        tree_str = trace_system.visualize_tree()
        
        # 验证包含关键节点
        assert "root" in tree_str
        # 应该显示分叉结构
        assert tree_str is not None
        assert len(tree_str) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
