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
单次生成模式集成测试

测试 TraceSystem 和 FileSystemState 在单次生成模式下的集成：
- coder_only_workflow
- default_workflow (Designer → Coder → Verifier)
- verifier_only_workflow
- connect_all_workflow
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
)


class TestCoderOnlyWorkflow:
    """测试 coder_only_workflow 模式与 Trace 系统集成"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        """创建并初始化 TraceSystem"""
        ts = TraceSystem("test_coder_only", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_coder_only_basic_flow(self, trace_system):
        """测试 coder_only 基本流程：Coder → Verifier → Conductor"""
        # 模拟 Coder 阶段
        coder_action = {"type": "call_coder", "strategy": "basic"}
        coder_result = {
            "success": True,
            "code": "import triton\n@triton.jit\ndef kernel(...): ...",
            "lines": 50,
        }
        coder_node = trace_system.add_node(coder_action, coder_result)
        
        # 验证节点创建
        assert coder_node is not None
        assert trace_system.get_current_node() == coder_node
        
        # 模拟 Verifier 阶段
        verifier_action = {"type": "call_verifier"}
        verifier_result = {
            "success": True,
            "compile_success": True,
            "accuracy_pass": True,
            "performance": 0.75,
        }
        verifier_node = trace_system.add_node(verifier_action, verifier_result)
        
        # 模拟 Conductor 阶段
        conductor_action = {"type": "conductor_decide", "decision": "accept"}
        conductor_result = {"final_status": "completed", "accepted": True}
        conductor_node = trace_system.add_node(conductor_action, conductor_result)
        
        # 验证路径
        path = trace_system.get_path_to_node(conductor_node)
        assert len(path) == 4  # root -> coder -> verifier -> conductor
        assert path[0] == "root"
        assert path[-1] == conductor_node
        
        # 验证完整历史
        history = trace_system.get_full_action_history(conductor_node)
        assert len(history) == 3
        assert history[0].tool_name == "call_coder"
        assert history[1].tool_name == "call_verifier"
        assert history[2].tool_name == "conductor_decide"
    
    def test_coder_only_with_retry(self, trace_system):
        """测试 coder_only 带重试：失败后重试"""
        # 第一次 Coder 尝试 - 失败
        action1 = {"type": "call_coder", "attempt": 1}
        result1 = {"success": False, "error": "compile_error"}
        node1 = trace_system.add_node(action1, result1)
        
        # 验证失败后重试 - 成功
        action2 = {"type": "call_coder", "attempt": 2, "fix": "add_import"}
        result2 = {"success": True, "code": "..."}
        node2 = trace_system.add_node(action2, result2)
        
        # Verifier
        action3 = {"type": "call_verifier"}
        result3 = {"success": True, "performance": 0.8}
        node3 = trace_system.add_node(action3, result3)
        
        # 验证历史包含所有尝试
        history = trace_system.get_full_action_history(node3)
        assert len(history) == 3
        # 第一次失败也在历史中
        assert history[0].result.get("success") is False
        assert history[1].result.get("success") is True
    
    def test_coder_only_state_persistence(self, trace_system, temp_dir):
        """测试状态持久化"""
        # 执行一些操作
        node1 = trace_system.add_node(
            {"type": "call_coder"},
            {"success": True, "code": "..."}
        )
        
        # 保存 thinking 状态
        fs = trace_system.fs
        thinking = ThinkingState(
            node_id=node1,
            turn=1,
            current_plan={"goal": "generate softmax kernel"},
            latest_thinking="Using online softmax algorithm",
        )
        fs.save_thinking(node1, thinking)
        
        # 创建新的 TraceSystem 实例（模拟重启）
        trace_system2 = TraceSystem("test_coder_only", base_dir=temp_dir)
        # 不调用 initialize()，自动加载现有数据
        
        # 验证状态恢复
        assert trace_system2.get_current_node() == node1
        
        loaded_thinking = trace_system2.fs.load_thinking(node1)
        assert loaded_thinking is not None
        assert loaded_thinking.latest_thinking == "Using online softmax algorithm"


class TestDefaultWorkflow:
    """测试 default_workflow 模式：Designer → Coder → Verifier"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_default_workflow", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_default_workflow_full_flow(self, trace_system):
        """测试完整的 Designer → Coder → Verifier 流程"""
        # Designer 阶段
        designer_action = {"type": "call_designer"}
        designer_result = {
            "success": True,
            "design": {
                "algorithm": "online_softmax",
                "block_size": 256,
                "memory_strategy": "shared_memory",
            },
        }
        designer_node = trace_system.add_node(designer_action, designer_result)
        
        # 保存设计方案到 thinking
        fs = trace_system.fs
        thinking = ThinkingState(
            node_id=designer_node,
            turn=1,
            current_plan={
                "algorithm": "online_softmax",
                "steps": ["block_reduction", "normalize"],
            },
            latest_thinking="Chose online softmax for numerical stability",
        )
        fs.save_thinking(designer_node, thinking)
        
        # Coder 阶段
        coder_action = {"type": "call_coder", "design_ref": designer_node}
        coder_result = {
            "success": True,
            "code": "@triton.jit\ndef softmax_kernel(...): ...",
            "lines": 80,
        }
        coder_node = trace_system.add_node(coder_action, coder_result)
        
        # 保存代码文件
        fs.save_node_state(coder_node, NodeState(
            node_id=coder_node,
            turn=2,
            status="running",
        ))
        fs.save_code_file(coder_node, "softmax_kernel.py", coder_result["code"])
        
        # Verifier 阶段
        verifier_action = {"type": "call_verifier"}
        verifier_result = {
            "success": True,
            "compile_success": True,
            "accuracy": 0.9999,
            "performance": 0.85,
        }
        verifier_node = trace_system.add_node(verifier_action, verifier_result)
        
        # 验证完整流程
        path = trace_system.get_path_to_node(verifier_node)
        assert len(path) == 4  # root -> designer -> coder -> verifier
        
        history = trace_system.get_full_action_history(verifier_node)
        assert len(history) == 3
        assert history[0].tool_name == "call_designer"
        assert history[1].tool_name == "call_coder"
        assert history[2].tool_name == "call_verifier"
    
    def test_default_workflow_designer_revision(self, trace_system):
        """测试 Designer 修改设计后的分叉"""
        # Designer 初始设计
        design1 = trace_system.add_node(
            {"type": "call_designer", "version": 1},
            {"success": True, "design": {"algorithm": "basic_softmax"}}
        )
        
        # Coder 实现
        code1 = trace_system.add_node(
            {"type": "call_coder"},
            {"success": True, "code": "..."}
        )
        
        # Verifier 发现性能问题
        verify1 = trace_system.add_node(
            {"type": "call_verifier"},
            {"success": True, "performance": 0.5}  # 性能不达标
        )
        
        # 回到 Designer 节点，创建新设计分叉
        trace_system.switch_node(design1)
        
        # 新的设计方案
        design2 = trace_system.add_node(
            {"type": "call_designer", "version": 2, "optimization": "online"},
            {"success": True, "design": {"algorithm": "online_softmax"}}
        )
        
        # 在新设计上继续
        code2 = trace_system.add_node(
            {"type": "call_coder"},
            {"success": True, "code": "...optimized..."}
        )
        
        verify2 = trace_system.add_node(
            {"type": "call_verifier"},
            {"success": True, "performance": 0.9}  # 性能达标
        )
        
        # 验证两条路径
        path1 = trace_system.get_path_to_node(verify1)
        path2 = trace_system.get_path_to_node(verify2)
        
        assert len(path1) == 4
        assert len(path2) == 4
        
        # 对比两条路径
        comparison = trace_system.compare_nodes(verify1, verify2)
        assert comparison["fork_point"] == design1  # 分叉点是 design1


class TestVerifierOnlyWorkflow:
    """测试 verifier_only_workflow 模式"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_verifier_only", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_verifier_only_basic(self, trace_system):
        """测试仅验证模式"""
        # 仅 Verifier 阶段
        verifier_action = {
            "type": "call_verifier",
            "input_code_path": "/path/to/existing_kernel.py",
        }
        verifier_result = {
            "success": True,
            "compile_success": True,
            "accuracy": 0.9998,
            "performance": 0.92,
            "performance_vs_baseline": 1.15,
        }
        verifier_node = trace_system.add_node(verifier_action, verifier_result)
        
        # 验证
        path = trace_system.get_path_to_node(verifier_node)
        assert len(path) == 2  # root -> verifier
        
        history = trace_system.get_full_action_history(verifier_node)
        assert len(history) == 1
        assert history[0].tool_name == "call_verifier"


class TestConnectAllWorkflow:
    """测试 connect_all_workflow 模式（全连接）"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_connect_all", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_connect_all_with_multiple_iterations(self, trace_system):
        """测试全连接模式的多轮迭代"""
        # 第一轮：Designer → Coder → Verifier（失败）
        d1 = trace_system.add_node(
            {"type": "call_designer", "round": 1},
            {"success": True, "design": {"v": 1}}
        )
        c1 = trace_system.add_node(
            {"type": "call_coder", "round": 1},
            {"success": True, "code": "v1"}
        )
        v1 = trace_system.add_node(
            {"type": "call_verifier", "round": 1},
            {"success": False, "error": "accuracy_fail"}
        )
        
        # 第二轮：继续优化
        d2 = trace_system.add_node(
            {"type": "call_designer", "round": 2, "feedback": "improve_accuracy"},
            {"success": True, "design": {"v": 2}}
        )
        c2 = trace_system.add_node(
            {"type": "call_coder", "round": 2},
            {"success": True, "code": "v2"}
        )
        v2 = trace_system.add_node(
            {"type": "call_verifier", "round": 2},
            {"success": True, "accuracy": 0.999}
        )
        
        # 验证完整路径
        path = trace_system.get_path_to_node(v2)
        assert len(path) == 7  # root + 6 nodes
        
        history = trace_system.get_full_action_history(v2)
        assert len(history) == 6
        
        # 验证迭代历史可追溯
        rounds = [h.arguments.get("round") for h in history]
        assert rounds == [1, 1, 1, 2, 2, 2]
    
    def test_connect_all_with_fork_for_alternative(self, trace_system):
        """测试全连接模式中的分叉探索"""
        # 初始设计
        d1 = trace_system.add_node(
            {"type": "call_designer"},
            {"success": True, "design": {"approach": "A"}}
        )
        c1 = trace_system.add_node(
            {"type": "call_coder"},
            {"success": True}
        )
        v1 = trace_system.add_node(
            {"type": "call_verifier"},
            {"success": True, "performance": 0.7}
        )
        
        # 回到设计阶段尝试另一种方案
        trace_system.switch_node(d1)
        
        d2 = trace_system.add_node(
            {"type": "call_designer", "alternative": True},
            {"success": True, "design": {"approach": "B"}}
        )
        c2 = trace_system.add_node(
            {"type": "call_coder"},
            {"success": True}
        )
        v2 = trace_system.add_node(
            {"type": "call_verifier"},
            {"success": True, "performance": 0.9}
        )
        
        # 验证两条路径
        comparison = trace_system.compare_nodes(v1, v2)
        assert comparison["fork_point"] == d1
        
        # 验证 metrics 对比
        node_v1 = trace_system.get_node(v1)
        node_v2 = trace_system.get_node(v2)
        assert node_v2["result"]["performance"] > node_v1["result"]["performance"]


class TestResumeFromCheckpoint:
    """测试断点续跑功能"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_resume_single_mode_execution(self, temp_dir):
        """测试单次模式执行中断后恢复"""
        # 第一次执行（模拟中断）
        ts1 = TraceSystem("test_resume_single", base_dir=temp_dir)
        ts1.initialize()
        
        # 执行部分流程
        d1 = ts1.add_node(
            {"type": "call_designer"},
            {"success": True, "design": {"algorithm": "softmax"}}
        )
        c1 = ts1.add_node(
            {"type": "call_coder"},
            {"success": True, "code": "..."}
        )
        
        # 保存 pending tools（模拟中断前的状态）
        pending = ts1.fs.load_pending_tools(c1)
        pending.add_pending_tool(PendingTool(
            tool_call_id="call_123",
            tool_name="call_verifier",
            arguments={},
        ))
        ts1.fs.save_pending_tools(c1, pending)
        
        last_node = ts1.get_current_node()
        
        # 模拟重启 - 创建新实例
        ts2 = TraceSystem("test_resume_single", base_dir=temp_dir)
        
        # 验证恢复
        assert ts2.get_current_node() == last_node
        
        # 检查 pending tools
        pending2 = ts2.fs.load_pending_tools(c1)
        assert len(pending2.pending_tools) == 1
        assert pending2.pending_tools[0].tool_name == "call_verifier"
        
        # 继续执行
        v1 = ts2.add_node(
            {"type": "call_verifier"},
            {"success": True, "performance": 0.85}
        )
        
        # 标记 pending tool 完成
        ts2.fs.mark_tool_completed(c1, "call_123")
        
        # 验证完整历史
        history = ts2.get_full_action_history(v1)
        assert len(history) == 3
    
    def test_resume_from_specific_node(self, temp_dir):
        """测试从指定节点恢复"""
        ts = TraceSystem("test_resume_specific", base_dir=temp_dir)
        ts.initialize()
        
        # 执行一系列操作
        n1 = ts.add_node({"type": "step1"}, {"success": True})
        n2 = ts.add_node({"type": "step2"}, {"success": True})
        n3 = ts.add_node({"type": "step3"}, {"success": False})
        
        # 切换回 n2 重试
        ts.switch_node(n2)
        
        # 从 n2 创建新分支
        n4 = ts.add_node({"type": "step3_retry"}, {"success": True})
        
        # 验证当前节点
        assert ts.get_current_node() == n4
        
        # 验证树结构
        node_n2 = ts.get_node(n2)
        assert n3 in node_n2.children
        assert n4 in node_n2.children


class TestCodeFileManagement:
    """测试代码文件管理在集成场景中的使用"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_code_evolution_across_nodes(self, temp_dir):
        """测试代码在节点间的演进"""
        ts = TraceSystem("test_code_evolution", base_dir=temp_dir)
        ts.initialize()
        fs = ts.fs
        
        # 初始代码
        n1 = ts.add_node(
            {"type": "call_coder", "version": 1},
            {"success": True}
        )
        fs.save_node_state(n1, NodeState(node_id=n1, turn=1, status="completed"))
        fs.save_code_file(n1, "kernel.py", "# version 1\ndef kernel(): pass")
        
        # 优化代码
        n2 = ts.add_node(
            {"type": "call_coder", "version": 2, "optimization": "vectorize"},
            {"success": True}
        )
        fs.save_node_state(n2, NodeState(node_id=n2, turn=2, status="completed"))
        # 复制并修改代码
        original_code = fs.load_code_file(n1, "kernel.py")
        optimized_code = original_code.replace("version 1", "version 2 - vectorized")
        fs.save_code_file(n2, "kernel.py", optimized_code)
        
        # 验证两个版本的代码
        code_v1 = fs.load_code_file(n1, "kernel.py")
        code_v2 = fs.load_code_file(n2, "kernel.py")
        
        assert "version 1" in code_v1
        assert "version 2 - vectorized" in code_v2
        assert code_v1 != code_v2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
