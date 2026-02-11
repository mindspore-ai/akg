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
TraceSystem 单元测试
"""

import json
import pytest
import shutil
import tempfile
from pathlib import Path

from akg_agents.core_v2.filesystem import (
    TraceSystem,
    FileSystemState,
    NodeNotFoundError,
    TraceNotInitializedError,
)


class TestTraceSystemInitialization:
    """测试 TraceSystem 初始化"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace(self, temp_dir):
        """创建 TraceSystem 实例"""
        return TraceSystem("test_task_001", base_dir=temp_dir)
    
    def test_initialize_creates_trace_json(self, trace, temp_dir):
        """测试 initialize 创建 trace.json"""
        trace.initialize()
        
        trace_file = Path(temp_dir) / "conversations" / "test_task_001" / "trace.json"
        assert trace_file.exists()
    
    def test_initialize_creates_root_node(self, trace):
        """测试 initialize 创建 root 节点"""
        trace.initialize()
        
        root = trace.get_node("root")
        assert root.node_id == "root"
        assert root.parent_id is None
        assert root.children == []
    
    def test_initialize_sets_current_node_to_root(self, trace):
        """测试 initialize 设置当前节点为 root"""
        trace.initialize()
        
        assert trace.get_current_node() == "root"
    
    def test_initialize_force_reinitialize(self, trace):
        """测试强制重新初始化"""
        trace.initialize()
        
        # 添加节点
        trace.add_node({"type": "test"}, {"success": True})
        
        # 强制重新初始化
        trace.initialize(force=True)
        
        assert trace.get_current_node() == "root"
        assert len(trace.trace.tree) == 1  # 只有 root
    
    def test_load_existing_trace(self, trace, temp_dir):
        """测试加载已存在的 Trace"""
        trace.initialize()
        trace.add_node({"type": "test"}, {"success": True})
        node_id = trace.get_current_node()
        
        # 创建新的 TraceSystem 实例
        trace2 = TraceSystem("test_task_001", base_dir=temp_dir)
        trace2.initialize()
        
        # 应该加载现有状态
        assert trace2.get_current_node() == node_id


class TestNodeOperations:
    """测试节点操作"""
    
    @pytest.fixture
    def trace(self):
        """创建初始化的 TraceSystem"""
        temp = tempfile.mkdtemp()
        trace = TraceSystem("test_task_002", base_dir=temp)
        trace.initialize()
        yield trace
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_add_single_node(self, trace):
        """测试添加单个节点"""
        node_id = trace.add_node(
            action={"type": "call_designer"},
            result={"success": True, "output": "设计方案..."}
        )
        
        assert node_id == "node_001"
        assert trace.get_current_node() == "node_001"
        
        node = trace.get_node(node_id)
        assert node.parent_id == "root"
        assert node.action["type"] == "call_designer"
        assert node.result["success"] is True
    
    def test_add_multiple_nodes_chain(self, trace):
        """测试链式添加多个节点"""
        node1 = trace.add_node({"type": "call_designer"}, {"success": True})
        node2 = trace.add_node({"type": "call_coder"}, {"success": True})
        node3 = trace.add_node({"type": "verify"}, {"success": True, "performance": 0.85})
        
        assert node1 == "node_001"
        assert node2 == "node_002"
        assert node3 == "node_003"
        
        # 验证父子关系
        assert trace.get_node(node1).parent_id == "root"
        assert trace.get_node(node2).parent_id == node1
        assert trace.get_node(node3).parent_id == node2
        
        # 验证 children
        assert trace.get_node("root").children == [node1]
        assert trace.get_node(node1).children == [node2]
        assert trace.get_node(node2).children == [node3]
    
    def test_switch_node(self, trace):
        """测试切换节点"""
        node1 = trace.add_node({"type": "call_designer"}, {"success": True})
        node2 = trace.add_node({"type": "call_coder"}, {"success": True})
        
        trace.switch_node(node1)
        
        assert trace.get_current_node() == node1
    
    def test_switch_to_nonexistent_node_raises_error(self, trace):
        """测试切换到不存在的节点抛出异常"""
        with pytest.raises(NodeNotFoundError):
            trace.switch_node("nonexistent_node")
    
    def test_get_nonexistent_node_raises_error(self, trace):
        """测试获取不存在的节点抛出异常"""
        with pytest.raises(NodeNotFoundError):
            trace.get_node("nonexistent_node")


class TestAutoFork:
    """测试自动分叉"""
    
    @pytest.fixture
    def trace(self):
        """创建初始化的 TraceSystem"""
        temp = tempfile.mkdtemp()
        trace = TraceSystem("test_task_003", base_dir=temp)
        trace.initialize()
        yield trace
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_auto_fork_when_adding_to_node_with_children(self, trace):
        """测试在有子节点的节点上添加时自动分叉"""
        # 创建初始路径
        node1 = trace.add_node({"type": "call_designer"}, {"success": True})
        node2 = trace.add_node({"type": "call_coder"}, {"success": True})
        node3 = trace.add_node({"type": "verify"}, {"success": True, "performance": 0.65})
        
        # 切换回 node2
        trace.switch_node(node2)
        
        # 添加新节点（应该创建分叉）
        node4 = trace.add_node(
            {"type": "call_coder", "strategy": "shared_memory"},
            {"success": True, "performance": 0.85}
        )
        
        # 验证分叉
        assert node4 == "node_004"
        assert trace.get_node(node2).children == [node3, node4]
        
        # node3 和 node4 有相同的父节点
        assert trace.get_node(node3).parent_id == node2
        assert trace.get_node(node4).parent_id == node2
    
    def test_multiple_forks(self, trace):
        """测试多次分叉"""
        node1 = trace.add_node({"type": "design"}, {"success": True})
        
        # 从 node1 创建第一个分叉
        node2 = trace.add_node({"type": "code_v1"}, {"success": True})
        
        # 切换回 node1，创建第二个分叉
        trace.switch_node(node1)
        node3 = trace.add_node({"type": "code_v2"}, {"success": True})
        
        # 切换回 node1，创建第三个分叉
        trace.switch_node(node1)
        node4 = trace.add_node({"type": "code_v3"}, {"success": True})
        
        # 验证 node1 有三个子节点
        assert len(trace.get_node(node1).children) == 3
        assert set(trace.get_node(node1).children) == {node2, node3, node4}


class TestPathOperations:
    """测试路径操作"""
    
    @pytest.fixture
    def trace(self):
        """创建初始化的 TraceSystem"""
        temp = tempfile.mkdtemp()
        trace = TraceSystem("test_task_004", base_dir=temp)
        trace.initialize()
        yield trace
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_get_path_to_node(self, trace):
        """测试获取到节点的路径"""
        node1 = trace.add_node({"type": "step1"}, {"success": True})
        node2 = trace.add_node({"type": "step2"}, {"success": True})
        node3 = trace.add_node({"type": "step3"}, {"success": True})
        
        path = trace.get_path_to_node(node3)
        
        assert path == ["root", node1, node2, node3]
    
    def test_get_path_to_root(self, trace):
        """测试获取到 root 的路径"""
        path = trace.get_path_to_node("root")
        
        assert path == ["root"]
    
    def test_get_path_to_nonexistent_node(self, trace):
        """测试获取不存在节点的路径 - 应该返回空列表"""
        path = trace.get_path_to_node("nonexistent_node")
        
        assert path == []
    
    def test_get_node_depth(self, trace):
        """测试获取节点深度"""
        node1 = trace.add_node({"type": "step1"}, {"success": True})
        node2 = trace.add_node({"type": "step2"}, {"success": True})
        
        assert trace.get_node_depth("root") == 0
        assert trace.get_node_depth(node1) == 1
        assert trace.get_node_depth(node2) == 2


class TestActionHistory:
    """测试动作历史"""
    
    @pytest.fixture
    def trace(self):
        """创建初始化的 TraceSystem"""
        temp = tempfile.mkdtemp()
        trace = TraceSystem("test_task_005", base_dir=temp)
        trace.initialize()
        yield trace
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_get_full_action_history(self, trace):
        """测试获取完整动作历史"""
        node1 = trace.add_node({"type": "call_designer"}, {"success": True})
        node2 = trace.add_node({"type": "call_coder"}, {"success": True})
        node3 = trace.add_node({"type": "verify"}, {"success": True})
        
        history = trace.get_full_action_history(node3)
        
        assert len(history) == 3
        assert history[0].tool_name == "call_designer"
        assert history[1].tool_name == "call_coder"
        assert history[2].tool_name == "verify"
    
    def test_get_full_action_history_nonexistent_node(self, trace):
        """测试获取不存在节点的动作历史 - 应该返回空列表"""
        history = trace.get_full_action_history("nonexistent_node")
        
        assert history == []
    
    def test_full_action_history_with_fork(self, trace):
        """测试有分叉时的完整动作历史"""
        # 创建分叉
        node1 = trace.add_node({"type": "design"}, {"success": True})
        node2 = trace.add_node({"type": "code_v1"}, {"success": True})
        
        trace.switch_node(node1)
        node3 = trace.add_node({"type": "code_v2"}, {"success": True})
        
        # node2 的历史
        history2 = trace.get_full_action_history(node2)
        assert len(history2) == 2
        assert history2[0].tool_name == "design"
        assert history2[1].tool_name == "code_v1"
        
        # node3 的历史
        history3 = trace.get_full_action_history(node3)
        assert len(history3) == 2
        assert history3[0].tool_name == "design"
        assert history3[1].tool_name == "code_v2"


class TestNodeComparison:
    """测试节点对比"""
    
    @pytest.fixture
    def trace(self):
        """创建初始化的 TraceSystem"""
        temp = tempfile.mkdtemp()
        trace = TraceSystem("test_task_006", base_dir=temp)
        trace.initialize()
        yield trace
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_compare_nodes(self, trace):
        """测试对比节点"""
        # 创建分叉路径
        node1 = trace.add_node(
            {"type": "design"}, 
            {"success": True},
            metrics={"token_used": 1000}
        )
        node2 = trace.add_node(
            {"type": "code_v1"}, 
            {"success": True},
            metrics={"token_used": 2000}
        )
        node3 = trace.add_node(
            {"type": "verify"}, 
            {"success": True, "performance": 0.65},
            metrics={"token_used": 500, "performance": 0.65}
        )
        
        trace.switch_node(node2)
        node4 = trace.add_node(
            {"type": "code_v2"}, 
            {"success": True},
            metrics={"token_used": 2500}
        )
        node5 = trace.add_node(
            {"type": "verify"}, 
            {"success": True, "performance": 0.85},
            metrics={"token_used": 500, "performance": 0.85}
        )
        
        # 对比 node3 和 node5
        comparison = trace.compare_nodes(node3, node5)
        
        assert comparison["fork_point"] == node2
        assert comparison["path_1"] == ["root", node1, node2, node3]
        assert comparison["path_2"] == ["root", node1, node2, node4, node5]
        assert comparison["metrics_1"]["performance"] == 0.65
        assert comparison["metrics_2"]["performance"] == 0.85


class TestLeafNodes:
    """测试叶节点"""
    
    @pytest.fixture
    def trace(self):
        """创建初始化的 TraceSystem"""
        temp = tempfile.mkdtemp()
        trace = TraceSystem("test_task_007", base_dir=temp)
        trace.initialize()
        yield trace
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_get_all_leaf_nodes(self, trace):
        """测试获取所有叶节点"""
        node1 = trace.add_node({"type": "step1"}, {"success": True})
        node2 = trace.add_node({"type": "step2"}, {"success": True})
        
        trace.switch_node(node1)
        node3 = trace.add_node({"type": "step2_alt"}, {"success": True})
        
        leaves = trace.get_all_leaf_nodes()
        
        assert set(leaves) == {node2, node3}
    
    def test_get_best_leaf_node(self, trace):
        """测试获取最优叶节点"""
        node1 = trace.add_node(
            {"type": "step1"}, 
            {"success": True},
            metrics={"performance": 0.65}
        )
        
        trace.switch_node("root")
        node2 = trace.add_node(
            {"type": "step1_alt"}, 
            {"success": True},
            metrics={"performance": 0.85}
        )
        
        best = trace.get_best_leaf_node("performance")
        
        assert best == node2


class TestParallelForks:
    """测试并行分叉"""
    
    @pytest.fixture
    def trace(self):
        """创建初始化的 TraceSystem"""
        temp = tempfile.mkdtemp()
        trace = TraceSystem("test_task_008", base_dir=temp)
        trace.initialize()
        yield trace
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_create_parallel_forks(self, trace):
        """测试创建并行分叉"""
        node1 = trace.add_node({"type": "step1"}, {"success": True})
        
        # 创建 3 个并行分叉
        forks = trace.create_parallel_forks(
            n=3,
            action_template={"type": "parallel_step"}
        )
        
        assert len(forks) == 3
        
        # 验证所有分叉的父节点都是 node1
        for fork_id in forks:
            node = trace.get_node(fork_id)
            assert node.parent_id == node1
            assert node.action["type"] == "parallel_step"
            assert node.action["fork_index"] in [0, 1, 2]
        
        # 验证 node1 有 3 个子节点
        assert len(trace.get_node(node1).children) == 3
    
    def test_complete_fork(self, trace):
        """测试完成分叉节点"""
        node1 = trace.add_node({"type": "step1"}, {"success": True})
        
        forks = trace.create_parallel_forks(
            n=2,
            action_template={"type": "parallel"}
        )
        
        # 完成第一个分叉
        trace.complete_fork(
            forks[0],
            result={"success": True, "performance": 0.8},
            metrics={"performance": 0.8}
        )
        
        node = trace.get_node(forks[0])
        assert node.result["performance"] == 0.8
        assert node.state_snapshot["status"] == "completed"


class TestResumeTask:
    """测试任务恢复"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_get_resume_info(self, temp_dir):
        """测试获取恢复信息"""
        trace = TraceSystem("test_task_010", base_dir=temp_dir)
        trace.initialize()
        
        # 添加一些节点
        trace.add_node({"type": "step1"}, {"success": True})
        trace.add_node({"type": "step2"}, {"success": True})
        current_node = trace.get_current_node()
        
        # 获取恢复信息
        info = trace.get_resume_info()
        
        assert info["task_id"] == "test_task_010"
        assert info["current_node"] == current_node
        assert info["state"] is not None
        assert len(info["action_history"]) == 2
        assert len(info["path"]) == 3  # root + 2 nodes
    
    def test_resume_from_existing_trace(self, temp_dir):
        """测试从现有 Trace 恢复"""
        # 第一次运行
        trace1 = TraceSystem("test_task_011", base_dir=temp_dir)
        trace1.initialize()
        trace1.add_node({"type": "step1"}, {"success": True})
        trace1.add_node({"type": "step2"}, {"success": True})
        original_node = trace1.get_current_node()
        
        # 模拟重启 - 创建新实例
        trace2 = TraceSystem("test_task_011", base_dir=temp_dir)
        trace2.initialize()  # 应该加载现有状态
        
        # 验证恢复
        assert trace2.get_current_node() == original_node
        
        history = trace2.get_full_action_history(original_node)
        assert len(history) == 2
        
        # 继续添加节点
        new_node = trace2.add_node({"type": "step3"}, {"success": True})
        
        history2 = trace2.get_full_action_history(new_node)
        assert len(history2) == 3
    
    def test_resume_preserves_action_counter(self, temp_dir):
        """测试恢复时动作计数器正确更新 - 避免 ID 冲突"""
        # 第一次运行 - 创建多个节点和动作
        trace1 = TraceSystem("test_task_action_counter", base_dir=temp_dir)
        trace1.initialize()
        
        node1 = trace1.add_node({"type": "step1"}, {"success": True})
        node2 = trace1.add_node({"type": "step2"}, {"success": True})
        node3 = trace1.add_node({"type": "step3"}, {"success": True})
        
        # 记录最后一个动作 ID
        history1 = trace1.get_full_action_history(node3)
        last_action_id = history1[-1].action_id
        
        # 模拟重启 - 创建新实例
        trace2 = TraceSystem("test_task_action_counter", base_dir=temp_dir)
        trace2.initialize()  # 应该加载现有状态并更新计数器
        
        # 添加新节点
        node4 = trace2.add_node({"type": "step4"}, {"success": True})
        
        # 获取新动作的 ID
        history2 = trace2.get_full_action_history(node4)
        new_action_id = history2[-1].action_id
        
        # 验证新动作 ID 大于之前的动作 ID（没有冲突）
        last_num = int(last_action_id.split("_")[1])
        new_num = int(new_action_id.split("_")[1])
        assert new_num > last_num, f"Action ID collision: {new_action_id} should be > {last_action_id}"
        
        # 验证所有动作 ID 唯一
        all_history = trace2.get_full_action_history(node4)
        action_ids = [a.action_id for a in all_history]
        assert len(action_ids) == len(set(action_ids)), "Duplicate action IDs found"


class TestNodeStatus:
    """测试节点状态更新"""
    
    @pytest.fixture
    def trace(self):
        """创建初始化的 TraceSystem"""
        temp = tempfile.mkdtemp()
        trace = TraceSystem("test_task_012", base_dir=temp)
        trace.initialize()
        yield trace
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_mark_node_completed(self, trace):
        """测试标记节点完成"""
        node_id = trace.add_node({"type": "step1"}, {"success": True})
        
        trace.mark_node_completed(node_id, metrics={"performance": 0.9})
        
        node = trace.get_node(node_id)
        assert node.state_snapshot["status"] == "completed"
        assert node.metrics["performance"] == 0.9
    
    def test_mark_node_failed(self, trace):
        """测试标记节点失败"""
        node_id = trace.add_node({"type": "step1"}, {"success": False})
        
        trace.mark_node_failed(node_id, error="Compilation error")
        
        node = trace.get_node(node_id)
        assert node.state_snapshot["status"] == "failed"
        assert node.result["error"] == "Compilation error"
    
    def test_update_node_result(self, trace):
        """测试更新节点结果"""
        node_id = trace.add_node({"type": "step1"}, {"success": True})
        
        trace.update_node_result(
            node_id,
            result={"success": True, "performance": 0.85},
            metrics={"token_used": 1500}
        )
        
        node = trace.get_node(node_id)
        assert node.result["performance"] == 0.85
        assert node.metrics["token_used"] == 1500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
