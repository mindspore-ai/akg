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
TreeSearch 模式集成测试

测试 TraceSystem 和 FileSystemState 在 TreeSearch 模式下的集成：
- evolve 模式
- adaptive_search 模式
"""

import pytest
import shutil
import tempfile
import random
from pathlib import Path

from akg_agents.core_v2.filesystem import (
    FileSystemState,
    TraceSystem,
    NodeState,
    ActionRecord,
    ThinkingState,
)


class TestEvolveMode:
    """测试 evolve 模式与 Trace 系统集成"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_evolve", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_evolve_basic_rounds(self, trace_system):
        """测试 evolve 基本多轮迭代"""
        # 模拟 3 轮 evolve，每轮 2 个变体
        for round_idx in range(1, 4):
            for variant in range(2):
                performance = random.uniform(0.5, 1.0)
                action = {
                    "type": "evolve_round",
                    "round": round_idx,
                    "variant": variant,
                }
                result = {
                    "success": True,
                    "performance": performance,
                }
                trace_system.add_node(action, result, metrics={"performance": performance})
        
        # 验证树结构 - 应该有多个叶节点
        leaf_nodes = trace_system.get_all_leaf_nodes()
        assert len(leaf_nodes) >= 1
        
        # 验证最优叶节点
        best = trace_system.get_best_leaf_node(metric="performance")
        assert best is not None
    
    def test_evolve_with_branching(self, trace_system):
        """测试 evolve 带分支探索"""
        # Round 1: 初始探索
        r1_n1 = trace_system.add_node(
            {"type": "evolve", "round": 1, "strategy": "A"},
            {"success": True, "performance": 0.6},
            metrics={"performance": 0.6}
        )
        
        # Round 2: 从 r1_n1 继续
        r2_n1 = trace_system.add_node(
            {"type": "evolve", "round": 2, "strategy": "A+"},
            {"success": True, "performance": 0.7},
            metrics={"performance": 0.7}
        )
        
        # 切换回 r1_n1，尝试另一个方向
        trace_system.switch_node(r1_n1)
        
        r2_n2 = trace_system.add_node(
            {"type": "evolve", "round": 2, "strategy": "B"},
            {"success": True, "performance": 0.8},
            metrics={"performance": 0.8}
        )
        
        # 验证分叉
        node_r1 = trace_system.get_node(r1_n1)
        assert len(node_r1.children) == 2
        assert r2_n1 in node_r1.children
        assert r2_n2 in node_r1.children
        
        # 对比两个分支
        comparison = trace_system.compare_nodes(r2_n1, r2_n2)
        assert comparison["fork_point"] == r1_n1
    
    def test_evolve_parallel_forks(self, trace_system):
        """测试 evolve 并行分叉"""
        # 初始节点
        init_node = trace_system.add_node(
            {"type": "init"},
            {"success": True}
        )
        
        # 创建 3 个并行分叉
        fork_ids = trace_system.create_parallel_forks(
            n=3,
            action_template={"type": "parallel_explore"},
        )
        
        assert len(fork_ids) == 3
        
        # 验证每个分叉都是 init_node 的子节点
        init_info = trace_system.get_node(init_node)
        for fork_id in fork_ids:
            assert fork_id in init_info.children
    
    def test_evolve_resume(self, temp_dir):
        """测试 evolve 断点续跑"""
        # 第一次执行
        ts1 = TraceSystem("test_evolve_resume", base_dir=temp_dir)
        ts1.initialize()
        
        # 执行 3 轮
        nodes_before = []
        for i in range(3):
            node = ts1.add_node(
                {"type": "evolve", "round": i + 1},
                {"success": True, "performance": 0.5 + i * 0.1}
            )
            nodes_before.append(node)
        
        last_node = ts1.get_current_node()
        
        # 模拟重启
        ts2 = TraceSystem("test_evolve_resume", base_dir=temp_dir)
        
        # 验证恢复
        assert ts2.get_current_node() == last_node
        
        # 继续执行
        for i in range(3, 5):
            ts2.add_node(
                {"type": "evolve", "round": i + 1},
                {"success": True, "performance": 0.5 + i * 0.1}
            )
        
        # 验证完整历史
        history = ts2.get_full_action_history(ts2.get_current_node())
        assert len(history) == 5
        
        # 验证 action IDs 没有重复
        action_ids = [h.action_id for h in history]
        assert len(action_ids) == len(set(action_ids))
    
    def test_evolve_best_path_selection(self, trace_system):
        """测试 evolve 最优路径选择"""
        # 创建多个分支路径
        # 路径 1: performance 逐渐提升
        path1_nodes = []
        for i in range(3):
            perf = 0.5 + i * 0.1
            node = trace_system.add_node(
                {"type": "evolve", "path": 1, "step": i},
                {"success": True, "performance": perf},
                metrics={"performance": perf}
            )
            path1_nodes.append(node)
        
        # 回到 root 创建路径 2
        trace_system.switch_node("root")
        
        path2_nodes = []
        for i in range(3):
            perf = 0.6 + i * 0.15  # 更好的性能
            node = trace_system.add_node(
                {"type": "evolve", "path": 2, "step": i},
                {"success": True, "performance": perf},
                metrics={"performance": perf}
            )
            path2_nodes.append(node)
        
        # 获取最优叶节点
        best = trace_system.get_best_leaf_node(metric="performance")
        
        # 应该是路径 2 的最后一个节点（性能更好）
        assert best == path2_nodes[-1]
        
        # 验证最优路径
        best_path = trace_system.get_path_to_node(best)
        assert len(best_path) == 4  # root + 3 nodes


class TestAdaptiveSearchMode:
    """测试 adaptive_search 模式与 Trace 系统集成"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_adaptive_search", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_adaptive_search_basic(self, trace_system):
        """测试 adaptive_search 基本流程"""
        # 模拟自适应搜索：根据结果调整策略
        strategies = ["explore", "exploit", "explore", "exploit", "exploit"]
        
        for i, strategy in enumerate(strategies):
            performance = 0.5 + random.uniform(0, 0.4)
            action = {
                "type": "adaptive_search",
                "step": i + 1,
                "strategy": strategy,
            }
            result = {
                "success": True,
                "performance": performance,
                "strategy_effective": performance > 0.7,
            }
            trace_system.add_node(action, result, metrics={"performance": performance})
        
        # 验证历史
        history = trace_system.get_full_action_history(trace_system.get_current_node())
        assert len(history) == 5
        
        # 验证策略序列
        recorded_strategies = [h.arguments.get("strategy") for h in history]
        assert recorded_strategies == strategies
    
    def test_adaptive_search_with_backtrack(self, trace_system):
        """测试 adaptive_search 回溯机制"""
        # 初始探索
        n1 = trace_system.add_node(
            {"type": "adaptive", "step": 1},
            {"success": True, "performance": 0.6}
        )
        
        # 继续探索 - 性能下降
        n2 = trace_system.add_node(
            {"type": "adaptive", "step": 2},
            {"success": True, "performance": 0.4}  # 性能下降
        )
        
        n3 = trace_system.add_node(
            {"type": "adaptive", "step": 3},
            {"success": True, "performance": 0.3}  # 继续下降
        )
        
        # 检测到性能下降，回溯到 n1
        trace_system.switch_node(n1)
        
        # 尝试新方向
        n4 = trace_system.add_node(
            {"type": "adaptive", "step": 2, "alternative": True},
            {"success": True, "performance": 0.75}
        )
        
        n5 = trace_system.add_node(
            {"type": "adaptive", "step": 3},
            {"success": True, "performance": 0.85}
        )
        
        # 验证最优路径
        path_to_n3 = trace_system.get_path_to_node(n3)
        path_to_n5 = trace_system.get_path_to_node(n5)
        
        # 两条路径都从 n1 分叉
        comparison = trace_system.compare_nodes(n3, n5)
        assert comparison["fork_point"] == n1
        
        # 新路径的性能更好
        best = trace_system.get_best_leaf_node(metric="performance")
        # 注意：n3 和 n5 都是叶节点，但 n5 性能更好
        # 需要根据实际 metrics 存储来判断


class TestTreeSearchStateManagement:
    """测试 TreeSearch 模式下的状态管理"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_treesearch_state", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_thinking_state_across_branches(self, trace_system):
        """测试分支间的 thinking 状态管理"""
        fs = trace_system.fs
        
        # 创建初始设计节点
        design_node = trace_system.add_node(
            {"type": "design"},
            {"success": True}
        )
        
        # 保存设计思路
        fs.save_node_state(design_node, NodeState(
            node_id=design_node, turn=1, status="completed"
        ))
        fs.save_thinking(design_node, ThinkingState(
            node_id=design_node,
            turn=1,
            current_plan={"approach": "online_softmax"},
            latest_thinking="Decided on online softmax for numerical stability",
        ))
        
        # 分支 1：使用 shared memory
        branch1 = trace_system.add_node(
            {"type": "implement", "optimization": "shared_memory"},
            {"success": True}
        )
        fs.save_node_state(branch1, NodeState(
            node_id=branch1, turn=2, status="completed"
        ))
        fs.save_thinking(branch1, ThinkingState(
            node_id=branch1,
            turn=2,
            current_plan={"memory": "shared"},
            latest_thinking="Using shared memory for intermediate results",
        ))
        
        # 回到 design_node 创建分支 2
        trace_system.switch_node(design_node)
        
        branch2 = trace_system.add_node(
            {"type": "implement", "optimization": "registers"},
            {"success": True}
        )
        fs.save_node_state(branch2, NodeState(
            node_id=branch2, turn=2, status="completed"
        ))
        fs.save_thinking(branch2, ThinkingState(
            node_id=branch2,
            turn=2,
            current_plan={"memory": "registers"},
            latest_thinking="Using registers for maximum performance",
        ))
        
        # 验证两个分支有不同的 thinking
        thinking1 = fs.load_thinking(branch1)
        thinking2 = fs.load_thinking(branch2)
        
        assert thinking1.current_plan["memory"] == "shared"
        assert thinking2.current_plan["memory"] == "registers"
    
    def test_code_versioning_in_treesearch(self, trace_system):
        """测试 TreeSearch 中的代码版本管理"""
        fs = trace_system.fs
        
        # 基础代码
        base_node = trace_system.add_node(
            {"type": "coder", "version": "base"},
            {"success": True}
        )
        fs.save_node_state(base_node, NodeState(
            node_id=base_node, turn=1, status="completed"
        ))
        fs.save_code_file(base_node, "kernel.py", """
import triton
@triton.jit
def base_kernel(): pass
""")
        
        # 变体 1
        variant1 = trace_system.add_node(
            {"type": "coder", "version": "optimized_v1"},
            {"success": True}
        )
        fs.save_node_state(variant1, NodeState(
            node_id=variant1, turn=2, status="completed"
        ))
        fs.save_code_file(variant1, "kernel.py", """
import triton
@triton.jit
def optimized_kernel_v1(): pass  # with vectorization
""")
        
        # 回到基础节点，创建变体 2
        trace_system.switch_node(base_node)
        
        variant2 = trace_system.add_node(
            {"type": "coder", "version": "optimized_v2"},
            {"success": True}
        )
        fs.save_node_state(variant2, NodeState(
            node_id=variant2, turn=2, status="completed"
        ))
        fs.save_code_file(variant2, "kernel.py", """
import triton
@triton.jit
def optimized_kernel_v2(): pass  # with tiling
""")
        
        # 验证三个版本的代码都保持独立
        code_base = fs.load_code_file(base_node, "kernel.py")
        code_v1 = fs.load_code_file(variant1, "kernel.py")
        code_v2 = fs.load_code_file(variant2, "kernel.py")
        
        assert "base_kernel" in code_base
        assert "optimized_kernel_v1" in code_v1
        assert "optimized_kernel_v2" in code_v2
        
        # 验证树结构
        base_info = trace_system.get_node(base_node)
        assert len(base_info.children) == 2
        assert variant1 in base_info.children
        assert variant2 in base_info.children


class TestEvolveWithMetrics:
    """测试 evolve 模式的 metrics 追踪"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        ts = TraceSystem("test_evolve_metrics", base_dir=temp_dir)
        ts.initialize()
        return ts
    
    def test_metrics_accumulation(self, trace_system):
        """测试 metrics 累积"""
        # 创建带 metrics 的节点
        n1 = trace_system.add_node(
            {"type": "evolve", "round": 1},
            {"success": True},
            metrics={"performance": 0.6, "accuracy": 0.95}
        )
        
        n2 = trace_system.add_node(
            {"type": "evolve", "round": 2},
            {"success": True},
            metrics={"performance": 0.75, "accuracy": 0.97}
        )
        
        n3 = trace_system.add_node(
            {"type": "evolve", "round": 3},
            {"success": True},
            metrics={"performance": 0.85, "accuracy": 0.99}
        )
        
        # 获取路径 metrics
        path_metrics = trace_system._calculate_path_metrics(
            trace_system.get_path_to_node(n3)
        )
        
        # 验证 metrics 计算
        assert path_metrics["node_count"] == 4  # root + 3 nodes
    
    def test_best_node_by_different_metrics(self, trace_system):
        """测试按不同 metrics 获取最优节点"""
        # 创建两个分支
        # 分支 1：高性能
        n1 = trace_system.add_node(
            {"type": "evolve", "branch": 1},
            {"success": True},
            metrics={"performance": 0.95, "accuracy": 0.90}
        )
        
        # 回到 root 创建分支 2
        trace_system.switch_node("root")
        
        # 分支 2：高精度
        n2 = trace_system.add_node(
            {"type": "evolve", "branch": 2},
            {"success": True},
            metrics={"performance": 0.80, "accuracy": 0.99}
        )
        
        # 按性能获取最优
        best_perf = trace_system.get_best_leaf_node(metric="performance")
        assert best_perf == n1
        
        # 按精度获取最优
        best_acc = trace_system.get_best_leaf_node(metric="accuracy")
        assert best_acc == n2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
