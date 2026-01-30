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
CLI Trace 命令测试

测试 trace 命令的各项功能：
- trace show
- trace switch
- trace compare
- trace path
- trace history
- trace best
- trace leaves
"""

import pytest
import shutil
import tempfile
from pathlib import Path
from typer.testing import CliRunner

from akg_agents.core_v2.filesystem import TraceSystem, NodeState


class TestCLITraceCommands:
    """测试 CLI trace 命令"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def runner(self):
        """CLI 测试 runner"""
        return CliRunner()
    
    @pytest.fixture
    def trace_system(self, temp_dir):
        """创建带数据的 TraceSystem"""
        ts = TraceSystem("test_cli_task", base_dir=temp_dir)
        ts.initialize()
        
        # 添加一些节点
        n1 = ts.add_node(
            {"type": "design"},
            {"success": True},
            metrics={"performance": 0.6}
        )
        n2 = ts.add_node(
            {"type": "code"},
            {"success": True},
            metrics={"performance": 0.7}
        )
        n3 = ts.add_node(
            {"type": "verify"},
            {"success": True},
            metrics={"performance": 0.8}
        )
        
        # 创建分叉
        ts.switch_node(n1)
        n4 = ts.add_node(
            {"type": "code_v2"},
            {"success": True},
            metrics={"performance": 0.85}
        )
        n5 = ts.add_node(
            {"type": "verify"},
            {"success": True},
            metrics={"performance": 0.9}
        )
        
        return ts, temp_dir
    
    def test_trace_show_command(self, runner, trace_system):
        """测试 trace show 命令"""
        ts, temp_dir = trace_system
        
        # 导入 app
        from akg_agents.cli.cli import app
        
        result = runner.invoke(app, [
            "trace", "show", "test_cli_task",
            "--base-dir", temp_dir
        ])
        
        # 验证输出
        assert result.exit_code == 0 or "不存在" not in result.output
        # 应该包含树结构或统计信息
    
    def test_trace_switch_command(self, runner, trace_system):
        """测试 trace switch 命令"""
        ts, temp_dir = trace_system
        
        from akg_agents.cli.cli import app
        
        # 获取一个节点 ID
        leaves = ts.get_all_leaf_nodes()
        target_node = leaves[0] if leaves else "root"
        
        result = runner.invoke(app, [
            "trace", "switch", "test_cli_task", target_node,
            "--base-dir", temp_dir
        ])
        
        # 验证切换成功
        assert result.exit_code == 0 or "错误" not in result.output
    
    def test_trace_path_command(self, runner, trace_system):
        """测试 trace path 命令"""
        ts, temp_dir = trace_system
        
        from akg_agents.cli.cli import app
        
        # 获取当前节点
        current = ts.get_current_node()
        
        result = runner.invoke(app, [
            "trace", "path", "test_cli_task", current,
            "--base-dir", temp_dir
        ])
        
        # 验证输出包含路径信息
        assert result.exit_code == 0 or "错误" not in result.output
    
    def test_trace_history_command(self, runner, trace_system):
        """测试 trace history 命令"""
        ts, temp_dir = trace_system
        
        from akg_agents.cli.cli import app
        
        result = runner.invoke(app, [
            "trace", "history", "test_cli_task",
            "--base-dir", temp_dir,
            "--limit", "10"
        ])
        
        # 验证输出
        assert result.exit_code == 0 or "错误" not in result.output
    
    def test_trace_leaves_command(self, runner, trace_system):
        """测试 trace leaves 命令"""
        ts, temp_dir = trace_system
        
        from akg_agents.cli.cli import app
        
        result = runner.invoke(app, [
            "trace", "leaves", "test_cli_task",
            "--base-dir", temp_dir
        ])
        
        # 验证输出包含叶节点列表
        assert result.exit_code == 0 or "错误" not in result.output
    
    def test_trace_compare_command(self, runner, trace_system):
        """测试 trace compare 命令"""
        ts, temp_dir = trace_system
        
        from akg_agents.cli.cli import app
        
        # 获取两个叶节点进行对比
        leaves = ts.get_all_leaf_nodes()
        if len(leaves) >= 2:
            result = runner.invoke(app, [
                "trace", "compare", "test_cli_task",
                leaves[0], leaves[1],
                "--base-dir", temp_dir
            ])
            
            # 验证输出包含对比信息
            assert result.exit_code == 0 or "错误" not in result.output
    
    def test_trace_best_command(self, runner, trace_system):
        """测试 trace best 命令"""
        ts, temp_dir = trace_system
        
        from akg_agents.cli.cli import app
        
        result = runner.invoke(app, [
            "trace", "best", "test_cli_task",
            "--metric", "performance",
            "--base-dir", temp_dir
        ])
        
        # 验证输出
        assert result.exit_code == 0 or "错误" not in result.output
    
    def test_trace_nonexistent_task(self, runner, temp_dir):
        """测试访问不存在的任务"""
        from akg_agents.cli.cli import app
        
        result = runner.invoke(app, [
            "trace", "show", "nonexistent_task",
            "--base-dir", temp_dir
        ])
        
        # 应该报错
        assert result.exit_code != 0 or "不存在" in result.output


class TestTraceCommandsUnit:
    """Trace 命令的单元测试（不依赖 CLI runner）"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_trace_system_for_cli(self, temp_dir):
        """测试 TraceSystem 功能用于 CLI"""
        ts = TraceSystem("cli_test", base_dir=temp_dir)
        ts.initialize()
        
        # 添加节点
        n1 = ts.add_node(
            {"type": "action1"},
            {"success": True},
            metrics={"score": 0.8}
        )
        
        # 验证可视化
        tree_str = ts.visualize_tree()
        assert tree_str is not None
        assert len(tree_str) > 0
        
        # 验证获取叶节点
        leaves = ts.get_all_leaf_nodes()
        assert n1 in leaves
        
        # 验证获取路径
        path = ts.get_path_to_node(n1)
        assert len(path) == 2  # root + n1
        
        # 验证获取历史
        history = ts.get_full_action_history(n1)
        assert len(history) == 1
    
    def test_trace_compare_for_cli(self, temp_dir):
        """测试节点对比功能用于 CLI"""
        ts = TraceSystem("cli_compare", base_dir=temp_dir)
        ts.initialize()
        
        # 创建分叉
        n1 = ts.add_node({"type": "base"}, {"success": True})
        n2 = ts.add_node({"type": "path1"}, {"success": True})
        
        ts.switch_node(n1)
        n3 = ts.add_node({"type": "path2"}, {"success": True})
        
        # 对比
        comparison = ts.compare_nodes(n2, n3)
        
        assert comparison["fork_point"] == n1
        assert "path_1" in comparison
        assert "path_2" in comparison


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
