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
- trace fork
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
try:
    from akg_agents.cli.cli import app
    from typer.testing import CliRunner
    # 尝试触发 Typer 的参数解析，如果类型提示不兼容会在此处或 invoke 时抛出 RuntimeError
    # 注意：这里使用 dummy 执行来确保 Typer 已经完成了对 app 的分析
    runner = CliRunner()
    result = runner.invoke(app, ["--help"], catch_exceptions=False)
    CLI_AVAILABLE = True
except Exception:
    app = None
    CLI_AVAILABLE = False

from akg_agents.core_v2.filesystem import TraceSystem, NodeState


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI interface not compatible with current environment (str | None issues)")
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

    @pytest.mark.skip(reason="trace 顶层命令未实现，使用交互模式内 /trace slash command")
    def test_trace_show_command(self, runner, trace_system):
        """测试 trace show 命令"""
        ts, temp_dir = trace_system
        
        # 使用全局 app (可能是 Mock)
        
        result = runner.invoke(app, [
            "trace", "show", "test_cli_task",
            "--base-dir", temp_dir
        ])
        
        # 验证输出
        assert result.exit_code == 0
        # 应该包含树结构或统计信息

    @pytest.mark.skip(reason="trace 顶层命令未实现，使用交互模式内 /trace slash command")
    def test_trace_fork_command(self, runner, trace_system):
        """测试 trace fork 命令"""
        ts, temp_dir = trace_system
        
        from akg_agents.cli.cli import app
        
        # 先创建一个 ask_user 节点
        ts.add_node(
            action={"type": "ask_user", "arguments": {"message": "测试问题"}},
            result={"status": "responded", "user_response": "测试回答", "message": "测试问题"},
        )
        ask_node = ts.get_current_node()
        
        result = runner.invoke(app, [
            "trace", "fork", "test_cli_task", ask_node,
            "--base-dir", temp_dir
        ])
        
        # 验证 fork 成功
        assert result.exit_code == 0

    @pytest.mark.skip(reason="trace 顶层命令未实现，使用交互模式内 /trace slash command")
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
        assert result.exit_code == 0

    @pytest.mark.skip(reason="trace 顶层命令未实现，使用交互模式内 /trace slash command")
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
        assert result.exit_code == 0

    @pytest.mark.skip(reason="trace 顶层命令未实现，使用交互模式内 /trace slash command")
    def test_trace_leaves_command(self, runner, trace_system):
        """测试 trace leaves 命令"""
        ts, temp_dir = trace_system
        
        from akg_agents.cli.cli import app
        
        result = runner.invoke(app, [
            "trace", "leaves", "test_cli_task",
            "--base-dir", temp_dir
        ])
        
        # 验证输出包含叶节点列表
        assert result.exit_code == 0

    @pytest.mark.skip(reason="trace 顶层命令未实现，使用交互模式内 /trace slash command")
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
            assert result.exit_code == 0

    @pytest.mark.skip(reason="trace 顶层命令未实现，使用交互模式内 /trace slash command")
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
        assert result.exit_code == 0

    @pytest.mark.skip(reason="trace 顶层命令未实现，使用交互模式内 /trace slash command")
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

    def test_get_conversation_history(self, temp_dir):
        """测试从 action_history 提取对话历史"""
        ts = TraceSystem("test_history", base_dir=temp_dir)
        ts.initialize()

        # 模拟 ask_user 节点
        ts.add_node(
            action={"type": "ask_user", "arguments": {"user_input": "请生成 add kernel"}},
            result={"message": "请确认 block_size: 128"},
            metrics={}
        )

        # 模拟 finish 节点
        ts.add_node(
            action={"type": "finish"},
            result={"message": "已完成，性能: 0.85"},
            metrics={}
        )

        # 获取对话历史
        history = ts.get_conversation_history(max_turns=10)

        # 验证
        assert len(history) == 3  # user + assistant (ask_user) + assistant (finish)
        assert history[0]["role"] == "user"
        assert "add kernel" in history[0]["content"]
        assert history[1]["role"] == "assistant"
        assert "block_size" in history[1]["content"]
        assert history[2]["role"] == "assistant"
        assert "0.85" in history[2]["content"]


class TestSessionResolverUnit:
    """会话 ID 解析器单元测试（精确 + 前缀匹配）"""

    @pytest.fixture
    def conversations_dir(self):
        temp = tempfile.mkdtemp()
        conversations = Path(temp) / "conversations"
        conversations.mkdir(parents=True, exist_ok=True)
        yield conversations
        shutil.rmtree(temp, ignore_errors=True)

    def test_resolve_session_dir_exact_match_priority(self, conversations_dir):
        """精确匹配优先于前缀匹配"""
        from akg_agents.cli.commands.trace import resolve_session_dir, SessionResolveError

        (conversations_dir / "cli_abc").mkdir()
        (conversations_dir / "cli_abcdef").mkdir()

        sid, session_dir = resolve_session_dir("abc", conversations_dir=conversations_dir)

        assert sid == "abc"
        assert session_dir == conversations_dir / "cli_abc"

    def test_resolve_session_dir_unique_prefix(self, conversations_dir):
        """唯一前缀时自动恢复"""
        from akg_agents.cli.commands.trace import resolve_session_dir, SessionResolveError

        (conversations_dir / "cli_50b111f7-aaaa").mkdir()
        (conversations_dir / "cli_60c222f8-bbbb").mkdir()

        sid, session_dir = resolve_session_dir("50b1", conversations_dir=conversations_dir)

        assert sid == "50b111f7-aaaa"
        assert session_dir == conversations_dir / "cli_50b111f7-aaaa"

    def test_resolve_session_dir_ambiguous_prefix(self, conversations_dir):
        """多前缀匹配时报歧义并返回候选"""
        from akg_agents.cli.commands.trace import resolve_session_dir, SessionResolveError

        (conversations_dir / "cli_1234-abcd").mkdir()
        (conversations_dir / "cli_1234-efgh").mkdir()

        with pytest.raises(SessionResolveError) as exc_info:
            resolve_session_dir("1234", conversations_dir=conversations_dir)

        err = exc_info.value
        assert err.reason == "ambiguous"
        assert err.candidates == ["1234-abcd", "1234-efgh"]

    def test_resolve_session_dir_not_found(self, conversations_dir):
        """无匹配时报不存在"""
        from akg_agents.cli.commands.trace import resolve_session_dir, SessionResolveError

        (conversations_dir / "cli_exists-only").mkdir()

        with pytest.raises(SessionResolveError) as exc_info:
            resolve_session_dir("missing", conversations_dir=conversations_dir)

        err = exc_info.value
        assert err.reason == "not_found"
        assert err.normalized_session_id == "missing"
        assert err.expected_path == conversations_dir / "cli_missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
