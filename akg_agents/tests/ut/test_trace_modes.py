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
Trace + FileSystem 模式集成测试

含：单次生成模式 / 对话模式 / TreeSearch 模式 / SessionResumeError / CLI Session 解析
"""

import json
import random
import shutil
import tempfile

import pytest
from pathlib import Path

from akg_agents.core_v2.filesystem import (
    FileSystemState,
    TraceSystem,
    NodeState,
    ActionRecord,
    ThinkingState,
    PendingTool,
    PendingToolsState,
    SessionResumeError,
)


# ================================================================
#  通用 fixture
# ================================================================

@pytest.fixture
def ts(tmp_path):
    t = TraceSystem("mode_test", base_dir=str(tmp_path))
    t.initialize()
    return t


# ================================================================
#  1. 单次生成模式
# ================================================================

class TestSingleMode:
    def test_coder_only_flow(self, ts):
        """Coder → Verifier → Conductor 链"""
        c = ts.add_node({"type": "call_coder"}, {"success": True, "code": "..."})
        v = ts.add_node({"type": "call_verifier"}, {"success": True, "perf": 0.75})
        d = ts.add_node({"type": "conductor_decide", "decision": "accept"}, {"accepted": True})
        path = ts.get_path_to_node(d)
        assert len(path) == 4
        h = ts.get_full_action_history(d)
        assert [a.tool_name for a in h] == ["call_coder", "call_verifier", "conductor_decide"]

    def test_coder_retry(self, ts):
        """失败后重试"""
        ts.add_node({"type": "call_coder", "attempt": 1}, {"success": False, "error": "compile"})
        ts.add_node({"type": "call_coder", "attempt": 2}, {"success": True, "code": "..."})
        v = ts.add_node({"type": "call_verifier"}, {"success": True})
        h = ts.get_full_action_history(v)
        assert len(h) == 3
        assert h[0].result["success"] is False
        assert h[1].result["success"] is True

    def test_default_workflow_full(self, ts):
        """Designer → Coder → Verifier 完整流程"""
        ts.add_node({"type": "call_designer"}, {"success": True, "design": {"algo": "online"}})
        c = ts.add_node({"type": "call_coder"}, {"success": True, "code": "..."})
        ts.fs.save_node_state(c, NodeState(node_id=c, turn=2, status="running"))
        ts.fs.save_code_file(c, "softmax.py", "@triton.jit\ndef kernel(): ...")
        v = ts.add_node({"type": "call_verifier"}, {"success": True, "perf": 0.85})
        h = ts.get_full_action_history(v)
        assert [a.tool_name for a in h] == ["call_designer", "call_coder", "call_verifier"]

    def test_designer_revision_fork(self, ts):
        """设计修改产生分叉"""
        d1 = ts.add_node({"type": "call_designer", "v": 1}, {"success": True})
        ts.add_node({"type": "call_coder"}, {"success": True})
        v1 = ts.add_node({"type": "call_verifier"}, {"success": True, "perf": 0.5})
        ts.switch_node(d1)
        ts.add_node({"type": "call_designer", "v": 2}, {"success": True})
        ts.add_node({"type": "call_coder"}, {"success": True})
        v2 = ts.add_node({"type": "call_verifier"}, {"success": True, "perf": 0.9})
        cmp = ts.compare_nodes(v1, v2)
        assert cmp["fork_point"] == d1

    def test_connect_all_multi_round(self, ts):
        """全连接多轮迭代"""
        for rnd in range(1, 3):
            ts.add_node({"type": "call_designer", "round": rnd}, {"success": True})
            ts.add_node({"type": "call_coder", "round": rnd}, {"success": True})
            ts.add_node({"type": "call_verifier", "round": rnd}, {"success": rnd == 2})
        h = ts.get_full_action_history(ts.get_current_node())
        assert len(h) == 6
        rounds = [a.arguments.get("round") for a in h]
        assert rounds == [1, 1, 1, 2, 2, 2]

    def test_state_persistence(self, tmp_path):
        """thinking 状态持久化"""
        ts1 = TraceSystem("sp", base_dir=str(tmp_path))
        ts1.initialize()
        n = ts1.add_node({"type": "call_coder"}, {"success": True})
        ts1.fs.save_thinking(n, ThinkingState(
            node_id=n, turn=1, current_plan={"goal": "softmax"},
            latest_thinking="online softmax"))
        ts2 = TraceSystem("sp", base_dir=str(tmp_path))
        assert ts2.get_current_node() == n
        assert ts2.fs.load_thinking(n).latest_thinking == "online softmax"


# ================================================================
#  2. 对话模式
# ================================================================

class TestDialogMode:
    def test_basic_flow(self, ts):
        """用户请求 → Designer → 确认 → Coder"""
        ts.add_node({"type": "user_request"}, {"response": "I'll design..."})
        ts.add_node({"type": "call_designer"}, {"success": True})
        ts.add_node({"type": "user_confirm"}, {"next_step": "implement"})
        c = ts.add_node({"type": "call_coder"}, {"success": True})
        assert len(ts.get_full_action_history(c)) == 4
        assert len(ts.get_path_to_node(c)) == 5

    def test_user_feedback_fork(self, ts):
        """用户反馈产生的分叉"""
        d1 = ts.add_node({"type": "call_designer", "v": 1}, {"design": {"approach": "basic"}})
        ts.add_node({"type": "user_feedback"}, {"action": "redesign"})
        d2 = ts.add_node({"type": "call_designer", "v": 2}, {"design": {"approach": "opt"}})
        h = ts.get_full_action_history(d2)
        assert h[1].tool_name == "user_feedback"

    def test_switch_and_fork(self, ts):
        """切换节点并创建分叉"""
        ts.add_node({"type": "call_designer"}, {"success": True})
        coder = ts.add_node({"type": "call_coder"}, {"success": True})
        v1 = ts.add_node({"type": "verify"}, {"perf": 0.65})
        ts.switch_node(coder)
        ts.add_node({"type": "call_coder", "strategy": "shared_memory"}, {"success": True})
        v2 = ts.add_node({"type": "verify"}, {"perf": 0.85})
        cmp = ts.compare_nodes(v1, v2)
        assert cmp["fork_point"] == coder

    def test_multi_exploration(self, ts):
        """多方案探索"""
        base = ts.add_node({"type": "design"}, {"success": True})
        for approach in ("A", "B", "C"):
            ts.switch_node(base)
            ts.add_node({"type": "implement", "approach": approach}, {"perf": 0.6})
        assert len(ts.get_node(base).children) == 3
        assert len(ts.get_all_leaf_nodes()) == 3


class TestHITL:
    def test_pending_tools_lifecycle(self, ts):
        """pending tools 的完整生命周期"""
        fs = ts.fs
        n = ts.add_node({"type": "call_designer"}, {"success": True})
        fs.save_node_state(n, NodeState(node_id=n, turn=1, status="waiting"))
        pending = PendingToolsState(node_id=n, turn=1)
        pending.add_pending_tool(PendingTool(tool_call_id="c1", tool_name="ask_user", arguments={}))
        pending.add_pending_tool(PendingTool(tool_call_id="c2", tool_name="file_write", arguments={}))
        fs.save_pending_tools(n, pending)

        loaded = fs.load_pending_tools(n)
        assert len(loaded.pending_tools) == 2

        fs.mark_tool_completed(n, "c1")
        loaded = fs.load_pending_tools(n)
        assert sum(1 for t in loaded.pending_tools if t.status == "completed") == 1

        fs.mark_tool_completed(n, "c2")
        loaded = fs.load_pending_tools(n)
        assert all(t.status == "completed" for t in loaded.pending_tools)

    def test_resume_with_pending(self, tmp_path):
        """带 pending tools 的断点续跑"""
        ts1 = TraceSystem("rp", base_dir=str(tmp_path))
        ts1.initialize()
        n = ts1.add_node({"type": "action"}, {"success": True})
        ts1.fs.save_node_state(n, NodeState(node_id=n, turn=1, status="waiting"))
        p = PendingToolsState(node_id=n, turn=1)
        p.add_pending_tool(PendingTool(tool_call_id="p1", tool_name="ask_user", arguments={}))
        ts1.fs.save_pending_tools(n, p)

        ts2 = TraceSystem("rp", base_dir=str(tmp_path))
        loaded = ts2.fs.load_pending_tools(n)
        assert loaded.pending_tools[0].status == "pending"
        ts2.fs.mark_tool_completed(n, "p1")
        ts2.add_node({"type": "continue"}, {"success": True})
        assert len(ts2.get_full_action_history(ts2.get_current_node())) == 2


class TestThinkingEvolution:
    def test_thinking_across_turns(self, ts):
        """thinking 跨 turn 演进"""
        fs = ts.fs
        n1 = ts.add_node({"type": "design"}, {"success": True})
        fs.save_node_state(n1, NodeState(node_id=n1, turn=1, status="done"))
        fs.save_thinking(n1, ThinkingState(
            node_id=n1, turn=1, current_plan={"goal": "softmax"},
            latest_thinking="basic design"))
        n2 = ts.add_node({"type": "user_feedback"}, {"feedback": "opt"})
        fs.save_node_state(n2, NodeState(node_id=n2, turn=2, status="done"))
        fs.update_thinking(n2, latest_thinking="user wants opt", decision="switch")
        n3 = ts.add_node({"type": "optimize"}, {"success": True})
        fs.save_node_state(n3, NodeState(node_id=n3, turn=3, status="done"))
        fs.update_thinking(n3, latest_thinking="shared mem", decision="use shmem")
        assert len(fs.load_thinking(n3).decision_history) == 2

    def test_thinking_branch_independence(self, ts):
        """不同分支 thinking 独立"""
        fs = ts.fs
        base = ts.add_node({"type": "base"}, {"success": True})
        fs.save_node_state(base, NodeState(node_id=base, turn=1, status="done"))
        fs.save_thinking(base, ThinkingState(
            node_id=base, turn=1, current_plan={"approach": "?"},
            latest_thinking="considering"))

        ba = ts.add_node({"type": "a"}, {"success": True})
        fs.save_thinking(ba, ThinkingState(
            node_id=ba, turn=2, current_plan={"approach": "A"}, latest_thinking="A"))

        ts.switch_node(base)
        bb = ts.add_node({"type": "b"}, {"success": True})
        fs.save_thinking(bb, ThinkingState(
            node_id=bb, turn=2, current_plan={"approach": "B"}, latest_thinking="B"))

        assert fs.load_thinking(ba).current_plan["approach"] == "A"
        assert fs.load_thinking(bb).current_plan["approach"] == "B"


# ================================================================
#  3. TreeSearch 模式
# ================================================================

class TestEvolveMode:
    def test_branching(self, ts):
        """evolve 分支探索"""
        r1 = ts.add_node({"type": "evolve", "round": 1, "s": "A"}, {"perf": 0.6},
                          metrics={"perf": 0.6})
        ts.add_node({"type": "evolve", "round": 2, "s": "A+"}, {"perf": 0.7},
                     metrics={"perf": 0.7})
        ts.switch_node(r1)
        ts.add_node({"type": "evolve", "round": 2, "s": "B"}, {"perf": 0.8},
                     metrics={"perf": 0.8})
        assert len(ts.get_node(r1).children) == 2

    def test_parallel_forks(self, ts):
        """evolve 并行分叉"""
        init = ts.add_node({"type": "init"}, {"success": True})
        forks = ts.create_parallel_forks(n=3, action_template={"type": "parallel"})
        assert len(forks) == 3
        assert all(f in ts.get_node(init).children for f in forks)

    def test_resume(self, tmp_path):
        """evolve 断点续跑"""
        ts1 = TraceSystem("ev_re", base_dir=str(tmp_path))
        ts1.initialize()
        for i in range(3):
            ts1.add_node({"type": "evolve", "round": i+1}, {"perf": 0.5+i*0.1})
        last = ts1.get_current_node()

        ts2 = TraceSystem("ev_re", base_dir=str(tmp_path))
        assert ts2.get_current_node() == last
        for i in range(3, 5):
            ts2.add_node({"type": "evolve", "round": i+1}, {"perf": 0.5+i*0.1})
        h = ts2.get_full_action_history(ts2.get_current_node())
        assert len(h) == 5
        assert len(set(a.action_id for a in h)) == 5

    def test_best_path_selection(self, ts):
        """最优路径选择"""
        for i in range(3):
            ts.add_node({"type": "evolve"}, {"perf": 0.5+i*0.1}, metrics={"perf": 0.5+i*0.1})
        ts.switch_node("root")
        for i in range(3):
            ts.add_node({"type": "evolve"}, {"perf": 0.6+i*0.15}, metrics={"perf": 0.6+i*0.15})
        best = ts.get_best_leaf_node(metric="perf")
        best_path = ts.get_path_to_node(best)
        assert len(best_path) == 4

    def test_best_by_different_metrics(self, ts):
        """按不同指标选最优"""
        n1 = ts.add_node({"type": "x"}, {}, metrics={"perf": 0.95, "acc": 0.90})
        ts.switch_node("root")
        n2 = ts.add_node({"type": "x"}, {}, metrics={"perf": 0.80, "acc": 0.99})
        assert ts.get_best_leaf_node(metric="perf") == n1
        assert ts.get_best_leaf_node(metric="acc") == n2


class TestAdaptiveSearch:
    def test_basic(self, ts):
        """adaptive_search 策略记录"""
        strategies = ["explore", "exploit", "explore"]
        for i, s in enumerate(strategies):
            ts.add_node({"type": "adaptive", "step": i+1, "strategy": s},
                        {"perf": 0.5+random.uniform(0, 0.4)})
        h = ts.get_full_action_history(ts.get_current_node())
        assert [a.arguments["strategy"] for a in h] == strategies

    def test_backtrack(self, ts):
        """回溯机制"""
        n1 = ts.add_node({"type": "adaptive", "step": 1}, {"perf": 0.6})
        ts.add_node({"type": "adaptive", "step": 2}, {"perf": 0.4})
        n3 = ts.add_node({"type": "adaptive", "step": 3}, {"perf": 0.3})
        ts.switch_node(n1)
        ts.add_node({"type": "adaptive", "step": 2, "alt": True}, {"perf": 0.75})
        n5 = ts.add_node({"type": "adaptive", "step": 3}, {"perf": 0.85})
        cmp = ts.compare_nodes(n3, n5)
        assert cmp["fork_point"] == n1

    def test_code_versioning(self, ts):
        """代码版本管理"""
        fs = ts.fs
        base = ts.add_node({"type": "coder"}, {"success": True})
        fs.save_node_state(base, NodeState(node_id=base, turn=1, status="done"))
        fs.save_code_file(base, "k.py", "def base_kernel(): pass")

        v1 = ts.add_node({"type": "coder", "v": "opt1"}, {"success": True})
        fs.save_code_file(v1, "k.py", "def opt_kernel_v1(): pass")

        ts.switch_node(base)
        v2 = ts.add_node({"type": "coder", "v": "opt2"}, {"success": True})
        fs.save_code_file(v2, "k.py", "def opt_kernel_v2(): pass")

        assert "base_kernel" in fs.load_code_file(base, "k.py")
        assert "opt_kernel_v1" in fs.load_code_file(v1, "k.py")
        assert "opt_kernel_v2" in fs.load_code_file(v2, "k.py")
        assert len(ts.get_node(base).children) == 2


# ================================================================
#  4. SessionResumeError
# ================================================================

class TestSessionResumeError:
    @pytest.fixture
    def normal_session(self, tmp_path):
        ts = TraceSystem("resume_err", base_dir=str(tmp_path))
        ts.initialize(task_input="测试任务")
        ts.fs.save_node_state("root", NodeState(
            node_id="root", turn=0, status="init",
            task_info={"task_id": "resume_err", "task_input": "测试任务"},
            agent_info={"agent_name": "test"}))
        n1 = ts.add_node({"type": "call_coder"}, {"success": True})
        return ts, n1, tmp_path

    def test_corrupt_trace_json(self, normal_session):
        ts, _, tmp = normal_session
        trace_file = tmp / "conversations" / ts.task_id / "trace.json"
        trace_file.write_text("{ INVALID !!!", encoding="utf-8")
        ts2 = TraceSystem(ts.task_id, base_dir=str(tmp))
        with pytest.raises(Exception):
            ts2.trace

    def test_error_attributes(self):
        cause = ValueError("bad")
        err = SessionResumeError("abc", "field xyz missing", cause=cause)
        assert err.session_id == "abc"
        assert err.detail == "field xyz missing"
        assert err.cause is cause
        assert "abc" in str(err) and "xyz" in str(err)


# ================================================================
#  5. CLI Session 解析
# ================================================================

class TestSessionResolver:
    @pytest.fixture
    def conversations_dir(self, tmp_path):
        d = tmp_path / "conversations"
        d.mkdir()
        return d

    def test_exact_match_priority(self, conversations_dir):
        from akg_agents.cli.commands.trace import resolve_session_dir, SessionResolveError
        (conversations_dir / "cli_abc").mkdir()
        (conversations_dir / "cli_abcdef").mkdir()
        sid, sdir = resolve_session_dir("abc", conversations_dir=conversations_dir)
        assert sid == "abc"
        assert sdir == conversations_dir / "cli_abc"

    def test_unique_prefix(self, conversations_dir):
        from akg_agents.cli.commands.trace import resolve_session_dir
        (conversations_dir / "cli_50b111f7-aaaa").mkdir()
        (conversations_dir / "cli_60c222f8-bbbb").mkdir()
        sid, _ = resolve_session_dir("50b1", conversations_dir=conversations_dir)
        assert sid == "50b111f7-aaaa"

    def test_ambiguous_prefix(self, conversations_dir):
        from akg_agents.cli.commands.trace import resolve_session_dir, SessionResolveError
        (conversations_dir / "cli_1234-abcd").mkdir()
        (conversations_dir / "cli_1234-efgh").mkdir()
        with pytest.raises(SessionResolveError) as exc:
            resolve_session_dir("1234", conversations_dir=conversations_dir)
        assert exc.value.reason == "ambiguous"

    def test_not_found(self, conversations_dir):
        from akg_agents.cli.commands.trace import resolve_session_dir, SessionResolveError
        (conversations_dir / "cli_exists").mkdir()
        with pytest.raises(SessionResolveError) as exc:
            resolve_session_dir("missing", conversations_dir=conversations_dir)
        assert exc.value.reason == "not_found"


class TestConversationHistory:
    def test_get_conversation_history(self, ts):
        ts.add_node(
            action={"type": "ask_user", "arguments": {"user_input": "请生成 add kernel"}},
            result={"message": "请确认 block_size: 128"})
        ts.add_node(action={"type": "finish"}, result={"message": "已完成，性能: 0.85"})
        history = ts.get_conversation_history(max_turns=10)
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert "add kernel" in history[0]["content"]
