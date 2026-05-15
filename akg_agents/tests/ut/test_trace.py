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

含：初始化/节点操作/分叉/路径/动作历史/叶节点/状态/恢复/
    .traceconfig/blame/compression/可视化/集成 (CoW + 状态恢复)
"""

import json
import shutil
import tempfile

import pytest
from pathlib import Path

from akg_agents.core_v2.filesystem import (
    TraceSystem,
    FileSystemState,
    NodeState,
    ActionRecord,
    ActionHistoryFact,
    NodeNotFoundError,
    TraceNotInitializedError,
    ActionHistoryCompressed,
    TraceSystemError,
)
from akg_agents.core_v2.filesystem.compressor import ActionCompressor
from akg_agents.core_v2.filesystem.trace_visualizer import (
    format_node_text,
    format_node_detail_rich,
    visualize_rich,
    visualize_text,
    _short_id,
    _summarize_result,
)


# ================================================================
#  通用 Fixture
# ================================================================

@pytest.fixture
def trace(tmp_path):
    ts = TraceSystem("test_task", base_dir=str(tmp_path))
    ts.initialize()
    return ts


@pytest.fixture
def env(tmp_path):
    task_id = "integration_task"
    fs = FileSystemState(task_id, base_dir=str(tmp_path))
    trace = TraceSystem(task_id, base_dir=str(tmp_path))
    fs.initialize_task()
    trace.initialize()
    return fs, trace, task_id


# ================================================================
#  1. 初始化
# ================================================================

class TestInit:
    def test_creates_trace_json(self, tmp_path):
        ts = TraceSystem("t1", base_dir=str(tmp_path))
        ts.initialize()
        assert (tmp_path / "conversations" / "t1" / "trace.json").exists()

    def test_creates_root(self, trace):
        root = trace.get_node("root")
        assert root.node_id == "root"
        assert root.parent_id is None
        assert root.children == []
        assert trace.get_current_node() == "root"

    def test_force_reinit(self, trace):
        trace.add_node({"type": "x"}, {"ok": True})
        trace.initialize(force=True)
        assert trace.get_current_node() == "root"
        assert len(trace.trace.tree) == 1

    def test_load_existing(self, tmp_path):
        t1 = TraceSystem("t", base_dir=str(tmp_path))
        t1.initialize()
        t1.add_node({"type": "a"}, {"ok": True})
        nid = t1.get_current_node()

        t2 = TraceSystem("t", base_dir=str(tmp_path))
        t2.initialize()
        assert t2.get_current_node() == nid


# ================================================================
#  2. 节点操作
# ================================================================

class TestNodeOps:
    def test_add_single(self, trace):
        nid = trace.add_node({"type": "call_designer"}, {"success": True, "output": "..."})
        assert nid == "node_001"
        assert trace.get_current_node() == "node_001"
        node = trace.get_node(nid)
        assert node.parent_id == "root"
        assert node.action["type"] == "call_designer"

    def test_chain(self, trace):
        n1 = trace.add_node({"type": "a"}, {"ok": True})
        n2 = trace.add_node({"type": "b"}, {"ok": True})
        n3 = trace.add_node({"type": "c"}, {"ok": True})
        assert trace.get_node(n1).parent_id == "root"
        assert trace.get_node(n2).parent_id == n1
        assert trace.get_node(n3).parent_id == n2
        assert trace.get_node("root").children == [n1]

    def test_switch(self, trace):
        n1 = trace.add_node({"type": "a"}, {"ok": True})
        trace.add_node({"type": "b"}, {"ok": True})
        trace.switch_node(n1)
        assert trace.get_current_node() == n1

    def test_switch_nonexistent(self, trace):
        with pytest.raises(NodeNotFoundError):
            trace.switch_node("nope")

    def test_get_nonexistent(self, trace):
        with pytest.raises(NodeNotFoundError):
            trace.get_node("nope")


# ================================================================
#  3. 分叉
# ================================================================

class TestFork:
    def test_auto_fork(self, trace):
        n1 = trace.add_node({"type": "a"}, {"ok": True})
        n2 = trace.add_node({"type": "b"}, {"ok": True})
        trace.switch_node(n1)
        n3 = trace.add_node({"type": "c"}, {"ok": True})
        assert trace.get_node(n1).children == [n2, n3]
        assert trace.get_node(n2).parent_id == n1
        assert trace.get_node(n3).parent_id == n1

    def test_multiple_forks(self, trace):
        n1 = trace.add_node({"type": "d"}, {"ok": True})
        trace.add_node({"type": "v1"}, {"ok": True})
        trace.switch_node(n1)
        trace.add_node({"type": "v2"}, {"ok": True})
        trace.switch_node(n1)
        trace.add_node({"type": "v3"}, {"ok": True})
        assert len(trace.get_node(n1).children) == 3

    def test_parallel_forks(self, trace):
        n1 = trace.add_node({"type": "s"}, {"ok": True})
        forks = trace.create_parallel_forks(n=3, action_template={"type": "p"})
        assert len(forks) == 3
        for fid in forks:
            assert trace.get_node(fid).parent_id == n1
        assert len(trace.get_node(n1).children) == 3

    def test_complete_fork(self, trace):
        trace.add_node({"type": "s"}, {"ok": True})
        forks = trace.create_parallel_forks(n=2, action_template={"type": "p"})
        trace.complete_fork(forks[0], result={"ok": True, "perf": 0.8}, metrics={"perf": 0.8})
        assert trace.get_node(forks[0]).result["perf"] == 0.8
        assert trace.get_node(forks[0]).state_snapshot["status"] == "completed"


# ================================================================
#  4. 路径 & 动作历史
# ================================================================

class TestPathAndHistory:
    def test_path_to_node(self, trace):
        n1 = trace.add_node({"type": "a"}, {"ok": True})
        n2 = trace.add_node({"type": "b"}, {"ok": True})
        n3 = trace.add_node({"type": "c"}, {"ok": True})
        assert trace.get_path_to_node(n3) == ["root", n1, n2, n3]
        assert trace.get_path_to_node("root") == ["root"]
        assert trace.get_path_to_node("nope") == []

    def test_depth(self, trace):
        n1 = trace.add_node({"type": "a"}, {"ok": True})
        n2 = trace.add_node({"type": "b"}, {"ok": True})
        assert trace.get_node_depth("root") == 0
        assert trace.get_node_depth(n1) == 1
        assert trace.get_node_depth(n2) == 2

    def test_full_history(self, trace):
        trace.add_node({"type": "call_designer"}, {"ok": True})
        trace.add_node({"type": "call_coder"}, {"ok": True})
        n3 = trace.add_node({"type": "verify"}, {"ok": True})
        h = trace.get_full_action_history(n3)
        assert len(h) == 3
        assert [a.tool_name for a in h] == ["call_designer", "call_coder", "verify"]

    def test_history_with_fork(self, trace):
        n1 = trace.add_node({"type": "design"}, {"ok": True})
        n2 = trace.add_node({"type": "code_v1"}, {"ok": True})
        trace.switch_node(n1)
        n3 = trace.add_node({"type": "code_v2"}, {"ok": True})
        h2 = trace.get_full_action_history(n2)
        h3 = trace.get_full_action_history(n3)
        assert [a.tool_name for a in h2] == ["design", "code_v1"]
        assert [a.tool_name for a in h3] == ["design", "code_v2"]

    def test_history_nonexistent(self, trace):
        assert trace.get_full_action_history("nope") == []


# ================================================================
#  5. 叶节点 & 对比
# ================================================================

class TestLeafAndCompare:
    def test_leaves(self, trace):
        n1 = trace.add_node({"type": "s"}, {"ok": True})
        n2 = trace.add_node({"type": "a"}, {"ok": True})
        trace.switch_node(n1)
        n3 = trace.add_node({"type": "b"}, {"ok": True})
        assert set(trace.get_all_leaf_nodes()) == {n2, n3}

    def test_best_leaf(self, trace):
        trace.add_node({"type": "a"}, {"ok": True}, metrics={"perf": 0.65})
        trace.switch_node("root")
        n2 = trace.add_node({"type": "b"}, {"ok": True}, metrics={"perf": 0.85})
        assert trace.get_best_leaf_node("perf") == n2

    def test_compare(self, trace):
        n1 = trace.add_node({"type": "d"}, {"ok": True}, metrics={"tok": 1000})
        n2 = trace.add_node({"type": "c1"}, {"ok": True}, metrics={"tok": 2000})
        n3 = trace.add_node({"type": "v"}, {"ok": True, "perf": 0.65}, metrics={"tok": 500, "performance": 0.65})
        trace.switch_node(n2)
        trace.add_node({"type": "c2"}, {"ok": True}, metrics={"tok": 2500})
        n5 = trace.add_node({"type": "v"}, {"ok": True, "perf": 0.85}, metrics={"tok": 500, "performance": 0.85})
        cmp = trace.compare_nodes(n3, n5)
        assert cmp["fork_point"] == n2
        assert cmp["metrics_1"]["performance"] == 0.65
        assert cmp["metrics_2"]["performance"] == 0.85


# ================================================================
#  6. 节点状态 & 恢复
# ================================================================

class TestNodeStatus:
    def test_mark_completed(self, trace):
        nid = trace.add_node({"type": "s"}, {"ok": True})
        trace.mark_node_completed(nid, metrics={"perf": 0.9})
        assert trace.get_node(nid).state_snapshot["status"] == "completed"
        assert trace.get_node(nid).metrics["perf"] == 0.9

    def test_mark_failed(self, trace):
        nid = trace.add_node({"type": "s"}, {"ok": False})
        trace.mark_node_failed(nid, error="Compile error")
        assert trace.get_node(nid).state_snapshot["status"] == "failed"
        assert trace.get_node(nid).result["error"] == "Compile error"

    def test_update_result(self, trace):
        nid = trace.add_node({"type": "s"}, {"ok": True})
        trace.update_node_result(nid, result={"ok": True, "perf": 0.85}, metrics={"tok": 1500})
        n = trace.get_node(nid)
        assert n.result["perf"] == 0.85
        assert n.metrics["tok"] == 1500


class TestResume:
    def test_resume_info(self, tmp_path):
        ts = TraceSystem("t_resume", base_dir=str(tmp_path))
        ts.initialize()
        ts.add_node({"type": "s1"}, {"ok": True})
        ts.add_node({"type": "s2"}, {"ok": True})
        info = ts.get_resume_info()
        assert info["task_id"] == "t_resume"
        assert len(info["action_history"]) == 2
        assert len(info["path"]) == 3

    def test_resume_from_existing(self, tmp_path):
        t1 = TraceSystem("t_re", base_dir=str(tmp_path))
        t1.initialize()
        t1.add_node({"type": "s1"}, {"ok": True})
        t1.add_node({"type": "s2"}, {"ok": True})
        orig = t1.get_current_node()

        t2 = TraceSystem("t_re", base_dir=str(tmp_path))
        t2.initialize()
        assert t2.get_current_node() == orig
        n = t2.add_node({"type": "s3"}, {"ok": True})
        assert len(t2.get_full_action_history(n)) == 3

    def test_action_counter_preserved(self, tmp_path):
        t1 = TraceSystem("t_ac", base_dir=str(tmp_path))
        t1.initialize()
        for _ in range(3):
            t1.add_node({"type": "x"}, {"ok": True})
        last_id = t1.get_full_action_history(t1.get_current_node())[-1].action_id

        t2 = TraceSystem("t_ac", base_dir=str(tmp_path))
        t2.initialize()
        t2.add_node({"type": "y"}, {"ok": True})
        new_id = t2.get_full_action_history(t2.get_current_node())[-1].action_id
        assert int(new_id.split("_")[1]) > int(last_id.split("_")[1])


# ================================================================
#  7. .traceconfig
# ================================================================

class TestTraceConfig:
    def _write_traceconfig(self, ts, content):
        """Write .traceconfig to the correct task_dir location."""
        cfg = ts.fs.task_dir / ".traceconfig"
        cfg.write_text(content, encoding="utf-8")
        ts._load_trace_config()

    def test_default_only_code(self, tmp_path):
        ts = TraceSystem("tc_def", base_dir=str(tmp_path))
        ts.initialize()
        fs = ts.fs
        fs.save_code_file("root", "main.py", "print('hello')")
        log_dir = fs.get_node_dir("root") / "logs"
        log_dir.mkdir()
        (log_dir / "app.log").write_text("log")
        n1 = ts.add_node({"type": "t"}, {"r": "ok"})
        assert fs.load_code_file(n1, "main.py") == "print('hello')"
        assert not (fs.get_node_dir(n1) / "logs").exists()

    def test_include_logs(self, tmp_path):
        ts = TraceSystem("tc_log", base_dir=str(tmp_path))
        ts.initialize()
        self._write_traceconfig(ts, "code/\nlogs/\n")
        fs = ts.fs
        fs.save_code_file("root", "main.py", "print('hello')")
        log_dir = fs.get_node_dir("root") / "logs"
        log_dir.mkdir()
        (log_dir / "app.log").write_text("log content")
        n1 = ts.add_node({"type": "t"}, {"r": "ok"})
        assert (fs.get_node_dir(n1) / "logs" / "app.log").read_text() == "log content"

    def test_exclude_pattern(self, tmp_path):
        ts = TraceSystem("tc_exc", base_dir=str(tmp_path))
        ts.initialize()
        self._write_traceconfig(ts, "code/\n!**/*.tmp\n")
        fs = ts.fs
        fs.save_code_file("root", "main.py", "ok")
        fs.save_code_file("root", "temp.tmp", "ignore")
        n1 = ts.add_node({"type": "t"}, {"r": "ok"})
        assert fs.load_code_file(n1, "main.py") == "ok"
        assert not (fs.get_node_dir(n1) / "code" / "temp.tmp").exists()

    def test_glob_pattern(self, tmp_path):
        ts = TraceSystem("tc_glob", base_dir=str(tmp_path))
        ts.initialize()
        self._write_traceconfig(ts, "code/\nartifacts/*.json\n")
        fs = ts.fs
        rt = fs.get_node_dir("root")
        (rt / "artifacts").mkdir()
        (rt / "artifacts" / "result.json").write_text("{}")
        (rt / "artifacts" / "data.bin").write_text("binary")
        n1 = ts.add_node({"type": "t"}, {"r": "ok"})
        n1_dir = fs.get_node_dir(n1)
        assert (n1_dir / "artifacts" / "result.json").exists()
        assert not (n1_dir / "artifacts" / "data.bin").exists()


# ================================================================
#  8. blame
# ================================================================

class TestBlame:
    def test_linear_create_and_modify(self, trace):
        n1 = trace.add_node({"type": "generate"}, {"r": "ok"})
        trace.fs.save_code_file(n1, "main.py", "v1\n")
        n2 = trace.add_node({"type": "modify"}, {"r": "ok"})
        trace.fs.save_code_file(n2, "main.py", "v2\n")
        n3 = trace.add_node({"type": "analyze"}, {"r": "ok"})
        records = trace.blame_file(n3, "main.py")
        assert len(records) == 2
        assert records[0]["change_type"] == "created"
        assert records[1]["change_type"] == "modified"
        assert records[0]["checksum"] != records[1]["checksum"]

    def test_file_at_root(self, trace):
        trace.fs.save_code_file("root", "cfg.py", "c={}\n")
        n1 = trace.add_node({"type": "s"}, {"r": "ok"})
        records = trace.blame_file(n1, "cfg.py")
        assert len(records) == 1
        assert records[0]["node_id"] == "root"

    def test_not_found(self, trace):
        n1 = trace.add_node({"type": "s"}, {"r": "ok"})
        assert trace.blame_file(n1, "nope.py") == []

    def test_fork_independent(self, trace):
        n1 = trace.add_node({"type": "gen"}, {"r": "ok"})
        trace.fs.save_code_file(n1, "m.py", "base\n")
        n2 = trace.add_node({"type": "mod_a"}, {"r": "ok"})
        trace.fs.save_code_file(n2, "m.py", "A\n")
        trace.switch_node(n1)
        n3 = trace.add_node({"type": "mod_b"}, {"r": "ok"})
        trace.fs.save_code_file(n3, "m.py", "B\n")
        ra = trace.blame_file(n2, "m.py")
        rb = trace.blame_file(n3, "m.py")
        assert ra[1]["action"] == "mod_a"
        assert rb[1]["action"] == "mod_b"
        assert ra[1]["checksum"] != rb[1]["checksum"]

    def test_deletion(self, trace):
        n1 = trace.add_node({"type": "gen"}, {"r": "ok"})
        trace.fs.save_code_file(n1, "t.py", "tmp\n")
        n2 = trace.add_node({"type": "cleanup"}, {"r": "ok"})
        state = trace.fs.load_node_state(n2)
        if "code/t.py" in state.file_state:
            del state.file_state["code/t.py"]
        trace.fs.save_node_state(n2, state)
        records = trace.blame_file(n2, "t.py")
        assert records[-1]["change_type"] == "deleted"

    def test_same_content_no_change(self, trace):
        n1 = trace.add_node({"type": "gen"}, {"r": "ok"})
        trace.fs.save_code_file(n1, "m.py", "same\n")
        n2 = trace.add_node({"type": "mod"}, {"r": "ok"})
        trace.fs.save_code_file(n2, "m.py", "same\n")
        assert len(trace.blame_file(n2, "m.py")) == 1

    def test_all_files(self, trace):
        n1 = trace.add_node({"type": "gen"}, {"r": "ok"})
        trace.fs.save_code_file(n1, "main.py", "m v1\n")
        trace.fs.save_code_file(n1, "utils.py", "u v1\n")
        n2 = trace.add_node({"type": "mod"}, {"r": "ok"})
        trace.fs.save_code_file(n2, "main.py", "m v2\n")
        n3 = trace.add_node({"type": "add"}, {"r": "ok"})
        trace.fs.save_code_file(n3, "helper.py", "h v1\n")
        result = trace.blame_all_files(n3)
        assert set(result.keys()) == {"main.py", "utils.py", "helper.py"}
        assert len(result["main.py"]) == 2
        assert len(result["utils.py"]) == 1


# ================================================================
#  9. Compression (async)
# ================================================================

class MockLLMClient:
    def __init__(self):
        self.generate_calls = []

    async def generate(self, messages, **kwargs):
        self.generate_calls.append({"messages": messages, "kwargs": kwargs})
        return {
            "content": "用户请求生成 ReLU 算子。首先调用 op_task_builder 生成了 PyTorch 任务代码。",
            "usage": {"total_tokens": 150},
        }


@pytest.fixture
def mock_llm():
    return MockLLMClient()


def _action(aid, tool, result=None):
    return ActionRecord(action_id=aid, tool_name=tool,
                        arguments={}, result=result or {})


class TestCompression:
    @pytest.mark.asyncio
    async def test_short_history_no_compress(self, mock_llm):
        compressor = ActionCompressor(mock_llm)
        history = [_action("a1", "builder"), _action("a2", "designer")]
        compressed = await compressor.compress_history(history)
        assert len(compressed) == 2
        assert len(mock_llm.generate_calls) == 0

    @pytest.mark.asyncio
    async def test_long_history_compress(self, mock_llm):
        compressor = ActionCompressor(mock_llm)
        history = [
            _action("a1", "call_op_task_builder"),
            _action("a2", "call_designer"),
            _action("a3", "call_coder_only"),
            _action("a4", "call_kernel_verifier", {"error": "syntax"}),
            _action("a5", "call_coder_only"),
            _action("a6", "call_kernel_verifier", {"perf": 120.5}),
        ]
        compressed = await compressor.compress_history(history)
        assert len(compressed) < len(history)
        assert compressed[0].tool_name == "history_summary"
        assert compressed[-1].action_id == "a6"
        assert len(mock_llm.generate_calls) == 1

    @pytest.mark.asyncio
    async def test_integration_compress_and_cache(self, trace, mock_llm):
        for i in range(6):
            trace.add_node({"type": f"step_{i}"}, {"r": f"v{i}"})
        nid = trace.get_current_node()

        c1 = await trace.get_compressed_history_for_llm(mock_llm, nid, max_tokens=2000)
        assert len(c1) < 6
        assert c1[0].tool_name == "history_summary"

        mock_llm.generate_calls.clear()
        c2 = await trace.get_compressed_history_for_llm(mock_llm, nid)
        assert len(c2) < 6
        assert len(mock_llm.generate_calls) == 0


# ================================================================
#  10. 可视化
# ================================================================

@pytest.fixture
def linear_trace(tmp_path):
    ts = TraceSystem("viz_lin", base_dir=str(tmp_path))
    ts.initialize()
    for i in range(11):
        ts.add_node(action={"type": f"step_{i+1}"}, result={})
    ts.switch_node("node_007")
    return ts


@pytest.fixture
def fork2_trace(tmp_path):
    ts = TraceSystem("viz_f2", base_dir=str(tmp_path))
    ts.initialize()
    n1 = ts.add_node(action={"type": "plan"}, result={})
    ts.add_node(action={"type": "gen_v1"}, result={})
    ts.add_node(action={"type": "compile"}, result={})
    ts.add_node(action={"type": "profile"}, result={})
    ts.switch_node(n1)
    ts.add_node(action={"type": "gen_v2"}, result={})
    ts.add_node(action={"type": "compile_v2"}, result={})
    ts.switch_node("node_004")
    return ts


@pytest.fixture
def fork8_trace(tmp_path):
    ts = TraceSystem("viz_f8", base_dir=str(tmp_path))
    ts.initialize()
    n1 = ts.add_node(action={"type": "plan"}, result={})
    branches = []
    for i in range(8):
        ts.switch_node(n1)
        b1 = ts.add_node(action={"type": f"gen_v{i+1}"}, result={})
        b2 = ts.add_node(action={"type": "compile"}, result={})
        b3 = ts.add_node(action={"type": "profile"}, result={})
        branches.append((b1, b2, b3))
    ts.switch_node(branches[3][2])
    return ts, branches


class TestVisualizeHelpers:
    def test_short_id(self):
        assert _short_id("node_005") == "005"
        assert _short_id("root") == "root"

    def test_summarize_result(self):
        assert _summarize_result(None) == ""
        assert "✅" in _summarize_result({"success": True})
        assert "性能" in _summarize_result({"performance": 0.85})


class TestVisualizeRich:
    def test_linear(self, linear_trace):
        markup = str(linear_trace.visualize_tree_rich())
        assert "007" in markup
        assert "当前" in markup
        assert "⋮" in markup

    def test_fork2(self, fork2_trace):
        markup = str(fork2_trace.visualize_tree_rich())
        assert "004" in markup and "当前" in markup
        assert "↳" in markup and "分支" in markup

    def test_fork8(self, fork8_trace):
        ts, _ = fork8_trace
        markup = str(ts.visualize_tree_rich())
        assert "7 个分支" in markup

    def test_view_other_branch(self, fork8_trace):
        ts, _ = fork8_trace
        markup = str(ts.visualize_tree_rich(focus_node="node_002"))
        assert "gen_v1" in markup
        assert "◀" in markup


class TestVisualizeText:
    def test_linear(self, linear_trace):
        t = linear_trace.visualize_tree()
        assert "007" in t and "当前" in t

    def test_fork2(self, fork2_trace):
        t = fork2_trace.visualize_tree()
        assert "↳" in t and "分支" in t

    def test_fork8(self, fork8_trace):
        ts, _ = fork8_trace
        assert "7 个分支" in ts.visualize_tree()


# ================================================================
#  11. fork_ask_user
# ================================================================

class TestForkAskUser:
    @pytest.fixture
    def ask_trace(self, tmp_path):
        ts = TraceSystem("vu_ask", base_dir=str(tmp_path))
        ts.initialize()
        ts.add_node(
            action={"type": "kernelgen", "arguments": {"code": "v1"}},
            result={"success": True})
        ts.add_node(
            action={"type": "ask_user", "arguments": {"message": "要不要改一下策略？"}},
            result={"status": "responded", "user_response": "改一下", "message": "要不要改一下策略？"})
        ts.add_node(
            action={"type": "kernelgen", "arguments": {"code": "v2"}},
            result={"success": True})
        return ts

    def test_creates_sibling(self, ask_trace):
        new_id = ask_trace.fork_ask_user("node_002")
        original = ask_trace.get_node("node_002")
        forked = ask_trace.get_node(new_id)
        assert forked.parent_id == original.parent_id
        parent = ask_trace.get_node(original.parent_id)
        assert "node_002" in parent.children and new_id in parent.children

    def test_preserves_original(self, ask_trace):
        ask_trace.fork_ask_user("node_002")
        assert ask_trace.get_node("node_002").result["user_response"] == "改一下"
        assert "node_003" in ask_trace.get_node("node_002").children

    def test_resets_result(self, ask_trace):
        new_id = ask_trace.fork_ask_user("node_002")
        assert ask_trace.get_node(new_id).result["status"] == "waiting"

    def test_rejects_non_ask_user(self, ask_trace):
        with pytest.raises(TraceSystemError, match="ask_user"):
            ask_trace.fork_ask_user("node_001")

    def test_switches_current(self, ask_trace):
        new_id = ask_trace.fork_ask_user("node_002")
        assert ask_trace.get_current_node() == new_id


# ================================================================
#  12. Show 详情
# ================================================================

class TestShowDetail:
    @pytest.fixture
    def detail_trace(self, tmp_path):
        ts = TraceSystem("det", base_dir=str(tmp_path))
        ts.initialize()
        ts.add_node(
            action={"type": "ask_user", "arguments": {"message": "请选择优化策略"}},
            result={"status": "responded", "user_response": "使用共享内存", "message": "请选择优化策略"})
        ts.add_node(
            action={"type": "profile_kernel", "arguments": {"kernel_name": "matmul", "device": "gpu"}},
            result={"status": "success", "speedup": "2.5x", "gflops": 120},
            metrics={"duration_ms": 350})
        return ts

    def test_ask_user_detail(self, detail_trace):
        d = detail_trace.get_node_detail("node_001")
        assert "Agent 提问" in d and "请选择优化策略" in d
        assert "用户回答" in d and "使用共享内存" in d

    def test_tool_detail(self, detail_trace):
        d = detail_trace.get_node_detail("node_002")
        assert "profile_kernel" in d and "matmul" in d and "duration_ms" in d

    def test_rich_detail(self, detail_trace):
        markup = str(format_node_detail_rich(detail_trace, "node_001"))
        assert "Agent 提问" in markup

    def test_nonexistent(self, detail_trace):
        markup = str(format_node_detail_rich(detail_trace, "node_999"))
        assert "不存在" in markup

    def test_path_detail(self, detail_trace):
        d = detail_trace.get_path_detail("node_002")
        assert "路径" in d and "步数" in d and "Token" in d

    def test_ask_user_icon(self, detail_trace):
        node = detail_trace.get_node("node_001")
        assert "👤" in format_node_text("node_001", node)

    def test_non_ask_user_icon(self, detail_trace):
        node = detail_trace.get_node("node_002")
        assert "👤" not in format_node_text("node_002", node)


# ================================================================
#  13. 集成：CoW + 路径兼容 + 状态恢复
# ================================================================

class TestIntegration:
    def test_cow_breaks_link(self, env):
        fs, trace, _ = env
        fs.save_code_file("root", "code.py", "v_root")
        root_ino = (fs.get_code_snapshot_dir("root") / "code.py").stat().st_ino
        nodes = [trace.add_node(action={"i": i}, result={}) for i in range(3)]
        for nid in nodes:
            assert (fs.get_code_snapshot_dir(nid) / "code.py").stat().st_ino == root_ino
        fs.save_code_file(nodes[-1], "code.py", "modified")
        assert (fs.get_code_snapshot_dir(nodes[-1]) / "code.py").stat().st_ino != root_ino
        for nid in nodes[:-1]:
            assert (fs.get_code_snapshot_dir(nid) / "code.py").read_text() == "v_root"

    def test_verify_integrity(self, env):
        fs, trace, _ = env
        fs.save_code_file("root", "code.py", "correct")
        n1 = trace.add_node(action={"type": "t"}, result={})
        assert fs.verify_snapshot_integrity("root") == []
        (fs.get_code_snapshot_dir(n1) / "code.py").write_text("POLLUTED")
        assert len(fs.verify_snapshot_integrity(n1)) > 0

    def test_mixed_separators(self, env):
        fs, trace, _ = env
        fs.save_code_file("root", "src/core/main.py", "c1")
        fs.save_code_file("root", "scripts\\test.py", "c2")
        n = trace.add_node(action={}, result={})
        assert fs.load_code_file(n, "src/core/main.py") == "c1"
        assert fs.load_code_file(n, "scripts/test.py") == "c2"

    def test_resume_info_completeness(self, env):
        fs, trace, task_id = env
        fs.save_node_state("root", NodeState(
            node_id="root", turn=0, status="init",
            task_info={"task_id": task_id, "task_input": "relu"}))
        a1 = ActionRecord(action_id="n1", tool_name="ask_user",
                          arguments={"message": "确认"}, result={"status": "responded"})
        n1 = trace.add_node({"type": "ask_user"}, {"status": "responded"})
        fs.save_action_history_fact(n1, ActionHistoryFact(
            node_id=n1, parent_node_id="root", turn=1, actions=[a1]))
        info = trace.get_resume_info()
        for k in ("task_id", "current_node", "state", "action_history", "path"):
            assert k in info
