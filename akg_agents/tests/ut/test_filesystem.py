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
FileSystemState 单元测试（含 diff/merge 和 snapshot 能力）
"""

import pytest
from pathlib import Path

from akg_agents.core_v2.filesystem import (
    FileSystemState,
    TraceSystem,
    NodeState,
    ActionRecord,
    ActionHistoryFact,
    ThinkingState,
    PendingTool,
    PendingToolsState,
    NodeNotFoundError,
)


@pytest.fixture
def fs(tmp_path):
    fs = FileSystemState("test_task", base_dir=str(tmp_path))
    fs.initialize_task()
    return fs


# ==================== 初始化 ====================

class TestInitialization:
    def test_creates_directories(self, fs):
        assert fs.task_dir.exists()
        assert fs.nodes_dir.exists()
        assert fs.logs_dir.exists()
        assert (fs.nodes_dir / "root").exists()

    def test_creates_root_node(self, fs):
        root = fs.load_node_state("root")
        assert root.node_id == "root"
        assert root.turn == 0
        assert root.status == "init"

    def test_sets_current_node(self, fs):
        assert fs.get_current_node() == "root"

    def test_force_reinitialize(self, fs):
        fs.save_node_state("n1", NodeState(node_id="n1", turn=1, status="running"))
        fs.initialize_task(force=True)
        assert not fs.node_exists("n1")
        assert fs.node_exists("root")

    def test_task_exists(self, tmp_path):
        fs = FileSystemState("t", base_dir=str(tmp_path))
        assert not fs.task_exists()
        fs.initialize_task()
        assert fs.task_exists()


# ==================== 节点状态 ====================

class TestNodeState:
    def test_save_load(self, fs):
        state = NodeState(node_id="n1", turn=1, status="running",
                          agent_info={"name": "K"}, metrics={"tok": 100})
        fs.save_node_state("n1", state)
        loaded = fs.load_node_state("n1")
        assert loaded.status == "running"
        assert loaded.agent_info["name"] == "K"
        assert loaded.metrics["tok"] == 100

    def test_load_nonexistent_raises(self, fs):
        with pytest.raises(NodeNotFoundError):
            fs.load_node_state("nope")

    def test_update(self, fs):
        fs.save_node_state("n1", NodeState(node_id="n1", turn=1, status="running"))
        updated = fs.update_node_state("n1", turn=2, status="done", metrics={"p": 0.85})
        assert updated.turn == 2
        assert updated.status == "done"
        assert updated.metrics["p"] == 0.85

    def test_copy(self, fs):
        fs.save_node_state("n1", NodeState(node_id="n1", turn=1, status="ok",
                                           metrics={"tok": 1000}))
        new = fs.copy_node_state("n1", "n2")
        assert new.node_id == "n2"
        assert new.metrics["tok"] == 1000


# ==================== 代码文件 ====================

class TestCodeFiles:
    def test_save_load(self, fs):
        fs.save_node_state("n1", NodeState(node_id="n1", turn=1, status="ok"))
        fs.save_code_file("n1", "k.cu", "__global__ void k(){}")
        assert fs.load_code_file("n1", "k.cu") == "__global__ void k(){}"

    def test_updates_file_state(self, fs):
        fs.save_node_state("n1", NodeState(node_id="n1", turn=1, status="ok"))
        code = "content"
        fs.save_code_file("n1", "k.cu", code)
        state = fs.load_node_state("n1")
        assert "code/k.cu" in state.file_state
        assert state.file_state["code/k.cu"]["size"] == len(code)

    def test_list(self, fs):
        fs.save_node_state("n1", NodeState(node_id="n1", turn=1, status="ok"))
        fs.save_code_file("n1", "a.py", "1")
        fs.save_code_file("n1", "b.py", "2")
        files = fs.list_code_files("n1")
        assert "a.py" in files and "b.py" in files

    def test_nested_dirs(self, fs):
        fs.save_code_file("root", "a/b/c/deep.py", "# deep")
        loaded = fs.load_code_file("root", "a/b/c/deep.py")
        assert loaded.strip() == "# deep"

    def test_overwrite(self, fs):
        fs.save_code_file("root", "k.py", "v1")
        fs.save_code_file("root", "k.py", "v2")
        assert fs.load_code_file("root", "k.py").strip() == "v2"

    def test_load_nonexistent(self, fs):
        with pytest.raises(FileNotFoundError):
            fs.load_code_file("root", "nope.py")


# ==================== 动作历史 ====================

class TestActionHistory:
    def test_save_load(self, fs):
        a = ActionRecord(action_id="a1", tool_name="call_coder",
                         arguments={"s": "opt"}, result={"ok": True})
        h = ActionHistoryFact(node_id="n1", parent_node_id="root", turn=1)
        h.add_action(a)
        fs.save_action_history_fact("n1", h)
        loaded = fs.load_action_history_fact("n1")
        assert len(loaded.actions) == 1
        assert loaded.actions[0].tool_name == "call_coder"

    def test_load_nonexistent_returns_empty(self, fs):
        h = fs.load_action_history_fact("nope")
        assert len(h.actions) == 0

    def test_append(self, fs):
        a1 = ActionRecord(action_id="a1", tool_name="designer", result={"ok": True})
        a2 = ActionRecord(action_id="a2", tool_name="coder", result={"ok": True})
        fs.append_action("n1", a1, parent_node_id="root", turn=1)
        fs.append_action("n1", a2, turn=1)
        h = fs.load_action_history_fact("n1")
        assert len(h.actions) == 2

    def test_incremental_no_duplicate(self, fs):
        h1 = ActionHistoryFact(node_id="n1", parent_node_id="root", turn=1)
        h1.add_action(ActionRecord(action_id="a1", tool_name="d", result={}))
        fs.save_action_history_fact("n1", h1)
        h2 = ActionHistoryFact(node_id="n2", parent_node_id="n1", turn=2)
        h2.add_action(ActionRecord(action_id="a2", tool_name="c", result={}))
        fs.save_action_history_fact("n2", h2)
        loaded = fs.load_action_history_fact("n2")
        assert len(loaded.actions) == 1
        assert loaded.actions[0].action_id == "a2"


# ==================== Thinking ====================

class TestThinking:
    def test_save_load(self, fs):
        t = ThinkingState(node_id="n1", turn=1,
                          current_plan={"goal": "softmax"},
                          latest_thinking="优化内存")
        fs.save_thinking("n1", t)
        loaded = fs.load_thinking("n1")
        assert loaded.current_plan["goal"] == "softmax"
        assert loaded.latest_thinking == "优化内存"

    def test_load_nonexistent(self, fs):
        assert fs.load_thinking("nope") is None


# ==================== Pending Tools ====================

class TestPendingTools:
    def test_save_load(self, fs):
        tool = PendingTool(tool_call_id="c1", tool_name="file_write",
                           arguments={"path": "out.cu"})
        p = PendingToolsState(node_id="n1", turn=1)
        p.add_pending_tool(tool)
        fs.save_pending_tools("n1", p)
        loaded = fs.load_pending_tools("n1")
        assert len(loaded.pending_tools) == 1
        assert loaded.pending_tools[0].tool_name == "file_write"

    def test_mark_completed(self, fs):
        tool = PendingTool(tool_call_id="c1", tool_name="fw", arguments={})
        fs.add_pending_tool("n1", tool, turn=1)
        fs.mark_tool_completed("n1", "c1")
        loaded = fs.load_pending_tools("n1")
        assert loaded.pending_tools[0].status == "completed"

    def test_mark_completed_raise_if_not_found(self, fs):
        from akg_agents.core_v2.filesystem import FileSystemStateError
        tool = PendingTool(tool_call_id="c1", tool_name="fw", arguments={})
        fs.add_pending_tool("n1", tool, turn=1)
        with pytest.raises(FileSystemStateError):
            fs.mark_tool_completed("n1", "nope", raise_if_not_found=True)

    def test_clear(self, fs):
        fs.save_node_state("n1", NodeState(node_id="n1", turn=1, status="ok"))
        tool = PendingTool(tool_call_id="c1", tool_name="fw", arguments={})
        fs.add_pending_tool("n1", tool, turn=1)
        fs.clear_pending_tools("n1")
        assert len(fs.load_pending_tools("n1").pending_tools) == 0


# ==================== System Prompts ====================

class TestSystemPrompts:
    def test_save_load(self, fs):
        fs.save_system_prompt("n1", turn=1, prompt="你是 coder")
        assert fs.load_system_prompt("n1", turn=1) == "你是 coder"

    def test_nonexistent(self, fs):
        assert fs.load_system_prompt("n1", turn=1) is None

    def test_get_latest(self, fs):
        fs.save_system_prompt("n1", turn=1, prompt="P1")
        fs.save_system_prompt("n1", turn=2, prompt="P2")
        assert fs.get_latest_system_prompt("n1") == "P2"


# ==================== Workspace Switch & Export ====================

class TestWorkspaceSwitch:
    def test_switch_restores_snapshot(self, fs):
        fs.save_code_file("root", "k.py", "v1")
        fs.copy_node_state("root", "a")
        fs.set_current_node("a")
        fs.save_code_file("a", "k.py", "v2-a")
        fs.set_current_node("root")
        fs.copy_node_state("root", "b")
        fs.set_current_node("b")
        fs.save_code_file("b", "k.py", "v2-b")

        fs.set_current_node("a")
        assert (fs.workspace_dir / "k.py").read_text().strip() == "v2-a"
        fs.set_current_node("b")
        assert (fs.workspace_dir / "k.py").read_text().strip() == "v2-b"
        fs.set_current_node("root")
        assert (fs.workspace_dir / "k.py").read_text().strip() == "v1"

    def test_export_node_code(self, fs, tmp_path):
        fs.save_code_file("root", "k.py", "# kernel")
        fs.copy_node_state("root", "exp")
        fs.set_current_node("exp")
        fs.save_code_file("exp", "test.py", "# test")
        export_dir = tmp_path / "export"
        fs.export_node_code("exp", str(export_dir))
        assert (export_dir / "k.py").exists()
        assert (export_dir / "test.py").read_text().strip() == "# test"


# ==================== Diff & Merge ====================

class TestDiffMerge:
    @pytest.fixture
    def env(self, tmp_path):
        fs = FileSystemState("diff_task", base_dir=str(tmp_path))
        fs.initialize_task()
        trace = TraceSystem("diff_task", base_dir=str(tmp_path))
        trace.initialize()
        return fs, trace

    def test_diff_file(self, env):
        fs, _ = env
        fs.save_code_file("root", "main.py", "line 1\nline 2\nline 3\n")
        fs.copy_node_state("root", "nb")
        fs.save_code_file("nb", "main.py", "line 1\nline 2 mod\nline 3\nline 4\n")
        diff = fs.diff_file("root", "nb", "main.py")
        assert "+line 2 mod" in diff
        assert "+line 4" in diff

    def test_diff_nodes_patch(self, env):
        fs, _ = env
        fs.save_code_file("root", "a.py", "a\n")
        fs.save_code_file("root", "shared.py", "shared\n")
        fs.copy_node_state("root", "nb")
        fs.save_code_file("nb", "b.py", "b\n")
        fs.save_code_file("nb", "a.py", "a mod\n")
        patch = fs.diff_nodes("root", "nb")
        content = patch.read_text()
        assert "+++ b/a.py" in content
        assert "+++ b/b.py" in content
        assert "shared.py" not in content

    def test_find_lca(self, env):
        _, trace = env
        n1 = trace.add_node({"t": "a"}, {"r": "a"})
        n2 = trace.add_node({"t": "b"}, {"r": "b"})
        n3 = trace.add_node({"t": "c"}, {"r": "c"})
        trace.switch_node(n1)
        n4 = trace.add_node({"t": "d"}, {"r": "d"})
        assert trace.find_lca(n3, n4) == n1

    def test_merge_no_conflict(self, env):
        fs, trace = env
        fs.save_code_file("root", "main.py", "base\n")
        n1 = trace.add_node({"t": "a"}, {"r": "ok"})
        fs.save_code_file(n1, "a.py", "a\n")
        trace.switch_node("root")
        n2 = trace.add_node({"t": "b"}, {"r": "ok"})
        fs.save_code_file(n2, "b.py", "b\n")
        m = trace.merge_nodes(n2, n1)
        assert fs.load_code_file(m, "main.py") == "base\n"
        assert fs.load_code_file(m, "a.py") == "a\n"
        assert fs.load_code_file(m, "b.py") == "b\n"

    def test_merge_conflict(self, env):
        fs, trace = env
        fs.save_code_file("root", "m.py", "base\n")
        n1 = trace.add_node({"t": "a"}, {"r": "a"})
        fs.save_code_file(n1, "m.py", "A\n")
        trace.switch_node("root")
        n2 = trace.add_node({"t": "b"}, {"r": "b"})
        fs.save_code_file(n2, "m.py", "B\n")
        m = trace.merge_nodes(n2, n1)
        content = fs.load_code_file(m, "m.py")
        assert "<<<<<<< YOURS" in content
        assert "=======" in content
        assert ">>>>>>> THEIRS" in content

    def test_auto_merge_different_regions(self, env):
        fs, trace = env
        fs.save_code_file("root", "m.py", "l1\nl2\nl3\nl4\n")
        n1 = trace.add_node({"t": "top"}, {"r": "ok"})
        fs.save_code_file(n1, "m.py", "l1 MOD A\nl2\nl3\nl4\n")
        trace.switch_node("root")
        n2 = trace.add_node({"t": "bot"}, {"r": "ok"})
        fs.save_code_file(n2, "m.py", "l1\nl2\nl3\nl4 MOD B\n")
        m = trace.merge_nodes(n2, n1)
        content = fs.load_code_file(m, "m.py")
        assert "l1 MOD A" in content
        assert "l4 MOD B" in content
        assert "<<<<<<< YOURS" not in content
