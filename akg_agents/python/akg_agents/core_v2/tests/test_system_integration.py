import pytest
import os
import shutil
import tempfile
from pathlib import Path
from akg_agents.core_v2.filesystem.state import FileSystemState
from akg_agents.core_v2.filesystem.trace_system import TraceSystem
from akg_agents.core_v2.filesystem.models import NodeState

class TestSystemIntegration:
    @pytest.fixture
    def setup_system(self):
        """Setup a clean environment for each test."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="akg_integration_test_"))
        self.task_id = "integration_task"
        self.fs = FileSystemState(self.task_id, base_dir=str(self.test_dir))
        self.trace = TraceSystem(self.task_id, base_dir=str(self.test_dir))
        
        self.fs.initialize_task()
        self.trace.initialize()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_end_to_end_workflow(self, setup_system):
        """
        Simulate a complex workflow:
        1. Root: Create base file.
        2. Node A: Fork from root, modify file.
        3. Node B: Fork from root, add new file.
        4. Node C: Merge Node B into Node A.
        5. Verify final state and integrity.
        """
        # 1. Root
        self.fs.save_code_file("root", "main.py", "base content\n")
        
        # 2. Node A (Fork from root)
        node_a = self.trace.add_node(action={"name": "edit_main"}, result={})
        self.trace.switch_node(node_a)
        self.fs.save_code_file(node_a, "main.py", "modified content\n")
        
        # 3. Node B (Fork from root)
        self.trace.switch_node("root")
        node_b = self.trace.add_node(action={"name": "add_util"}, result={})
        self.trace.switch_node(node_b)
        self.fs.save_code_file(node_b, "util.py", "util content\n")
        
        # 4. Node C (Merge B into A)
        self.trace.switch_node(node_a)
        node_c = self.trace.merge_nodes(node_a, node_b)
        self.trace.switch_node(node_c)
        
        # 5. Verify
        # main.py should be the one from node_a (modified)
        # util.py should be the one from node_b (added)
        assert self.fs.load_code_file(node_c, "main.py") == "modified content\n"
        assert self.fs.load_code_file(node_c, "util.py") == "util content\n"
        
        # Check integrity of workspace
        workspace_files = list(self.fs.workspace_dir.iterdir())
        workspace_basenames = [f.name for f in workspace_files if f.is_file()]
        assert "main.py" in workspace_basenames
        assert "util.py" in workspace_basenames

    def test_space_efficiency_hardlinks(self, setup_system):
        """Verify that unmodified files use hard links."""
        self.fs.save_code_file("root", "large.py", "A" * 1024)
        
        root_path = self.fs.get_code_snapshot_dir("root") / "large.py"
        initial_links = root_path.stat().st_nlink
        
        # Create 5 generations without modifying large.py
        current = "root"
        for i in range(5):
            next_node = self.trace.add_node(action={"idx": i}, result={})
            current = next_node
            
        final_path = self.fs.get_code_snapshot_dir(current) / "large.py"
        
        # Check hard link count. It should be 7 (root + 5 nodes + workspace if root is still active, 
        # but here we keep adding nodes which are snapshots).
        # Actually in copy_node_state, we link them.
        assert final_path.stat().st_nlink >= 6
        assert final_path.stat().st_ino == root_path.stat().st_ino
        
        # Modify in the last node and verify it breaks link
        self.trace.switch_node(current)
        self.fs.save_code_file(current, "large.py", "modified")
        
        assert final_path.stat().st_ino != root_path.stat().st_ino
        assert root_path.stat().st_nlink == 5 # Still 5 nodes sharing it

    def test_path_separator_compatibility(self, setup_system):
        """Verify that both / and \\ are handled correctly in filenames."""
        # Using Mixed separators
        self.fs.save_code_file("root", "src/core/main.py", "content1")
        self.fs.save_code_file("root", "scripts\\test.py", "content2")
        
        node_a = self.trace.add_node(action={}, result={})
        self.trace.switch_node(node_a)
        
        assert self.fs.load_code_file(node_a, "src/core/main.py") == "content1"
        assert self.fs.load_code_file(node_a, "scripts/test.py") == "content2" # Should normalize to /
        
        # Verify workspace structure
        assert (self.fs.workspace_dir / "src/core/main.py").exists()
        assert (self.fs.workspace_dir / "scripts/test.py").exists()

    def test_large_history_performance(self, setup_system):
        """Simulate a deep tree to check for TraceSystem performance issues."""
        current = "root"
        for i in range(50):
            current = self.trace.add_node(action={"type": "step", "params": {"step": i}}, result={"val": i})
            
        history = self.trace.get_full_action_history(current)
        assert len(history) == 50
        assert history[-1].arguments["step"] == 49
        
    def test_merge_conflict_integrity(self, setup_system):
        """Verify conflict state doesn't break the system and status is correct."""
        self.fs.save_code_file("root", "conflict.py", "line 1\n")
        
        node_a = self.trace.add_node(action={"side": "a"}, result={})
        self.trace.switch_node(node_a)
        self.fs.save_code_file(node_a, "conflict.py", "line 1 modified A\n")
        
        self.trace.switch_node("root")
        node_b = self.trace.add_node(action={"side": "b"}, result={})
        self.trace.switch_node(node_b)
        self.fs.save_code_file(node_b, "conflict.py", "line 1 modified B\n")
        
        node_c = self.trace.merge_nodes(node_a, node_b)
        
        state = self.fs.load_node_state(node_c)
        assert state.status == "conflict"
        
        content = self.fs.load_code_file(node_c, "conflict.py")
        assert "<<<<<<<" in content
        assert "=======" in content
        assert ">>>>>>>" in content

    def test_direct_write_pollutes_hardlinks(self, setup_system):
        """
        回归测试：验证直接用 Path.write_text() 写入 code/ 目录会污染硬链接。
        
        这个测试模拟了之前 react_agent._save_node_result 中的 bug：
        直接用 write_text() 写入 code/code.py，导致所有共享硬链接的节点
        都被修改为最后一次写入的内容。
        
        修复方案：所有对 code/ 目录的写入必须通过 fs.save_code_file()。
        """
        # 1. root 写入初始代码
        self.fs.save_code_file("root", "code.py", "version_root")
        root_path = self.fs.get_code_snapshot_dir("root") / "code.py"
        
        # 2. 创建 3 代节点（不修改代码，通过硬链接继承）
        nodes = []
        for i in range(3):
            nid = self.trace.add_node(action={"idx": i}, result={})
            nodes.append(nid)
        
        # 3. 验证所有节点共享同一 inode
        root_inode = root_path.stat().st_ino
        for nid in nodes:
            p = self.fs.get_code_snapshot_dir(nid) / "code.py"
            assert p.stat().st_ino == root_inode, f"{nid} should share inode with root"
        
        # 4. 用 save_code_file（正确方式）修改最后一个节点
        last_node = nodes[-1]
        self.fs.save_code_file(last_node, "code.py", "version_modified")
        
        last_path = self.fs.get_code_snapshot_dir(last_node) / "code.py"
        
        # 5. 验证 CoW 正确工作：最后节点的 inode 应该不同
        assert last_path.stat().st_ino != root_inode, \
            "save_code_file should break hardlink (new inode)"
        
        # 6. 验证其他节点没被污染
        assert root_path.read_text() == "version_root", \
            "root snapshot should not be polluted"
        for nid in nodes[:-1]:
            p = self.fs.get_code_snapshot_dir(nid) / "code.py"
            assert p.read_text() == "version_root", \
                f"{nid} snapshot should not be polluted"
            assert p.stat().st_ino == root_inode, \
                f"{nid} should still share original inode"
        
        # 7. 验证修改后的节点内容正确
        assert last_path.read_text() == "version_modified"

    def test_verify_snapshot_integrity(self, setup_system):
        """
        测试 verify_snapshot_integrity 能检测到硬链接污染。
        """
        # 1. 正常流程：save_code_file 写入
        self.fs.save_code_file("root", "code.py", "correct_content")
        n1 = self.trace.add_node(action={"type": "test"}, result={})
        
        # 2. 正常情况下完整性应该是 OK 的
        corrupted = self.fs.verify_snapshot_integrity("root")
        assert corrupted == [], f"Should be clean, got: {corrupted}"
        
        # 3. 模拟外部直接写入（绕过 CoW）—— 这是错误的做法
        snapshot_file = self.fs.get_code_snapshot_dir(n1) / "code.py"
        snapshot_file.write_text("POLLUTED_CONTENT")
        
        # 4. 完整性检查应该检测到 n1 被污染
        corrupted = self.fs.verify_snapshot_integrity(n1)
        assert len(corrupted) > 0, "Should detect corruption"
        assert "checksum mismatch" in corrupted[0]
        
        # 5. 同时 root 也被污染了（因为共享 inode）
        corrupted_root = self.fs.verify_snapshot_integrity("root")
        assert len(corrupted_root) > 0, \
            "Root should also be corrupted (shared inode was modified in-place)"


class TestAgentStateRestore:
    """测试 ReActAgent 的状态恢复功能"""
    
    @pytest.fixture
    def setup_trace(self):
        """创建一个带有历史记录的 trace 环境"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="akg_restore_test_"))
        self.task_id = "restore_test_task"
        self.fs = FileSystemState(self.task_id, base_dir=str(self.test_dir))
        self.trace = TraceSystem(self.task_id, base_dir=str(self.test_dir))
        
        self.fs.initialize_task()
        self.trace.initialize()
        
        # 创建 root 节点并保存 task_input
        from akg_agents.core_v2.filesystem.models import ActionRecord, ActionHistoryFact
        
        root_state = NodeState(
            node_id="root",
            turn=0,
            status="init",
            task_info={
                "task_id": self.task_id,
                "task_input": "生成 relu 算子",
            },
        )
        self.fs.save_node_state("root", root_state)
        
        # 添加几个节点并保存 action history
        node1 = self.trace.add_node(
            action={"type": "ask_user", "arguments": {"message": "确认参数"}},
            result={"status": "responded", "user_response": "默认"},
        )
        
        # 保存 node1 的 action_history_fact
        action1 = ActionRecord(
            action_id=node1,
            tool_name="ask_user",
            arguments={"message": "确认参数"},
            result={"status": "responded"},
        )
        history1 = ActionHistoryFact(
            node_id=node1,
            parent_node_id="root",
            turn=1,
            actions=[action1],
        )
        self.fs.save_action_history_fact(node1, history1)
        
        node2 = self.trace.add_node(
            action={"type": "call_op_task_builder", "arguments": {"user_input": "生成 relu 算子"}},
            result={"status": "success"},
        )
        
        action2 = ActionRecord(
            action_id=node2,
            tool_name="call_op_task_builder",
            arguments={"user_input": "生成 relu 算子"},
            result={"status": "success", "op_name": "relu"},
        )
        history2 = ActionHistoryFact(
            node_id=node2,
            parent_node_id=node1,
            turn=2,
            actions=[action2],
        )
        self.fs.save_action_history_fact(node2, history2)
        
        self.node1 = node1
        self.node2 = node2
        
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_restore_history_from_trace(self, setup_trace):
        """验证 tool_executor.history 可以从 trace 恢复"""
        resume_info = self.trace.get_resume_info()
        
        action_history = resume_info.get("action_history", [])
        assert len(action_history) == 2, f"应有 2 条历史记录，实际 {len(action_history)}"
        assert action_history[0].tool_name == "ask_user"
        assert action_history[1].tool_name == "call_op_task_builder"
    
    def test_restore_original_user_input_from_root(self, setup_trace):
        """验证 _original_user_input 可以从 root 节点的 task_info 恢复"""
        root_state = self.fs.load_node_state("root")
        task_input = root_state.task_info.get("task_input", "")
        assert task_input == "生成 relu 算子", f"task_input 应为 '生成 relu 算子'，实际 '{task_input}'"
    
    def test_restore_original_user_input_fallback(self, setup_trace):
        """验证当 root 没有 task_input 时，可以从 action_history 回退恢复"""
        # 清除 root 的 task_input
        root_state = self.fs.load_node_state("root")
        root_state.task_info["task_input"] = ""
        self.fs.save_node_state("root", root_state)
        
        # 从 action_history 中找到 user_input
        resume_info = self.trace.get_resume_info()
        action_history = resume_info.get("action_history", [])
        
        found_user_input = None
        for action in action_history:
            args = action.arguments if hasattr(action, 'arguments') else {}
            if args.get("user_input"):
                found_user_input = args["user_input"]
                break
        
        assert found_user_input == "生成 relu 算子", \
            f"应从 action_history 回退恢复 user_input，实际 '{found_user_input}'"
    
    def test_resume_info_completeness(self, setup_trace):
        """验证 get_resume_info 返回的信息完整性"""
        resume_info = self.trace.get_resume_info()
        
        assert "task_id" in resume_info
        assert "current_node" in resume_info
        assert "state" in resume_info
        assert "action_history" in resume_info
        assert "pending_tools" in resume_info
        assert "thinking" in resume_info
        assert "path" in resume_info
        
        assert resume_info["task_id"] == self.task_id
        assert resume_info["current_node"] == self.node2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
