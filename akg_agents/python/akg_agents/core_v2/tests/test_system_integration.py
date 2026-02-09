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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
