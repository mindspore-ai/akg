import os
import shutil
import pytest
from pathlib import Path
from akg_agents.core_v2.filesystem.state import FileSystemState
from akg_agents.core_v2.filesystem.trace_system import TraceSystem
from akg_agents.core_v2.filesystem.models import NodeState

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

def test_diff_file(temp_dir):
    task_id = "test_diff_file"
    fs = FileSystemState(task_id, base_dir=str(temp_dir))
    fs.initialize_task()
    
    # Create node A
    node_a = "root"
    fs.save_code_file(node_a, "main.py", "line 1\nline 2\nline 3\n")
    
    # Create node B (forked)
    node_b = "node_b"
    fs.copy_node_state(node_a, node_b)
    fs.save_code_file(node_b, "main.py", "line 1\nline 2 modified\nline 3\nline 4\n")
    
    # Test diff
    diff = fs.diff_file(node_a, node_b, "main.py")
    assert "+line 2 modified" in diff
    assert "+line 4" in diff
    assert "-line 2" in diff

def test_diff_nodes_persistent_patch(temp_dir):
    task_id = "test_diff_nodes"
    fs = FileSystemState(task_id, base_dir=str(temp_dir))
    fs.initialize_task()
    
    # Node A
    node_a = "root"
    fs.save_code_file(node_a, "a.py", "content a\n")
    fs.save_code_file(node_a, "shared.py", "shared\n")
    
    # Node B
    node_b = "node_b"
    fs.copy_node_state(node_a, node_b)
    fs.save_code_file(node_b, "b.py", "content b\n") # New file
    fs.save_code_file(node_b, "a.py", "content a modified\n") # Modified
    # shared.py remains same (checksum match should skip it)
    
    patch_path = fs.diff_nodes(node_a, node_b)
    assert patch_path.exists()
    content = patch_path.read_text()
    
    assert "+++ b/a.py" in content
    assert "+++ b/b.py" in content
    assert "shared.py" not in content # Should be skipped due to checksum match

def test_find_lca(temp_dir):
    task_id = "test_lca"
    trace = TraceSystem(task_id, base_dir=str(temp_dir))
    trace.initialize()
    
    # root -> n1 -> n2 -> n3
    #            -> n4
    n1 = trace.add_node({"type": "a"}, {"res": "a"})
    n2 = trace.add_node({"type": "b"}, {"res": "b"})
    n3 = trace.add_node({"type": "c"}, {"res": "c"})
    
    trace.switch_node(n1)
    n4 = trace.add_node({"type": "d"}, {"res": "d"})
    
    assert trace.find_lca(n3, n4) == n1
    assert trace.find_lca(n2, n3) == n2
    assert trace.find_lca(n1, "root") == "root"

def test_merge_no_conflict(temp_dir):
    task_id = "test_merge_no_conflict"
    trace = TraceSystem(task_id, base_dir=str(temp_dir))
    trace.initialize()
    
    # root: main.py="base"
    trace.fs.save_code_file("root", "main.py", "base\n")
    
    # branch A: add file_a.py
    n1 = trace.add_node({"type": "branch_a"}, {"res": "ok"})
    trace.fs.save_code_file(n1, "file_a.py", "a\n")
    
    # branch B: add file_b.py (fork from root)
    trace.switch_node("root")
    n2 = trace.add_node({"type": "branch_b"}, {"res": "ok"})
    trace.fs.save_code_file(n2, "file_b.py", "b\n")
    
    # Merge n1 into n2
    merge_node = trace.merge_nodes(n2, n1)
    
    # Check Result
    assert trace.fs.load_code_file(merge_node, "main.py") == "base\n"
    assert trace.fs.load_code_file(merge_node, "file_a.py") == "a\n"
    assert trace.fs.load_code_file(merge_node, "file_b.py") == "b\n"
    
    state = trace.fs.load_node_state(merge_node)
    assert state.status == "completed"

def test_merge_with_conflict(temp_dir):
    task_id = "test_merge_conflict"
    trace = TraceSystem(task_id, base_dir=str(temp_dir))
    trace.initialize()
    
    trace.fs.save_code_file("root", "main.py", "base\n")
    
    # branch A: change main.py to "A"
    n1 = trace.add_node({"type": "a"}, {"res": "a"})
    trace.fs.save_code_file(n1, "main.py", "A\n")
    
    # branch B: change main.py to "B"
    trace.switch_node("root")
    n2 = trace.add_node({"type": "b"}, {"res": "b"})
    trace.fs.save_code_file(n2, "main.py", "B\n")
    
    # Merge
    merge_node = trace.merge_nodes(n2, n1)
    
    content = trace.fs.load_code_file(merge_node, "main.py")
    assert "<<<<<<< YOURS" in content
    assert "B" in content
    assert "=======" in content
    assert "A" in content
    assert ">>>>>>> THEIRS" in content
    
    result = trace.get_node(merge_node).result
    assert result["status"] == "conflict"
    assert "main.py" in result["conflicts"]
