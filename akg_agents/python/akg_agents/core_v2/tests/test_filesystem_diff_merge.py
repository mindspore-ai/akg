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


def test_merge_auto_merge_different_regions(temp_dir):
    """
    核心新能力: 双方修改不同区域时，merge3 能自动合并，无冲突

    base:
        line 1
        line 2
        line 3
        line 4

    branch A: 修改 line 1
    branch B: 修改 line 4

    结果: 自动合并两个修改，无冲突
    """
    trace = TraceSystem("test_auto_merge", base_dir=str(temp_dir))
    trace.initialize()

    base_content = "line 1\nline 2\nline 3\nline 4\n"
    trace.fs.save_code_file("root", "main.py", base_content)

    # branch A: 修改 line 1
    n1 = trace.add_node({"type": "modify_top"}, {"res": "ok"})
    trace.fs.save_code_file(n1, "main.py", "line 1 MODIFIED BY A\nline 2\nline 3\nline 4\n")

    # branch B: 修改 line 4 (fork from root)
    trace.switch_node("root")
    n2 = trace.add_node({"type": "modify_bottom"}, {"res": "ok"})
    trace.fs.save_code_file(n2, "main.py", "line 1\nline 2\nline 3\nline 4 MODIFIED BY B\n")

    # Merge
    merge_node = trace.merge_nodes(n2, n1)
    content = trace.fs.load_code_file(merge_node, "main.py")

    # 两个修改都应该保留，无冲突标记
    assert "line 1 MODIFIED BY A" in content
    assert "line 4 MODIFIED BY B" in content
    assert "<<<<<<< YOURS" not in content

    # 状态应该是 completed，不是 conflict
    result = trace.get_node(merge_node).result
    assert result["status"] == "completed"


def test_merge_partial_conflict(temp_dir):
    """
    双方修改同一行导致冲突，但其他区域的修改应自动合并

    base:
        header
        shared line
        footer

    branch A: 修改 header, 修改 shared line
    branch B: 修改 shared line (不同), 修改 footer

    结果: header/footer 自动合并, shared line 冲突
    """
    trace = TraceSystem("test_partial_conflict", base_dir=str(temp_dir))
    trace.initialize()

    base_content = "header\nshared line\nfooter\n"
    trace.fs.save_code_file("root", "main.py", base_content)

    # branch A
    n1 = trace.add_node({"type": "a"}, {"res": "a"})
    trace.fs.save_code_file(n1, "main.py", "header A\nshared A\nfooter\n")

    # branch B
    trace.switch_node("root")
    n2 = trace.add_node({"type": "b"}, {"res": "b"})
    trace.fs.save_code_file(n2, "main.py", "header\nshared B\nfooter B\n")

    merge_node = trace.merge_nodes(n2, n1)
    content = trace.fs.load_code_file(merge_node, "main.py")

    # shared line 应该有冲突标记
    assert "<<<<<<< YOURS" in content
    assert "=======" in content
    assert ">>>>>>> THEIRS" in content

    result = trace.get_node(merge_node).result
    assert result["status"] == "conflict"


def test_merge_add_lines_no_conflict(temp_dir):
    """
    双方各自在不同位置新增行

    base:  line 1 / line 2
    A:     NEW A / line 1 / line 2
    B:     line 1 / line 2 / NEW B

    结果: 自动合并
    """
    trace = TraceSystem("test_add_lines", base_dir=str(temp_dir))
    trace.initialize()

    trace.fs.save_code_file("root", "main.py", "line 1\nline 2\n")

    n1 = trace.add_node({"type": "add_top"}, {"res": "ok"})
    trace.fs.save_code_file(n1, "main.py", "NEW A\nline 1\nline 2\n")

    trace.switch_node("root")
    n2 = trace.add_node({"type": "add_bottom"}, {"res": "ok"})
    trace.fs.save_code_file(n2, "main.py", "line 1\nline 2\nNEW B\n")

    merge_node = trace.merge_nodes(n2, n1)
    content = trace.fs.load_code_file(merge_node, "main.py")

    assert "NEW A" in content
    assert "NEW B" in content
    assert "line 1" in content
    assert "line 2" in content
    assert "<<<<<<< YOURS" not in content
