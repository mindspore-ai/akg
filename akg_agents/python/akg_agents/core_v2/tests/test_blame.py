"""
Tests for TraceSystem.blame_file and blame_all_files

跨节点代码演化追踪功能的测试用例：
- 线性路径上的文件创建/修改追踪
- 分叉路径上的独立演化
- 文件删除检测
- 多文件追踪 (blame_all_files)
- root 节点无文件时的边界情况
- 不存在的文件
"""

import pytest
from akg_agents.core_v2.filesystem.trace_system import TraceSystem


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


# ─── blame_file: 线性路径 ───────────────────────────────────────


def test_blame_linear_create_and_modify(temp_dir):
    """root -> n1(创建) -> n2(修改) -> n3(不变)"""
    trace = TraceSystem("blame_linear", base_dir=str(temp_dir))
    trace.initialize()

    # n1: 创建 main.py
    n1 = trace.add_node({"type": "generate"}, {"res": "ok"})
    trace.fs.save_code_file(n1, "main.py", "version 1\n")

    # n2: 修改 main.py
    n2 = trace.add_node({"type": "modify"}, {"res": "ok"})
    trace.fs.save_code_file(n2, "main.py", "version 2\n")

    # n3: 不修改 main.py (copy_node_state 保留了 file_state)
    n3 = trace.add_node({"type": "analyze"}, {"res": "ok"})

    # ─── 验证 ───
    records = trace.blame_file(n3, "main.py")
    assert len(records) == 2

    # 第一条: n1 创建
    assert records[0]["node_id"] == n1
    assert records[0]["change_type"] == "created"
    assert records[0]["action"] == "generate"
    assert records[0]["checksum"] is not None

    # 第二条: n2 修改
    assert records[1]["node_id"] == n2
    assert records[1]["change_type"] == "modified"
    assert records[1]["action"] == "modify"
    assert records[1]["checksum"] != records[0]["checksum"]


def test_blame_file_created_at_root(temp_dir):
    """文件在 root 节点就存在"""
    trace = TraceSystem("blame_at_root", base_dir=str(temp_dir))
    trace.initialize()

    # 在 root 节点直接保存文件
    trace.fs.save_code_file("root", "config.py", "cfg = {}\n")

    # n1: 不修改
    n1 = trace.add_node({"type": "step"}, {"res": "ok"})

    records = trace.blame_file(n1, "config.py")
    assert len(records) == 1
    assert records[0]["node_id"] == "root"
    assert records[0]["change_type"] == "created"


def test_blame_file_not_found(temp_dir):
    """追踪一个从未出现的文件，应返回空列表"""
    trace = TraceSystem("blame_nofile", base_dir=str(temp_dir))
    trace.initialize()

    n1 = trace.add_node({"type": "step"}, {"res": "ok"})

    records = trace.blame_file(n1, "nonexistent.py")
    assert records == []


# ─── blame_file: 分叉路径 ───────────────────────────────────────


def test_blame_fork_independent_evolution(temp_dir):
    """
    root -> n1(创建) -> n2(branch A: 修改)
                     -> n3(branch B: 不同修改)

    blame n2 和 blame n3 应该有不同的路径
    """
    trace = TraceSystem("blame_fork", base_dir=str(temp_dir))
    trace.initialize()

    # n1: 创建 main.py
    n1 = trace.add_node({"type": "generate"}, {"res": "ok"})
    trace.fs.save_code_file(n1, "main.py", "base\n")

    # branch A: n2 修改
    n2 = trace.add_node({"type": "modify_a"}, {"res": "ok"})
    trace.fs.save_code_file(n2, "main.py", "branch A\n")

    # branch B: fork from n1, n3 不同修改
    trace.switch_node(n1)
    n3 = trace.add_node({"type": "modify_b"}, {"res": "ok"})
    trace.fs.save_code_file(n3, "main.py", "branch B\n")

    # ─── blame n2 (path: root -> n1 -> n2) ───
    records_a = trace.blame_file(n2, "main.py")
    assert len(records_a) == 2
    assert records_a[0]["node_id"] == n1
    assert records_a[0]["change_type"] == "created"
    assert records_a[1]["node_id"] == n2
    assert records_a[1]["change_type"] == "modified"
    assert records_a[1]["action"] == "modify_a"

    # ─── blame n3 (path: root -> n1 -> n3) ───
    records_b = trace.blame_file(n3, "main.py")
    assert len(records_b) == 2
    assert records_b[0]["node_id"] == n1
    assert records_b[0]["change_type"] == "created"
    assert records_b[1]["node_id"] == n3
    assert records_b[1]["change_type"] == "modified"
    assert records_b[1]["action"] == "modify_b"

    # checksum 应该不同 (branch A vs branch B)
    assert records_a[1]["checksum"] != records_b[1]["checksum"]


# ─── blame_file: 文件删除 ──────────────────────────────────────


def test_blame_file_deletion(temp_dir):
    """文件被创建后在后续节点中删除"""
    trace = TraceSystem("blame_delete", base_dir=str(temp_dir))
    trace.initialize()

    # n1: 创建
    n1 = trace.add_node({"type": "generate"}, {"res": "ok"})
    trace.fs.save_code_file(n1, "temp.py", "temporary\n")

    # n2: 删除文件 (通过从 file_state 中移除)
    n2 = trace.add_node({"type": "cleanup"}, {"res": "ok"})
    state = trace.fs.load_node_state(n2)
    if "code/temp.py" in state.file_state:
        del state.file_state["code/temp.py"]
    trace.fs.save_node_state(n2, state)

    records = trace.blame_file(n2, "temp.py")
    assert len(records) == 2
    assert records[0]["change_type"] == "created"
    assert records[0]["node_id"] == n1
    assert records[1]["change_type"] == "deleted"
    assert records[1]["node_id"] == n2
    assert records[1]["checksum"] is None


# ─── blame_file: 多次修改 ──────────────────────────────────────


def test_blame_multiple_modifications(temp_dir):
    """文件经历多次修改: 创建 -> 修改1 -> 修改2 -> 修改3"""
    trace = TraceSystem("blame_multi", base_dir=str(temp_dir))
    trace.initialize()

    versions = ["v1\n", "v2\n", "v3\n", "v4\n"]
    node_ids = []

    for i, content in enumerate(versions):
        action_type = "generate" if i == 0 else "modify"
        nid = trace.add_node({"type": action_type}, {"res": f"v{i+1}"})
        trace.fs.save_code_file(nid, "evolving.py", content)
        node_ids.append(nid)

    records = trace.blame_file(node_ids[-1], "evolving.py")
    assert len(records) == 4

    assert records[0]["change_type"] == "created"
    for i in range(1, 4):
        assert records[i]["change_type"] == "modified"

    # 每条记录的 checksum 都不同
    checksums = [r["checksum"] for r in records]
    assert len(set(checksums)) == 4


# ─── blame_all_files ────────────────────────────────────────────


def test_blame_all_files(temp_dir):
    """追踪路径上所有文件的演化"""
    trace = TraceSystem("blame_all", base_dir=str(temp_dir))
    trace.initialize()

    # n1: 创建两个文件
    n1 = trace.add_node({"type": "generate"}, {"res": "ok"})
    trace.fs.save_code_file(n1, "main.py", "main v1\n")
    trace.fs.save_code_file(n1, "utils.py", "utils v1\n")

    # n2: 只修改 main.py
    n2 = trace.add_node({"type": "modify"}, {"res": "ok"})
    trace.fs.save_code_file(n2, "main.py", "main v2\n")

    # n3: 添加新文件
    n3 = trace.add_node({"type": "add"}, {"res": "ok"})
    trace.fs.save_code_file(n3, "helper.py", "helper v1\n")

    result = trace.blame_all_files(n3)

    # 应该有 3 个文件
    assert set(result.keys()) == {"helper.py", "main.py", "utils.py"}

    # main.py: 2 条记录 (created + modified)
    assert len(result["main.py"]) == 2
    assert result["main.py"][0]["change_type"] == "created"
    assert result["main.py"][1]["change_type"] == "modified"

    # utils.py: 1 条记录 (created, 未修改)
    assert len(result["utils.py"]) == 1
    assert result["utils.py"][0]["change_type"] == "created"

    # helper.py: 1 条记录 (created)
    assert len(result["helper.py"]) == 1
    assert result["helper.py"][0]["change_type"] == "created"
    assert result["helper.py"][0]["node_id"] == n3


# ─── 边界: 只有 root ─────────────────────────────────────────


def test_blame_only_root(temp_dir):
    """只有 root 节点（无文件）"""
    trace = TraceSystem("blame_root_only", base_dir=str(temp_dir))
    trace.initialize()

    records = trace.blame_file("root", "anything.py")
    assert records == []


def test_blame_all_files_empty(temp_dir):
    """路径上没有任何文件"""
    trace = TraceSystem("blame_empty", base_dir=str(temp_dir))
    trace.initialize()

    n1 = trace.add_node({"type": "noop"}, {"res": "ok"})
    result = trace.blame_all_files(n1)
    assert result == {}


# ─── 边界: 同内容覆写不算修改 ────────────────────────────────


def test_blame_same_content_no_change(temp_dir):
    """用相同内容覆写文件不应该产生额外的 blame 记录"""
    trace = TraceSystem("blame_same", base_dir=str(temp_dir))
    trace.initialize()

    n1 = trace.add_node({"type": "generate"}, {"res": "ok"})
    trace.fs.save_code_file(n1, "main.py", "same content\n")

    # n2: 同样的内容
    n2 = trace.add_node({"type": "modify"}, {"res": "ok"})
    trace.fs.save_code_file(n2, "main.py", "same content\n")

    records = trace.blame_file(n2, "main.py")
    # 只有 1 条 "created"，因为 checksum 相同
    assert len(records) == 1
    assert records[0]["change_type"] == "created"
