# Copyright 2026 Huawei Technologies Co., Ltd
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
Trace 可视化模块测试

覆盖场景:
1. 线性链路 — 上方省略、当前标记、上下 N 层
2. 2 分叉 — 分叉标注
3. 8 分叉 — 多兄弟省略
4. 嵌套 8x8 分叉 — 多层分叉标注
5. 查看其他分支 — focus_node 非当前路径，◀ 标记
6. 当前节点是分叉点 — 子分支提示
7. 纯文本版本
8. 节点详情 / 路径详情（从 test_trace_system 迁移）
"""

import shutil
import tempfile

import pytest

from akg_agents.core_v2.filesystem import TraceSystem, TraceSystemError
from akg_agents.core_v2.filesystem.trace_visualizer import (
    format_node_text,
    format_node_detail_rich,
    visualize_rich,
    visualize_text,
    _short_id,
    _summarize_result,
)


# ==================== Fixtures ====================

@pytest.fixture
def trace():
    """创建初始化的 TraceSystem"""
    temp = tempfile.mkdtemp(prefix="akg_viz_test_")
    ts = TraceSystem("viz_test", base_dir=temp)
    ts.initialize()
    yield ts
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def linear_trace():
    """线性链路: root -> 001 -> 002 -> ... -> 011 (11 个节点)"""
    temp = tempfile.mkdtemp(prefix="akg_viz_linear_")
    ts = TraceSystem("viz_linear", base_dir=temp)
    ts.initialize()
    for i in range(11):
        ts.add_node(action={"type": f"step_{i+1}"}, result={})
    # 当前在 node_007
    ts.switch_node("node_007")
    yield ts
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def fork2_trace():
    """2 分叉: root -> 001 plan -> (002 gen_v1 -> 003 -> 004) | (005 gen_v2 -> 006)"""
    temp = tempfile.mkdtemp(prefix="akg_viz_fork2_")
    ts = TraceSystem("viz_fork2", base_dir=temp)
    ts.initialize()
    n1 = ts.add_node(action={"type": "plan"}, result={})
    ts.add_node(action={"type": "gen_v1"}, result={})
    ts.add_node(action={"type": "compile"}, result={})
    ts.add_node(action={"type": "profile"}, result={})
    ts.switch_node(n1)
    ts.add_node(action={"type": "gen_v2"}, result={})
    ts.add_node(action={"type": "compile_v2"}, result={})
    # 当前在 node_004 (gen_v1 分支末尾)
    ts.switch_node("node_004")
    yield ts
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def fork8_trace():
    """8 分叉: root -> 001 plan -> 8 个分支，每个分支 3 步"""
    temp = tempfile.mkdtemp(prefix="akg_viz_fork8_")
    ts = TraceSystem("viz_fork8", base_dir=temp)
    ts.initialize()
    n1 = ts.add_node(action={"type": "plan"}, result={})
    branches = []
    for i in range(8):
        ts.switch_node(n1)
        b1 = ts.add_node(action={"type": f"gen_v{i+1}"}, result={})
        b2 = ts.add_node(action={"type": "compile"}, result={})
        b3 = ts.add_node(action={"type": "profile"}, result={})
        branches.append((b1, b2, b3))
    # 当前在第 4 个分支末尾
    ts.switch_node(branches[3][2])
    yield ts, branches
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def nested_8x8_trace():
    """嵌套 8x8 分叉"""
    temp = tempfile.mkdtemp(prefix="akg_viz_nested_")
    ts = TraceSystem("viz_nested", base_dir=temp)
    ts.initialize()
    n1 = ts.add_node(action={"type": "ask_user"}, result={})
    branches_l1 = []
    for i in range(8):
        ts.switch_node(n1)
        b = ts.add_node(action={"type": f"plan_{i+1}"}, result={})
        branches_l1.append(b)
    # 在第 3 个分支上再 8 分叉
    branches_l2 = []
    for i in range(8):
        ts.switch_node(branches_l1[2])
        b = ts.add_node(action={"type": f"gen_{i+1}"}, result={})
        branches_l2.append(b)
    # 当前在第 3 分支的第 5 子分支
    ts.switch_node(branches_l2[4])
    leaf = ts.add_node(action={"type": "optimize"}, result={})
    yield ts, branches_l1, branches_l2
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def fork_at_current_trace():
    """当前节点是分叉点"""
    temp = tempfile.mkdtemp(prefix="akg_viz_fork_cur_")
    ts = TraceSystem("viz_fork_cur", base_dir=temp)
    ts.initialize()
    ts.add_node(action={"type": "plan"}, result={})
    ts.add_node(action={"type": "design"}, result={})
    n3 = ts.add_node(action={"type": "code"}, result={})
    ts.add_node(action={"type": "test_v1"}, result={})
    ts.switch_node(n3)
    ts.add_node(action={"type": "test_v2"}, result={})
    ts.switch_node(n3)
    ts.add_node(action={"type": "test_v3"}, result={})
    ts.switch_node(n3)  # 当前回到分叉点
    yield ts
    shutil.rmtree(temp, ignore_errors=True)


# ==================== 工具函数测试 ====================

class TestHelpers:
    """测试辅助函数"""

    def test_short_id(self):
        assert _short_id("node_005") == "005"
        assert _short_id("root") == "root"
        assert _short_id("node_123") == "123"

    def test_format_node_text_root(self):
        assert format_node_text("root", None) == "root"

    def test_summarize_result_empty(self):
        assert _summarize_result(None) == ""
        assert _summarize_result({}) == ""

    def test_summarize_result_success(self):
        result = _summarize_result({"success": True})
        assert "✅" in result

    def test_summarize_result_performance(self):
        result = _summarize_result({"performance": 0.85})
        assert "性能" in result

    def test_summarize_result_lines(self):
        result = _summarize_result({"lines": 120})
        assert "120" in result


# ==================== 场景测试: Rich 可视化 ====================

class TestVisualizeRich:
    """Rich 可视化场景测试"""

    def test_linear_default_view(self, linear_trace):
        """场景 1: 线性链路，/trace (当前=007, N=4)"""
        text = linear_trace.visualize_tree_rich()
        markup = text.markup if hasattr(text, 'markup') else str(text)

        # 当前节点标记
        assert "007" in markup
        assert "当前" in markup

        # 上方应该有省略提示（root, 001, 002 被省略）
        assert "⋮" in markup
        assert "上方" in markup

        # 向上 4 层: 003, 004, 005, 006 应该可见
        assert "003" in markup
        assert "006" in markup

        # 向下 4 层: 008, 009, 010, 011 应该可见
        assert "008" in markup
        assert "011" in markup

    def test_linear_root_view(self, linear_trace):
        """线性链路，/trace root"""
        text = linear_trace.visualize_tree_rich(focus_node="root")
        markup = text.markup if hasattr(text, 'markup') else str(text)

        assert "root" in markup
        assert "001" in markup

    def test_linear_custom_depth(self, linear_trace):
        """线性链路，/trace -n 2"""
        text = linear_trace.visualize_tree_rich(depth=2)
        markup = text.markup if hasattr(text, 'markup') else str(text)

        # 向上 2 层: 005, 006 可见，004 不可见
        assert "005" in markup
        assert "006" in markup
        assert "007" in markup

    def test_fork2(self, fork2_trace):
        """场景 2: 2 分叉"""
        text = fork2_trace.visualize_tree_rich()
        markup = text.markup if hasattr(text, 'markup') else str(text)

        # 当前路径: root -> 001 -> 002 -> 003 -> 004
        assert "001" in markup
        assert "plan" in markup
        assert "004" in markup
        assert "当前" in markup

        # 分叉标注
        assert "↳" in markup
        assert "分支" in markup
        # 兄弟 005 应该在标注中
        assert "005" in markup

    def test_fork8(self, fork8_trace):
        """场景 3: 8 分叉"""
        ts, branches = fork8_trace
        text = ts.visualize_tree_rich()
        markup = text.markup if hasattr(text, 'markup') else str(text)

        # 当前路径可见
        assert "001" in markup
        assert "plan" in markup
        assert "当前" in markup

        # 分叉标注: 另有 7 个分支
        assert "7 个分支" in markup
        assert "↳" in markup

    def test_fork8_root_view(self, fork8_trace):
        """8 分叉，/trace root"""
        ts, _ = fork8_trace
        text = ts.visualize_tree_rich(focus_node="root")
        markup = text.markup if hasattr(text, 'markup') else str(text)

        assert "root" in markup
        assert "7 个分支" in markup

    def test_nested_8x8(self, nested_8x8_trace):
        """场景 4: 嵌套 8x8 分叉"""
        ts, _, _ = nested_8x8_trace
        text = ts.visualize_tree_rich()
        markup = text.markup if hasattr(text, 'markup') else str(text)

        # 两层分叉都应该有标注
        assert markup.count("↳") >= 2
        assert "optimize" in markup
        assert "当前" in markup

    def test_view_other_branch(self, fork8_trace):
        """场景 5: /trace 002 查看其他分支"""
        ts, _ = fork8_trace
        text = ts.visualize_tree_rich(focus_node="node_002")
        markup = text.markup if hasattr(text, 'markup') else str(text)

        # 应该展示 002 所在路径
        assert "002" in markup
        assert "gen_v1" in markup

        # 兄弟列表中应该有 ◀ 标记（当前节点所在分支）
        assert "◀" in markup

    def test_fork_at_current(self, fork_at_current_trace):
        """场景 6: 当前节点是分叉点"""
        text = fork_at_current_trace.visualize_tree_rich()
        markup = text.markup if hasattr(text, 'markup') else str(text)

        assert "003" in markup
        assert "code" in markup
        assert "当前" in markup

        # 子分支提示
        assert "子分支" in markup
        assert "004" in markup
        assert "005" in markup
        assert "006" in markup


# ==================== 场景测试: 纯文本可视化 ====================

class TestVisualizeText:
    """纯文本可视化测试"""

    def test_linear_text(self, linear_trace):
        """纯文本线性链路"""
        text = linear_trace.visualize_tree()

        assert "007" in text
        assert "当前" in text
        assert "⋮" in text

    def test_fork2_text(self, fork2_trace):
        """纯文本 2 分叉"""
        text = fork2_trace.visualize_tree()

        assert "↳" in text
        assert "分支" in text
        assert "004" in text

    def test_fork8_text(self, fork8_trace):
        """纯文本 8 分叉"""
        ts, _ = fork8_trace
        text = ts.visualize_tree()

        assert "7 个分支" in text

    def test_text_with_focus(self, fork8_trace):
        """纯文本指定焦点"""
        ts, _ = fork8_trace
        text = ts.visualize_tree(focus_node="node_002")

        assert "002" in text
        assert "gen_v1" in text

    def test_text_with_depth(self, linear_trace):
        """纯文本指定深度"""
        text = linear_trace.visualize_tree(depth=2)

        assert "007" in text
        assert "当前" in text


# ==================== 迁移自 test_trace_system.py 的测试 ====================

class TestNodeAndPathDetail:
    """节点详情 / 路径详情测试（从 TestTreeVisualization 迁移）"""

    @pytest.fixture
    def trace(self):
        temp = tempfile.mkdtemp(prefix="akg_viz_detail_")
        ts = TraceSystem("viz_detail", base_dir=temp)
        ts.initialize()
        yield ts
        shutil.rmtree(temp, ignore_errors=True)

    def test_visualize_tree_basic(self, trace):
        """基本可视化"""
        trace.add_node({"type": "call_designer"}, {"success": True})
        trace.add_node({"type": "call_coder"}, {"success": True, "lines": 120})
        trace.add_node({"type": "verify"}, {"success": True, "performance": 0.65})

        tree_str = trace.visualize_tree()

        assert "当前" in tree_str
        assert "call_designer" in tree_str
        assert "call_coder" in tree_str
        assert "verify" in tree_str

    def test_visualize_tree_with_fork(self, trace):
        """分叉可视化"""
        node1 = trace.add_node({"type": "design"}, {"success": True})
        trace.add_node({"type": "code_v1"}, {"success": True})

        trace.switch_node(node1)
        trace.add_node({"type": "code_v2"}, {"success": True})

        tree_str = trace.visualize_tree()
        assert "↳" in tree_str or "分支" in tree_str

    def test_get_node_detail(self, trace):
        """节点详情"""
        trace.add_node(
            {"type": "call_coder", "params": {"strategy": "optimize"}},
            {"success": True, "lines": 150},
            metrics={"token_used": 2000},
        )

        detail = trace.get_node_detail("node_001")
        assert "节点详情" in detail
        assert "call_coder" in detail
        assert "token_used" in detail

    def test_get_path_detail(self, trace):
        """路径详情"""
        trace.add_node(
            {"type": "step1"},
            {"success": True},
            metrics={"token_used": 1000},
        )
        trace.add_node(
            {"type": "step2"},
            {"success": True},
            metrics={"token_used": 2000},
        )

        detail = trace.get_path_detail("node_002")
        assert "路径" in detail
        assert "步数" in detail
        assert "Token" in detail


# ==================== Fork ask_user 测试 ====================

class TestForkAskUser:
    """fork_ask_user 功能测试"""

    @pytest.fixture
    def ask_user_trace(self):
        """创建包含 ask_user 节点的 trace"""
        temp = tempfile.mkdtemp(prefix="akg_viz_fork_ask_")
        ts = TraceSystem("viz_fork_ask", base_dir=temp)
        ts.initialize()
        # root -> 001 (kernelgen) -> 002 (ask_user, responded) -> 003 (kernelgen)
        ts.add_node(
            action={"type": "kernelgen", "arguments": {"code": "v1"}},
            result={"success": True},
        )
        ts.add_node(
            action={"type": "ask_user", "arguments": {"message": "要不要改一下策略？"}},
            result={"status": "responded", "user_response": "改一下", "message": "要不要改一下策略？"},
        )
        ts.add_node(
            action={"type": "kernelgen", "arguments": {"code": "v2"}},
            result={"success": True},
        )
        yield ts
        shutil.rmtree(temp, ignore_errors=True)

    def test_fork_creates_sibling(self, ask_user_trace):
        """fork 后新节点与原节点同父"""
        ts = ask_user_trace
        new_id = ts.fork_ask_user("node_002")

        # 新节点的父节点 == 原节点的父节点
        original = ts.get_node("node_002")
        forked = ts.get_node(new_id)
        assert forked.parent_id == original.parent_id

        # 父节点的 children 包含原节点和新节点
        parent = ts.get_node(original.parent_id)
        assert "node_002" in parent.children
        assert new_id in parent.children

    def test_fork_preserves_original(self, ask_user_trace):
        """fork 后原节点和子链不受影响"""
        ts = ask_user_trace
        ts.fork_ask_user("node_002")

        # 原节点不变
        original = ts.get_node("node_002")
        assert original.result["status"] == "responded"
        assert original.result["user_response"] == "改一下"

        # 原节点的子节点不变
        assert "node_003" in original.children

    def test_fork_copies_agent_question(self, ask_user_trace):
        """fork 后新节点的 action.arguments.message 与原节点相同"""
        ts = ask_user_trace
        new_id = ts.fork_ask_user("node_002")

        forked = ts.get_node(new_id)
        original = ts.get_node("node_002")

        assert forked.action["type"] == "ask_user"
        assert forked.action["arguments"]["message"] == original.action["arguments"]["message"]

    def test_fork_resets_result(self, ask_user_trace):
        """fork 后新节点 result.status == 'waiting'"""
        ts = ask_user_trace
        new_id = ts.fork_ask_user("node_002")

        forked = ts.get_node(new_id)
        assert forked.result["status"] == "waiting"
        assert forked.result["message"] == "要不要改一下策略？"

    def test_fork_rejects_non_ask_user(self, ask_user_trace):
        """对非 ask_user 节点 fork 应报错"""
        ts = ask_user_trace
        with pytest.raises(TraceSystemError, match="ask_user"):
            ts.fork_ask_user("node_001")

    def test_fork_switches_current_to_new_node(self, ask_user_trace):
        """fork 后当前节点切换到新节点"""
        ts = ask_user_trace
        new_id = ts.fork_ask_user("node_002")

        assert ts.get_current_node() == new_id


# ==================== Show 详情测试 ====================

class TestShowDetail:
    """节点详情视图测试"""

    @pytest.fixture
    def detail_trace(self):
        """创建用于详情测试的 trace"""
        temp = tempfile.mkdtemp(prefix="akg_viz_detail_show_")
        ts = TraceSystem("viz_detail_show", base_dir=temp)
        ts.initialize()
        # ask_user 节点
        ts.add_node(
            action={"type": "ask_user", "arguments": {"message": "请选择优化策略"}},
            result={"status": "responded", "user_response": "使用共享内存", "message": "请选择优化策略"},
        )
        # 工具节点
        ts.add_node(
            action={"type": "profile_kernel", "arguments": {"kernel_name": "matmul", "device": "gpu"}},
            result={"status": "success", "speedup": "2.5x", "gflops": 120},
            metrics={"duration_ms": 350},
        )
        yield ts
        shutil.rmtree(temp, ignore_errors=True)

    def test_show_ask_user_detail(self, detail_trace):
        """ask_user 节点详情包含 agent 提问和用户回答"""
        detail = detail_trace.get_node_detail("node_001")

        assert "Agent 提问" in detail
        assert "请选择优化策略" in detail
        assert "用户回答" in detail
        assert "使用共享内存" in detail
        assert "fork" in detail  # 应该有 fork 提示

    def test_show_tool_detail(self, detail_trace):
        """工具节点详情包含参数和结果"""
        detail = detail_trace.get_node_detail("node_002")

        assert "profile_kernel" in detail
        assert "kernel_name" in detail
        assert "matmul" in detail
        assert "speedup" in detail
        assert "2.5x" in detail
        assert "duration_ms" in detail

    def test_show_ask_user_rich_detail(self, detail_trace):
        """Rich 版本的 ask_user 详情"""
        text = format_node_detail_rich(detail_trace, "node_001")
        markup = text.markup if hasattr(text, 'markup') else str(text)

        assert "Agent 提问" in markup
        assert "请选择优化策略" in markup
        assert "用户回答" in markup
        assert "使用共享内存" in markup
        assert "fork" in markup

    def test_show_tool_rich_detail(self, detail_trace):
        """Rich 版本的工具节点详情"""
        text = format_node_detail_rich(detail_trace, "node_002")
        markup = text.markup if hasattr(text, 'markup') else str(text)

        assert "profile_kernel" in markup
        assert "kernel_name" in markup
        assert "matmul" in markup

    def test_show_nonexistent_node(self, detail_trace):
        """查看不存在的节点"""
        text = format_node_detail_rich(detail_trace, "node_999")
        markup = text.markup if hasattr(text, 'markup') else str(text)

        assert "不存在" in markup


# ==================== ask_user 图标测试 ====================

class TestAskUserIcon:
    """ask_user 节点图标测试"""

    @pytest.fixture
    def icon_trace(self):
        temp = tempfile.mkdtemp(prefix="akg_viz_icon_")
        ts = TraceSystem("viz_icon", base_dir=temp)
        ts.initialize()
        ts.add_node(
            action={"type": "ask_user", "arguments": {"message": "test?"}},
            result={"status": "waiting", "message": "test?"},
        )
        ts.add_node(
            action={"type": "kernelgen"},
            result={"success": True},
        )
        yield ts
        shutil.rmtree(temp, ignore_errors=True)

    def test_ask_user_has_special_icon(self, icon_trace):
        """ask_user 节点使用 👤 图标"""
        node = icon_trace.get_node("node_001")
        text = format_node_text("node_001", node)
        assert "👤" in text

    def test_non_ask_user_no_special_icon(self, icon_trace):
        """非 ask_user 节点不使用 👤 图标"""
        node = icon_trace.get_node("node_002")
        text = format_node_text("node_002", node)
        assert "👤" not in text
