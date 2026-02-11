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
Trace 树可视化模块

设计思路（单路径 + 分叉标注）：
- 始终只展示一条完整路径（焦点节点所在的路径）
- 路径上遇到分叉点 → 下方加一行 ↳ 另有 N 个分支 (id...)
- 以焦点节点为中心，向上/向下各展示 depth 层
- 超出范围 → 上方显示 ⋮ 省略提示

命令映射：
  /trace              焦点=当前节点, depth=4
  /trace root         焦点=root, depth=4
  /trace 005          焦点=node_005, depth=4
  /trace -n 8         焦点=当前节点, depth=8
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .trace_system import TraceSystem


# ==================== 默认参数 ====================

DEFAULT_DEPTH = 4


# ==================== 节点格式化 ====================

def format_node_text(node_id: str, node) -> str:
    """格式化节点显示文本（纯文本，无样式）"""
    if node_id == "root":
        return "root"

    short_id = node_id.replace("node_", "")
    action = node.action or {}
    action_type = action.get("type", "")

    # 状态图标：ask_user 节点用特殊图标
    if action_type == "ask_user":
        icon = "👤"
    else:
        state = node.state_snapshot or {}
        status = state.get("status", "")
        icon = {"completed": "✅", "running": "🔄", "failed": "❌"}.get(status, "○")

    text = f"{icon} {short_id}"
    if action_type:
        text += f" {action_type}"

    # 摘要：根据 action 类型提取关键信息
    summary = _extract_summary(action_type, action, node.result)
    if summary:
        text += f" — {summary}"

    return text


# 摘要最大字符数
_SUMMARY_MAX_LEN = 40


def _extract_summary(action_type: str, action: Dict, result: Optional[Dict]) -> str:
    """
    根据 action 类型从 action/result 中提取一行摘要

    策略：
    - ask_user: 优先显示 user_response（用户说了什么），回退到 message 开头
    - call_op_task_builder: op_name + status
    - use_kernelgen_only_workflow: status
    - profile_kernel: speedup 或 output 摘要
    - 其他: 尝试从 result 中提取 status/output
    """
    args = action.get("arguments", {}) or {}
    res = result or {}

    summary = ""

    if action_type == "ask_user":
        # 优先显示用户的回复
        user_resp = res.get("user_response", "")
        if user_resp and user_resp != "None":
            summary = _truncate(str(user_resp))
        else:
            # 回退到 agent 发给用户的消息开头
            msg = args.get("message", "") or res.get("message", "")
            if msg:
                # 取第一行非空内容
                first_line = _first_meaningful_line(str(msg))
                summary = _truncate(first_line)
            # 标记等待状态
            if res.get("status") == "waiting":
                summary = summary + " (等待中)" if summary else "(等待回复)"

    elif action_type == "call_op_task_builder":
        op_name = res.get("op_name", "")
        st = res.get("status", "")
        if op_name:
            summary = f"{op_name}"
            if st and st != "success":
                summary += f" ({st})"
        elif st:
            summary = st

    elif action_type == "use_kernelgen_only_workflow":
        st = res.get("status", "")
        if st:
            summary = st

    elif action_type == "profile_kernel":
        speedup = res.get("speedup")
        if speedup is not None:
            try:
                sp = float(speedup)
                if sp >= 1:
                    summary = f"加速 {sp:.2f}x"
                else:
                    summary = f"减速 {1/sp:.2f}x"
            except (ValueError, TypeError, ZeroDivisionError):
                pass
        if not summary:
            output = res.get("output", "")
            if output:
                summary = _truncate(str(output))

    else:
        # 通用回退：尝试 result.status 或 result.output
        st = res.get("status", "")
        if st:
            summary = st
        output = res.get("output", "")
        if output and not summary:
            summary = _truncate(str(output))

    return summary


def _truncate(text: str, max_len: int = _SUMMARY_MAX_LEN) -> str:
    """截断文本，保留前 max_len 个字符"""
    text = text.strip().replace("\n", " ").replace("\r", "")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _first_meaningful_line(text: str) -> str:
    """取第一行有意义的内容（跳过空行和纯符号行）"""
    for line in text.split("\n"):
        line = line.strip()
        if line and not all(c in "─═━-*#=>" for c in line):
            return line
    return text[:60]


def _short_id(node_id: str) -> str:
    """node_005 -> 005, root -> root"""
    if node_id == "root":
        return "root"
    return node_id.replace("node_", "")


def _summarize_result(result: Optional[Dict]) -> str:
    """总结执行结果"""
    if result is None:
        return ""

    summaries = []

    if "success" in result:
        status = "✅" if result["success"] else "❌"
        summaries.append(status)

    if "performance" in result:
        perf = result["performance"]
        if isinstance(perf, (int, float)):
            summaries.append(f"性能: {perf:.1%}" if perf <= 1 else f"性能: {perf}")

    if "output" in result:
        output = str(result["output"])
        if len(output) > 30:
            output = output[:30] + "..."
        summaries.append(output)

    if "lines" in result:
        summaries.append(f"{result['lines']} 行")

    return " | ".join(summaries) if summaries else ""


# ==================== 路径工具 ====================

def _get_path_to_root(ts: "TraceSystem", node_id: str) -> List[str]:
    """获取从 root 到 node_id 的路径（含两端）"""
    try:
        return ts.get_path_to_node(node_id)
    except Exception:
        return [node_id]


def _find_branch_containing(
    ts: "TraceSystem", parent_children: List[str], target_node: str
) -> Optional[str]:
    """在 parent 的 children 中找到包含 target_node 的那个分支的起点 child_id"""
    target_path = set(_get_path_to_root(ts, target_node))
    for child_id in parent_children:
        if child_id in target_path:
            return child_id
    return None


# ==================== 纯文本可视化 ====================

def visualize_text(ts: "TraceSystem", focus_node: Optional[str] = None, depth: int = DEFAULT_DEPTH) -> str:
    """
    纯文本可视化（无 Rich markup）

    Args:
        focus_node: 焦点节点 ID，默认为当前节点
        depth: 向上/向下展示的层数

    Returns:
        树的字符串表示
    """
    current_node = ts.trace.current_node
    if focus_node is None:
        focus_node = current_node

    lines: List[str] = []
    _render_path_view(ts, focus_node, current_node, depth, lines, use_rich=False)

    lines.append("")
    lines.append("提示: /trace <id> 查看分支 | /trace show <id> 详情 | /trace fork <id> 分叉")

    return "\n".join(lines)


# ==================== Rich 可视化 ====================

def visualize_rich(ts: "TraceSystem", focus_node: Optional[str] = None, depth: int = DEFAULT_DEPTH):
    """
    Rich markup 可视化（用于 CLI 终端）

    Args:
        focus_node: 焦点节点 ID，默认为当前节点
        depth: 向上/向下展示的层数

    Returns:
        rich.text.Text 对象
    """
    from rich.text import Text

    current_node = ts.trace.current_node
    if focus_node is None:
        focus_node = current_node

    lines: List[str] = []
    _render_path_view(ts, focus_node, current_node, depth, lines, use_rich=True)

    return Text.from_markup("\n".join(lines))


# ==================== 核心渲染逻辑 ====================

def _render_path_view(
    ts: "TraceSystem",
    focus_node: str,
    current_node: str,
    depth: int,
    lines: List[str],
    use_rich: bool,
) -> None:
    """
    渲染单路径视图

    策略：
    1. 计算焦点节点到 root 的完整路径
    2. 确定显示范围（向上 depth 层，向下 depth 层）
    3. 沿路径逐节点渲染，遇到分叉点标注兄弟
    4. 到达焦点节点后，继续向下渲染子路径（沿当前节点方向）
    """
    # 1. 焦点节点到 root 的路径
    focus_path = _get_path_to_root(ts, focus_node)  # [root, ..., focus_node]

    # 2. 确定向上的显示范围
    focus_idx = len(focus_path) - 1  # focus_node 在路径中的索引
    start_idx = max(0, focus_idx - depth)

    # 上方省略提示
    if start_idx > 0:
        omitted = focus_path[:start_idx]
        omitted_ids = ", ".join(_short_id(n) for n in omitted)
        if use_rich:
            lines.append(f"[dim]  ⋮ 上方 {len(omitted)} 个节点 ({omitted_ids})[/dim]")
        else:
            lines.append(f"  ⋮ 上方 {len(omitted)} 个节点 ({omitted_ids})")
        lines.append("│")

    # 3. 渲染路径上的节点（从 start_idx 到 focus_node）
    visible_path = focus_path[start_idx:]

    for i, node_id in enumerate(visible_path):
        node = ts.trace.get_node(node_id)
        if node is None:
            continue

        # 渲染节点本身
        _render_node_line(ts, node_id, node, current_node, focus_path, lines, use_rich)

        # 检查分叉：如果这个节点有多个子节点，且路径上的下一个节点是其中之一
        children = node.children or []
        if len(children) > 1 and (i + 1) < len(visible_path):
            next_on_path = visible_path[i + 1]
            siblings = [c for c in children if c != next_on_path]
            _render_fork_hint(ts, siblings, current_node, lines, use_rich)

        # 连线到下一个节点
        if (i + 1) < len(visible_path):
            lines.append("│")

    # 4. 焦点节点之后：继续向下渲染
    _render_below_focus(ts, focus_node, current_node, depth, lines, use_rich)


def _render_node_line(
    ts: "TraceSystem",
    node_id: str,
    node,
    current_node: str,
    focus_path: List[str],
    lines: List[str],
    use_rich: bool,
) -> None:
    """渲染一个节点行"""
    text = format_node_text(node_id, node)
    is_current = node_id == current_node
    on_path = node_id in focus_path

    if use_rich:
        if is_current:
            lines.append(f"[bold yellow]{text} ◀ 当前[/bold yellow]")
        elif on_path:
            lines.append(f"[bold]{text}[/bold]")
        else:
            lines.append(text)
    else:
        marker = " ◀ 当前" if is_current else ""
        lines.append(f"{text}{marker}")


def _render_fork_hint(
    ts: "TraceSystem",
    sibling_ids: List[str],
    current_node: str,
    lines: List[str],
    use_rich: bool,
) -> None:
    """渲染分叉提示行：↳ 另有 N 个分支 (id...)"""
    current_path_set = set(_get_path_to_root(ts, current_node))

    id_parts = []
    for sid in sibling_ids:
        short = _short_id(sid)
        # 如果当前节点在这个兄弟分支中，标记 ◀
        if sid in current_path_set:
            if use_rich:
                id_parts.append(f"[bold]{short}◀[/bold]")
            else:
                id_parts.append(f"{short}◀")
        else:
            id_parts.append(short)

    ids_str = ", ".join(id_parts)
    count = len(sibling_ids)

    if use_rich:
        lines.append(f"│ [dim]↳ 另有 {count} 个分支 ({ids_str})[/dim]")
    else:
        lines.append(f"│ ↳ 另有 {count} 个分支 ({ids_str})")


def _render_below_focus(
    ts: "TraceSystem",
    focus_node: str,
    current_node: str,
    depth: int,
    lines: List[str],
    use_rich: bool,
) -> None:
    """
    渲染焦点节点之下的子路径

    策略：
    - 如果焦点节点只有一个子节点 → 继续沿单链路向下
    - 如果焦点节点有多个子节点 → 标注分叉，沿当前节点方向继续
    - 最多向下 depth 层
    """
    node = ts.trace.get_node(focus_node)
    if node is None:
        return

    children = node.children or []
    if not children:
        return

    current_path_set = set(_get_path_to_root(ts, current_node))

    # 确定要沿着哪个子节点继续
    if len(children) == 1:
        next_child = children[0]
    else:
        # 多个子节点：标注分叉
        # 优先沿当前节点方向
        next_child = _find_branch_containing(ts, children, current_node)
        if next_child is None:
            # 当前节点不在任何子分支中（焦点就是当前节点或更深处）
            # 只显示分叉提示，不继续
            if use_rich:
                ids_str = ", ".join(_short_id(c) for c in children)
                lines.append(f"│ [dim]↳ 有 {len(children)} 个子分支 ({ids_str})[/dim]")
            else:
                ids_str = ", ".join(_short_id(c) for c in children)
                lines.append(f"│ ↳ 有 {len(children)} 个子分支 ({ids_str})")
            return

        # 标注其他兄弟
        siblings = [c for c in children if c != next_child]
        _render_fork_hint(ts, siblings, current_node, lines, use_rich)

    # 沿 next_child 继续向下渲染，最多 depth 层
    _render_downward(ts, next_child, current_node, current_path_set, depth, lines, use_rich)


def _render_downward(
    ts: "TraceSystem",
    node_id: str,
    current_node: str,
    current_path_set: Set[str],
    remaining_depth: int,
    lines: List[str],
    use_rich: bool,
) -> None:
    """沿单路径向下渲染，最多 remaining_depth 层"""
    if remaining_depth <= 0:
        # 还有更多节点但超出范围
        node = ts.trace.get_node(node_id)
        if node is not None:
            if use_rich:
                lines.append("│")
                lines.append("[dim]  ⋮ 下方还有更多节点[/dim]")
            else:
                lines.append("│")
                lines.append("  ⋮ 下方还有更多节点")
        return

    node = ts.trace.get_node(node_id)
    if node is None:
        return

    # 连线
    lines.append("│")

    # 渲染节点
    text = format_node_text(node_id, node)
    is_current = node_id == current_node

    if use_rich:
        if is_current:
            lines.append(f"[bold yellow]{text} ◀ 当前[/bold yellow]")
        elif node_id in current_path_set:
            lines.append(f"[bold]{text}[/bold]")
        elif not node.children:
            lines.append(f"[green]{text}[/green]")
        else:
            lines.append(text)
    else:
        marker = " ◀ 当前" if is_current else ""
        lines.append(f"{text}{marker}")

    # 继续向下
    children = node.children or []
    if not children:
        return

    if len(children) == 1:
        _render_downward(ts, children[0], current_node, current_path_set, remaining_depth - 1, lines, use_rich)
    else:
        # 分叉：选择当前路径方向
        next_child = _find_branch_containing(ts, children, current_node)
        if next_child is None:
            # 当前节点不在任何子分支中，只显示分叉提示
            if use_rich:
                ids_str = ", ".join(_short_id(c) for c in children)
                lines.append(f"│ [dim]↳ 有 {len(children)} 个子分支 ({ids_str})[/dim]")
            else:
                ids_str = ", ".join(_short_id(c) for c in children)
                lines.append(f"│ ↳ 有 {len(children)} 个子分支 ({ids_str})")
            return

        siblings = [c for c in children if c != next_child]
        _render_fork_hint(ts, siblings, current_node, lines, use_rich)
        _render_downward(ts, next_child, current_node, current_path_set, remaining_depth - 1, lines, use_rich)


# ==================== 节点详情视图 ====================

_DETAIL_TRUNCATE = 100


def format_node_detail_rich(ts: "TraceSystem", node_id: str):
    """
    生成节点详情的 Rich Text 视图

    ask_user 节点：展示 agent 提问 + 用户回答 + fork 提示
    工具节点：展示输入参数摘要 + 执行结果 + 性能数据

    Returns:
        rich.text.Text 对象
    """
    from rich.text import Text

    node = ts.trace.get_node(node_id)
    if node is None:
        return Text.from_markup(f"[red]节点 '{node_id}' 不存在[/red]")

    action = node.action or {}
    result = node.result or {}
    action_type = action.get("type", "")
    args = action.get("arguments", {}) or {}

    lines: List[str] = []

    # 头部
    lines.append(f"[bold]节点: {node_id}[/bold]")
    lines.append(f"[dim]父节点: {node.parent_id or 'None'} | "
                 f"子节点: {', '.join(node.children) if node.children else '无'} | "
                 f"时间: {node.timestamp or '?'}[/dim]")
    lines.append("")

    if action_type == "ask_user":
        _detail_ask_user(lines, node_id, action, args, result)
    else:
        _detail_tool(lines, node_id, action_type, args, result)

    # 指标
    if node.metrics:
        lines.append("")
        lines.append("[bold]指标[/bold]")
        for k, v in node.metrics.items():
            lines.append(f"  {k}: {v}")

    return Text.from_markup("\n".join(lines))


def _detail_ask_user(lines: List[str], node_id: str, action: Dict, args: Dict, result: Dict) -> None:
    """ask_user 节点详情"""
    message = args.get("message", result.get("message", ""))
    user_resp = result.get("user_response", "")
    status = result.get("status", "?")

    # Agent 提问
    lines.append("[bold cyan]Agent 提问:[/bold cyan]")
    if message:
        # 保留换行，但限制总长度
        msg_display = message[:300] + "..." if len(message) > 300 else message
        for line in msg_display.split("\n"):
            lines.append(f"  {_escape_rich(line)}")
    else:
        lines.append("  [dim](无消息)[/dim]")

    lines.append("")

    # 用户回答
    if status == "responded" and user_resp:
        lines.append("[bold green]用户回答:[/bold green]")
        resp_display = user_resp[:300] + "..." if len(user_resp) > 300 else user_resp
        for line in resp_display.split("\n"):
            lines.append(f"  {_escape_rich(line)}")
    elif status == "waiting":
        lines.append("[bold yellow]状态: 等待用户回答[/bold yellow]")
    elif status == "skipped":
        lines.append("[bold dim]状态: 已跳过（用户未回答）[/bold dim]")
    else:
        lines.append(f"[dim]状态: {status}[/dim]")

    lines.append("")
    lines.append(f"[dim]提示: /trace fork {_short_id(node_id)} — 在此决策点创建新分支，重新回答[/dim]")


def _detail_tool(lines: List[str], node_id: str, action_type: str, args: Dict, result: Dict) -> None:
    """工具节点详情"""
    lines.append(f"[bold cyan]工具: {action_type}[/bold cyan]")
    lines.append("")

    # 输入参数
    if args:
        lines.append("[bold]输入参数:[/bold]")
        for k, v in args.items():
            v_str = str(v) if v is not None else "None"
            if v_str.startswith("read_json_file("):
                v_str = "[dim](从文件读取)[/dim]"
            elif len(v_str) > _DETAIL_TRUNCATE:
                v_str = v_str[:_DETAIL_TRUNCATE] + "..."
            lines.append(f"  {k}: {_escape_rich(v_str)}")
        lines.append("")

    # 执行结果
    if result:
        lines.append("[bold]执行结果:[/bold]")
        for k, v in result.items():
            v_str = str(v) if v is not None else "None"
            if len(v_str) > _DETAIL_TRUNCATE:
                v_str = v_str[:_DETAIL_TRUNCATE] + "..."
            lines.append(f"  {k}: {_escape_rich(v_str)}")


def _escape_rich(text: str) -> str:
    """转义 Rich markup 中的特殊字符"""
    return text.replace("[", "\\[").replace("]", "\\]")


# ==================== 路径上 ask_user 详情 ====================

def collect_current_node_detail(
    ts: "TraceSystem",
) -> Optional[Dict]:
    """
    收集当前节点的完整信息（仅当当前节点是 ask_user 时）

    Returns:
        dict 包含 node_id, short_id, message, user_response, status；
        如果当前节点不是 ask_user 则返回 None
    """
    current_node = ts.trace.current_node
    if not current_node or current_node == "root":
        return None

    node = ts.get_node(current_node)
    if not node or not node.action:
        return None
    if node.action.get("type") != "ask_user":
        return None

    action = node.action
    result = node.result or {}
    args = action.get("arguments", {}) or {}
    message = args.get("message", result.get("message", ""))
    user_resp = result.get("user_response", "")
    status = result.get("status", "?")

    return {
        "node_id": current_node,
        "short_id": _short_id(current_node),
        "message": message,
        "user_response": user_resp,
        "status": status,
    }


def print_current_node_detail_rich(detail: Optional[Dict], console) -> None:
    """
    在 trace Panel 下方打印当前节点的完整信息（仅 ask_user 节点）

    Args:
        detail: collect_current_node_detail 的返回值
        console: Rich Console 实例
    """
    if not detail:
        return

    status = detail["status"]
    if status == "responded":
        status_tag = "[green]已回答[/green]"
    elif status == "waiting":
        status_tag = "[yellow]等待中[/yellow]"
    elif status == "skipped":
        status_tag = "[dim]已跳过[/dim]"
    else:
        status_tag = f"[dim]{status}[/dim]"

    console.print()
    console.print(f"[bold cyan]── 👤 {detail['short_id']} ask_user[/bold cyan] ({status_tag})")

    if detail["message"]:
        console.print("[bold]Agent 提问:[/bold]")
        console.print(f"  {detail['message']}")

    if status == "responded" and detail["user_response"]:
        console.print("[bold]用户回答:[/bold]")
        console.print(f"  {detail['user_response']}")

    console.print()
