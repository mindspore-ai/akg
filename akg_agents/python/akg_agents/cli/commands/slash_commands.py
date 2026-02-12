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

"""斜杠命令系统 - 基于装饰器的命令注册和调度"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
import json
from typing import Callable, Optional, List
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class CommandCategory(Enum):
    """命令分类"""
    CONTROL = "控制"
    INFO = "信息"
    SYSTEM = "系统"
    DEBUG = "调试"


@dataclass
class SlashCommand:
    """斜杠命令定义"""
    name: str
    description: str
    handler: Callable
    category: CommandCategory = CommandCategory.CONTROL
    aliases: List[str] = field(default_factory=list)
    usage: str = ""
    examples: List[str] = field(default_factory=list)
    args_required: bool = False
    is_blocking: bool = False
    scene: str = "all"


class SlashCommandRegistry:
    """命令注册表（单例）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._commands = {}
            cls._instance._categories = {}
        return cls._instance
    
    def register(self, cmd: SlashCommand):
        """注册命令"""
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd
        
        # 按分类组织
        if cmd.category not in self._categories:
            self._categories[cmd.category] = []
        if cmd not in self._categories[cmd.category]:
            self._categories[cmd.category].append(cmd)
    
    def get(self, name: str) -> Optional[SlashCommand]:
        """获取命令"""
        return self._commands.get(name.lower())
    
    def list_all(self) -> List[SlashCommand]:
        """列出所有命令（去重）"""
        seen = set()
        result = []
        for cmd in self._commands.values():
            if id(cmd) not in seen:
                seen.add(id(cmd))
                result.append(cmd)
        return result
    
    def list_by_category(self, category: CommandCategory) -> List[SlashCommand]:
        """按分类列出命令"""
        return self._categories.get(category, [])


# 全局注册表
_registry = SlashCommandRegistry()


def slash_command(
    name: str,
    desc: str,
    category: CommandCategory = CommandCategory.CONTROL,
    aliases: List[str] = None,
    usage: str = "",
    examples: List[str] = None,
    is_blocking: bool = False,
    scene: str = "all",
):
    """命令注册装饰器"""
    def decorator(func):
        _registry.register(SlashCommand(
            name=name,
            description=desc,
            handler=func,
            category=category,
            aliases=aliases or [],
            usage=usage or f"/{name} [参数]",
            examples=examples or [],
            is_blocking=is_blocking,
            scene=scene,
        ))
        return func
    return decorator


def get_registry() -> SlashCommandRegistry:
    """获取全局注册表"""
    return _registry


# ============= 命令实现 =============

@slash_command(
    'help',
    '显示帮助信息',
    category=CommandCategory.INFO,
    aliases=['h', '?'],
    usage='/help [命令名]',
    examples=['/help', '/help trace'],
    is_blocking=False
)
async def cmd_help(runner, args: List[str]):
    """显示命令帮助"""
    runner_scene = getattr(runner, "scene", "all")
    def _scene_allowed(cmd: SlashCommand) -> bool:
        return cmd.scene == "all" or cmd.scene == runner_scene

    if args:
        # 显示特定命令的详细帮助
        cmd_name = args[0].lstrip('/')
        cmd = _registry.get(cmd_name)
        if cmd and _scene_allowed(cmd):
            # 构建详细帮助内容
            aliases_text = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
            content_lines = [
                f"[bold cyan]/{cmd.name}{aliases_text}[/bold cyan]",
                "",
                f"  {cmd.description}",
                "",
            ]
            
            if cmd.usage:
                content_lines.append(f"  [yellow]用法:[/yellow] {cmd.usage}")
            
            # 显示 handler docstring 中的详细说明（子命令等）
            docstring = (cmd.handler.__doc__ or "").strip()
            if docstring:
                # 跳过第一行（通常和 description 重复），取后续行作为详细说明
                doc_lines = docstring.split("\n")
                detail_lines = [l.strip() for l in doc_lines[1:] if l.strip()]
                if detail_lines:
                    content_lines.append("")
                    content_lines.append("  [yellow]子命令:[/yellow]")
                    for dl in detail_lines:
                        content_lines.append(f"    {dl}")
            
            if cmd.examples:
                content_lines.append("")
                content_lines.append("  [yellow]示例:[/yellow]")
                for ex in cmd.examples:
                    content_lines.append(f"    {ex}")
            
            content_lines.append("")
            content_lines.append(f"  [dim]分类: {cmd.category.value}[/dim]")
            
            console.print(Panel(
                "\n".join(content_lines),
                border_style="dim",
                padding=(1, 2),
            ))
        else:
            console.print(f"[red]❌ 未知命令: {cmd_name}[/red]")
    else:
        # 显示所有命令（更美观的格式）
        content_lines = []
        
        # 基础提示
        content_lines.extend([
            "[bold]基础功能:[/bold]",
            "  • 输入斜杠命令: 使用 / 开头执行命令（例如 /help, /trace）",
            "  • Tab 自动补全: 输入 / 后按 Tab 键查看所有可用命令",
            "  • 上下箭头: 浏览历史命令",
            "  • Ctrl+C 取消: 清空输入 / 取消任务 / 退出",
            "",
        ])
        
        # 按分类显示命令
        content_lines.append("[bold]命令:[/bold]")
        
        for cat in CommandCategory:
            cmds = _registry.list_by_category(cat)
            cmds = [cmd for cmd in cmds if _scene_allowed(cmd)]
            if not cmds:
                continue
            
            # 分类标题
            content_lines.append(f"  [cyan]{cat.value}:[/cyan]")
            
            # 每个命令
            for cmd in cmds:
                aliases_text = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
                content_lines.append(f"    /{cmd.name}{aliases_text} - {cmd.description}")
            
            content_lines.append("")
        
        # 底部提示
        content_lines.extend([
            "[dim]💡 提示: 使用 /help <命令名> 查看命令详细帮助[/dim]",
        ])
        
        console.print(Panel(
            "\n".join(content_lines),
            border_style="dim",
            padding=(1, 2),
        ))


@slash_command(
    'trace',
    '查看 trace 树节点、详情、创建分叉',
    category=CommandCategory.INFO,
    aliases=['t'],
    usage='/trace [<id>|root|show <id>|node <id>|history|fork <id>] [-n depth]',
    examples=['/trace', '/trace root', '/trace 005', '/trace -n 8', '/trace show 003', '/trace fork 005'],
    is_blocking=False
)
async def cmd_trace(runner, args: List[str]):
    """查看 trace 树节点、查看详情、创建分叉
    
    子命令:
      /trace                   — 以当前节点为中心，上下各 4 层（路径视图）
      /trace root              — 以 root 为起点向下展示
      /trace <id>              — 以指定节点为中心展示其所在路径
      /trace -n <N>            — 调整展示范围为上下各 N 层
      /trace show <id>         — 查看指定节点详情（action、result、参数）
      /trace node <id>         — 同 show（兼容旧命令）
      /trace history           — 查看当前节点的完整动作历史
      /trace fork <id>         — 在 ask_user 节点创建分叉（重新回答）
    """
    # 获取 trace：优先从已创建的 agent 获取，否则用 session_id 直接构建
    trace = None
    try:
        executor = getattr(runner, "cli", None)
        react_executor = getattr(executor, "_react_executor", None) if executor else None
        agent = getattr(react_executor, "_agent", None) if react_executor else None
        trace = getattr(agent, "trace", None) if agent else None
    except Exception:
        pass
    
    # 回退：agent 尚未创建时，直接用 session_id 构建 TraceSystem
    if trace is None:
        try:
            executor = getattr(runner, "cli", None)
            session_id = getattr(executor, "session_id", None) if executor else None
            if session_id:
                from akg_agents.core_v2.filesystem import TraceSystem
                task_id = f"cli_{session_id}"
                ts = TraceSystem(task_id=task_id)
                if ts.fs.task_exists():
                    ts.initialize(force=False)
                    trace = ts
        except Exception:
            pass
    
    if trace is None:
        console.print("[yellow]当前会话无 trace 信息可显示[/yellow]")
        console.print("[dim]可能原因: 尚未发送过请求，或 session 目录不存在[/dim]")
        return
    
    # 解析参数：提取 subcmd, focus_node, depth
    subcmd = "show"
    focus_node = None
    depth = 4
    
    remaining = list(args)
    
    # 解析 -n <depth>
    if "-n" in remaining:
        n_idx = remaining.index("-n")
        if n_idx + 1 < len(remaining):
            try:
                depth = int(remaining[n_idx + 1])
            except ValueError:
                pass
            remaining = remaining[:n_idx] + remaining[n_idx + 2:]
        else:
            remaining = remaining[:n_idx]
    
    if remaining:
        first = remaining[0]
        if first in ("node", "history", "fork"):
            subcmd = first
        elif first == "show":
            # /trace show <id> → 节点详情
            if len(remaining) >= 2:
                subcmd = "node"
                focus_node = remaining[1]
            else:
                subcmd = "tree"
        elif first == "root":
            subcmd = "tree"
            focus_node = "root"
        elif first.isdigit():
            # /trace 005 → 以 node_005 为焦点的路径视图
            subcmd = "tree"
            focus_node = f"node_{first.zfill(3)}"
        elif first.startswith("node_"):
            subcmd = "tree"
            focus_node = first
    
    # 规范化 focus_node
    if focus_node and focus_node != "root" and not focus_node.startswith("node_"):
        if focus_node.isdigit():
            focus_node = f"node_{focus_node.zfill(3)}"
    
    if subcmd in ("show", "tree"):
        # 显示 trace 树（单路径 + 分叉标注）
        try:
            current = trace.get_current_node()
            rich_tree = trace.visualize_tree_rich(focus_node=focus_node, depth=depth)
            
            # 先打印当前节点的完整信息（仅 ask_user），再打印 trace Panel
            from akg_agents.core_v2.filesystem.trace_visualizer import (
                collect_current_node_detail,
                print_current_node_detail_rich,
            )
            detail = collect_current_node_detail(trace)
            print_current_node_detail_rich(detail, console)
            
            # 构建标题
            if focus_node and focus_node != current:
                subtitle_text = f"焦点: [cyan]{focus_node}[/cyan] | 当前: [cyan]{current}[/cyan]"
            else:
                subtitle_text = f"当前节点: [cyan]{current}[/cyan]"
            
            console.print(Panel(
                rich_tree,
                title=f"[bold]Trace: {trace.task_id}[/bold]",
                subtitle=subtitle_text,
                padding=(1, 2),
            ))
            
            total_nodes = len(trace.trace.tree)
            console.print(
                f"[dim]总节点: {total_nodes} | 当前: {current} | 范围: ±{depth} 层[/dim]"
            )
            console.print(
                f"[dim]操作: /trace <id> 查看分支 | /trace show <id> 详情 | /trace fork <id> 分叉[/dim]"
            )
            
        except Exception as e:
            console.print(f"[red]显示 trace 失败: {e}[/red]")
    
    elif subcmd == "node":
        # 显示指定节点详情
        # 支持 /trace show <id> 和 /trace node <id>
        raw_id = focus_node or (args[1] if len(args) >= 2 else None)
        if not raw_id:
            console.print("[yellow]用法: /trace show <id> 或 /trace node <id>[/yellow]")
            return
        
        # 规范化 node_id
        if raw_id != "root" and not raw_id.startswith("node_"):
            if raw_id.isdigit():
                raw_id = f"node_{raw_id.zfill(3)}"
        node_id = raw_id
        
        try:
            from akg_agents.core_v2.filesystem.trace_visualizer import format_node_detail_rich
            detail_text = format_node_detail_rich(trace, node_id)
            console.print(Panel(
                detail_text,
                title=f"[bold]节点详情: {node_id}[/bold]",
                padding=(1, 2),
            ))
        except Exception as e:
            console.print(f"[red]查看节点失败: {e}[/red]")
    
    elif subcmd == "history":
        # 显示当前节点的动作历史
        try:
            from rich.table import Table as RichTable
            current = trace.get_current_node()
            history = trace.get_full_action_history(current)
            
            if not history:
                console.print(f"[yellow]节点 '{current}' 没有动作历史[/yellow]")
                return
            
            table = RichTable(title=f"动作历史: {current}")
            table.add_column("#", style="dim", width=4)
            table.add_column("Tool", style="green")
            table.add_column("Timestamp", style="dim")
            
            for i, action in enumerate(history[-20:], max(1, len(history) - 19)):
                table.add_row(
                    str(i),
                    action.tool_name,
                    action.timestamp[:19] if action.timestamp else "-",
                )
            
            console.print(table)
            if len(history) > 20:
                console.print(f"[dim]显示最近 20 条，共 {len(history)} 条[/dim]")
        except Exception as e:
            console.print(f"[red]查看历史失败: {e}[/red]")
    
    elif subcmd == "fork" and len(args) >= 2:
        # 在 ask_user 节点创建分叉
        raw_node_id = args[1]
        # 支持简写：005 -> node_005
        if raw_node_id.isdigit():
            node_id = f"node_{raw_node_id.zfill(3)}"
        elif raw_node_id.startswith("node_"):
            node_id = raw_node_id
        else:
            node_id = raw_node_id
        
        try:
            # 验证目标节点存在
            node_info = trace.get_node(node_id)
            if not node_info:
                console.print(f"[red]节点 '{node_id}' 不存在[/red]")
                return
            
            # 执行 fork
            new_node_id = trace.fork_ask_user(node_id)
            
            # 同步更新 agent 内存状态（如果 agent 已创建）
            agent = None
            try:
                executor = getattr(runner, "cli", None)
                react_executor = getattr(executor, "_react_executor", None) if executor else None
                agent = getattr(react_executor, "_agent", None) if react_executor else None
            except Exception:
                pass
            
            if agent is not None:
                # 更新 agent 的 current_node_id 到新节点
                agent.current_node_id = new_node_id
                
                # 重建 tool_executor.history（从 root 到新节点的路径）
                new_history = trace.get_full_action_history(new_node_id)
                agent.tool_executor.history = list(new_history)
            
            # 显示 agent 的原始提问
            action = node_info.action or {}
            message = (action.get("arguments") or {}).get("message", "")
            
            console.print(f"[green]已创建分叉:[/green] [cyan]{node_id}[/cyan] → 新节点 [cyan]{new_node_id}[/cyan]")
            console.print(f"[dim]原节点和历史完全保留，新节点等待用户回答[/dim]")
            if message:
                console.print()
                console.print("[bold cyan]Agent 的提问:[/bold cyan]")
                for line in message.split("\n"):
                    console.print(f"  {line}")
                console.print()
                console.print("[yellow]请直接输入您的回答（作为新分支的起点）[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Fork 失败: {e}[/red]")
    
    else:
        console.print("[yellow]用法: /trace [show <id>|node <id>|history|fork <id>][/yellow]")
        console.print("[dim]  /trace             - 显示 trace 树（路径视图）[/dim]")
        console.print("[dim]  /trace <id>        - 以指定节点为焦点展示路径[/dim]")
        console.print("[dim]  /trace show <id>   - 查看节点详情（action、result、参数）[/dim]")
        console.print("[dim]  /trace history     - 查看动作历史[/dim]")
        console.print("[dim]  /trace fork <id>   - 在 ask_user 节点创建分叉（支持简写: 005 → node_005）[/dim]")


@slash_command(
    'parallel',
    '启动并行任务',
    category=CommandCategory.CONTROL,
    aliases=['p'],
    usage='/parallel [数量] [子任务类型]',
    examples=['/parallel 4 call_design', '/parallel 3 call_codegen'],
    is_blocking=True,
    scene='ops'
)
async def cmd_parallel(runner, args: List[str]):
    """启动并行任务（ops 场景专用）"""
    if len(args) < 2:
        console.print("[yellow]用法: /parallel [数量] [子任务类型][/yellow]")
        console.print("[dim]示例: /parallel 4 call_design[/dim]")
        console.print("[dim]示例: /parallel 3 call_codegen[/dim]")
        return
    
    try:
        count = int(args[0])
        subtask = args[1]
    except ValueError:
        console.print("[red]错误: 数量必须是整数[/red]")
        return
    
    console.print(f"[cyan]🚀 启动并行任务: {count} x {subtask}[/cyan]")
    console.print("[yellow]ℹ️  并行任务功能开发中...[/yellow]")


@slash_command(
    'save',
    '保存当前结果',
    category=CommandCategory.SYSTEM,
    aliases=['s'],
    usage='/save [路径]',
    examples=['/save', '/save ./output/result.py'],
    is_blocking=False
)
async def cmd_save(runner, args: List[str]):
    """保存当前结果到文件（功能开发中）"""
    output_path = args[0] if args else None
    console.print("[yellow]ℹ️  保存功能开发中...[/yellow]")
    if output_path:
        console.print(f"[dim]目标路径: {output_path}[/dim]")



# @slash_command(
#     'plan',
#     '进入 plan 模式（common 场景）',
#     category=CommandCategory.CONTROL,
#     aliases=['planning'],
#     usage='/plan',
#     examples=['/plan'],
#     is_blocking=True,
#     scene='common'
# )
# async def cmd_plan(runner, args: List[str]):
#     """Manually enter plan mode for common."""
#     if args:
#         console.print("[red]❌ /plan 不支持参数[/red]")
#         return
#     if not hasattr(runner, "cli") or not hasattr(runner.cli, "enter_common_plan_mode"):
#         console.print("[yellow]ℹ️  当前会话未启动 common，请先输入一次请求[/yellow]")
#         return
#
#     result = runner.cli.enter_common_plan_mode()
#     if isinstance(result, dict):
#         if hasattr(runner, "_render_agent_messages"):
#             runner._render_agent_messages(result)
#         else:
#             msg = str(result.get("display_message") or "").strip()
#             if msg:
#                 console.print(msg)
#         auto_input = result.get("auto_input")
#         if auto_input and hasattr(runner, "_process_input"):
#             await runner._process_input(str(auto_input), print_input=False)
#         return
#
#     console.print("[red]❌ /plan 执行失败[/red]")


@slash_command(
    'compact',
    '压缩上下文',
    category=CommandCategory.SYSTEM,
    aliases=['cmp'],
    usage='/compact',
    examples=['/compact'],
    is_blocking=False,
    scene='common'
)
async def cmd_compact(runner, args: List[str]):
    """Compact common conversation context."""
    if not hasattr(runner, "cli") or not hasattr(runner.cli, "compact_common_history"):
        console.print("[yellow]ℹ️  当前会话未启动 common，请先输入一次请求[/yellow]")
        return
    if args:
        console.print("[red]❌ /compact 不支持参数[/red]")
        return
    result = await runner.cli.compact_common_history()
    if not result or not result.get("ok"):
        reason = (result or {}).get("reason") or "unknown"
        detail = (result or {}).get("error")
        if reason == "not_started":
            console.print("[yellow]ℹ️  当前会话未启动 common，请先输入一次请求[/yellow]")
        elif reason == "no_messages":
            console.print("[yellow]ℹ️  当前没有可压缩的上下文[/yellow]")
        else:
            if detail:
                console.print(f"[red]❌ Compact 失败: {reason} ({detail})[/red]")
            else:
                console.print(f"[red]❌ Compact 失败: {reason}[/red]")
        return

    summary = str(result.get("summary") or "").strip()
    console.print("[green]✅ Session compacted[/green]")
    if summary:
        console.print(Panel(summary, border_style="dim", padding=(1, 2)))
    else:
        console.print("[yellow]ℹ️  未生成摘要内容[/yellow]")


# 已移除 /clear 命令（用户反馈没用）
# @slash_command(
#     'clear',
#     '清空屏幕',
#     category=CommandCategory.SYSTEM,
#     aliases=['cls'],
#     usage='/clear',
#     examples=['/clear'],
#     is_blocking=False
# )
# async def cmd_clear(runner, args: List[str]):
#     """清空终端屏幕"""
#     os.system('clear' if os.name != 'nt' else 'cls')


@slash_command(
    'exit',
    '退出程序',
    category=CommandCategory.SYSTEM,
    aliases=['quit', 'q'],
    usage='/exit',
    examples=['/exit'],
    is_blocking=False
)
async def cmd_exit(runner, args: List[str]):
    """退出 CLI"""
    console.print("[yellow]👋 再见！[/yellow]")
    # 设置退出标志
    if hasattr(runner, '_should_exit'):
        runner._should_exit = True


@slash_command(
    'log',
    '查看日志',
    category=CommandCategory.DEBUG,
    aliases=['l'],
    usage='/log [级别] [行数]',
    examples=['/log', '/log error', '/log info 20'],
    is_blocking=False
)
async def cmd_log(runner, args: List[str]):
    """查看日志文件（功能开发中）"""
    level = args[0] if args else 'all'
    lines = int(args[1]) if len(args) > 1 and args[1].isdigit() else 50
    console.print(f"[yellow]ℹ️  日志查看功能开发中...[/yellow]")
    console.print(f"[dim]级别: {level}, 行数: {lines}[/dim]")


@slash_command(
    'list_tools',
    '列出当前可用工具',
    category=CommandCategory.INFO,
    aliases=['tools'],
    usage='/list_tools',
    examples=['/list_tools'],
    is_blocking=False,
    scene='common'
)
async def cmd_list_tools(runner, args: List[str]):
    """列出可用工具（common 场景）"""
    tools = []
    if hasattr(runner, "cli") and hasattr(runner.cli, "get_common_tools"):
        tools = runner.cli.get_common_tools() or []

    if not tools:
        console.print("[yellow]ℹ️  暂无工具信息（请先运行一次任务）[/yellow]")
        return

    table = Table(title="Common Tools", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")

    for tool in tools:
        name = getattr(tool, "name", "") or ""
        desc = getattr(tool, "description", "") or ""
        table.add_row(str(name), str(desc))

    console.print(table)


@slash_command(
    'display_last_raw_llm_input',
    '显示上一次 LLM 原始输入',
    category=CommandCategory.DEBUG,
    aliases=['last_input', 'last_llm_input'],
    usage='/display_last_raw_llm_input',
    examples=['/display_last_raw_llm_input'],
    is_blocking=False,
    scene='common'
)
async def cmd_display_last_raw_llm_input(runner, args: List[str]):
    """显示上一轮 LLM 输入（common 场景）"""
    payload = None
    if hasattr(runner, "cli") and hasattr(runner.cli, "get_last_raw_llm_input"):
        payload = runner.cli.get_last_raw_llm_input()
    if not payload:
        console.print("[yellow]ℹ️  暂无 LLM 输入记录（请先运行一次任务）[/yellow]")
        return
    try:
        content = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        content = str(payload)
    console.print(Panel(content, border_style="dim", padding=(1, 2)))

@slash_command(
    'mock_wait_30',
    '测试命令：等待30秒（测试阻塞）',
    category=CommandCategory.DEBUG,
    aliases=['mock30', 'wait30'],
    usage='/mock_wait_30',
    examples=['/mock_wait_30'],
    is_blocking=True  # 🔥 阻塞命令，用于测试
)
async def cmd_mock_wait_30(runner, args: List[str]):
    """Mock 命令：等待30秒（用于测试阻塞行为）"""
    import asyncio
    console.print("[cyan]⏱️  开始等待 30 秒...[/cyan]")
    console.print("[dim]（测试阻塞行为：运行期间输入框应该被禁用）[/dim]")
    
    for i in range(30, 0, -1):
        console.print(f"[yellow]倒计时: {i} 秒...[/yellow]")
        await asyncio.sleep(1)
    
    console.print("[green]✅ 等待完成！[/green]")


# ============= 命令调度器 =============

@dataclass
class CommandResult:
    """命令执行结果"""
    handled: bool
    is_blocking: bool
    error: str = ""


class CommandDispatcher:
    """命令调度器"""
    
    def __init__(self, registry: SlashCommandRegistry):
        self.registry = registry
        self.console = Console()
    
    async def dispatch(self, runner, text: str) -> CommandResult:
        """调度命令执行"""
        text = text.strip()
        
        # 不是斜杠命令，返回未处理
        if not text.startswith('/'):
            return CommandResult(handled=False, is_blocking=True)
        
        # 解析命令和参数
        parts = text[1:].split()
        if not parts:
            return CommandResult(handled=True, is_blocking=False)
        
        cmd_name = parts[0].lower()
        cmd_args = parts[1:] if len(parts) > 1 else []
        
        # 查找命令
        cmd = self.registry.get(cmd_name)
        if not cmd:
            self.console.print(f"[red]❌ 未知命令: /{cmd_name}[/red]")
            self.console.print("输入 [cyan]/help[/cyan] 查看可用命令")
            return CommandResult(handled=True, is_blocking=False)

        runner_scene = getattr(runner, "scene", "all")
        if cmd.scene != "all" and cmd.scene != runner_scene:
            self.console.print(
                f"[yellow]ℹ️  命令 /{cmd.name} 仅在 {cmd.scene} 场景可用[/yellow]"
            )
            return CommandResult(handled=True, is_blocking=False)
        
        # 执行命令
        try:
            result = cmd.handler(runner, cmd_args)
            if hasattr(result, '__await__'):
                await result
        except Exception as e:
            self.console.print(f"[red]❌ 命令执行失败: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return CommandResult(handled=True, is_blocking=False, error=str(e))
        
        # 根据命令的 is_blocking 属性决定是否阻塞输入
        return CommandResult(handled=True, is_blocking=cmd.is_blocking)


def create_dispatcher() -> CommandDispatcher:
    """创建全局命令调度器"""
    return CommandDispatcher(get_registry())
