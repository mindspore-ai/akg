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
    if args:
        # 显示特定命令的详细帮助
        cmd_name = args[0].lstrip('/')
        cmd = _registry.get(cmd_name)
        if cmd:
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
    '查看执行轨迹',
    category=CommandCategory.CONTROL,
    aliases=['t'],
    usage='/trace [trace_id|all]',
    examples=['/trace', '/trace all', '/trace abc123'],
    is_blocking=False
)
async def cmd_trace(runner, args: List[str]):
    """显示 trace 信息（功能开发中）"""
    console.print("[yellow]ℹ️  Trace 功能开发中...[/yellow]")
    if args:
        console.print(f"[dim]参数: {' '.join(args)}[/dim]")


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


@slash_command(
    'history',
    '查看命令历史',
    category=CommandCategory.INFO,
    aliases=['hist'],
    usage='/history [n]',
    examples=['/history', '/history 10'],
    is_blocking=False
)
async def cmd_history(runner, args: List[str]):
    """显示命令历史（功能开发中）"""
    count = int(args[0]) if args and args[0].isdigit() else 10
    console.print(f"[yellow]ℹ️  命令历史功能开发中...[/yellow]")
    console.print(f"[dim]将显示最近 {count} 条命令[/dim]")


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
