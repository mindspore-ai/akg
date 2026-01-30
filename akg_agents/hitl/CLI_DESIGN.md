# AIKG CLI 交互层设计文档

> 基于 prompt_toolkit 3.0 的斜杠命令式交互
> 设计时间: 2026-01-25
> 参考: qwen-agent 命令式交互风格

---

## 设计目标

### 核心理念
**只保留输入框和流式瀑布输出，统一采用斜杠命令交互**

- ✅ **简洁直观**: `/command` 风格命令，清晰明确
- ✅ **强大补全**: Tab 自动补全命令和参数，带描述提示
- ✅ **易于扩展**: 装饰器注册新命令，无需修改核心代码
- ✅ **通用与专用**: 通用命令（所有场景）+ 专用命令（特定场景）
- ✅ **智能阻塞**: 🔥 查询类命令不阻塞输入，执行类命令自动阻塞

### 对比现状

| 项目 | 现状 | 重构后 |
|------|------|--------|
| 交互方式 | 自然语言输入 | 斜杠命令 + 自然语言混合 |
| 界面显示 | Top5 实现历史框 + 面板 | 只保留输入框和流式瀑布输出 |
| 命令支持 | 简单命令识别（保存/退出） | 通用命令 + 专用命令系统 |
| 自动补全 | 历史补全 | 命令补全 + 历史补全 |
| 可扩展性 | 硬编码 | 装饰器注册 |
| 🔥 输入阻塞 | 任何输入都阻塞 | 查询类命令不阻塞 |

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI 交互层架构                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  用户输入层                                                       │
│  ┌──────────────────────────────────────────┐                  │
│  │  prompt_toolkit Buffer                   │                  │
│  │  • FileHistory (命令历史)                │                  │
│  │  • SlashCommandCompleter (自动补全)      │                  │
│  │  • KeyBindings (快捷键)                  │                  │
│  └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  命令解析层                                                       │
│  ┌──────────────────────────────────────────┐                  │
│  │  输入文本解析                              │                  │
│  │  ├─ 斜杠命令? → CommandDispatcher        │                  │
│  │  └─ 普通输入? → Agent 执行流程            │                  │
│  └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  命令执行层                                                       │
│  ┌──────────────────────────────────────────┐                  │
│  │  SlashCommandRegistry                    │                  │
│  │  ├─ /trace   → TraceCommand              │                  │
│  │  ├─ /parallel → ParallelCommand          │                  │
│  │  ├─ /help    → HelpCommand               │                  │
│  │  ├─ /mode    → ModeCommand               │                  │
│  │  ├─ /save    → SaveCommand               │                  │
│  │  ├─ /history → HistoryCommand            │                  │
│  │  ├─ /clear   → ClearCommand              │                  │
│  │  └─ ...      → 可扩展                     │                  │
│  └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  显示层                                                           │
│  ┌──────────────────────────────────────────┐                  │
│  │  SimplePanelRenderer (简化面板)          │                  │
│  │  • 当前状态显示                           │                  │
│  │  • 命令提示（F2 切换显示/隐藏）           │                  │
│  │  • 实时更新（无 Top5 框架）               │                  │
│  └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 通用命令 vs 专用命令 🔥

### 设计理念

**为什么要区分？**
- **通用命令**：所有场景都需要（如 `/help`, `/trace`, `/save`）
- **专用命令**：特定场景专用（如 ops 的 `/parallel`，common 的其他命令）

### 命令注册方式

```python
# 通用命令：所有场景都注册
@slash_command('help', '显示帮助信息', is_blocking=False)
async def cmd_help(runner, args):
    ...

# ops 专用命令：只在 ops 场景注册
@slash_command('parallel', '启动并行任务', is_blocking=True, scene='ops')
async def cmd_parallel(runner, args):
    ...

# common 专用命令：只在 common 场景注册
@slash_command('refactor', '启动重构任务', is_blocking=True, scene='common')
async def cmd_refactor(runner, args):
    ...
```

### 场景说明

| 场景 | 命令类型 | 示例 |
|------|----------|------|
| **all** | 通用命令 | `/help`, `/trace`, `/save`, `/exit` |
| **ops** | 算子场景 | `/parallel 3 call_codegen` |
| **common** | 通用场景 | （待扩展） |

---

## 命令阻塞机制 🔥

### 设计理念

**问题**：当前 CLI 在输入命令后就禁用输入，但查询类命令（如 `/help`, `/trace`）应该立即返回，不应禁用输入。

**解决方案**：引入 `is_blocking` 属性，区分阻塞和非阻塞命令。

### 命令分类

#### 通用命令（所有场景可用）

| 类型 | 示例命令 | is_blocking | 行为 |
|------|----------|-------------|------|
| **查询类** | `/help`, `/trace`, `/history`, `/log` | `False` | 立即返回结果，不禁用输入 |
| **即时操作** | `/clear`, `/save`, `/exit` | `False` | 瞬间完成，不禁用输入 |

#### 专用命令（特定场景）

| 场景 | 命令 | is_blocking | 说明 |
|------|------|-------------|------|
| **ops** | `/parallel` | `True` | 并行代码生成（如：`/parallel 3 call_codegen`） |
| **common** | （待扩展） | - | 通用场景的专用命令 |

#### 特殊输入

| 类型 | 示例 | is_blocking | 行为 |
|------|------|-------------|------|
| **自然语言** | "生成一个 matmul 算子" | `True` | 调用 Agent，禁用输入 |
| **Ctrl+C** | - | - | 取消当前操作 |

### 工作流程

```
用户输入
    ↓
CommandDispatcher.dispatch()
    ↓
解析输入
    ├─ 普通输入 → CommandResult(handled=False, is_blocking=True)
    │              → 调用 Agent，禁用输入
    │
    └─ 斜杠命令 → 执行命令 → CommandResult(handled=True, is_blocking=?)
                                ↓
                    ┌───────────┴────────────┐
                    │                        │
              is_blocking=False        is_blocking=True
                    ↓                        ↓
            立即返回，继续接受输入       等待完成，禁用输入
```

### 用户体验对比

**改进前**：
```
用户> /help
[等待 Agent 响应... 输入被禁用]  ← 体验不佳
显示帮助信息
[输入重新启用]
```

**改进后**：
```
用户> /help
立即显示帮助信息  ← 瞬间响应
用户> /trace        ← 可以立即输入下一个命令
立即显示 trace
用户> 生成一个 matmul 算子  ← 这时才禁用输入
[Agent 工作中... 输入被禁用]
```

---

## 核心组件设计

### 1. 斜杠命令系统

#### 命令注册（装饰器风格）

```python
# cli/commands/slash_commands.py

from dataclasses import dataclass, field
from typing import Callable, Optional, List
from enum import Enum

class CommandCategory(Enum):
    """命令分类"""
    CONTROL = "控制"      # 控制流命令 (trace, parallel, cancel)
    INFO = "信息"         # 信息查询 (help, history, status)
    SYSTEM = "系统"       # 系统操作 (save, clear, exit)
    DEBUG = "调试"        # 调试命令 (debug, log)

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
    is_blocking: bool = False  # 🔥 是否阻塞输入（True=禁用输入，False=立即返回）
    scene: str = "all"  # 🔥 适用场景（"all"=通用命令，"ops"=ops专用，"common"=common专用）
    
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
    is_blocking: bool = False,  # 🔥 默认非阻塞（查询类命令）
    scene: str = "all",  # 🔥 适用场景（"all"=通用，"ops"=ops专用，"common"=common专用）
):
    """命令注册装饰器
    
    Args:
        is_blocking: 是否阻塞用户输入
            - False (默认): 查询类命令，立即返回，不禁用输入
            - True: 执行类命令，需要等待完成，禁用输入期间
        scene: 适用场景
            - "all" (默认): 通用命令，所有场景可用
            - "ops": ops 场景专用
            - "common": common 场景专用
    """
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
```

#### 核心命令实现

```python
# cli/commands/slash_commands.py (续)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# ============= 信息查询命令 =============

@slash_command(
    'help',
    '显示帮助信息',
    category=CommandCategory.INFO,
    aliases=['h', '?'],
    usage='/help [命令名]',
    examples=['/help', '/help trace'],
    is_blocking=False  # 🔥 查询类，不阻塞
)
async def cmd_help(runner, args: List[str]):
    """显示命令帮助"""
    if args:
        # 显示特定命令的详细帮助
        cmd_name = args[0].lstrip('/')
        cmd = _registry.get(cmd_name)
        if cmd:
            console.print(Panel(
                f"[bold cyan]{cmd.name}[/bold cyan]\n\n"
                f"{cmd.description}\n\n"
                f"[yellow]用法:[/yellow] {cmd.usage}\n"
                f"[yellow]别名:[/yellow] {', '.join(cmd.aliases) if cmd.aliases else '无'}\n"
                f"[yellow]分类:[/yellow] {cmd.category.value}\n\n"
                + (f"[yellow]示例:[/yellow]\n" + "\n".join(f"  • {ex}" for ex in cmd.examples) if cmd.examples else ""),
                title="命令帮助",
                border_style="cyan"
            ))
        else:
            console.print(f"[red]未知命令: {cmd_name}[/red]")
    else:
        # 显示所有命令（按分类）
        for cat in CommandCategory:
            cmds = _registry.list_by_category(cat)
            if not cmds:
                continue
            
            table = Table(title=f"📋 {cat.value}命令", show_header=True, header_style="bold magenta")
            table.add_column("命令", style="cyan", width=20)
            table.add_column("说明", style="green")
            table.add_column("别名", style="yellow", width=15)
            
            for cmd in cmds:
                aliases_str = ', '.join(cmd.aliases) if cmd.aliases else '-'
                table.add_row(f"/{cmd.name}", cmd.description, aliases_str)
            
            console.print(table)
            console.print()

# ============= 控制流命令 =============

@slash_command(
    'trace',
    '查看执行轨迹',
    category=CommandCategory.CONTROL,
    aliases=['t'],
    usage='/trace [trace_id|all]',
    examples=['/trace', '/trace all', '/trace abc123'],
    is_blocking=False  # 🔥 查询类，不阻塞
)
async def cmd_trace(runner, args: List[str]):
    """显示 trace 信息"""
    trace_system = getattr(runner.cli, 'trace_system', None)
    if not trace_system:
        console.print("[yellow]Trace 系统未初始化[/yellow]")
        return
    
    if not args or args[0] == 'all':
        # 显示所有 trace
        traces = trace_system.list_all()
        if not traces:
            console.print("[yellow]暂无 trace 记录[/yellow]")
            return
        
        table = Table(title="Trace 列表", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("时间", style="green")
        table.add_column("状态", style="yellow")
        table.add_column("描述", style="white")
        
        for trace in traces:
            table.add_row(
                trace.id[:8],
                trace.timestamp.strftime("%H:%M:%S"),
                trace.status,
                trace.description[:50]
            )
        console.print(table)
    else:
        # 显示特定 trace 详情
        trace_id = args[0]
        trace = trace_system.get(trace_id)
        if trace:
            console.print(Panel(
                trace.to_rich_text(),
                title=f"Trace {trace_id[:8]}",
                border_style="cyan"
            ))
        else:
            console.print(f"[red]未找到 trace: {trace_id}[/red]")

@slash_command(
    'parallel',
    '启动并行任务',
    category=CommandCategory.CONTROL,
    aliases=['p'],
    usage='/parallel [数量] [子任务类型]',
    examples=['/parallel 4 call_design', '/parallel 3 call_codegen'],
    is_blocking=True,  # 🔥 执行类，启动 Agent 并行跑代码
    scene='ops'  # 🔥 ops 场景专用
)
async def cmd_parallel(runner, args: List[str]):
    """启动并行任务（ops 场景专用）
    
    用法：
        /parallel [数量] [子任务类型]
    
    示例：
        /parallel 4 call_design    - 并行设计 4 个方案
        /parallel 3 call_codegen   - 并行生成 3 个代码实现
    """
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
    
    # TODO: 调用并行执行逻辑
    # await runner.cli.execute_parallel_task(count, subtask)
    console.print("[green]✓[/green] 并行任务已启动")

# ============= 系统操作命令 =============

@slash_command(
    'save',
    '保存当前结果',
    category=CommandCategory.SYSTEM,
    aliases=['s'],
    usage='/save [路径]',
    examples=['/save', '/save ./output/result.py'],
    is_blocking=False  # 🔥 保存操作通常很快，不阻塞
)
async def cmd_save(runner, args: List[str]):
    """保存当前结果到文件"""
    output_path = args[0] if args else None
    # 调用现有的保存逻辑
    console.print(f"[green]✓[/green] 结果已保存{f'到 {output_path}' if output_path else ''}")

@slash_command(
    'history',
    '查看命令历史',
    category=CommandCategory.INFO,
    aliases=['hist'],
    usage='/history [n]',
    examples=['/history', '/history 10'],
    is_blocking=False  # 🔥 查询类，不阻塞
)
async def cmd_history(runner, args: List[str]):
    """显示命令历史"""
    count = int(args[0]) if args and args[0].isdigit() else 10
    # 从 FileHistory 读取
    console.print(f"[cyan]最近 {count} 条命令历史[/cyan]")
    # TODO: 实现历史记录读取

@slash_command(
    'clear',
    '清空屏幕',
    category=CommandCategory.SYSTEM,
    aliases=['cls'],
    usage='/clear',
    examples=['/clear'],
    is_blocking=False  # 🔥 即时操作，不阻塞
)
async def cmd_clear(runner, args: List[str]):
    """清空终端屏幕"""
    import os
    os.system('clear' if os.name != 'nt' else 'cls')

@slash_command(
    'exit',
    '退出程序',
    category=CommandCategory.SYSTEM,
    aliases=['quit', 'q'],
    usage='/exit',
    examples=['/exit'],
    is_blocking=False  # 🔥 手动停止任务，立即响应
)
async def cmd_exit(runner, args: List[str]):
    """退出 CLI"""
    console.print("[yellow]再见！[/yellow]")
    # 设置退出标志
    if hasattr(runner, '_should_exit'):
        runner._should_exit = True

# ============= 调试命令 =============

@slash_command(
    'log',
    '查看日志',
    category=CommandCategory.DEBUG,
    aliases=['l'],
    usage='/log [级别] [行数]',
    examples=['/log', '/log error', '/log info 20'],
    is_blocking=False  # 🔥 查询类，不阻塞
)
async def cmd_log(runner, args: List[str]):
    """查看日志文件"""
    level = args[0] if args else 'all'
    lines = int(args[1]) if len(args) > 1 and args[1].isdigit() else 50
    
    console.print(f"[cyan]最近 {lines} 条日志 (级别: {level})[/cyan]")
    # TODO: 从日志文件读取
```

---

### 2. 自动补全系统

```python
# cli/ui/completers.py

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from typing import Iterable

class SlashCommandCompleter(Completer):
    """斜杠命令自动补全器"""
    
    def __init__(self, registry):
        self.registry = registry
    
    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        
        # 只在输入斜杠开头时提示命令
        if text.startswith('/'):
            word = text[1:]  # 去掉前导 /
            
            for cmd in self.registry.list_all():
                # 匹配命令名或别名
                if cmd.name.startswith(word):
                    yield self._make_completion(cmd, word, False)
                else:
                    for alias in cmd.aliases:
                        if alias.startswith(word):
                            yield self._make_completion(cmd, word, True, alias)
                            break
    
    def _make_completion(self, cmd, word: str, is_alias: bool, alias: str = None) -> Completion:
        """构造补全项"""
        display_name = alias if is_alias else cmd.name
        display_text = f'/{display_name}'
        
        # 构建描述信息
        meta_parts = [cmd.description]
        if is_alias:
            meta_parts.append(f'(→ {cmd.name})')
        if cmd.category:
            meta_parts.append(f'[{cmd.category.value}]')
        
        return Completion(
            text=display_name,
            start_position=-len(word),
            display=display_text,
            display_meta=' '.join(meta_parts),
            style='fg:ansigreen bold' if not is_alias else 'fg:ansiyellow'
        )

class EnhancedCompleter(Completer):
    """增强的自动补全器（组合多种补全源）"""
    
    def __init__(self, command_registry):
        self.slash_completer = SlashCommandCompleter(command_registry)
        # 可以添加更多补全器：参数补全、路径补全等
    
    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        
        # 斜杠命令补全
        if text.startswith('/'):
            yield from self.slash_completer.get_completions(document, complete_event)
        
        # TODO: 其他补全逻辑（如命令参数、文件路径等）
```

---

### 3. 命令调度器

```python
# cli/commands/slash_commands.py (续)

@dataclass
class CommandResult:
    """命令执行结果"""
    handled: bool        # 是否处理了输入（True=斜杠命令，False=普通输入）
    is_blocking: bool    # 是否阻塞输入（True=禁用输入，False=继续接受输入）
    error: str = ""      # 错误信息（如果有）

class CommandDispatcher:
    """命令调度器"""
    
    def __init__(self, registry: SlashCommandRegistry):
        self.registry = registry
        self.console = Console()
    
    async def dispatch(self, runner, text: str) -> CommandResult:
        """
        调度命令执行
        
        Returns:
            CommandResult: 包含处理状态和阻塞信息
        """
        text = text.strip()
        
        # 不是斜杠命令，返回未处理（让 Runner 按普通输入处理，且阻塞输入）
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
            self.console.print(f"[red]未知命令: /{cmd_name}[/red]")
            self.console.print("输入 [cyan]/help[/cyan] 查看可用命令")
            return CommandResult(handled=True, is_blocking=False)  # 未知命令不阻塞
        
        # 执行命令
        try:
            result = cmd.handler(runner, cmd_args)
            if hasattr(result, '__await__'):
                await result
        except Exception as e:
            self.console.print(f"[red]命令执行失败: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return CommandResult(handled=True, is_blocking=False, error=str(e))
        
        # 🔥 关键：根据命令的 is_blocking 属性决定是否阻塞输入
        return CommandResult(handled=True, is_blocking=cmd.is_blocking)

def create_dispatcher() -> CommandDispatcher:
    """创建全局命令调度器"""
    return CommandDispatcher(get_registry())
```

---

### 4. 简化面板

```python
# cli/ui/panel_simple.py

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from datetime import datetime

class StreamOutputRenderer:
    """流式瀑布输出渲染器（只保留输入框和输出）"""
    
    def __init__(self):
        self.console = Console()
        self._current_phase = ""
        self._current_op = ""
    
    def update(self, phase: str = "", op_name: str = ""):
        """更新当前状态（用于流式输出）"""
        if phase:
            self._current_phase = phase
        if op_name:
            self._current_op = op_name
    
    def render_output(self, chunk: str):
        """渲染流式输出块"""
        self.console.print(chunk, end="")
    
    def render_header(self, scene: str = "ops", **kwargs):
        """渲染启动头部
        
        Args:
            scene: 场景名称（"ops" 或 "common"）
            **kwargs: 场景相关配置（如 backend, arch, dsl）
        """
        lines = [
            "  🤖 AIKG - AKG Agents",
            "",
            "入门提示：",
            "1. 输入需求描述或使用斜杠命令",
            "2. /help 查看所有命令",
            "3. Tab 键自动补全命令",
            "4. Ctrl+C 取消操作",
            "",
        ]
        
        # 显示场景信息
        if scene == "ops":
            config_parts = []
            if 'backend' in kwargs:
                config_parts.append(f"backend={kwargs['backend']}")
            if 'arch' in kwargs:
                config_parts.append(f"arch={kwargs['arch']}")
            if 'dsl' in kwargs:
                config_parts.append(f"dsl={kwargs['dsl']}")
            if config_parts:
                lines.append(f"已加载：ops 场景，{', '.join(config_parts)}")
        elif scene == "common":
            lines.append("已加载：common 场景")
        
        for line in lines:
            self.console.print(line)
```

---

## 集成到现有 Runner

### 修改 InteractiveOpRunner

```python
# cli/commands/op/runners.py (关键修改)

from akg_agents.cli.commands.slash_commands import create_dispatcher
from akg_agents.cli.ui.completers import EnhancedCompleter
from akg_agents.cli.ui.panel_simple import SimplePanelRenderer

class InteractiveOpRunner:
    def __init__(self, ...):
        # ... 现有初始化 ...
        
        # 🔥 新增：命令调度器和流式输出渲染器
        self._command_dispatcher = create_dispatcher()
        self._stream_renderer = StreamOutputRenderer()
    
    async def _run_tui_app(self, ...):
        # ... 现有代码 ...
        
        def _accept_input(buff: Buffer) -> None:
            text = buff.text
            buff.text = ""
            
            # 🔥 新增：先尝试作为斜杠命令处理
            asyncio.create_task(self._handle_input(text))
        
        # 🔥 修改：使用增强补全器
        from akg_agents.cli.commands.slash_commands import get_registry
        buffer = Buffer(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=EnhancedCompleter(get_registry()),  # 新增
            multiline=True,
            accept_handler=_accept_input,
        )
        
        # 🔥 移除：不再需要面板相关代码
        # 只保留输入框和流式输出
        
        # ... 其余现有代码 ...
    
    async def _handle_input(self, text: str):
        """处理用户输入（斜杠命令或普通输入）"""
        # 尝试作为斜杠命令处理
        result = await self._command_dispatcher.dispatch(self, text)
        
        if not result.handled:
            # 不是斜杠命令，按原有逻辑处理（阻塞输入）
            state.is_generating = True
            state.loading_text = "运行中..."
            app.invalidate()
            input_queue.put_nowait(text)
        elif result.is_blocking:
            # 是阻塞命令，禁用输入
            state.is_generating = True
            state.loading_text = "执行中..."
            app.invalidate()
        else:
            # 🔥 非阻塞命令：立即完成，不禁用输入
            # 什么都不做，用户可以继续输入下一个命令
            pass
```

---

## 使用示例

### 启动后的界面

```
  █████╗ ██╗  ██╗ ██████╗ 
 ██╔══██╗██║ ██╔╝██╔════╝ 
 ███████║█████╔╝ ██║  ███╗
 ██╔══██║██╔═██╗ ██║   ██║
 ██║  ██║██║  ██╗╚██████╔╝
 ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ 

🤖 AKG Agents

入门提示：
1. 输入需求描述或使用斜杠命令
2. /help 查看所有命令
3. Tab 键自动补全命令
4. Ctrl+C 取消操作

已加载：ops 场景，backend=cuda, arch=a100, dsl=triton
────────────────────────────────────────────────────────────────
 * /help
────────────────────────────────────────────────────────────────
📋 控制命令
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ 命令               ┃ 说明                               ┃ 别名          ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ /trace             │ 查看执行轨迹                       │ t             │
│ /parallel          │ 启动并行任务 (ops 场景专用)       │ p             │
└────────────────────┴────────────────────────────────────┴───────────────┘

📋 信息命令
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ 命令               ┃ 说明                     ┃ 别名          ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ /help              │ 显示帮助信息             │ h, ?          │
│ /history           │ 查看命令历史             │ hist          │
└────────────────────┴──────────────────────────┴───────────────┘

📋 系统命令
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ 命令               ┃ 说明                     ┃ 别名          ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ /save              │ 保存当前结果             │ s             │
│ /clear             │ 清空屏幕                 │ cls           │
│ /exit              │ 退出程序                 │ quit, q       │
└────────────────────┴──────────────────────────┴───────────────┘

💡 提示：使用 Ctrl+C 取消当前操作

📋 调试命令
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ 命令               ┃ 说明                     ┃ 别名          ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ /log               │ 查看日志                 │ l             │
└────────────────────┴──────────────────────────┴───────────────┘

────────────────────────────────────────────────────────────────
 * 
```

### Tab 自动补全演示

```
 * /tr<Tab>
 
→ 自动补全为：
 * /trace  查看执行轨迹 [控制]

继续输入：
 * /trace all
 
→ 显示所有 trace 列表
```

### 非阻塞命令演示 🔥

```
已加载：ops 场景，backend=cuda, arch=a100, dsl=triton
────────────────────────────────────────────────────────────────
 * /help
────────────────────────────────────────────────────────────────
[立即显示帮助信息，输入框保持可用]

📋 控制命令
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ 命令               ┃ 说明                     ┃ 别名          ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ /trace             │ 查看执行轨迹             │ t             │
...

────────────────────────────────────────────────────────────────
 * /trace                    ← 🔥 可以立即输入，不需等待
────────────────────────────────────────────────────────────────
[立即显示 trace 列表，输入框保持可用]

Trace 列表
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID     ┃ 时间     ┃ 状态   ┃ 描述                   ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ abc123 │ 10:30:15 │ 成功   │ matmul 算子生成        │
└────────┴──────────┴────────┴────────────────────────┘

────────────────────────────────────────────────────────────────
 * /history                  ← 🔥 可以连续输入多个查询命令
────────────────────────────────────────────────────────────────
[立即显示历史记录，输入框保持可用]

最近 10 条命令历史
1. /help
2. /trace
3. 生成一个 matmul 算子
...

────────────────────────────────────────────────────────────────
 * /parallel 3 call_codegen   ← 🔥 启动并行Agent（ops专用），阻塞输入
────────────────────────────────────────────────────────────────
[Agent 工作中... 输入被禁用]

🚀 启动并行任务: 3 x call_codegen
🤖 并行任务 1: 生成代码...
🤖 并行任务 2: 生成代码...
🤖 并行任务 3: 生成代码...
✓ 3 个实现已生成

────────────────────────────────────────────────────────────────
 * /save                     ← 🔥 完成后输入恢复，可继续操作
────────────────────────────────────────────────────────────────
✓ 结果已保存

────────────────────────────────────────────────────────────────
 * [Ctrl+C]                  ← 🔥 取消操作（用 Ctrl+C，不需要 /cancel）
────────────────────────────────────────────────────────────────
⚠️ 操作已被用户取消
```

---

## 扩展新命令

### 添加新命令只需 3 步

```python
# 1. 导入装饰器
from akg_agents.cli.commands.slash_commands import slash_command, CommandCategory

# 2. 使用装饰器定义命令
@slash_command(
    'profile',                          # 命令名
    '查看性能分析',                      # 描述
    category=CommandCategory.INFO,      # 分类
    aliases=['prof', 'perf'],           # 别名
    usage='/profile [算子名]',          # 用法
    examples=['/profile matmul'],       # 示例
    is_blocking=False                   # 🔥 是否阻塞输入（查询类=False）
)
async def cmd_profile(runner, args: List[str]):
    """性能分析命令"""
    op_name = args[0] if args else None
    # 实现逻辑
    ...

# 3. 无需注册，装饰器自动完成
# 命令立即可用，自动补全也会包含
```

### 如何选择 is_blocking 值？

**is_blocking=False**（默认，推荐）：
- ✅ 查询类命令：`/help`, `/trace`, `/history`, `/log`
- ✅ 配置类命令：`/mode`
- ✅ 即时操作：`/clear`, `/save`, `/cancel`, `/exit`

**is_blocking=True**：
- ✅ 执行类命令：`/parallel`（启动 Agent 跑代码）
- ✅ 自然语言输入（调用 Agent）

**经验法则**：
- 如果命令在 100ms 内完成 → `is_blocking=False`
- 如果命令需要启动 Agent 执行任务 → `is_blocking=True`
- 不确定时 → 使用默认值 `False`（更好的用户体验）

**核心命令总结**：

#### 通用命令

| 命令 | is_blocking | 类型 | 说明 |
|------|-------------|------|------|
| `/help` | `False` | 查询 | 显示帮助信息 |
| `/trace` | `False` | 查询 | 查看执行轨迹 |
| `/history` | `False` | 查询 | 查看命令历史 |
| `/log` | `False` | 查询 | 查看日志 |
| `/save` | `False` | 操作 | 保存结果 |
| `/clear` | `False` | 操作 | 清空屏幕 |
| `/exit` | `False` | 控制 | 退出程序 |

#### ops 场景专用命令

| 命令 | is_blocking | 类型 | 说明 |
|------|-------------|------|------|
| `/parallel` | `True` | 执行 | 启动并行任务（如：`/parallel 3 call_codegen`） |

#### 特殊操作

| 操作 | 说明 |
|------|------|
| `Ctrl+C` | 取消当前操作 |

---

## 实施计划

### 阶段 1: 基础框架 (1天)
- [ ] 实现 `SlashCommandRegistry` 和装饰器
- [ ] 实现 `CommandDispatcher` 和 `CommandResult`
- [ ] 🔥 实现阻塞/非阻塞机制（`is_blocking` 属性）
- [ ] 集成到 `InteractiveOpRunner._accept_input`
- [ ] 实现基础命令：`/help`, `/exit`
- [ ] 测试非阻塞命令（输入不被禁用）

### 阶段 2: 自动补全 (1天)
- [ ] 实现 `SlashCommandCompleter`
- [ ] 实现 `EnhancedCompleter`
- [ ] 集成到 `Buffer`
- [ ] 测试补全效果

### 阶段 3: 核心命令 (1-2天)
- [ ] 实现 `/trace` 命令（通用命令）
- [ ] 实现 `/parallel` 命令（ops 专用命令）
- [ ] 实现 `/save` 命令（通用命令）
- [ ] 实现 `/history` 命令（通用命令）
- [ ] 实现 `/log` 命令（通用命令）
- [ ] 实现 `/clear` 命令（通用命令）
- [ ] 实现 `/exit` 命令（通用命令）

### 阶段 4: 界面简化 (0.5天)
- [ ] 实现 `StreamOutputRenderer`（流式瀑布输出）
- [ ] 移除 `KernelImplListPlugin` 相关代码
- [ ] 移除 Top5 框架相关代码
- [ ] 移除 F2 面板切换功能
- [ ] 只保留输入框和流式输出

### 阶段 5: Common 场景支持 (0.5天)
- [ ] 复用命令系统到 `akg_cli common`
- [ ] 调整 Logo 和提示文本
- [ ] 测试两个场景的一致性

### 阶段 6: 测试和优化 (1天)
- [ ] 功能测试
- [ ] 用户体验测试
- [ ] 性能优化
- [ ] 文档完善

**总计**: 约 3-5 天

---

## 技术栈

- **prompt_toolkit 3.0.52**: TUI 框架（已安装）
- **typer 0.21.0**: CLI 参数解析（已安装）
- **rich 13.9.4**: 美化输出（已安装）

**✅ 无需新增依赖**

---

## 关键设计决策总结

### 1. 界面简化
- ❌ 移除：Top5 框架、面板切换（F2）
- ✅ 保留：输入框 + 流式瀑布输出

### 2. 命令分类
- **通用命令**（7个）：所有场景可用
  - `/help`, `/trace`, `/history`, `/log`, `/save`, `/clear`, `/exit`
- **专用命令**：特定场景专用
  - ops: `/parallel [数量] [子任务类型]`
  - common: （待扩展）

### 3. 取消操作
- ❌ 不使用 `/cancel` 命令
- ✅ 使用 `Ctrl+C` 取消当前操作

### 4. 阻塞机制
- **非阻塞**（7个通用命令）：立即返回，可连续输入
- **阻塞**（1个专用命令）：`/parallel` 启动 Agent 时阻塞

### 5. 命令注册
```python
# 通用命令
@slash_command('help', ..., scene='all')  # 默认

# 专用命令
@slash_command('parallel', ..., scene='ops')
```

---

## 成功标准

### 核心功能
- [ ] 支持核心通用命令（7个：help, trace, history, log, save, clear, exit）
- [ ] 支持 ops 专用命令（1个：parallel）
- [ ] Tab 自动补全可用且准确
- [ ] 命令历史可用（上下箭头）
- [ ] Ctrl+C 取消当前操作

### 界面设计
- [ ] 只保留输入框和流式瀑布输出
- [ ] 移除所有 Top5 框架代码
- [ ] 移除 F2 面板切换功能

### 架构设计
- [ ] 区分通用命令和专用命令
- [ ] `akg_cli op` 和 `akg_cli common` 共享通用命令
- [ ] 扩展新命令只需添加装饰器函数

### 交互体验
- [ ] 🔥 查询类命令不阻塞输入（可连续输入多个命令）
- [ ] 🔥 `/parallel` 启动并行 Agent 正确阻塞输入
- [ ] 🔥 `/parallel` 用法：`/parallel [数量] [子任务类型]`
- [ ] 🔥 `/exit` 立即响应，不阻塞
- [ ] 用户反馈良好

---

**更新时间**: 2026-01-25
**状态**: 设计完成，待实施
**负责人**: TBD
