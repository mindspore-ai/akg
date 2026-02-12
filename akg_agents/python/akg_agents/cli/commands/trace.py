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

"""
Trace 命令模块

提供 trace 树的查看、分叉和对比功能：
- /trace show [task_id] - 显示 trace 树
- /trace fork <task_id> <node_id> - 在 ask_user 节点创建分叉
- /trace compare <task_id> <node1> <node2> - 对比两个节点
- /trace path <task_id> <node_id> - 显示到节点的路径
- /trace history <task_id> [node_id] - 显示动作历史
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from typing import Optional

from ..service import CLIAppServices


def resolve_session_dir(session_id: str):
    """根据 session_id 解析会话目录路径
    
    Args:
        session_id: session_id 或完整 task_id（如 cli_xxx）
        
    Returns:
        (normalized_session_id, session_dir_path) 元组
    """
    from pathlib import Path
    sid = session_id
    if sid.startswith("cli_"):
        sid = sid[4:]
    session_dir = Path.home() / ".akg" / "conversations" / f"cli_{sid}"
    return sid, session_dir


def load_session_domain(session_dir) -> str:
    """从会话目录中读取 domain 信息
    
    查找策略：
    1. 优先从 root/state.json 的 task_info.domain 读取
    2. 回退：如果有 dsl/framework 等 op 特征字段，推断为 "op"
    
    Args:
        session_dir: 会话目录 Path 对象
        
    Returns:
        domain 字符串，如 "op"、"common"；未知时返回空字符串
    """
    import json
    
    root_state_file = session_dir / "nodes" / "root" / "state.json"
    if root_state_file.exists():
        try:
            rs = json.loads(root_state_file.read_text())
            ti = rs.get("task_info", {})
            if ti.get("domain"):
                return ti["domain"]
            # 旧会话兼容：有 dsl 字段说明是 op 领域
            if ti.get("dsl") or ti.get("backend"):
                return "op"
        except Exception:
            pass
    
    # 回退：从 trace.json 的 action arguments 推断
    trace_file = session_dir / "trace.json"
    if trace_file.exists():
        try:
            trace_data = json.loads(trace_file.read_text())
            tree = trace_data.get("tree", {})
            for node_id in sorted(tree.keys()):
                node = tree[node_id]
                args = (node.get("action") or {}).get("arguments") or {}
                if args.get("dsl") and args.get("framework"):
                    return "op"
        except Exception:
            pass
    
    return ""


def create_trace_app(console: Console, services: CLIAppServices) -> typer.Typer:
    """创建 trace 子命令应用"""
    
    trace_app = typer.Typer(help="Trace 树管理命令")
    
    @trace_app.command("show")
    def show_trace(
        task_id: str = typer.Argument(..., help="任务 ID"),
        focus: Optional[str] = typer.Argument(None, help="焦点节点 ID（如 root、005、node_005），默认为当前节点"),
        base_dir: Optional[str] = typer.Option(None, "--base-dir", "-d", help="基础目录"),
        depth: int = typer.Option(4, "--depth", "-n", help="向上/向下展示的层数"),
    ):
        """显示 trace 树结构（单路径视图 + 分叉标注）"""
        try:
            from akg_agents.core_v2.filesystem import TraceSystem
            
            ts = TraceSystem(task_id, base_dir=base_dir)
            
            # 检查任务是否存在
            if not ts.fs.task_exists():
                console.print(f"[red]任务 '{task_id}' 不存在[/red]")
                raise typer.Exit(code=1)
            
            # 规范化 focus_node
            focus_node = None
            if focus:
                if focus == "root":
                    focus_node = "root"
                elif focus.isdigit():
                    focus_node = f"node_{focus.zfill(3)}"
                elif focus.startswith("node_"):
                    focus_node = focus
                else:
                    focus_node = focus
            
            current = ts.get_current_node()
            rich_tree = ts.visualize_tree_rich(focus_node=focus_node, depth=depth)
            
            # 先打印当前节点的完整信息（仅 ask_user），再打印 trace Panel
            from akg_agents.core_v2.filesystem.trace_visualizer import (
                collect_current_node_detail,
                print_current_node_detail_rich,
            )
            detail = collect_current_node_detail(ts)
            print_current_node_detail_rich(detail, console)
            
            if focus_node and focus_node != current:
                subtitle_text = f"焦点: [cyan]{focus_node}[/cyan] | 当前: [cyan]{current}[/cyan]"
            else:
                subtitle_text = f"当前节点: [cyan]{current}[/cyan]"
            
            console.print(Panel(
                rich_tree,
                title=f"[bold]Trace: {task_id}[/bold]",
                subtitle=subtitle_text,
                padding=(1, 2),
            ))
            
            total_nodes = len(ts.trace.tree)
            console.print(
                f"[dim]总节点: {total_nodes} | 当前: {current} | 范围: ±{depth} 层[/dim]"
            )
            
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            raise typer.Exit(code=1)
    
    @trace_app.command("fork")
    def fork_node(
        task_id: str = typer.Argument(..., help="任务 ID"),
        node_id: str = typer.Argument(..., help="目标 ask_user 节点 ID"),
        base_dir: Optional[str] = typer.Option(None, "--base-dir", "-d", help="基础目录"),
    ):
        """在 ask_user 节点创建分叉（重新回答）"""
        try:
            from akg_agents.core_v2.filesystem import TraceSystem
            
            ts = TraceSystem(task_id, base_dir=base_dir)
            
            if not ts.fs.task_exists():
                console.print(f"[red]任务 '{task_id}' 不存在[/red]")
                raise typer.Exit(code=1)
            
            ts.initialize(force=False)
            
            # 规范化 node_id
            if node_id.isdigit():
                node_id = f"node_{node_id.zfill(3)}"
            
            new_node_id = ts.fork_ask_user(node_id)
            
            # 显示结果
            console.print(f"[green]✓[/green] 已创建分叉: [cyan]{node_id}[/cyan] → 新节点 [cyan]{new_node_id}[/cyan]")
            console.print(f"[dim]原节点和历史完全保留，新节点等待用户回答[/dim]")
            
            # 显示 agent 的原始提问
            node_info = ts.get_node(node_id)
            if node_info:
                action = node_info.action or {}
                message = (action.get("arguments") or {}).get("message", "")
                if message:
                    console.print()
                    console.print("[bold cyan]Agent 的提问:[/bold cyan]")
                    for line in message.split("\n"):
                        console.print(f"  {line}")
                
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            raise typer.Exit(code=1)
    
    @trace_app.command("compare")
    def compare_nodes(
        task_id: str = typer.Argument(..., help="任务 ID"),
        node1: str = typer.Argument(..., help="节点 1 ID"),
        node2: str = typer.Argument(..., help="节点 2 ID"),
        base_dir: Optional[str] = typer.Option(None, "--base-dir", "-d", help="基础目录"),
    ):
        """对比两个节点"""
        try:
            from akg_agents.core_v2.filesystem import TraceSystem
            
            ts = TraceSystem(task_id, base_dir=base_dir)
            
            if not ts.fs.task_exists():
                console.print(f"[red]任务 '{task_id}' 不存在[/red]")
                raise typer.Exit(code=1)
            
            comparison = ts.compare_nodes(node1, node2)
            
            console.print(Panel(
                f"分叉点: [cyan]{comparison.get('fork_point', 'None')}[/cyan]",
                title=f"[bold]节点对比: {node1} vs {node2}[/bold]",
            ))
            
            # 显示两条路径
            table = Table(title="路径对比")
            table.add_column("节点 1 路径", style="cyan")
            table.add_column("节点 2 路径", style="green")
            
            path1 = comparison.get("path_1", [])
            path2 = comparison.get("path_2", [])
            max_len = max(len(path1), len(path2))
            
            for i in range(max_len):
                p1 = path1[i] if i < len(path1) else ""
                p2 = path2[i] if i < len(path2) else ""
                # 标记分叉点
                if p1 == comparison.get("fork_point"):
                    p1 = f"[bold yellow]{p1}[/bold yellow] (分叉)"
                if p2 == comparison.get("fork_point"):
                    p2 = f"[bold yellow]{p2}[/bold yellow] (分叉)"
                table.add_row(p1, p2)
            
            console.print(table)
            
            # 显示 metrics 对比
            metrics1 = comparison.get("metrics_1", {})
            metrics2 = comparison.get("metrics_2", {})
            
            if metrics1 or metrics2:
                metrics_table = Table(title="Metrics 对比")
                metrics_table.add_column("指标", style="cyan")
                metrics_table.add_column("节点 1", style="blue")
                metrics_table.add_column("节点 2", style="green")
                
                all_keys = set(metrics1.keys()) | set(metrics2.keys())
                for key in sorted(all_keys):
                    v1 = str(metrics1.get(key, "-"))
                    v2 = str(metrics2.get(key, "-"))
                    metrics_table.add_row(key, v1, v2)
                
                console.print(metrics_table)
                
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            raise typer.Exit(code=1)
    
    @trace_app.command("path")
    def show_path(
        task_id: str = typer.Argument(..., help="任务 ID"),
        node_id: str = typer.Argument(..., help="目标节点 ID"),
        base_dir: Optional[str] = typer.Option(None, "--base-dir", "-d", help="基础目录"),
    ):
        """显示从 root 到指定节点的路径"""
        try:
            from akg_agents.core_v2.filesystem import TraceSystem
            
            ts = TraceSystem(task_id, base_dir=base_dir)
            
            if not ts.fs.task_exists():
                console.print(f"[red]任务 '{task_id}' 不存在[/red]")
                raise typer.Exit(code=1)
            
            path = ts.get_path_to_node(node_id)
            
            if not path:
                console.print(f"[red]节点 '{node_id}' 不存在或路径无效[/red]")
                raise typer.Exit(code=1)
            
            # 使用 Tree 显示路径
            tree = Tree(f"[bold cyan]root[/bold cyan]")
            current = tree
            for i, node in enumerate(path[1:], 1):  # 跳过 root
                style = "bold green" if node == node_id else "cyan"
                current = current.add(f"[{style}]{node}[/{style}]")
            
            console.print(Panel(tree, title=f"路径: root → {node_id}"))
            console.print(f"路径长度: {len(path)} 个节点")
            
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            raise typer.Exit(code=1)
    
    @trace_app.command("history")
    def show_history(
        task_id: str = typer.Argument(..., help="任务 ID"),
        node_id: Optional[str] = typer.Argument(None, help="节点 ID (默认为当前节点)"),
        base_dir: Optional[str] = typer.Option(None, "--base-dir", "-d", help="基础目录"),
        limit: int = typer.Option(20, "--limit", "-n", help="最多显示的动作数"),
    ):
        """显示节点的动作历史"""
        try:
            from akg_agents.core_v2.filesystem import TraceSystem
            
            ts = TraceSystem(task_id, base_dir=base_dir)
            
            if not ts.fs.task_exists():
                console.print(f"[red]任务 '{task_id}' 不存在[/red]")
                raise typer.Exit(code=1)
            
            target_node = node_id or ts.get_current_node()
            history = ts.get_full_action_history(target_node)
            
            if not history:
                console.print(f"[yellow]节点 '{target_node}' 没有动作历史[/yellow]")
                return
            
            table = Table(title=f"动作历史: {target_node}")
            table.add_column("#", style="dim", width=4)
            table.add_column("Action ID", style="cyan")
            table.add_column("Tool", style="green")
            table.add_column("Success", style="blue")
            table.add_column("Timestamp", style="dim")
            
            # 限制显示数量
            displayed = history[-limit:] if len(history) > limit else history
            start_idx = len(history) - len(displayed) + 1
            
            for i, action in enumerate(displayed, start_idx):
                success = "✓" if action.result.get("success", False) else "✗"
                success_style = "green" if action.result.get("success", False) else "red"
                table.add_row(
                    str(i),
                    action.action_id,
                    action.tool_name,
                    f"[{success_style}]{success}[/{success_style}]",
                    action.timestamp[:19] if action.timestamp else "-",
                )
            
            console.print(table)
            
            if len(history) > limit:
                console.print(f"[dim]显示最近 {limit} 条，共 {len(history)} 条[/dim]")
                
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            raise typer.Exit(code=1)
    
    @trace_app.command("best")
    def show_best_node(
        task_id: str = typer.Argument(..., help="任务 ID"),
        metric: str = typer.Option("performance", "--metric", "-m", help="评价指标"),
        base_dir: Optional[str] = typer.Option(None, "--base-dir", "-d", help="基础目录"),
    ):
        """显示最优叶节点"""
        try:
            from akg_agents.core_v2.filesystem import TraceSystem
            
            ts = TraceSystem(task_id, base_dir=base_dir)
            
            if not ts.fs.task_exists():
                console.print(f"[red]任务 '{task_id}' 不存在[/red]")
                raise typer.Exit(code=1)
            
            best = ts.get_best_leaf_node(metric=metric)
            
            if not best:
                console.print(f"[yellow]没有找到符合条件的叶节点[/yellow]")
                return
            
            node_info = ts.get_node(best)
            path = ts.get_path_to_node(best)
            
            console.print(Panel(
                f"[bold green]{best}[/bold green]",
                title=f"最优叶节点 (按 {metric})",
            ))
            
            console.print(f"路径长度: {len(path)} 个节点")
            if node_info.metrics:
                console.print(f"Metrics: {node_info.metrics}")
                
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            raise typer.Exit(code=1)
    
    @trace_app.command("leaves")
    def show_leaf_nodes(
        task_id: str = typer.Argument(..., help="任务 ID"),
        base_dir: Optional[str] = typer.Option(None, "--base-dir", "-d", help="基础目录"),
    ):
        """显示所有叶节点"""
        try:
            from akg_agents.core_v2.filesystem import TraceSystem
            
            ts = TraceSystem(task_id, base_dir=base_dir)
            
            if not ts.fs.task_exists():
                console.print(f"[red]任务 '{task_id}' 不存在[/red]")
                raise typer.Exit(code=1)
            
            leaves = ts.get_all_leaf_nodes()
            
            if not leaves:
                console.print("[yellow]没有叶节点[/yellow]")
                return
            
            table = Table(title="叶节点列表")
            table.add_column("#", style="dim", width=4)
            table.add_column("Node ID", style="cyan")
            table.add_column("路径长度", style="blue")
            table.add_column("Metrics", style="green")
            
            for i, leaf in enumerate(leaves, 1):
                path = ts.get_path_to_node(leaf)
                node_info = ts.get_node(leaf)
                metrics = node_info.metrics
                metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else "-"
                table.add_row(
                    str(i),
                    leaf,
                    str(len(path)),
                    metrics_str,
                )
            
            console.print(table)
            console.print(f"共 {len(leaves)} 个叶节点")
                
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            raise typer.Exit(code=1)
    
    return trace_app


def create_sessions_app(console: Console, services: CLIAppServices) -> typer.Typer:
    """创建 sessions 子命令应用"""
    
    sessions_app = typer.Typer(help="会话管理命令")
    
    @sessions_app.command("list")
    def list_sessions(
        limit: int = typer.Option(20, "--limit", "-n", help="最多显示的会话数"),
    ):
        """列出所有已保存的会话"""
        from pathlib import Path
        import json
        from datetime import datetime
        
        conversations_dir = Path.home() / ".akg" / "conversations"
        
        if not conversations_dir.exists():
            console.print("[yellow]暂无会话记录[/yellow]")
            console.print(f"[dim]会话目录: {conversations_dir}[/dim]")
            return
        
        # 收集所有会话信息
        sessions = []
        for session_dir in conversations_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            task_id = session_dir.name
            trace_file = session_dir / "trace.json"
            current_node_file = session_dir / "current_node.txt"
            
            info = {
                "task_id": task_id,
                "session_id": task_id[4:] if task_id.startswith("cli_") else task_id,
                "current_node": "",
                "node_count": 0,
                "created_at": "",
                "modified_at": "",
                "target": "",
                "domain": "",
            }
            
            # 读取 current_node
            if current_node_file.exists():
                try:
                    info["current_node"] = current_node_file.read_text().strip()
                except Exception:
                    pass
            
            # 读取 trace.json 获取节点数、创建时间和 target 配置
            if trace_file.exists():
                try:
                    with open(trace_file, "r") as f:
                        trace_data = json.load(f)
                    info["node_count"] = len(trace_data.get("tree", {}))
                    info["created_at"] = trace_data.get("created_at", "")[:19]
                    
                    # 提取 target 配置 (framework/backend/arch/dsl)
                    tree = trace_data.get("tree", {})
                    root_state_file = session_dir / "nodes" / "root" / "state.json"
                    target_found = False
                    if root_state_file.exists():
                        try:
                            rs = json.loads(root_state_file.read_text())
                            ti = rs.get("task_info", {})
                            if ti.get("dsl"):
                                parts = [p for p in [ti.get("framework"), ti.get("backend"), ti.get("arch"), ti.get("dsl")] if p]
                                info["target"] = "/".join(parts)
                                target_found = True
                        except Exception:
                            pass
                    # 回退：从 action arguments 找
                    if not target_found:
                        for nid in sorted(tree.keys()):
                            node = tree[nid]
                            args = (node.get("action") or {}).get("arguments") or {}
                            if args.get("dsl") and args.get("framework"):
                                parts = [p for p in [args.get("framework"), args.get("backend"), args.get("arch"), args.get("dsl")] if p]
                                info["target"] = "/".join(parts)
                                break
                except Exception:
                    pass
            
            # 使用公共函数读取 domain（含旧会话兼容推断）
            info["domain"] = load_session_domain(session_dir)
            
            # 获取目录修改时间
            try:
                mtime = session_dir.stat().st_mtime
                info["modified_at"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
            
            sessions.append(info)
        
        if not sessions:
            console.print("[yellow]暂无会话记录[/yellow]")
            return
        
        # 按修改时间倒序排列
        sessions.sort(key=lambda s: s["modified_at"], reverse=True)
        sessions = sessions[:limit]
        
        table = Table(title="会话列表")
        table.add_column("#", style="dim", width=4)
        table.add_column("Session ID", style="cyan")
        table.add_column("Domain", style="yellow")
        table.add_column("Target", style="magenta")
        table.add_column("当前节点", style="green")
        table.add_column("节点数", style="blue", justify="right")
        table.add_column("创建时间", style="dim")
        table.add_column("最后修改", style="dim")
        
        for i, s in enumerate(sessions, 1):
            table.add_row(
                str(i),
                s["session_id"],
                s["domain"] or "-",
                s["target"] or "-",
                s["current_node"],
                str(s["node_count"]),
                s["created_at"],
                s["modified_at"],
            )
        
        console.print(table)
        
        console.print(f"[dim]共 {len(sessions)} 个会话 | 恢复命令: akg_cli resume <session_id>[/dim]")
    
    return sessions_app


def register_trace_command(
    app: typer.Typer,
    console: Console,
    services: CLIAppServices,
) -> None:
    """注册 trace 和 sessions 子命令到主应用"""
    trace_app = create_trace_app(console, services)
    app.add_typer(trace_app, name="trace")
    
    sessions_app = create_sessions_app(console, services)
    app.add_typer(sessions_app, name="sessions")


def register_resume_command(
    app: typer.Typer,
    console: Console,
    services: CLIAppServices,
) -> None:
    """注册顶层 resume 命令：根据 session 中保存的 domain 自动路由到对应领域"""

    @app.command("resume")
    def resume_cmd(
        session_id: str = typer.Argument(
            ...,
            help="要恢复的会话 ID（如 50b111f7-...）或完整 task_id（如 cli_50b111f7-...）",
        ),
        # 通用参数（各 domain 都可能用到）
        auto_yes: bool = typer.Option(
            False, "--yes", "-y", help="自动确认所有提示，使用默认值"
        ),
        worker_url: Optional[str] = typer.Option(
            None, "--worker_url", "--worker-url",
            help="Worker Service 地址，逗号分隔",
        ),
        devices: Optional[str] = typer.Option(
            None, "--devices",
            help="本地设备列表，逗号分隔（如 0,1,2,3）。与 --worker_url 互斥。",
        ),
        stream: bool = typer.Option(
            True, "--stream/--no-stream", help="启用/关闭 LLM 流式输出"
        ),
        rag: bool = typer.Option(
            False, "--rag/--no-rag", help="启用/关闭 RAG"
        ),
        output_path: Optional[str] = typer.Option(
            None, "--output-path",
            help="保存目录根路径",
        ),
    ) -> None:
        """恢复已有会话（自动识别领域）。"""
        sid, session_dir = resolve_session_dir(session_id)

        if not session_dir.exists():
            console.print(f"[red]错误:[/red] 会话不存在: cli_{sid}")
            console.print(f"[dim]路径: {session_dir}[/dim]")
            console.print("[dim]使用 akg_cli sessions list 查看可用会话[/dim]")
            raise typer.Exit(code=2)

        domain = load_session_domain(session_dir)

        if domain == "op":
            # 路由到 op 领域
            from .op.orchestrator import OpCommandOrchestrator
            from .op.types import OpCommandArgs

            orchestrator = OpCommandOrchestrator(console=console, services=services)
            orchestrator.run(
                OpCommandArgs(
                    intent=None,
                    task_file=None,
                    framework="",
                    backend="",
                    arch="",
                    dsl="",
                    auto_yes=auto_yes,
                    worker_url=worker_url,
                    devices=devices,
                    stream=stream,
                    rag=rag,
                    output_path=output_path,
                    resume_session_id=sid,
                )
            )
        else:
            # 未知 domain 或未来扩展
            if domain:
                console.print(
                    f"[red]错误:[/red] 暂不支持恢复 domain='{domain}' 的会话"
                )
            else:
                console.print(
                    "[red]错误:[/red] 无法识别该会话的领域（domain），"
                    "可能是旧版本创建的会话"
                )
            console.print("[dim]当前支持的 domain: op[/dim]")
            raise typer.Exit(code=2)
