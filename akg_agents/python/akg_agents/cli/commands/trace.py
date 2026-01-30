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

提供 trace 树的查看、切换和对比功能：
- /trace show [task_id] - 显示 trace 树
- /trace switch <task_id> <node_id> - 切换到指定节点
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


def create_trace_app(console: Console, services: CLIAppServices) -> typer.Typer:
    """创建 trace 子命令应用"""
    
    trace_app = typer.Typer(help="Trace 树管理命令")
    
    @trace_app.command("show")
    def show_trace(
        task_id: str = typer.Argument(..., help="任务 ID"),
        base_dir: Optional[str] = typer.Option(None, "--base-dir", "-d", help="基础目录"),
    ):
        """显示 trace 树结构"""
        try:
            from akg_agents.core_v2.filesystem import TraceSystem
            
            ts = TraceSystem(task_id, base_dir=base_dir)
            
            # 检查任务是否存在
            if not ts.fs.task_exists():
                console.print(f"[red]任务 '{task_id}' 不存在[/red]")
                raise typer.Exit(code=1)
            
            # 获取树可视化
            tree_str = ts.visualize_tree()
            
            # 获取当前节点
            current = ts.get_current_node()
            
            console.print(Panel(
                tree_str,
                title=f"[bold]Trace Tree: {task_id}[/bold]",
                subtitle=f"当前节点: [cyan]{current}[/cyan]",
            ))
            
            # 显示统计信息
            leaf_nodes = ts.get_all_leaf_nodes()
            total_nodes = len(ts.trace.tree)
            
            table = Table(title="统计信息", show_header=False)
            table.add_column("指标", style="cyan")
            table.add_column("值", style="green")
            table.add_row("总节点数", str(total_nodes))
            table.add_row("叶节点数", str(len(leaf_nodes)))
            table.add_row("当前节点", current)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            raise typer.Exit(code=1)
    
    @trace_app.command("switch")
    def switch_node(
        task_id: str = typer.Argument(..., help="任务 ID"),
        node_id: str = typer.Argument(..., help="目标节点 ID"),
        base_dir: Optional[str] = typer.Option(None, "--base-dir", "-d", help="基础目录"),
    ):
        """切换到指定节点"""
        try:
            from akg_agents.core_v2.filesystem import TraceSystem
            
            ts = TraceSystem(task_id, base_dir=base_dir)
            
            if not ts.fs.task_exists():
                console.print(f"[red]任务 '{task_id}' 不存在[/red]")
                raise typer.Exit(code=1)
            
            old_node = ts.get_current_node()
            ts.switch_node(node_id)
            new_node = ts.get_current_node()
            
            console.print(f"[green]✓[/green] 已从 [cyan]{old_node}[/cyan] 切换到 [cyan]{new_node}[/cyan]")
            
            # 显示节点信息
            node_info = ts.get_node(new_node)
            if node_info:
                console.print(f"  父节点: {node_info.get('parent_id', 'None')}")
                console.print(f"  子节点数: {len(node_info.get('children', []))}")
                
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
            if node_info.get("metrics"):
                console.print(f"Metrics: {node_info['metrics']}")
                
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
                metrics = node_info.get("metrics", {})
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


def register_trace_command(
    app: typer.Typer,
    console: Console,
    services: CLIAppServices,
) -> None:
    """注册 trace 子命令到主应用"""
    trace_app = create_trace_app(console, services)
    app.add_typer(trace_app, name="trace")
