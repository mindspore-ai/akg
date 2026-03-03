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
会话管理命令模块

提供 sessions 和 resume 顶层命令，以及会话辅助函数。
Trace 树的查看、分叉、对比等功能在交互模式内部通过 /trace slash command 提供。
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
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


def register_sessions_command(
    app: typer.Typer,
    console: Console,
    services: CLIAppServices,
) -> None:
    """注册 sessions 子命令到主应用"""
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
        """恢复已有会话"""
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
            try:
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
            except Exception as e:
                from akg_agents.core_v2.filesystem.exceptions import SessionResumeError
                if isinstance(e, SessionResumeError):
                    console.print(f"\n[red]恢复会话失败:[/red] {e.detail}")
                    console.print(f"[dim]Session ID: {e.session_id}[/dim]")
                    if e.cause:
                        console.print(f"[dim]原始错误: {e.cause}[/dim]")
                    console.print(
                        "\n[yellow]提示:[/yellow] 代码修改可能导致旧 session 数据不兼容。"
                        "\n请使用 [cyan]akg_cli sessions list[/cyan] 查看会话，并开始新会话。"
                    )
                    raise typer.Exit(code=3)
                raise
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
