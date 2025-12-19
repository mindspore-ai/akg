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

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.table import Table
from textual import log

from ..service import CLIAppServices
from ai_kernel_generator.cli.cli.constants import DisplayStyle
from ..utils.ui_helpers import print_logo_once


def register_misc_commands(
    app: typer.Typer, console: Console, services: CLIAppServices
) -> None:
    def _parse_devices(devices_str: str) -> list[int]:
        devices_str = (devices_str or "").strip()
        if not devices_str:
            console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --devices 不能为空（例如 0 或 0,1,2,3）"
            )
            raise typer.Exit(code=2)

        try:
            device_ids = [
                int(x.strip()) for x in devices_str.split(",") if x.strip() != ""
            ]
        except (TypeError, ValueError):
            console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --devices 格式非法: {devices_str}（期望逗号分隔整数）"
            )
            raise typer.Exit(code=2)
        if not device_ids:
            console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --devices 解析结果为空: {devices_str}"
            )
            raise typer.Exit(code=2)
        if len(set(device_ids)) != len(device_ids):
            console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --devices 不允许重复: {devices_str}"
            )
            raise typer.Exit(code=2)
        if any(d < 0 for d in device_ids):
            console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --devices 不能包含负数: {devices_str}"
            )
            raise typer.Exit(code=2)
        return device_ids

    @app.command("list")
    def list_cmd() -> None:
        """列出 akg_cli 当前支持的功能"""
        print_logo_once(console)
        services.config.print_report_if_any(console)

        table = Table(title="AKG CLI 功能列表", show_header=True)
        table.add_column("子命令", style=DisplayStyle.CYAN)
        table.add_column("说明", style=DisplayStyle.YELLOW)

        table.add_row(
            "op", "算子生成入口（search/default 交互；或从 KernelBench 文件/目录导入）"
        )
        table.add_row("graph", "图优化入口（占位，后续实现）")
        table.add_row("worker", "启动/停止 Worker Service（akg_cli worker --start/--stop）")
        console.print(table)

    @app.command("graph")
    def graph_cmd() -> None:
        """图优化入口（占位）"""
        print_logo_once(console)
        services.config.print_report_if_any(console)
        console.print(
            f"[{DisplayStyle.YELLOW}]graph 功能尚未实现：后续将接入图优化流程。[/{DisplayStyle.YELLOW}]"
        )

    @app.command("worker")
    def worker_cmd(
        start: bool = typer.Option(False, "--start", help="启动 Worker Service（后台进程）"),
        stop: bool = typer.Option(False, "--stop", help="停止 Worker Service"),
        backend: str | None = typer.Option(
            None, "--backend", help="后端类型（cuda/ascend）。可省略，默认 cuda"
        ),
        arch: str | None = typer.Option(
            None, "--arch", help="硬件架构（如 a100/ascend910b4）。可省略，默认 a100"
        ),
        devices: str = typer.Option(
            "0", "--devices", help="设备列表，逗号分隔（如 0,1,2,3）"
        ),
        host: str = typer.Option("0.0.0.0", "--host", help="监听地址（默认 0.0.0.0）"),
        port: int = typer.Option(9001, "--port", help="监听端口（默认 9001）"),
    ) -> None:
        """启动/停止 Worker Service（后台进程）"""
        print_logo_once(console)
        services.config.print_report_if_any(console)

        if start == stop:
            console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] 必须且只能指定 --start 或 --stop 之一"
            )
            raise typer.Exit(code=2)

        if port <= 0 or port > 65535:
            console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --port 超出范围: {port}"
            )
            raise typer.Exit(code=2)

        if stop:
            services.worker_service.stop(console, port=port)
            return

        backend_n = (backend or "").strip().lower()
        if backend_n and backend_n not in ["cuda", "ascend"]:
            console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --backend 仅支持 cuda/ascend，当前为: {backend_n}"
            )
            raise typer.Exit(code=2)

        arch_n = (arch or "").strip()
        device_ids = _parse_devices(devices)

        host_n = (host or "").strip()
        if not host_n:
            console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --host 不能为空"
            )
            raise typer.Exit(code=2)

        # 若未指定 backend/arch，优先使用环境变量，否则使用默认值
        backend_show = (
            backend_n
            or (os.environ.get("WORKER_BACKEND") or "").strip().lower()
            or "cuda"
        )
        arch_show = (
            arch_n or (os.environ.get("WORKER_ARCH") or "").strip() or "a100"
        )

        info = Table(title="启动 Worker Service（后台）", show_header=False)
        info.add_column("项", style=DisplayStyle.CYAN, width=18)
        info.add_column("值", style=DisplayStyle.YELLOW)
        info.add_row("backend", backend_show)
        info.add_row("arch", arch_show)
        info.add_row("devices", ",".join(str(x) for x in device_ids))
        info.add_row("host", host_n)
        info.add_row("port", str(port))
        console.print(info)

        services.worker_service.start(
            console,
            backend=backend_show,
            arch=arch_show,
            devices=device_ids,
            host=host_n,
            port=port,
        )

    @app.command("resume")
    def resume_cmd(
        session_id: str = typer.Argument(
            ..., help="已保存的 TUI 会话 ID（Ctrl+C 退出时打印）"
        ),
        speed: float = typer.Option(
            0.0, "--speed", help="回放倍速：0=尽快；1=真实速度；>1 更快"
        ),
        llm_speed: float | None = typer.Option(
            None, "--llm-speed", help="LLM token 流（LLMStream）单独倍速"
        ),
        verify_speed: float | None = typer.Option(
            None, "--verify-speed", help="verifier 阶段单独倍速"
        ),
        max_sleep: float = typer.Option(
            2.0, "--max-sleep", help="单步最大 sleep（秒），避免回放卡住"
        ),
        min_sleep: float = typer.Option(
            0.0, "--min-sleep", help="单步最小 sleep（秒），用于限流"
        ),
    ) -> None:
        """恢复并重新展示一个已保存的 TUI 会话（按录制的 server->cli 消息流回放）。"""
        print_logo_once(console)
        services.config.print_report_if_any(console)

        sid = (session_id or "").strip()
        if not sid:
            console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] session-id 不能为空"
            )
            raise typer.Exit(code=2)

        try:
            from ai_kernel_generator.cli.cli.ui.tui.manager import TextualLayoutManager
            from ai_kernel_generator.cli.cli.ui.tui.session_store import (
                resolve_session_dir_for_resume,
            )
            from ai_kernel_generator.cli.cli.utils.message_recording import ReplaySpeed
            from ai_kernel_generator.cli.cli.utils.message_replay import (
                replay_recorded_messages,
            )
            from ai_kernel_generator.cli.cli.presenter import CLIPresenter

            mgr = TextualLayoutManager()
            presenter = CLIPresenter(console, use_stream=True, layout_manager=mgr)

            session_dir = resolve_session_dir_for_resume(sid)
            record_path = session_dir / "messages.jsonl"
            sp = ReplaySpeed(
                speed=float(speed),
                llm_speed=float(llm_speed) if llm_speed is not None else None,
                verify_speed=float(verify_speed) if verify_speed is not None else None,
                max_sleep=float(max_sleep),
                min_sleep=float(min_sleep),
            )

            async def _replay():
                presenter.print_workflow_start()
                res = await replay_recorded_messages(
                    presenter=presenter,
                    record_path=record_path,
                    speed=sp,
                )
                presenter.print_workflow_complete()
                try:
                    if res:
                        presenter.display_summary(res)
                except Exception as e:
                    log.warning("[Misc] presenter.display_summary failed", exc_info=e)

            mgr.run_replay(session_id=sid, workflow_task=_replay())
        except FileNotFoundError as e:
            console.print(f"[{DisplayStyle.RED}]未找到会话:[/{DisplayStyle.RED}] {e}")
            raise typer.Exit(code=2)
        except Exception as e:
            console.print(f"[{DisplayStyle.RED}]resume 失败:[/{DisplayStyle.RED}] {e}")
            raise typer.Exit(code=2)
