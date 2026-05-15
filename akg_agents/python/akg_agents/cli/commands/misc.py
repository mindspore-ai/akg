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

import logging

import typer
from rich.console import Console
from rich.table import Table

from akg_agents.cli.service import CLIAppServices

logger = logging.getLogger(__name__)
from akg_agents.cli.constants import DisplayStyle
from akg_agents.cli.utils.device_parser import parse_devices
from akg_agents.cli.utils.ui_helpers import print_logo_once
from akg_agents.cli.console import AKGConsole


def register_misc_commands(
    app: typer.Typer, console: Console, services: CLIAppServices
) -> None:
    @app.command("list")
    def list_cmd() -> None:
        """列出 akg_cli 当前支持的功能"""
        akg_console = AKGConsole(console)
        print_logo_once(akg_console)

        table = Table(title="MindSpore AKG CLI 功能列表", show_header=True)
        table.add_column("子命令", style=DisplayStyle.CYAN)
        table.add_column("说明", style=DisplayStyle.YELLOW)

        table.add_row(
            "op", "算子生成入口（search/default 交互；或从 KernelBench 文件/目录导入）"
        )
        table.add_row("common", "Common Agent(Demo)")
        table.add_row("graph", "Graph Optimization Agent(TODO)")
        table.add_row(
            "worker", "启动/停止 Worker Service（akg_cli worker --start/--stop）"
        )
        akg_console.print(table)

    @app.command("graph")
    def graph_cmd() -> None:
        """Graph Optimization Agent(TODO)"""
        akg_console = AKGConsole(console)
        print_logo_once(akg_console)
        akg_console.print(
            f"[{DisplayStyle.YELLOW}]graph 功能尚未实现：后续将接入图优化流程。[/{DisplayStyle.YELLOW}]"
        )

    @app.command("worker")
    def worker_cmd(
        start: bool = typer.Option(
            False, "--start", help="启动 Worker Service（后台进程）"
        ),
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
        akg_console = AKGConsole(console)
        print_logo_once(akg_console)

        if start == stop:
            akg_console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] 必须且只能指定 --start 或 --stop 之一"
            )
            raise typer.Exit(code=2)

        if port <= 0 or port > 65535:
            akg_console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --port 超出范围: {port}"
            )
            raise typer.Exit(code=2)

        if stop:
            services.worker_service.stop(akg_console, port=port)
            return

        backend_n = (backend or "").strip().lower()
        if backend_n and backend_n not in ["cuda", "ascend"]:
            akg_console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --backend 仅支持 cuda/ascend，当前为: {backend_n}"
            )
            raise typer.Exit(code=2)

        arch_n = (arch or "").strip()
        try:
            device_ids = parse_devices(devices)
        except ValueError as exc:
            akg_console.print(f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] {exc}")
            raise typer.Exit(code=2) from exc

        host_n = (host or "").strip()
        if not host_n:
            akg_console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --host 不能为空"
            )
            raise typer.Exit(code=2)

        # 若未指定 backend/arch，优先使用环境变量，否则使用默认值
        backend_show = (
            backend_n
            or (os.environ.get("WORKER_BACKEND") or "").strip().lower()
            or "cuda"
        )
        arch_show = arch_n or (os.environ.get("WORKER_ARCH") or "").strip() or "a100"

        info = Table(title="启动 Worker Service（后台）", show_header=False)
        info.add_column("项", style=DisplayStyle.CYAN, width=18)
        info.add_column("值", style=DisplayStyle.YELLOW)
        info.add_row("backend", backend_show)
        info.add_row("arch", arch_show)
        info.add_row("devices", ",".join(str(x) for x in device_ids))
        info.add_row("host", host_n)
        info.add_row("port", str(port))
        akg_console.print(info)

        services.worker_service.start(
            akg_console,
            backend=backend_show,
            arch=arch_show,
            devices=device_ids,
            host=host_n,
            port=port,
        )
