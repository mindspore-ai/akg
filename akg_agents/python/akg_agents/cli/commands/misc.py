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
        status: bool = typer.Option(
            False, "--status",
            help="探测 /api/v1/status（远端模式走本机 tunnel）",
        ),
        reconnect_tunnel: bool = typer.Option(
            False, "--reconnect-tunnel",
            help="只重连本机 ssh -L tunnel，不重启远端 daemon。"
                 "long-running batch 中 tunnel 静默断开时使用。仅 --remote-host 模式有效。",
        ),
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
        remote_host: str | None = typer.Option(
            None, "--remote-host",
            help="按 alias SSH 调起远端 worker + 本机 ssh -L tunnel。"
                 "alias 在 --remote-config 指向的 yaml 里 "
                 "`remote_worker.hosts.<alias>` 定义。",
        ),
        remote_config: str | None = typer.Option(
            None, "--remote-config",
            help="读 remote_worker.hosts 的 yaml 路径。默认 ./config.yaml。",
        ),
    ) -> None:
        """启动/停止 Worker Service（本地直跑或经 --remote-host SSH 远端跑）"""
        akg_console = AKGConsole(console)
        print_logo_once(akg_console)

        actions = sum(1 for x in (start, stop, status, reconnect_tunnel) if x)
        if actions != 1:
            akg_console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] 必须且只能指定 --start / --stop / --status / --reconnect-tunnel 之一"
            )
            raise typer.Exit(code=2)
        if reconnect_tunnel and not remote_host:
            akg_console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --reconnect-tunnel 只在 --remote-host 模式下有意义"
            )
            raise typer.Exit(code=2)

        if port <= 0 or port > 65535:
            akg_console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --port 超出范围: {port}"
            )
            raise typer.Exit(code=2)

        # --- Remote dispatch branch -----------------------------------
        if remote_host:
            from akg_agents.cli.service import worker_remote
            host_cfg = worker_remote.load_remote_host_config(
                remote_host, remote_config,
            )
            if host_cfg is None:
                akg_console.print(
                    f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] "
                    f"未在 {remote_config or './config.yaml'} 找到 "
                    f"remote_worker.hosts.{remote_host}"
                )
                raise typer.Exit(code=2)

            if status:
                rc = worker_remote.dispatch_status(remote_host, host_cfg, port)
                raise typer.Exit(code=rc)
            if reconnect_tunnel:
                rc = worker_remote.dispatch_reconnect_tunnel(
                    remote_host, host_cfg, port,
                )
                raise typer.Exit(code=rc)
            if stop:
                rc = worker_remote.dispatch_stop(remote_host, host_cfg, port)
                raise typer.Exit(code=rc)
            # start: same defaults as local branch
            backend_n = (backend or os.environ.get("WORKER_BACKEND") or
                         "cuda").strip().lower()
            arch_n = (arch or os.environ.get("WORKER_ARCH") or "a100").strip()
            rc = worker_remote.dispatch_start(
                remote_host, host_cfg,
                backend=backend_n, arch=arch_n,
                devices=devices, port=port,
            )
            raise typer.Exit(code=rc)

        # --- Local branch ---------------------------------------------
        if status:
            from akg_agents.cli.service import worker_remote
            rc = worker_remote.dispatch_status(
                "local", {"ssh_alias": "local"}, port,
            )
            raise typer.Exit(code=rc)
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
