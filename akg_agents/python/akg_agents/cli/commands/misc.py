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
            False, "--start", help="启动 Worker Service。幂等：daemon 已活则跳过 spawn，仅按需重建 tunnel"
        ),
        stop: bool = typer.Option(False, "--stop", help="停止 Worker Service"),
        status: bool = typer.Option(
            False, "--status",
            help="纯查询：探 /api/v1/status + /api/v1/health，不做任何 spawn / 重连副作用",
        ),
        backend: str | None = typer.Option(
            None, "--backend",
            help="后端 cuda/ascend。可省 —— 远端模式按 torch_npu 探针推断；本地落 cuda",
        ),
        arch: str | None = typer.Option(
            None, "--arch",
            help="硬件 arch（如 ascend910b4 / a100 / x86_64）。可省 —— 按 backend "
                 "走对应探针（ascend→npu-smi、cuda→nvidia-smi、cpu→"
                 "platform.machine）；探针失败再回落到 config.yaml 的 defaults.arch",
        ),
        devices: str | None = typer.Option(
            None, "--devices",
            help="device 列表，逗号分隔（如 0,1）。可省 —— 远端模式默认 0；本地落 0",
        ),
        port: int | None = typer.Option(
            None, "--port",
            help="监听端口。未传时按 ./config.yaml 的 worker.port 取；yaml 没设则回落 9001",
        ),
        remote_host: str | None = typer.Option(
            None, "--remote-host",
            help="按 alias SSH 调起远端 worker + 本机 ssh -L tunnel。"
                 "alias 在 ./config.yaml 里 `remote_worker.hosts.<alias>` 定义",
        ),
        dsl: str | None = typer.Option(
            None, "--dsl",
            help="目标 DSL（如 triton_ascend / ascendc_catlass）。可省 —— "
                 "默认从 config.yaml 的 defaults.dsl 读；用于判断"
                 "缺 triton 时是 fatal 还是 warn",
        ),
    ) -> None:
        """启动 / 停止 / 探活 Worker Service（本地直跑或经 --remote-host SSH 远端跑）"""
        akg_console = AKGConsole(console)
        actions = sum(1 for x in (start, stop, status) if x)
        if actions != 1:
            akg_console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] 必须且只能指定 --start / --stop / --status 之一"
            )
            raise typer.Exit(code=2)

        # Single source: WorkerConfig 一次读 yaml，自带 9001 默认。
        # Precedence: explicit --port > ./config.yaml worker.port > dataclass。
        from akg_agents.cli.service.worker_config import (
            WorkerConfig, probe_local_arch,
        )
        cfg = WorkerConfig.load(None)
        if port is None:
            port = cfg.port

        if port <= 0 or port > 65535:
            akg_console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --port 超出范围: {port}"
            )
            raise typer.Exit(code=2)

        if start:
            print_logo_once(akg_console)

        # --- Remote dispatch branch -----------------------------------
        if remote_host:
            from akg_agents.cli.service import remote_dispatch
            host_cfg = remote_dispatch.load_remote_host_config(remote_host, None)
            if host_cfg is None:
                akg_console.print(
                    f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] "
                    f"未在 ./config.yaml 找到 remote_worker.hosts.{remote_host}"
                )
                raise typer.Exit(code=2)

            if status:
                backend_p = (
                    backend or os.environ.get("WORKER_BACKEND") or cfg.backend
                ).strip().lower() or None
                dsl_p = (
                    dsl or os.environ.get("WORKER_DSL") or cfg.dsl or ""
                ).strip() or None
                rc = remote_dispatch.dispatch_status(
                    remote_host, host_cfg, port, backend=backend_p, dsl=dsl_p,
                )
                raise typer.Exit(code=rc)
            if stop:
                rc = remote_dispatch.dispatch_stop(remote_host, host_cfg, port)
                raise typer.Exit(code=rc)
            # start: 透传 None，dispatch_start 按远端探针填默认。env override
            # 仍生效（WORKER_BACKEND / WORKER_ARCH / WORKER_DSL），便于在
            # CI / docker 里不靠 npu-smi 强行指定。
            backend_p = (backend or os.environ.get("WORKER_BACKEND") or "").strip().lower() or None
            arch_p = (arch or os.environ.get("WORKER_ARCH") or "").strip() or None
            dsl_p = (dsl or os.environ.get("WORKER_DSL") or "").strip() or None
            rc = remote_dispatch.dispatch_start(
                remote_host, host_cfg,
                backend=backend_p, arch=arch_p,
                devices=devices, port=port, dsl=dsl_p,
            )
            raise typer.Exit(code=rc)

        # --- Local branch ---------------------------------------------
        if status:
            from akg_agents.cli.service import remote_dispatch
            backend_p = (
                backend or os.environ.get("WORKER_BACKEND") or cfg.backend
            ).strip().lower() or None
            dsl_p = (
                dsl or os.environ.get("WORKER_DSL") or cfg.dsl or ""
            ).strip() or None
            rc = remote_dispatch.dispatch_status(
                "local", {"ssh_alias": "local"}, port,
                backend=backend_p, dsl=dsl_p,
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
        # Precedence: CLI > env > config (defaults.devices) > dataclass。
        # cfg.devices 自带 "0" 兜底，所以这里不再写 or "0"。
        try:
            device_ids = parse_devices(devices or cfg.devices)
        except ValueError as exc:
            akg_console.print(f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] {exc}")
            raise typer.Exit(code=2) from exc

        backend_show = (
            backend_n
            or (os.environ.get("WORKER_BACKEND") or "").strip().lower()
            or cfg.backend
        )
        # Precedence: CLI > env > backend-specific local probe > cfg.arch
        # (yaml or dataclass default). Without the probe step, an Ascend
        # host with no --arch flag falls through to cfg.arch="a100" and the
        # daemon ValueError-loops on every eval. probe returns None on any
        # subprocess failure, then cfg.arch remains the fallback.
        arch_show = (
            arch_n
            or (os.environ.get("WORKER_ARCH") or "").strip()
            or probe_local_arch(backend_show, device_ids[0])
            or cfg.arch
        )

        # SSH 递归调用（remote_dispatch 起远端 daemon）下 AKG_CLI_QUIET=1，
        # 本机已经印过这些参数，远端再印一遍是噪音；跳过 Table。
        if os.environ.get("AKG_CLI_QUIET") != "1":
            info = Table(title="启动 Worker Service（后台）", show_header=False)
            info.add_column("项", style=DisplayStyle.CYAN, width=18)
            info.add_column("值", style=DisplayStyle.YELLOW)
            info.add_row("backend", backend_show)
            info.add_row("arch", arch_show)
            info.add_row("devices", ",".join(str(x) for x in device_ids))
            info.add_row("port", str(port))
            akg_console.print(info)

        # Bind address: honor WORKER_HOST env var so the remote-spawn
        # path (which exports WORKER_HOST=127.0.0.1) can pin loopback —
        # see _build_remote_start_cmd. Local invocation without env
        # defaults to 0.0.0.0 (operator on the NPU host might want LAN
        # access).
        host_show = (os.environ.get("WORKER_HOST") or "0.0.0.0").strip()
        # Propagate WorkerConfig.timing 到 worker_service.start —— 跟远端
        # SSH 路径对齐（remote_dispatch 也是 export 同名 env），让本机
        # 直跑也能从 config.yaml worker.* 拿到 timing，而不是走代码默认。
        for key, value in cfg.timing.as_env().items():
            os.environ.setdefault(key, value)
        from akg_agents.core.worker.eval_config import eval_defaults
        for key, value in eval_defaults().as_env().items():
            os.environ.setdefault(key, value)
        services.worker_service.start(
            akg_console,
            backend=backend_show,
            arch=arch_show,
            devices=device_ids,
            host=host_show,
            port=port,
        )
