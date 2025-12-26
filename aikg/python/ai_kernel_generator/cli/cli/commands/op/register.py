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

from typing import Optional

import typer
from rich.console import Console

from ai_kernel_generator.cli.cli.service import CLIAppServices
from .orchestrator import OpCommandOrchestrator
from .types import OpCommandArgs


def register_op_command(
    app: typer.Typer, console: Console, services: CLIAppServices
) -> None:
    @app.command("op")
    def op_cmd(
        ctx: typer.Context,
        intent: Optional[str] = typer.Option(
            None, "--intent", help="直接提供需求文本（不进入输入提示）"
        ),
        framework: Optional[str] = typer.Option(
            None,
            "--framework",
            help="必填（仅命令行）：框架（如 torch/mindspore）。",
        ),
        backend: Optional[str] = typer.Option(
            None,
            "--backend",
            help="必填（仅命令行）：后端（如 cuda/ascend/cpu）。",
        ),
        arch: Optional[str] = typer.Option(
            None,
            "--arch",
            help="必填（仅命令行）：架构（如 a100/ascend910b4/x86_64 等）。",
        ),
        dsl: Optional[str] = typer.Option(
            None,
            "--dsl",
            help="必填（仅命令行）：DSL（如 triton_cuda/triton_ascend/cpp 等）。",
        ),
        # 常用参数
        auto_yes: bool = typer.Option(
            False, "--yes", "-y", help="自动确认所有提示，使用默认值"
        ),
        server_url: Optional[str] = typer.Option(
            None,
            "--server-url",
            "-s",
            help="远程服务器地址（例如: http://192.168.1.100:9002）",
        ),
        worker_url: Optional[str] = typer.Option(
            None,
            "--worker_url",
            "--worker-url",
            help="必填：多个 Worker Service 地址，逗号分隔（例如: localhost:9001,1.2.3.4:9002）。请先执行 akg_cli worker start。",
        ),
        devices: Optional[str] = typer.Option(
            None,
            "--devices",
            help="本地设备列表，逗号分隔（如 0,1,2,3）。与 --worker_url 互斥。",
        ),
        stream: bool = typer.Option(
            True, "--stream/--no-stream", help="启用/关闭 LLM 流式输出（默认开启）"
        ),
        rag: bool = typer.Option(
            False, "--rag/--no-rag", help="启用/关闭 RAG（默认关闭）"
        ),
    ) -> None:
        """算子生成入口。"""

        orchestrator = OpCommandOrchestrator(console=console, services=services)
        orchestrator.run(
            ctx,
            OpCommandArgs(
                intent=intent,
                framework=framework or "",
                backend=backend or "",
                arch=arch or "",
                dsl=dsl or "",
                auto_yes=auto_yes,
                server_url=server_url,
                worker_url=worker_url,
                devices=devices,
                stream=stream,
                rag=rag,
            ),
        )
