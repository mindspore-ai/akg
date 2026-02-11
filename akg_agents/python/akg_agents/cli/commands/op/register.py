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

from akg_agents.cli.service import CLIAppServices
from .orchestrator import OpCommandOrchestrator
from .types import OpCommandArgs


def register_op_command(
    app: typer.Typer, console: Console, services: CLIAppServices
) -> None:
    @app.command("op")
    def op_cmd(
        intent: Optional[str] = typer.Option(
            None, "--intent", help="直接提供需求文本（不进入输入提示）"
        ),
        task_file: Optional[str] = typer.Option(
            None, 
            "--task-file",
            "--task_file",
            help="直接读取 task_desc 文件（KernelBench 格式），跳过 OpTaskBuilder 转换"
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
        output_path: Optional[str] = typer.Option(
            None,
            "--output-path",
            help="保存目录根路径（用于 saved_verifications，默认使用启动 akg_cli 的当前目录）",
        ),
        resume: Optional[str] = typer.Option(
            None,
            "--resume",
            help="恢复已有会话。传入 session_id（如 50b111f7-d898-43a3-8945-eed5fd17fdf1）或完整 task_id（如 cli_50b111f7-...）",
        ),
    ) -> None:
        """算子生成入口。"""

        orchestrator = OpCommandOrchestrator(console=console, services=services)
        orchestrator.run(
            OpCommandArgs(
                intent=intent,
                task_file=task_file,
                framework=framework or "",
                backend=backend or "",
                arch=arch or "",
                dsl=dsl or "",
                auto_yes=auto_yes,
                worker_url=worker_url,
                devices=devices,
                stream=stream,
                rag=rag,
                output_path=output_path,
                resume_session_id=resume,
            ),
        )
