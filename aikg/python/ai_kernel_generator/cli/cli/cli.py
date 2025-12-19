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

"""AIKG CLI - 入口（Typer app 组装层）。

说明：
- 保持对外入口不变：setup.py 仍指向 `ai_kernel_generator.cli.cli.cli:app`
- 具体命令实现位于 `commands/`
- 配置/worker/server/通知等通用逻辑位于 `service/`
"""

from __future__ import annotations

import typer
from rich.console import Console

from ai_kernel_generator.cli.cli.constants import DisplayStyle
from .service import (
    CLIAppServices,
    CLIConfigManager,
    JobInspector,
    KernelBenchStaticChecker,
    WorkerRegistry,
    WorkflowServerProcessManager,
    WorkerServiceProcessManager,
)
from .commands import register_misc_commands, register_op_command


app = typer.Typer(help="AIKG CLI - AI Kernel Generator", invoke_without_command=True)
console = Console()

services = CLIAppServices(
    config=CLIConfigManager(),
    workers=WorkerRegistry(),
    server=WorkflowServerProcessManager(),
    worker_service=WorkerServiceProcessManager(),
    kernelbench=KernelBenchStaticChecker(),
    jobs=JobInspector(),
)


@app.callback()
def _global_options(
    ctx: typer.Context,
) -> None:
    """全局参数入口（作用于所有子命令）。"""

    ctx.ensure_object(dict)
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(code=0)

    services.config.reset()
    services.workers.clear()

    # worker_url 仅作为 op 子命令参数处理


# 注册命令
register_op_command(app, console, services)
register_misc_commands(app, console, services)


if __name__ == "__main__":
    app()
