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
- 保持对外入口不变：setup.py 仍指向 `akg_agents.cli.cli:app`（注意：cli 包名保持不变）
- 具体命令实现位于 `commands/`
- 配置/worker/server/通知等通用逻辑位于 `service/`
"""

from __future__ import annotations

# 在导入任何其他模块之前设置日志级别环境变量
# 这样当 __init__.py 执行时就能读取到正确的日志级别
import os
# 支持 AKG_AGENTS_LOG_LEVEL 和 AIKG_LOG_LEVEL（兼容）
if "AKG_AGENTS_LOG_LEVEL" not in os.environ and "AIKG_LOG_LEVEL" not in os.environ:
    os.environ["AKG_AGENTS_LOG_LEVEL"] = "4"

import logging
# 立即配置日志级别，确保即使 __init__.py 已经执行过也能生效
from akg_agents.core_v2.config.settings import get_akg_env_var
log_level_str = get_akg_env_var("LOG_LEVEL", "4")
level_map = {
    "0": logging.DEBUG,
    "1": logging.INFO,
    "2": logging.WARNING,
    "3": logging.ERROR,
    "4": logging.CRITICAL,
}
log_level = level_map.get(log_level_str, logging.WARNING)
root_logger = logging.getLogger()
# Root 级别放开到 DEBUG，具体输出由各 handler 控制
root_logger.setLevel(logging.DEBUG)
for handler in root_logger.handlers:
    handler.setLevel(log_level)

# 添加文件日志（固定路径，覆盖写）
from pathlib import Path
log_dir = Path(os.path.expanduser("~/akg_agents_logs"))
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "akg_agents.log"
file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
root_logger.addHandler(file_handler)

import typer
import sys
from rich.console import Console

from .service import CLIAppServices, WorkerService
from .commands import (
    register_common_command,
    register_misc_commands,
    register_op_command,
    register_trace_command,
    register_resume_command,
)


app = typer.Typer(help="AIKG CLI - AKG Agents", invoke_without_command=True)
console = Console(
    file=sys.stdout,
    force_terminal=True,
    legacy_windows=False,
    color_system="auto",
)

services = CLIAppServices(
    worker_service=WorkerService(),
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

    services.worker_service.clear()


# 注册命令
register_op_command(app, console, services)
register_common_command(app, console, services)
register_misc_commands(app, console, services)
register_trace_command(app, console, services)
register_resume_command(app, console, services)


if __name__ == "__main__":
    app()
