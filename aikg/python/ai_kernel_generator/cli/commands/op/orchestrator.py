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

import asyncio
import os
from typing import List

import logging
import typer

log = logging.getLogger(__name__)

from ai_kernel_generator.cli.service import CLIAppServices
from ai_kernel_generator.cli.constants import DisplayStyle
from .runners import InteractiveOpRunner
from .types import OpCommandArgs, ResolvedTargetConfig
from ai_kernel_generator.cli.runtime import LocalExecutor
from ai_kernel_generator.cli.utils.ui_helpers import print_logo_once
from ai_kernel_generator.cli.service import validate_target_config
from ai_kernel_generator.cli.utils.device_parser import parse_devices
from ai_kernel_generator.cli.console import AKGConsole


class OpCommandOrchestrator:
    def __init__(self, *, console, services: CLIAppServices):
        self._raw_console = console  # 保存原始 console，用于创建 AKGConsole
        self.services = services

    def _build_target_config(self, args: OpCommandArgs) -> ResolvedTargetConfig:
        from ai_kernel_generator.cli.service import (
            normalize_backend,
            normalize_dsl,
            normalize_framework,
        )

        framework = normalize_framework((args.framework or "").strip()) or ""
        backend = normalize_backend((args.backend or "").strip()) or ""
        arch = (args.arch or "").strip()
        dsl = normalize_dsl((args.dsl or "").strip()) or ""

        return ResolvedTargetConfig(
            framework=framework,
            backend=backend,
            arch=arch,
            dsl=dsl,
        )

    def run(self, args: OpCommandArgs) -> None:
        # 创建 AKGConsole 用于所有输出
        akg_console = AKGConsole(self._raw_console, use_stream=args.stream)
        print_logo_once(akg_console)

        output_path = (args.output_path or "").strip()
        if not output_path:
            output_path = os.getcwd()
        output_path = os.path.abspath(os.path.expanduser(output_path))

        if args.worker_url and args.devices:
            akg_console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --devices 与 --worker_url 不能同时使用"
            )
            raise typer.Exit(code=2)

        device_ids: List[int] = []
        if args.devices:
            try:
                device_ids = parse_devices(args.devices)
            except ValueError as exc:
                akg_console.print(
                    f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] {exc}"
                )
                raise typer.Exit(code=2) from exc

        if args.worker_url:
            try:
                self.services.worker_service.workers = (
                    self.services.worker_service.parse_workers(args.worker_url)
                )
            except Exception as e:
                akg_console.print(
                    f"[{DisplayStyle.RED}]worker_url 参数解析失败:[/{DisplayStyle.RED}] {e}"
                )
                raise typer.Exit(code=2)

        executor: LocalExecutor | None = None
        try:
            target = self._build_target_config(args)
            errs = validate_target_config(
                target.framework, target.backend, target.arch, target.dsl
            )
            if errs:
                akg_console.print(
                    f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] 目标配置非法/不兼容：\n- "
                    + "\n- ".join(errs)
                )
                akg_console.print(
                    f"[{DisplayStyle.DIM}]提示：framework/backend/arch/dsl 必须通过命令行显式传入。\n"
                    f"示例：akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda[/{DisplayStyle.DIM}]"
                )
                raise typer.Exit(code=2)

            executor = LocalExecutor.create_for_cli(
                akg_console,
                auto_yes=args.auto_yes,
                use_stream=args.stream,
            )

            # interactive
            runner = InteractiveOpRunner(
                console=akg_console,
                services=self.services,
                cli=executor,
                auto_yes=args.auto_yes,
                target=target,
                intent=args.intent,
                rag=args.rag,
                output_path=output_path,
                device_ids=device_ids,
            )

            runner.run()

        except KeyboardInterrupt:
            akg_console.print(
                f"\n[{DisplayStyle.YELLOW}]用户中断：已尝试终止当前生成流程。[/{DisplayStyle.YELLOW}]"
            )
            try:
                if executor is not None:
                    asyncio.run(
                        executor.cancel_main_agent(reason="cancelled by ctrl+c")
                    )
            except Exception as e:
                log.warning("[Op] cancel_main_agent failed", exc_info=e)
            akg_console.print(
                f"[{DisplayStyle.DIM}]已发送 cancel 请求。日志请查看 log_dir。[/{DisplayStyle.DIM}]"
            )

        except typer.Exit:
            raise

        except Exception as e:
            akg_console.print(f"\n[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] {e}")
            raise
