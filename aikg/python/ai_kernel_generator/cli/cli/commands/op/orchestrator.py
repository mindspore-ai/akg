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
import socket
import urllib.parse
from typing import List

import typer
from textual import log

from ai_kernel_generator.cli.cli.service import CLIAppServices, ResolvedValue
from ai_kernel_generator.cli.cli.constants import (
    Defaults,
    DisplayStyle,
    EnvVar,
    UISymbol,
)
from .runners import InteractiveOpRunner
from .types import OpCommandArgs, ResolvedRuntimeOptions, ResolvedTargetConfig
from ai_kernel_generator.cli.client import CliClient
from ai_kernel_generator.cli.cli.utils.ui_helpers import print_logo_once
from ai_kernel_generator.cli.cli.service import validate_target_config
from ai_kernel_generator.cli.cli.utils.device_parser import parse_devices


class OpCommandOrchestrator:
    def __init__(self, *, console, services: CLIAppServices):
        self.console = console
        self.services = services

    @staticmethod
    def _port_available(port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
        finally:
            try:
                sock.close()
            except Exception:
                pass
        return True

    def _select_server_port(self) -> int:
        """选择 server 端口。"""
        for port in range(8000, 9001):
            if self._port_available(port):
                return port
        raise RuntimeError("8000-9000 范围内没有可用端口")

    @staticmethod
    def _is_local_server_url(server_url: str) -> bool:
        if not server_url:
            return False
        url = server_url.strip()
        if "://" not in url:
            url = f"http://{url}"
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").strip().lower()
        return host in ["localhost", "127.0.0.1", "::1"]

    def _resolve_runtime_options(
        self, ctx: typer.Context, args: OpCommandArgs
    ) -> ResolvedRuntimeOptions:
        cfg = self.services.config

        r_stream = cfg.resolve_from_sources(
            ctx,
            param_name="stream",
            cli_value=args.stream,
            config_keys=["stream", "use_stream"],
            env_keys=[EnvVar.WORKFLOW_USE_STREAM],
            default_value=True,
            cast=lambda v: bool(v)
            if isinstance(v, bool)
            else str(v).strip().lower() in ["1", "true", "yes", "on"],
        )
        stream = bool(r_stream.value)

        # server_url：允许 config/env 覆盖；为空则后续自动拉起本地 server
        server_url = args.server_url or self.services.get_server_url()

        return ResolvedRuntimeOptions(
            stream=stream,
            server_url=server_url,
        )

    def _resolve_target_config(
        self, args: OpCommandArgs
    ) -> tuple[ResolvedTargetConfig, List[ResolvedValue]]:
        resolved_items: List[ResolvedValue] = []

        from ai_kernel_generator.cli.cli.service import (
            normalize_backend,
            normalize_dsl,
            normalize_framework,
        )

        resolved_framework = normalize_framework((args.framework or "").strip())
        if not resolved_framework:
            resolved_framework = ""
            resolved_items.append(
                ResolvedValue("framework", "<缺失>", "cli(missing)", required=True)
            )
        else:
            resolved_items.append(
                ResolvedValue(
                    "framework",
                    resolved_framework,
                    "cli",
                    required=True,
                )
            )

        resolved_backend = normalize_backend((args.backend or "").strip())
        if not resolved_backend:
            resolved_backend = ""
            resolved_items.append(
                ResolvedValue("backend", "<缺失>", "cli(missing)", required=True)
            )
        else:
            resolved_items.append(
                ResolvedValue(
                    "backend",
                    resolved_backend,
                    "cli",
                    required=True,
                )
            )

        resolved_arch = (args.arch or "").strip()
        if not resolved_arch:
            resolved_arch = ""
            resolved_items.append(
                ResolvedValue("arch", "<缺失>", "cli(missing)", required=True)
            )
        else:
            resolved_items.append(
                ResolvedValue(
                    "arch",
                    resolved_arch,
                    "cli",
                    required=True,
                )
            )

        resolved_dsl = normalize_dsl((args.dsl or "").strip())
        if not resolved_dsl:
            resolved_dsl = ""
            resolved_items.append(
                ResolvedValue("dsl", "<缺失>", "cli(missing)", required=True)
            )
        else:
            resolved_items.append(
                ResolvedValue("dsl", resolved_dsl, "cli", required=True)
            )

        return (
            ResolvedTargetConfig(
                framework=resolved_framework,
                backend=resolved_backend,
                arch=resolved_arch,
                dsl=resolved_dsl,
            ),
            resolved_items,
        )

    def run(self, ctx: typer.Context, args: OpCommandArgs) -> None:
        print_logo_once(self.console)
        self.services.config.print_report_if_any(self.console)

        runtime = self._resolve_runtime_options(ctx, args)

        if args.worker_url and args.devices:
            self.console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] --devices 与 --worker_url 不能同时使用"
            )
            raise typer.Exit(code=2)

        device_ids: List[int] = []
        if args.devices:
            try:
                device_ids = parse_devices(args.devices)
            except ValueError as exc:
                self.console.print(
                    f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] {exc}"
                )
                raise typer.Exit(code=2) from exc

        if args.worker_url:
            try:
                self.services.workers.workers = self.services.workers.parse_workers(
                    args.worker_url
                )
            except Exception as e:
                self.console.print(
                    f"[{DisplayStyle.RED}]worker_url 参数解析失败:[/{DisplayStyle.RED}] {e}"
                )
                raise typer.Exit(code=2)

        if args.devices and runtime.server_url and not self._is_local_server_url(
            runtime.server_url
        ):
            self.console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] 使用 --devices 时仅支持本地 server（当前为: {runtime.server_url}）。"
            )
            raise typer.Exit(code=2)

        server_url = runtime.server_url
        auto_server_started = False

        cli: CliClient | None = None
        try:
            target, resolved_items = self._resolve_target_config(args)
            errs = validate_target_config(
                target.framework, target.backend, target.arch, target.dsl
            )
            if errs:
                self.console.print(
                    f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] 目标配置非法/不兼容：\n- "
                    + "\n- ".join(errs)
                )
                self.console.print(
                    f"[{DisplayStyle.DIM}]提示：framework/backend/arch/dsl 必须通过命令行显式传入（不会从配置文件读取）。\n"
                    f"示例：akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda[/{DisplayStyle.DIM}]"
                )
                raise typer.Exit(code=2)

            if not server_url:
                try:
                    port = self._select_server_port()
                except Exception as e:
                    self.console.print(
                        f"[{DisplayStyle.RED}]{UISymbol.ERROR} 无法选择可用端口: {e}[/{DisplayStyle.RED}]"
                    )
                    raise typer.Exit(code=1)
                try:
                    proc, _ = self.services.server.start(self.console, port=port)
                    auto_server_started = proc is not None
                except Exception as e:
                    log.error("[Orchestrator] start server failed", exc_info=e)
                    self.console.print(
                        f"[{DisplayStyle.RED}]{UISymbol.ERROR} 启动 server 失败: {e}[/{DisplayStyle.RED}]"
                    )
                    raise typer.Exit(code=1)
                server_url = f"http://localhost:{port}"
                self.console.print(
                    f"[{DisplayStyle.GREEN}]{UISymbol.DONE} 已自动启动本地 server: {server_url}[/{DisplayStyle.GREEN}]\n"
                )

            if args.devices:
                self.services.workers.register_local_devices(
                    self.console,
                    server_url,
                    backend=target.backend,
                    arch=target.arch,
                    devices=device_ids,
                )
            else:
                # 关键点：server 与 worker 是两回事；op 必须显式提供 worker_url。
                if not self.services.workers.workers:
                    self.console.print(
                        f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] 缺少 worker_url 或 devices。请先启动 worker（例如: akg_cli worker --start --devices 0 --port 9001），"
                        f"并在 op 命令中通过 --worker_url 指定（例如: --worker_url localhost:9001），或直接使用 --devices 注册本地 worker。"
                    )
                    raise typer.Exit(code=2)
                # 如果指定了 worker_url，则在提交任务前注册到 server
                self.services.workers.register_if_any(self.console, server_url)

            cli = CliClient.create_for_cli(
                self.console,
                auto_yes=args.auto_yes,
                server_url=server_url,
                timeout=600.0,
                use_stream=runtime.stream,
                default_user_input=Defaults.DEFAULT_USER_INPUT,
            )

            if self.services.config.config_path:
                if args.server_url:
                    server_source = "cli"
                elif runtime.server_url:
                    server_source = "config/env"
                else:
                    server_source = "auto"
                self.services.config.print_resolution_table(
                    self.console,
                    [
                        ResolvedValue(
                            "server_url",
                            server_url,
                            server_source,
                            required=True,
                        ),
                        ResolvedValue(
                            "worker_url",
                            self.services.workers.workers or [],
                            "cli" if self.services.workers.workers else "-",
                            required=False,
                        ),
                        *resolved_items,
                    ],
                    title="配置解析（op）",
                )

            # interactive
            runner = InteractiveOpRunner(
                console=self.console,
                services=self.services,
                cli=cli,
                server_url=server_url,
                auto_yes=args.auto_yes,
                target=target,
                intent=args.intent,
                rag=args.rag,
            )
            runner.run_textual()

        except KeyboardInterrupt:
            self.console.print(
                f"\n[{DisplayStyle.YELLOW}]用户中断：已断开连接并尝试终止当前生成流程。[/{DisplayStyle.YELLOW}]"
            )
            try:
                if cli is not None:
                    asyncio.run(cli.cancel_current_job(reason="cancelled by ctrl+c"))
                    job_id = getattr(cli, "current_job_id", None)
                    if isinstance(job_id, str) and job_id:
                        self.services.jobs.try_print_job_summary(
                            self.console, server_url, job_id
                        )
            except Exception as e:
                log.warning("[Op] cancel_current_job or job_summary failed", exc_info=e)
            self.console.print(
                f"[{DisplayStyle.DIM}]已发送 cancel 请求（若 server 支持）。日志请查看 server 端输出与 log_dir。[/{DisplayStyle.DIM}]"
            )

        except typer.Exit:
            raise

        except Exception as e:
            self.console.print(f"\n[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] {e}")
            try:
                if cli is not None:
                    job_id = getattr(cli, "current_job_id", None)
                    if isinstance(job_id, str) and job_id:
                        self.services.jobs.try_print_job_summary(
                            self.console, server_url, job_id
                        )
            except Exception as e2:
                log.warning("[Op] try_print_job_summary failed", exc_info=e2)
            raise

        finally:
            if auto_server_started:
                self.services.server.stop(self.console)
