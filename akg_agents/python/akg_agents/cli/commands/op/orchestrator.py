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

from akg_agents.cli.service import CLIAppServices
from akg_agents.cli.constants import DisplayStyle
from .runners import InteractiveOpRunner
from .types import OpCommandArgs, ResolvedTargetConfig
from akg_agents.cli.runtime import LocalExecutor
from akg_agents.cli.utils.ui_helpers import print_logo_once
from akg_agents.cli.service import validate_target_config
from akg_agents.cli.utils.device_parser import parse_devices
from akg_agents.cli.console import AKGConsole


class OpCommandOrchestrator:
    def __init__(self, *, console, services: CLIAppServices):
        self._raw_console = console  # 保存原始 console，用于创建 AKGConsole
        self.services = services

    def _build_target_config(self, args: OpCommandArgs) -> ResolvedTargetConfig:
        from akg_agents.cli.service import (
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

    @staticmethod
    def _load_target_from_trace(session_dir, console) -> "ResolvedTargetConfig | None":
        """从已有 session 的 trace 中恢复 target 配置 (framework/backend/arch/dsl)
        
        查找策略：
        1. 优先从 root 节点的 task_info 中获取（新版本保存了 dsl/backend/arch）
        2. 回退从 trace.json 的 action arguments 中查找
        """
        import json
        from pathlib import Path
        
        # 策略 1: 从 root state.json 的 task_info 获取
        root_state_file = session_dir / "nodes" / "root" / "state.json"
        if root_state_file.exists():
            try:
                root_state = json.loads(root_state_file.read_text())
                ti = root_state.get("task_info", {})
                if ti.get("dsl") and ti.get("backend"):
                    return ResolvedTargetConfig(
                        framework=ti.get("framework", "torch"),
                        backend=ti["backend"],
                        arch=ti.get("arch", ""),
                        dsl=ti["dsl"],
                    )
            except Exception:
                pass
        
        # 策略 2: 从 trace.json 的 action arguments 中查找
        trace_file = session_dir / "trace.json"
        if trace_file.exists():
            try:
                trace_data = json.loads(trace_file.read_text())
                tree = trace_data.get("tree", {})
                # 按节点编号顺序遍历，找到第一个包含 framework/dsl 的 action
                for node_id in sorted(tree.keys()):
                    node = tree[node_id]
                    action = node.get("action") or {}
                    args = action.get("arguments") or {}
                    if args.get("dsl") and args.get("framework"):
                        return ResolvedTargetConfig(
                            framework=args["framework"],
                            backend=args.get("backend", ""),
                            arch=args.get("arch", ""),
                            dsl=args["dsl"],
                        )
            except Exception:
                pass
        
        return None

    def run(self, args: OpCommandArgs) -> None:
        # 创建 AKGConsole 用于所有输出
        akg_console = AKGConsole(self._raw_console, use_stream=args.stream)
        print_logo_once(akg_console)

        output_path = (args.output_path or "").strip()
        if not output_path:
            output_path = os.getcwd()
        output_path = os.path.abspath(os.path.expanduser(output_path))

        # 处理 task_file 参数：读取文件内容作为 task_desc
        task_file_content: str | None = None
        if args.task_file:
            task_file_path = os.path.abspath(os.path.expanduser(args.task_file))
            if not os.path.isfile(task_file_path):
                akg_console.print(
                    f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] task_file 不存在: {task_file_path}"
                )
                raise typer.Exit(code=2)
            try:
                with open(task_file_path, "r", encoding="utf-8") as f:
                    task_file_content = f.read()
                akg_console.print(
                    f"[{DisplayStyle.DIM}]已读取 task_file: {task_file_path} ({len(task_file_content)} 字符)[/{DisplayStyle.DIM}]"
                )
            except Exception as e:
                akg_console.print(
                    f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] 读取 task_file 失败: {e}"
                )
                raise typer.Exit(code=2)

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
            # 处理 --resume 参数：支持 session_id 或完整 task_id
            resume_sid = args.resume_session_id
            resumed_target: ResolvedTargetConfig | None = None
            if resume_sid:
                # 如果传入的是 cli_xxx 格式，提取 session_id 部分
                if resume_sid.startswith("cli_"):
                    resume_sid = resume_sid[4:]
                # 验证 session 目录是否存在
                from pathlib import Path
                session_dir = Path.home() / ".akg" / "conversations" / f"cli_{resume_sid}"
                if not session_dir.exists():
                    akg_console.print(
                        f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] 会话不存在: cli_{resume_sid}\n"
                        f"[{DisplayStyle.DIM}]路径: {session_dir}[/{DisplayStyle.DIM}]"
                    )
                    raise typer.Exit(code=2)
                
                # 从 trace 中恢复 target 配置（framework/backend/arch/dsl）
                resumed_target = self._load_target_from_trace(session_dir, akg_console)
                if resumed_target:
                    akg_console.print(
                        f"[{DisplayStyle.DIM}]恢复会话: cli_{resume_sid} "
                        f"(framework={resumed_target.framework}, backend={resumed_target.backend}, "
                        f"arch={resumed_target.arch}, dsl={resumed_target.dsl})[/{DisplayStyle.DIM}]"
                    )
                else:
                    akg_console.print(
                        f"[{DisplayStyle.DIM}]恢复会话: cli_{resume_sid} (未找到 target 配置，使用命令行参数)[/{DisplayStyle.DIM}]"
                    )

            # 确定 target：resume 恢复的优先，命令行参数可覆盖
            if resumed_target and not (args.framework or args.backend or args.arch or args.dsl):
                # 完全使用恢复的 target
                target = resumed_target
            else:
                # 使用命令行参数（新建或覆盖）
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
                session_id=resume_sid,
            )

            # 显示 session_id，方便用户后续 --resume
            akg_console.print(
                f"[{DisplayStyle.DIM}]Session ID: {executor.session_id} "
                f"(恢复命令: akg_cli op --resume {executor.session_id} ...)[/{DisplayStyle.DIM}]"
            )

            # interactive
            runner = InteractiveOpRunner(
                console=akg_console,
                services=self.services,
                cli=executor,
                auto_yes=args.auto_yes,
                target=target,
                intent=args.intent,
                task_file_content=task_file_content,
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
            if executor is not None:
                akg_console.print(
                    f"[{DisplayStyle.DIM}]Session ID: {executor.session_id}[/{DisplayStyle.DIM}]"
                )
                akg_console.print(
                    f"[{DisplayStyle.DIM}]恢复命令: akg_cli op --resume {executor.session_id}[/{DisplayStyle.DIM}]"
                )
            akg_console.print(
                f"[{DisplayStyle.DIM}]已发送 cancel 请求。日志请查看 log_dir。[/{DisplayStyle.DIM}]"
            )

        except typer.Exit:
            raise

        except Exception as e:
            akg_console.print(f"\n[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] {e}")
            raise
