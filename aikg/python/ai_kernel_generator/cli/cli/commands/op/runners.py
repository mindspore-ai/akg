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
from rich.text import Text
from textual import log

from ai_kernel_generator.cli.cli.service import (
    CLIAppServices,
    validate_target_config,
)
from ai_kernel_generator.cli.cli.constants import DisplayStyle
from ai_kernel_generator.cli.client import CliClient
from ai_kernel_generator.cli.cli.ui.types import SyntaxBlockMainContent
from ai_kernel_generator.cli.cli.utils.i18n import t
from .types import ResolvedTargetConfig


class InteractiveOpRunner:
    def __init__(
        self,
        *,
        console,
        services: CLIAppServices,
        cli: CliClient,
        server_url: str,
        auto_yes: bool,
        target: ResolvedTargetConfig,
        intent: Optional[str],
    ):
        self.console = console
        self.services = services
        self.cli = cli
        self.server_url = server_url
        self.auto_yes = auto_yes
        self.target = target
        self.intent = intent

    async def _ask_yes_no(
        self,
        *,
        textual_manager,
        prompt: str,
        default_yes: bool = True,
        invalid_text_as: str | None = None,
    ) -> tuple[bool, str]:
        """询问 y/n；可选将“非 y/n 的文本输入”转成 (yes/no, text)。"""
        effective_prompt = prompt
        while True:
            ans = await textual_manager.await_user_input(
                f"{effective_prompt} (y/n)"
            )
            if not ans:
                effective_prompt = f"{prompt}（请输入 y 或 n）"
                continue

            ans_norm = ans.strip()
            ans_l = ans_norm.lower()
            if ans_l in ["y", "yes", "1", "true"]:
                return True, ""
            if ans_l in ["n", "no", "0", "false"]:
                return False, ""

            if invalid_text_as == "no":
                return False, ans_norm
            if invalid_text_as == "yes":
                return True, ans_norm

            # 严格模式：不默认为默认值，避免误触发；提示后重试
            effective_prompt = f"{prompt}（请输入 y 或 n）"

    async def _execute_main_agent_task_init(
        self,
        *,
        textual_manager,
        resolved: dict,
        user_input: str,
        first: bool,
    ) -> dict:
        textual_manager.set_input_enabled(False)
        try:
            self.cli.presenter.set_task_context(
                framework=resolved.get("framework", ""),
                backend=resolved.get("backend", ""),
                arch=resolved.get("arch", ""),
                dsl=resolved.get("dsl", ""),
                workflow_name=getattr(self.cli, "workflow_name", "") or "",
            )
        except Exception as e:
            log.warning("[OpRunner] set_task_context failed; continue", exc_info=e)

        self.cli.presenter.print_user_input(user_input)
        self.cli.presenter.print_workflow_start()
        res = await self.cli.execute_main_agent(
            action="start" if first else "revise",
            user_input=user_input,
            framework=resolved.get("framework", ""),
            backend=resolved.get("backend", ""),
            arch=resolved.get("arch", ""),
            dsl=resolved.get("dsl", ""),
            sub_workflow=None,
        )
        self.cli.presenter.print_workflow_complete()
        return res

    async def _maybe_start_generation(
        self,
        *,
        textual_manager,
        resolved: dict,
    ) -> dict:
        textual_manager.set_input_enabled(False)
        try:
            self.cli.presenter.set_task_context(
                framework=resolved.get("framework", ""),
                backend=resolved.get("backend", ""),
                arch=resolved.get("arch", ""),
                dsl=resolved.get("dsl", ""),
                workflow_name=getattr(self.cli, "workflow_name", "") or "",
            )
        except Exception as e:
            log.warning("[OpRunner] set_task_context failed; continue", exc_info=e)

        self.cli.presenter.print_user_input(t("ops.msg.start_generate_with_desc"))
        self.cli.presenter.print_workflow_start()
        res = await self.cli.execute_main_agent(
            action="confirm",
            user_input="",
            framework=resolved.get("framework", ""),
            backend=resolved.get("backend", ""),
            arch=resolved.get("arch", ""),
            dsl=resolved.get("dsl", ""),
            sub_workflow=None,
        )
        self.cli.presenter.print_workflow_complete()
        self.cli.presenter.display_summary(res)
        return res

    def _extract_task_init_payload(self, taskinit_res: dict) -> dict:
        meta = taskinit_res.get("metadata") or {}
        payload = meta.get("task_init") if isinstance(meta, dict) else {}
        return payload if isinstance(payload, dict) else {}

    def run_textual(self) -> None:
        textual_manager = self.cli.presenter.layout_manager
        if not textual_manager:
            self.console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] Textual 管理器未初始化"
            )
            raise typer.Exit(code=2)

        async def _workflow() -> None:
            resolved = {
                "framework": self.target.framework,
                "backend": self.target.backend,
                "arch": self.target.arch,
                "dsl": self.target.dsl,
            }

            # UI 启动后才能拿到 session_dir：在这里补上录制路径，生成 messages.jsonl
            try:
                sd = textual_manager.get_session_dir()
                if sd and getattr(self.cli, "_recorder", None) is None:
                    self.cli.set_record_path(str(sd / "messages.jsonl"))
            except Exception as e:
                log.debug("[OpRunner] set_record_path skipped", exc_info=e)

            # 验证基本配置
            errs = validate_target_config(
                resolved.get("framework", ""),
                resolved.get("backend", ""),
                resolved.get("arch", ""),
                resolved.get("dsl", ""),
            )
            if errs:
                self.cli.presenter._handlers._emit_main_global(
                    Text.from_markup(
                        f"[{DisplayStyle.BOLD_RED}]{t('ops.error.config_invalid')}[/{DisplayStyle.BOLD_RED}]\n- "
                        + "\n- ".join(errs)
                    )
                )
                raise typer.Exit(code=2)

            # 检查 worker
            backend = resolved.get("backend", "")
            arch = resolved.get("arch", "")
            if not self.services.workers.server_has_worker(
                self.server_url, backend, arch
            ):
                # Textual 场景下，server 可能是新启动的，之前的注册状态不存在。
                # 为了"一劳永逸"减少误伤：只要用户已通过 --worker_url 提供了 worker 列表，就先自动重试注册一次。
                if self.services.workers.workers:
                    self.services.workers.register_if_any(self.console, self.server_url)
                    if not self.services.workers.server_has_worker(
                        self.server_url, backend, arch
                    ):
                        self.cli.presenter._handlers._emit_main_global(
                            Text.from_markup(
                                f"[{DisplayStyle.BOLD_RED}]{t('ops.error.worker_still_missing', backend=backend, arch=arch)}[/{DisplayStyle.BOLD_RED}]"
                            )
                        )
                        raise typer.Exit(code=2)
                else:
                    self.cli.presenter._handlers._emit_main_global(
                        Text.from_markup(
                            f"[{DisplayStyle.BOLD_RED}]{t('ops.error.missing_worker')}[/{DisplayStyle.BOLD_RED}] "
                            f"backend={backend}, arch={arch}"
                            f"{t('ops.error.missing_worker_hint')}"
                        )
                    )
                    raise typer.Exit(code=2)

            base_intent = (self.intent or "").strip()
            while True:
                if not base_intent:
                    base_intent = await textual_manager.await_user_input(
                        t("ops.prompt.op_intent")
                    )
                if not base_intent:
                    self.cli.presenter._handlers._emit_main_global(
                        Text.from_markup(
                            f"[{DisplayStyle.YELLOW}]{t('ops.msg.intent_empty_exit')}[/{DisplayStyle.YELLOW}]"
                        )
                    )
                    return

                conversation_round = 0
                pending_input = base_intent
                main_agent_first = True
                while True:
                    conversation_round += 1
                    current_input = pending_input
                    taskinit_res = await self._execute_main_agent_task_init(
                        textual_manager=textual_manager,
                        resolved=resolved,
                        user_input=current_input,
                        first=main_agent_first,
                    )
                    main_agent_first = False

                    status = taskinit_res.get("task_init_status", "")
                    payload = self._extract_task_init_payload(taskinit_res)

                    if status == "ready":
                        generated = taskinit_res.get("generated_task_desc", "") or ""
                        op_name = taskinit_res.get("op_name", "") or ""
                        if not generated.strip():
                            # 与 plain 模式保持一致：ready 但没有产出 task_desc 属于异常
                            self.cli.presenter._handlers._emit_main_global(
                                Text.from_markup(
                                    f"[{DisplayStyle.BOLD_RED}]{t('ops.error.taskinit_ready_but_missing_desc')}[/{DisplayStyle.BOLD_RED}]"
                                )
                            )
                            return
                        self.cli.presenter._handlers._emit_main_global(
                            Text.from_markup(
                                f"[bold]{t('ops.msg.taskinit_ready_code')}[/bold]"
                            )
                        )
                        self.cli.presenter._handlers._emit_main_global(
                            SyntaxBlockMainContent(
                                code=str(generated or ""),
                                lexer_name="python",
                                add_end_separator=True,
                            )
                        )

                        if self.auto_yes:
                            ok_to_start, inline_extra = True, ""
                        else:
                            ok_to_start, inline_extra = await self._ask_yes_no(
                                textual_manager=textual_manager,
                                prompt=t("ops.prompt.can_start_generate"),
                                default_yes=True,
                                invalid_text_as="no",
                            )
                        if not ok_to_start:
                            extra = (inline_extra or "").strip()
                            if not extra:
                                extra = await textual_manager.await_user_input(
                                    t("ops.prompt.modify_intent")
                                )
                            if not extra:
                                self.cli.presenter._handlers._emit_main_global(
                                    Text.from_markup(
                                        f"[{DisplayStyle.YELLOW}]{t('ops.msg.extra_empty_exit')}[/{DisplayStyle.YELLOW}]"
                                    )
                                )
                                return
                            pending_input = extra
                            continue

                        gen_res = await self._maybe_start_generation(
                            textual_manager=textual_manager, resolved=resolved
                        )

                        verify = gen_res.get("verification_result")
                        op_name = gen_res.get("op_name", "")
                        total_time = gen_res.get("total_time", 0.0)
                        error = gen_res.get("error", "")
                        meta2 = gen_res.get("metadata") or {}
                        log_dir = (
                            meta2.get("log_dir") if isinstance(meta2, dict) else None
                        )
                        kernel_path = (
                            meta2.get("kernel_code_path")
                            if isinstance(meta2, dict)
                            else None
                        )
                        task_path = (
                            meta2.get("task_desc_path")
                            if isinstance(meta2, dict)
                            else None
                        )

                        self.cli.presenter._handlers._emit_main_global(
                            Text.from_markup(
                                f"[{DisplayStyle.BOLD_GREEN}]{t('ops.msg.generation_done', op=op_name, verify=('PASS' if verify else 'FAIL'), time=f'{total_time:.2f}s')}[/{DisplayStyle.BOLD_GREEN}]"
                            )
                        )
                        if (not verify) and error:
                            self.cli.presenter._handlers._emit_main_global(
                                Text.from_markup(
                                    f"[{DisplayStyle.BOLD_RED}]error:[/{DisplayStyle.BOLD_RED}] {error}"
                                )
                            )
                        if log_dir:
                            self.cli.presenter._handlers._emit_main_global(
                                Text.from_markup(f"[dim]log_dir: {log_dir}[/dim]")
                            )
                        if task_path:
                            self.cli.presenter._handlers._emit_main_global(
                                Text.from_markup(
                                    f"[dim]task_desc_path: {task_path}[/dim]"
                                )
                            )
                        if kernel_path:
                            self.cli.presenter._handlers._emit_main_global(
                                Text.from_markup(
                                    f"[dim]kernel_code_path: {kernel_path}[/dim]"
                                )
                            )

                        if self.auto_yes:
                            satisfied, inline_feedback = True, ""
                        else:
                            satisfied, inline_feedback = await self._ask_yes_no(
                                textual_manager=textual_manager,
                                prompt=t("ops.prompt.satisfied"),
                                default_yes=True,
                                invalid_text_as="no",
                            )
                        if satisfied:
                            # 这里不要直接把用户输入当成“新需求”，否则用户习惯性输入 y/n 会被送进 TaskInit
                            if self.auto_yes:
                                cont, inline_next_intent = True, ""
                            else:
                                cont, inline_next_intent = await self._ask_yes_no(
                                    textual_manager=textual_manager,
                                    prompt=t("ops.prompt.continue_new_op"),
                                    default_yes=False,
                                    invalid_text_as="yes",
                                )
                            if not cont:
                                return
                            next_intent = (inline_next_intent or "").strip()
                            if not next_intent:
                                next_intent = await textual_manager.await_user_input(
                                    t("ops.prompt.new_op_intent")
                                )
                            if not next_intent.strip():
                                return
                            base_intent = next_intent.strip()
                            break

                        feedback = (inline_feedback or "").strip()
                        if not feedback:
                            feedback = await textual_manager.await_user_input(
                                t("ops.prompt.feedback")
                            )
                        if not feedback:
                            return
                        pending_input = feedback
                        continue

                    if status in ["need_clarification", "need_modification"]:
                        question = (
                            payload.get("clarification_question")
                            or payload.get("agent_message")
                            or t("ops.msg.need_more_info_default")
                        )
                        textual_manager.set_input_enabled(True)
                        self.cli.presenter._handlers._emit_main_global(
                            Text.from_markup(
                                f"[{DisplayStyle.BOLD_YELLOW}]{t('ops.msg.need_more_info', question=question)}[/{DisplayStyle.BOLD_YELLOW}]"
                            )
                        )
                        answer = await textual_manager.await_user_input(
                            t("ops.prompt.your_supplement")
                        )
                        if not answer:
                            self.cli.presenter._handlers._emit_main_global(
                                Text.from_markup(
                                    f"[{DisplayStyle.YELLOW}]{t('ops.msg.extra_empty_exit')}[/{DisplayStyle.YELLOW}]"
                                )
                            )
                            return
                        pending_input = answer
                        continue

                    msg = payload.get("agent_message") or t(
                        "ops.msg.taskinit_cannot_continue_default"
                    )
                    self.cli.presenter._handlers._emit_main_global(
                        Text.from_markup(
                            f"[{DisplayStyle.RED}]{t('ops.msg.cannot_continue', status=status, msg=msg)}[/{DisplayStyle.RED}]"
                        )
                    )
                    return

        textual_manager.run(_workflow())
