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

    async def _execute_main_agent_action(
        self,
        *,
        textual_manager,
        resolved: dict,
        action: str,
        user_input: str,
        display_input: str | None = None,
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

        shown_input = display_input if display_input is not None else user_input
        if shown_input:
            self.cli.presenter.print_user_input(shown_input)
        self.cli.presenter.print_workflow_start()
        res = await self.cli.execute_main_agent(
            action=action,
            user_input=user_input,
            framework=resolved.get("framework", ""),
            backend=resolved.get("backend", ""),
            arch=resolved.get("arch", ""),
            dsl=resolved.get("dsl", ""),
            sub_workflow=None,
        )
        self.cli.presenter.print_workflow_complete()
        return res

    async def _execute_main_agent_task_init(
        self,
        *,
        textual_manager,
        resolved: dict,
        user_input: str,
        first: bool,
    ) -> dict:
        return await self._execute_main_agent_action(
            textual_manager=textual_manager,
            resolved=resolved,
            action="start" if first else "revise",
            user_input=user_input,
        )

    async def _maybe_start_generation(
        self,
        *,
        textual_manager,
        resolved: dict,
    ) -> dict:
        res = await self._execute_main_agent_action(
            textual_manager=textual_manager,
            resolved=resolved,
            action="confirm",
            user_input="",
            display_input=t("ops.msg.start_generate_with_desc"),
        )
        return res

    def _extract_task_init_payload(self, taskinit_res: dict) -> dict:
        meta = taskinit_res.get("metadata") or {}
        payload = meta.get("task_init") if isinstance(meta, dict) else {}
        return payload if isinstance(payload, dict) else {}

    def _extract_main_agent_meta(self, taskinit_res: dict) -> dict:
        meta = taskinit_res.get("metadata") or {}
        main_meta = meta.get("main_agent") if isinstance(meta, dict) else {}
        return main_meta if isinstance(main_meta, dict) else {}

    def _is_generation_result(self, task_res: dict) -> bool:
        main_meta = self._extract_main_agent_meta(task_res)
        current_step = str(main_meta.get("current_step") or "").strip().lower()
        if current_step in ["completed", "failed"]:
            return True
        kernel_code = str(task_res.get("kernel_code") or "").strip()
        return bool(kernel_code)

    def _build_resolved(self) -> dict:
        return {
            "framework": self.target.framework,
            "backend": self.target.backend,
            "arch": self.target.arch,
            "dsl": self.target.dsl,
        }

    def _emit_main_global(self, markup: str) -> None:
        self.cli.presenter._handlers._emit_main_global(Text.from_markup(markup))

    async def _maybe_set_record_path(self, textual_manager) -> None:
        # UI 启动后才能拿到 session_dir：在这里补上录制路径，生成 messages.jsonl
        try:
            sd = textual_manager.get_session_dir()
            if sd and getattr(self.cli, "_recorder", None) is None:
                self.cli.set_record_path(str(sd / "messages.jsonl"))
        except Exception as e:
            log.debug("[OpRunner] set_record_path skipped", exc_info=e)

    def _validate_config_or_exit(self, resolved: dict) -> None:
        errs = validate_target_config(
            resolved.get("framework", ""),
            resolved.get("backend", ""),
            resolved.get("arch", ""),
            resolved.get("dsl", ""),
        )
        if errs:
            self._emit_main_global(
                f"[{DisplayStyle.BOLD_RED}]{t('ops.error.config_invalid')}[/{DisplayStyle.BOLD_RED}]\n- "
                + "\n- ".join(errs)
            )
            raise typer.Exit(code=2)

    def _ensure_worker_or_exit(self, resolved: dict) -> None:
        backend = resolved.get("backend", "")
        arch = resolved.get("arch", "")
        if self.services.workers.server_has_worker(self.server_url, backend, arch):
            return
        # Textual 场景下，server 可能是新启动的，之前的注册状态不存在。
        # 为了"一劳永逸"减少误伤：只要用户已通过 --worker_url 提供了 worker 列表，就先自动重试注册一次。
        if self.services.workers.workers:
            self.services.workers.register_if_any(self.console, self.server_url)
            if self.services.workers.server_has_worker(
                self.server_url, backend, arch
            ):
                return
            self._emit_main_global(
                f"[{DisplayStyle.BOLD_RED}]{t('ops.error.worker_still_missing', backend=backend, arch=arch)}[/{DisplayStyle.BOLD_RED}]"
            )
            raise typer.Exit(code=2)

        self._emit_main_global(
            f"[{DisplayStyle.BOLD_RED}]{t('ops.error.missing_worker')}[/{DisplayStyle.BOLD_RED}] "
            f"backend={backend}, arch={arch}"
            f"{t('ops.error.missing_worker_hint')}"
        )
        raise typer.Exit(code=2)

    async def _prompt_base_intent(self, textual_manager, base_intent: str) -> str | None:
        if not base_intent:
            base_intent = await textual_manager.await_user_input(
                t("ops.prompt.op_intent")
            )
        if not base_intent:
            self._emit_main_global(
                f"[{DisplayStyle.YELLOW}]{t('ops.msg.intent_empty_exit')}[/{DisplayStyle.YELLOW}]"
            )
            return None
        return base_intent

    async def _handle_irrelevant_input(
        self, *, textual_manager, resolved: dict, payload: dict
    ) -> dict | None:
        msg = payload.get("agent_message") or t("ops.msg.need_more_info_default")
        textual_manager.set_input_enabled(True)
        self._emit_main_global(
            f"[{DisplayStyle.BOLD_YELLOW}]{msg}[/{DisplayStyle.BOLD_YELLOW}]"
        )
        next_input = await textual_manager.await_user_input(
            t("ops.prompt.op_intent")
        )
        if not next_input:
            return None
        return await self._execute_main_agent_action(
            textual_manager=textual_manager,
            resolved=resolved,
            action="auto",
            user_input=next_input,
        )

    async def _handle_generation_result(
        self, *, textual_manager, resolved: dict, task_res: dict
    ) -> dict | str | None:
        self.cli.presenter.display_summary(task_res)

        verify = task_res.get("verification_result")
        op_name = task_res.get("op_name", "")
        total_time = task_res.get("total_time", 0.0)
        error = task_res.get("error", "")
        meta2 = task_res.get("metadata") or {}
        log_dir = meta2.get("log_dir") if isinstance(meta2, dict) else None
        kernel_path = meta2.get("kernel_code_path") if isinstance(meta2, dict) else None
        task_path = meta2.get("task_desc_path") if isinstance(meta2, dict) else None

        self._emit_main_global(
            f"[{DisplayStyle.BOLD_GREEN}]{t('ops.msg.generation_done', op=op_name, verify=('PASS' if verify else 'FAIL'), time=f'{total_time:.2f}s')}[/{DisplayStyle.BOLD_GREEN}]"
        )
        if (verify is False) and error:
            self._emit_main_global(
                f"[{DisplayStyle.BOLD_RED}]error:[/{DisplayStyle.BOLD_RED}] {error}"
            )
        if log_dir:
            self._emit_main_global(f"[dim]log_dir: {log_dir}[/dim]")
        if task_path:
            self._emit_main_global(f"[dim]task_desc_path: {task_path}[/dim]")
        if kernel_path:
            self._emit_main_global(f"[dim]kernel_code_path: {kernel_path}[/dim]")

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
                return None
            next_intent = (inline_next_intent or "").strip()
            if not next_intent:
                next_intent = await textual_manager.await_user_input(
                    t("ops.prompt.new_op_intent")
                )
            if not next_intent.strip():
                return None
            return next_intent.strip()

        feedback = (inline_feedback or "").strip()
        if not feedback:
            feedback = await textual_manager.await_user_input(t("ops.prompt.feedback"))
        if not feedback:
            return None
        return await self._execute_main_agent_action(
            textual_manager=textual_manager,
            resolved=resolved,
            action="auto",
            user_input=feedback,
        )

    async def _handle_ready_status(
        self, *, textual_manager, resolved: dict, task_res: dict
    ) -> dict | None:
        generated = task_res.get("generated_task_desc", "") or ""
        if not generated.strip():
            self._emit_main_global(
                f"[{DisplayStyle.BOLD_RED}]{t('ops.error.taskinit_ready_but_missing_desc')}[/{DisplayStyle.BOLD_RED}]"
            )
            return None
        self._emit_main_global(f"[bold]{t('ops.msg.taskinit_ready_code')}[/bold]")
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
                self._emit_main_global(
                    f"[{DisplayStyle.YELLOW}]{t('ops.msg.extra_empty_exit')}[/{DisplayStyle.YELLOW}]"
                )
                return None
            return await self._execute_main_agent_action(
                textual_manager=textual_manager,
                resolved=resolved,
                action="auto",
                user_input=extra,
            )
        return await self._maybe_start_generation(
            textual_manager=textual_manager, resolved=resolved
        )

    async def _handle_clarification(
        self, *, textual_manager, resolved: dict, payload: dict
    ) -> dict | None:
        question = (
            payload.get("clarification_question")
            or payload.get("agent_message")
            or t("ops.msg.need_more_info_default")
        )
        textual_manager.set_input_enabled(True)
        self._emit_main_global(
            f"[{DisplayStyle.BOLD_YELLOW}]{t('ops.msg.need_more_info', question=question)}[/{DisplayStyle.BOLD_YELLOW}]"
        )
        answer = await textual_manager.await_user_input(
            t("ops.prompt.your_supplement")
        )
        if not answer:
            self._emit_main_global(
                f"[{DisplayStyle.YELLOW}]{t('ops.msg.extra_empty_exit')}[/{DisplayStyle.YELLOW}]"
            )
            return None
        return await self._execute_main_agent_action(
            textual_manager=textual_manager,
            resolved=resolved,
            action="auto",
            user_input=answer,
        )

    def _get_task_state(self, task_res: dict) -> tuple[str, dict, str]:
        status = str(task_res.get("task_init_status", "") or "").strip().lower()
        payload = self._extract_task_init_payload(task_res)
        main_meta = self._extract_main_agent_meta(task_res)
        current_step = str(main_meta.get("current_step") or "").strip().lower()
        return status, payload, current_step

    def _is_cancelled(self, status: str, current_step: str) -> bool:
        return current_step in ["cancelled"] or status in ["cancelled"]

    def _is_irrelevant(self, status: str, current_step: str) -> bool:
        return (
            current_step in ["irrelevant_input", "rejected_by_intent"]
            or status in ["irrelevant_input", "rejected_by_intent"]
        )

    def _needs_clarification(self, status: str) -> bool:
        return status in ["need_clarification", "need_modification"]

    def _handle_cancelled(self, payload: dict) -> None:
        msg = payload.get("agent_message") or ""
        if msg:
            self._emit_main_global(
                f"[{DisplayStyle.YELLOW}]{msg}[/{DisplayStyle.YELLOW}]"
            )

    def _emit_cannot_continue(self, status: str, payload: dict) -> None:
        msg = payload.get("agent_message") or t(
            "ops.msg.taskinit_cannot_continue_default"
        )
        self._emit_main_global(
            f"[{DisplayStyle.RED}]{t('ops.msg.cannot_continue', status=status, msg=msg)}[/{DisplayStyle.RED}]"
        )

    async def _dispatch_task_state(
        self, *, textual_manager, resolved: dict, task_res: dict
    ) -> dict | str | None:
        status, payload, current_step = self._get_task_state(task_res)

        if self._is_cancelled(status, current_step):
            self._handle_cancelled(payload)
            return None

        if self._is_irrelevant(status, current_step):
            return await self._handle_irrelevant_input(
                textual_manager=textual_manager,
                resolved=resolved,
                payload=payload,
            )

        if self._is_generation_result(task_res):
            return await self._handle_generation_result(
                textual_manager=textual_manager,
                resolved=resolved,
                task_res=task_res,
            )

        if status == "ready":
            return await self._handle_ready_status(
                textual_manager=textual_manager,
                resolved=resolved,
                task_res=task_res,
            )

        if self._needs_clarification(status):
            return await self._handle_clarification(
                textual_manager=textual_manager,
                resolved=resolved,
                payload=payload,
            )

        self._emit_cannot_continue(status, payload)
        return None

    async def _run_task_loop(
        self, *, textual_manager, resolved: dict, task_res: dict
    ) -> str | None:
        while True:
            result = await self._dispatch_task_state(
                textual_manager=textual_manager,
                resolved=resolved,
                task_res=task_res,
            )
            if isinstance(result, dict):
                task_res = result
                continue
            return result

    async def _run_workflow(self, textual_manager) -> None:
        resolved = self._build_resolved()

        await self._maybe_set_record_path(textual_manager)
        self._validate_config_or_exit(resolved)
        self._ensure_worker_or_exit(resolved)

        base_intent = (self.intent or "").strip()
        while True:
            base_intent = await self._prompt_base_intent(
                textual_manager, base_intent
            )
            if not base_intent:
                return
            task_res = await self._execute_main_agent_task_init(
                textual_manager=textual_manager,
                resolved=resolved,
                user_input=base_intent,
                first=True,
            )
            next_intent = await self._run_task_loop(
                textual_manager=textual_manager,
                resolved=resolved,
                task_res=task_res,
            )
            if not next_intent:
                return
            base_intent = next_intent

    def run_textual(self) -> None:
        textual_manager = self.cli.presenter.layout_manager
        if not textual_manager:
            self.console.print(
                f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] Textual 管理器未初始化"
            )
            raise typer.Exit(code=2)

        async def _workflow() -> None:
            await self._run_workflow(textual_manager)

        textual_manager.run(_workflow())
