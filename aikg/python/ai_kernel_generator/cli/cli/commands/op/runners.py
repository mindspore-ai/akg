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
from ai_kernel_generator.cli.cli.utils.i18n import t
from .input_commands import SlashCommandContext, build_default_slash_commands
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
        rag: bool = False,
    ):
        self.console = console
        self.services = services
        self.cli = cli
        self.server_url = server_url
        self.auto_yes = auto_yes
        self.target = target
        self.intent = intent
        self.rag = rag
        self._slash_commands = build_default_slash_commands()

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
            self.cli.presenter.print_workflow_start()
            res = await self.cli.execute_main_agent(
                action=action,
                user_input=user_input,
                framework=resolved.get("framework", ""),
                backend=resolved.get("backend", ""),
                arch=resolved.get("arch", ""),
                dsl=resolved.get("dsl", ""),
            rag=self.rag,
            )
            self.cli.presenter.print_workflow_complete()
            return res
        finally:
            # pragma: no cover - defensive cleanup
            try:
                textual_manager.set_input_enabled(True)
            except Exception:
                pass

    def _build_resolved(self) -> dict:
        return {
            "framework": self.target.framework,
            "backend": self.target.backend,
            "arch": self.target.arch,
            "dsl": self.target.dsl,
        }

    def _emit_main_global(self, content: str | Text) -> None:
        if isinstance(content, Text):
            payload = content
        else:
            payload = Text.from_markup(content)
        self.cli.presenter._handlers._emit_main_global(payload)

    def _emit_main(self, task_id: str, content: str | Text) -> None:
        if isinstance(content, Text):
            payload = content
        else:
            payload = Text.from_markup(content)
        self.cli.presenter._handlers._emit_main(task_id, payload)

    def _main_task_id(self) -> str:
        return str(getattr(self.cli.presenter, "main_task_id", "") or "main")

    def _emit_warning(self, message: str) -> None:
        self._emit_main_global(message)

    def _is_on_main_tab(self, textual_manager) -> bool:
        try:
            if hasattr(textual_manager, "is_on_main_tab"):
                return bool(textual_manager.is_on_main_tab())
        except Exception:
            return True
        return True

    def _main_tab_label(self, textual_manager) -> str:
        try:
            if hasattr(textual_manager, "get_main_task_id"):
                main_id = str(textual_manager.get_main_task_id() or "").strip()
                return main_id or "main"
        except Exception:
            return "main"
        return "main"

    async def _await_user_input(self, textual_manager, prompt: str) -> str:
        while True:
            user_input = await textual_manager.await_user_input("")
            if not user_input:
                return ""
            try:
                if self._slash_commands.handle(
                    user_input,
                    ctx=SlashCommandContext(
                        layout_manager=textual_manager,
                        reset_state=self.cli.presenter.reset_state,
                    ),
                ):
                    continue
            except Exception:
                pass
            if not self._is_on_main_tab(textual_manager):
                main_label = self._main_tab_label(textual_manager)
                self._emit_warning(t("tui.msg.input_only_main_warn", main=main_label))
                continue
            return user_input

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
        base_intent = (base_intent or "").strip()
        if not base_intent:
            base_intent = await self._await_user_input(
                textual_manager, t("ops.prompt.op_intent")
            )
        if not base_intent:
            self._emit_main_global(
                f"[{DisplayStyle.YELLOW}]{t('ops.msg.intent_empty_exit')}[/{DisplayStyle.YELLOW}]"
            )
            return None
        return base_intent.strip()

    def _render_agent_messages(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        display_message = str(state.get("display_message") or "").rstrip()
        if display_message:
            self._emit_main(self._main_task_id(), Text(display_message))
        hint_message = str(state.get("hint_message") or "").rstrip()
        if hint_message:
            self._emit_main(self._main_task_id(), Text(hint_message))

    async def _run_workflow(self, textual_manager) -> None:
        resolved = self._build_resolved()

        await self._maybe_set_record_path(textual_manager)
        self._validate_config_or_exit(resolved)
        self._ensure_worker_or_exit(resolved)

        user_input = await self._prompt_base_intent(
            textual_manager, (self.intent or "")
        )
        if not user_input:
            return

        action = "start"
        while True:
            state = await self._execute_main_agent_action(
                textual_manager=textual_manager,
                resolved=resolved,
                action=action,
                user_input=user_input,
            )
            self._render_agent_messages(state)
            if not isinstance(state, dict):
                return
            current_step = str(state.get("current_step") or "").strip().lower()
            if not state.get("should_continue", True) or current_step in [
                "cancelled",
                "saved",
                "error",
            ]:
                return

            user_input = await self._await_user_input(
                textual_manager, t("ops.prompt.op_intent")
            )
            if not user_input or not user_input.strip():
                continue
            user_input = user_input.strip()
            action = "continue"

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
