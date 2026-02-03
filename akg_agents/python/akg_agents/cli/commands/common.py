# Copyright 2026 Huawei Technologies Co., Ltd
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
from typing import Optional

import typer
from prompt_toolkit import PromptSession
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from akg_agents.cli.console import AKGConsole
from akg_agents.cli.constants import DisplayStyle
from akg_agents.cli.runtime import LocalExecutor
from akg_agents.cli.service import CLIAppServices
from akg_agents.cli.utils.ui_helpers import print_logo_once
from akg_agents.cli.commands.slash_commands import create_dispatcher


class CommonCommandRunner:
    def __init__(
        self,
        *,
        console: AKGConsole,
        cli: LocalExecutor,
        intent: Optional[str],
        exit_after_intent: bool = False,
    ) -> None:
        self.console = console
        self.cli = cli
        self.intent = intent
        self.exit_after_intent = bool(exit_after_intent)
        self.scene = "common"
        self._should_exit = False
        self._command_dispatcher = create_dispatcher()

    async def _execute_common_agent(self, *, user_input: str) -> dict:
        return await self.cli.execute_common_agent(
            user_input=user_input,
            use_stream=self.console.use_stream,
        )

    def _render_agent_messages(self, state: dict) -> None:
        if not isinstance(state, dict):
            return

        display_message = str(state.get("display_message") or "").rstrip()
        if display_message:
            try:
                ansi_text = Text.from_ansi(display_message)
                panel = Panel(
                    ansi_text,
                    box=ROUNDED,
                    border_style="dim",
                    padding=(0, 1),
                )
                self.console.print(panel)
            except Exception:
                try:
                    panel = Panel(
                        display_message,
                        box=ROUNDED,
                        border_style="dim",
                        padding=(0, 1),
                    )
                    self.console.print(panel, markup=False)
                except Exception:
                    self.console.print(" " + display_message, markup=False)

        hint_message = str(state.get("hint_message") or "").rstrip()
        if hint_message:
            try:
                ansi_text = Text.from_ansi(hint_message)
                ansi_text.stylize(DisplayStyle.DIM, 0, len(ansi_text))
                self.console.print(ansi_text)
            except Exception:
                self.console.print(
                    f"[{DisplayStyle.DIM}]{hint_message}[/{DisplayStyle.DIM}]"
                )

        current_step = str(state.get("current_step", "")).strip().lower()
        if current_step == "error":
            error_message = str(
                state.get("error")
                or state.get("error_message")
                or state.get("message")
                or ""
            ).rstrip()
            if error_message:
                self.console.print(
                    f"[{DisplayStyle.BOLD_RED}]Error:[/{DisplayStyle.BOLD_RED}] {error_message}"
                )

    async def _process_input(self, user_input: str, print_input: bool) -> bool:
        user_input = (user_input or "").strip()
        if not user_input:
            return True

        next_input = user_input
        next_print = print_input
        auto_hops = 0

        while True:
            if next_print:
                self.console.print_user_input(next_input)

            state = await self._execute_common_agent(user_input=next_input)
            self._render_agent_messages(state)

            if not isinstance(state, dict):
                return False

            auto_input = state.get("auto_input")
            if auto_input:
                auto_hops += 1
                if auto_hops > 3:
                    return True
                next_input = str(auto_input).strip()
                if not next_input:
                    return True
                next_print = False
                continue

            current_step = str(state.get("current_step", "")).strip().lower()
            if current_step in ["cancelled_by_user", "cancelled"]:
                return True
            if not state.get("should_continue", True) or current_step in ["error"]:
                return False
            return True

    async def _workflow(self) -> None:
        initial_input = (self.intent or "").strip()
        if initial_input:
            cont = await self._process_input(initial_input, print_input=True)
            if not cont or self.exit_after_intent:
                return

        # Enable slash-command tab completion for common.
        from akg_agents.cli.ui.completers import EnhancedCompleter
        from akg_agents.cli.commands.slash_commands import get_registry

        session = PromptSession(completer=EnhancedCompleter(get_registry()))
        while True:
            try:
                user_input = await session.prompt_async("common> ")
            except (EOFError, KeyboardInterrupt):
                return

            if user_input is None:
                continue
            text = str(user_input).strip()
            if not text:
                continue
            if text.startswith("/"):
                result = await self._command_dispatcher.dispatch(self, text)
                if self._should_exit:
                    return
                if result.handled:
                    continue
            if text.lower() in {"exit", "quit", ":q"}:
                return

            cont = await self._process_input(text, print_input=True)
            if not cont:
                return

    def run(self) -> None:
        try:
            asyncio.run(self._workflow())
        except KeyboardInterrupt:
            try:
                asyncio.run(self.cli.cancel_main_agent(reason="cancelled by ctrl+c"))
            except Exception:
                pass
            self.console.print(
                f"\n[{DisplayStyle.YELLOW}]Exited[/{DisplayStyle.YELLOW}]"
            )


def register_common_command(
    app: typer.Typer, console: Console, services: CLIAppServices
) -> None:
    @app.command("common")
    def common_cmd(
        intent: Optional[str] = typer.Option(
            None, "--intent", help="Provide the request directly (skip input prompt)"
        ),
        stream: bool = typer.Option(
            True, "--stream/--no-stream", help="Enable/disable LLM streaming output"
        ),
        yolo: bool = typer.Option(
            False, "--yolo", help="Auto-approve all tool executions"
        ),
        exit_after_intent: bool = typer.Option(
            False,
            "--once",
            "--exit-after-intent",
            help="Exit after processing --intent (non-interactive mode)",
        ),
    ) -> None:
        """Common entry (ReAct mode)."""
        akg_console = AKGConsole(console, use_stream=stream)
        print_logo_once(akg_console)

        executor = LocalExecutor.create_for_cli(
            akg_console,
            use_stream=stream,
            auto_approve_tools=yolo,
        )

        runner = CommonCommandRunner(
            console=akg_console,
            cli=executor,
            intent=intent,
            exit_after_intent=exit_after_intent,
        )
        runner.run()
