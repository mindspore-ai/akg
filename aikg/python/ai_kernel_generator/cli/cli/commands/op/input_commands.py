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

import re
from dataclasses import dataclass
from typing import Callable, Sequence


_COMMAND_RE = re.compile(r"^/([A-Za-z][\w-]*)(?:\s+|$)")


@dataclass(frozen=True)
class SlashCommandContext:
    layout_manager: object
    reset_state: Callable[[], None]


SlashCommandHandler = Callable[[SlashCommandContext, Sequence[str]], None]


class SlashCommandRegistry:
    """Slash command registry for input loop (runner layer)."""

    def __init__(self) -> None:
        self._handlers: dict[str, SlashCommandHandler] = {}

    def register(self, name: str, handler: SlashCommandHandler) -> None:
        key = str(name or "").strip().lower()
        if not key:
            raise ValueError("slash command name must be non-empty")
        self._handlers[key] = handler

    def handle(self, raw: str, *, ctx: SlashCommandContext) -> bool:
        parsed = self._parse(raw)
        if parsed is None:
            return False
        name, args = parsed
        handler = self._handlers.get(name)
        if handler is None:
            return False
        handler(ctx, args)
        return True

    def _parse(self, raw: str) -> tuple[str, list[str]] | None:
        text = str(raw or "")
        if not text.startswith("/"):
            return None
        match = _COMMAND_RE.match(text)
        if not match:
            return None
        name = match.group(1).lower()
        rest = text[match.end() :].strip()
        args = rest.split() if rest else []
        return name, args


def _handle_clear(ctx: SlashCommandContext, args: Sequence[str]) -> None:
    del args
    try:
        ctx.reset_state()
    except Exception:
        pass


def build_default_slash_commands() -> SlashCommandRegistry:
    registry = SlashCommandRegistry()
    registry.register("clear", _handle_clear)
    return registry
