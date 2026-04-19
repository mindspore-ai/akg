"""
Tiny Markdown helper used across the autoresearch agent.

Why a module at all? The prompt builder, skill renderer, compress
layer and feedback builder all emit short Markdown fragments. When
each call site grows its own ``lines.append(f"- {k}: {v}")`` style,
drift accumulates: some use `"- "`, some `"* "`, some code fences
have language tags, some don't. Centralising the one-liners here
makes diffs reviewable and keeps output consistent.

The API is small on purpose — just the patterns actually duplicated
three or more times in the existing codebase. Add helpers here only
when a new duplicate shows up.
"""

from __future__ import annotations

from typing import Iterable


class Markdown:
    """Stateless renderer — every method is a staticmethod."""

    @staticmethod
    def bullet_list(items: Iterable[str], indent: int = 0) -> str:
        """Dash-bulleted list joined by newlines. Empty-item tolerant."""
        prefix = "  " * indent + "- "
        return "\n".join(prefix + str(it) for it in items if str(it) != "")

    @staticmethod
    def kv_list(pairs: Iterable[tuple[str, object]], indent: int = 0) -> str:
        """``- key: value`` rendering for small metadata blocks."""
        prefix = "  " * indent + "- "
        return "\n".join(f"{prefix}{k}: {v}" for k, v in pairs)

    @staticmethod
    def code_block(content: str, lang: str = "") -> str:
        """Fenced code block. ``lang`` may be empty for plain text."""
        fence = "```" + lang
        return f"{fence}\n{content.rstrip()}\n```"

    @staticmethod
    def section(title: str, body: str, level: int = 2) -> str:
        """A ``##`` header followed by a blank line and body."""
        hashes = "#" * max(1, level)
        return f"{hashes} {title}\n\n{body}"

    @staticmethod
    def labeled_block(tag: str, body: str, attrs: dict | None = None) -> str:
        """XML-tagged block. Used for structured tool_result content."""
        attr_str = "".join(f' {k}="{v}"' for k, v in (attrs or {}).items())
        return f"<{tag}{attr_str}>\n{body}\n</{tag}>"
