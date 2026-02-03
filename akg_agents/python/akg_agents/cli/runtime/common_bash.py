from __future__ import annotations

import glob
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from akg_agents.cli.runtime.common_support import WorkspacePaths


@lru_cache(maxsize=1)
def get_bash_parser():
    try:
        from tree_sitter import Parser
        from tree_sitter_languages import get_language
    except Exception:
        return None
    try:
        parser = Parser()
        parser.set_language(get_language("bash"))
        return parser
    except Exception:
        return None


def ts_node_text(node: Any, source: bytes) -> str:
    try:
        return source[node.start_byte : node.end_byte].decode(errors="replace")
    except Exception:
        return ""


def walk_nodes(node: Any):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        children = list(getattr(current, "children", []) or [])
        for child in reversed(children):
            stack.append(child)


def strip_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def contains_glob(text: str) -> bool:
    return any(ch in text for ch in ("*", "?", "["))


def unwrap_command_words(words: list[str]) -> list[str]:
    idx = 0
    while idx < len(words):
        token = words[idx]
        if "=" in token and not token.startswith("-") and "/" not in token:
            idx += 1
            continue
        break
    wrappers = {"sudo", "env", "command", "builtin", "nice", "time", "stdbuf"}
    while idx < len(words) and words[idx] in wrappers:
        idx += 1
    return words[idx:]


def extract_command_words(node: Any, source: bytes) -> list[str]:
    words: list[str] = []
    for child in getattr(node, "children", []) or []:
        if child.type in {"command_name", "word", "string", "raw_string", "concatenation"}:
            text = strip_quotes(ts_node_text(child, source))
            if text:
                words.append(text)
    return words


def extract_redirect_targets(root: Any, source: bytes) -> list[str]:
    targets: list[str] = []
    for node in walk_nodes(root):
        if node.type != "redirect":
            continue
        target = _extract_redirect_target(node, source)
        if target:
            targets.append(target)
    return targets


def _extract_redirect_target(node: Any, source: bytes) -> str:
    for child in getattr(node, "children", []) or []:
        if child.type in {"word", "string", "raw_string", "concatenation"}:
            return strip_quotes(ts_node_text(child, source))
    return ""


def _expand_arg(arg: str) -> str:
    return os.path.expandvars(os.path.expanduser(arg))


def _has_dynamic_tokens(expanded: str) -> bool:
    return "$" in expanded or "`" in expanded


def _resolve_candidate(expanded: str, cwd: Path) -> Path:
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate


def _classify_glob(expanded: str, candidate: Path, paths: WorkspacePaths) -> tuple[str, str] | None:
    matches = [Path(p) for p in glob.glob(str(candidate), recursive=True)]
    if not matches:
        return ("unknown", expanded)
    for match in matches:
        ok, _ = paths.ensure_within(match)
        if not ok:
            return ("external", str(match))
    return None


def classify_path_arg(arg: str, cwd: Path, paths: WorkspacePaths) -> tuple[str, str] | None:
    value = strip_quotes(str(arg))
    if not value or value in {"-", "/dev/null"}:
        return None
    expanded = _expand_arg(value)
    if _has_dynamic_tokens(expanded):
        return ("unknown", value)
    candidate = _resolve_candidate(expanded, cwd)
    if contains_glob(expanded):
        return _classify_glob(expanded, candidate, paths)
    ok, _ = paths.ensure_within(candidate)
    if not ok:
        return ("external", str(candidate))
    return None


def analyze_bash_command(command: str, cwd: Path) -> dict[str, Any] | None:
    parser = get_bash_parser()
    if parser is None:
        return None
    source = command.encode()
    tree = parser.parse(source)
    root = tree.root_node
    commands: list[list[str]] = []
    for node in walk_nodes(root):
        if node.type == "command":
            words = extract_command_words(node, source)
            if words:
                commands.append(words)
    redirects = extract_redirect_targets(root, source)
    return {"commands": commands, "redirects": redirects}
