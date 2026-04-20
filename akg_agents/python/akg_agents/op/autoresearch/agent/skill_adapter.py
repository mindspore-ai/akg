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

"""
Autoresearch skill retrieval primitives - keyword generation + ranking.

This module owns the *primitives* of the keyword pipeline. The
end-to-end orchestration (refill in either replace or append mode,
SkillBuilder integration, dedup, diagnostic prints) lives one layer
up in ``skill_pool.SkillPool``.

What this module exports:

  - Category taxonomy shared between ranking and rendering
    (``TRACKABLE_PATTERN_CATEGORIES``, ``_CATEGORY_LAYER``,
    ``_canonical_category``, ``_get_catalog``).
  - LLM-driven keyword bag generation with deterministic fallback
    (``QueryKeywords``, ``generate_query_keywords``,
    ``extract_fallback_keywords``, ``_parse_and_validate_query_keywords``).
  - Score-based ranking that fuses the keyword bags with a per-stage
    category prior and operator-metadata bonuses
    (``rank_skills_by_keywords``).

Prompt-facing rendering (full / index / ranked-block) lives in
``skill_rendering.py`` and imports ``_CATEGORY_LAYER`` /
``_canonical_category`` from here so the sort order and the
"method -> guide / implementation -> example" collapse rules stay
single-sourced.
"""

import asyncio
import functools
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Tuple

from akg_agents.op.skill import OperatorSkillCatalog

logger = logging.getLogger(__name__)

TRACKABLE_PATTERN_CATEGORIES = frozenset({
    "guide", "example", "case", "method", "implementation",
})


_CONTENT_MATCH_CATEGORIES = frozenset({
    "guide", "example", "method", "implementation",
})
_CATEGORY_LAYER = {
    "fundamental": (0, 0),
    "reference": (0, 1),
    "guide": (1, 0),
    "method": (1, 1),
    "example": (2, 0),
    "implementation": (2, 1),
    "case": (3, 0),
}
_CATEGORY_PRIOR = {
    "optimize": {
        "guide": 5.0,
        "example": 4.0,
        "case": 3.0,
        "fundamental": 2.0,
        "reference": 1.0,
    },
    "debug": {
        "case": 5.0,
        "guide": 4.0,
        "example": 3.0,
        "fundamental": 2.0,
        "reference": 1.0,
    },
}

_FALLBACK_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "class",
    "def", "do", "does", "for", "from", "have", "if", "import", "in",
    "into", "is", "it", "its", "len", "list", "none", "not", "of",
    "on", "or", "out", "print", "return", "self", "set", "str", "that",
    "the", "their", "them", "this", "to", "true", "tuple", "type",
    "use", "used", "with", "you",
    "\u4e00\u4e2a", "\u4e00\u4e9b", "\u4e0d\u4f1a", "\u4e0d\u662f", "\u4e2d", "\u4e86", "\u4ece", "\u4f7f\u7528", "\u4ee3\u7801", "\u505a", "\u5230",
    "\u548c", "\u5728", "\u5982\u679c", "\u5c06", "\u5c31", "\u6211\u4eec", "\u628a", "\u662f", "\u66f4", "\u6709\u5173", "\u6ca1\u6709", "\u7528",
    "\u7684", "\u7740", "\u800c", "\u88ab", "\u8ba9", "\u8bf7", "\u8fd9\u4e2a", "\u8fd9\u79cd", "\u9700\u8981",
})
_BANNED_GENERIC = frozenset({
    "performance", "optimization", "optimize", "efficient", "fast",
    "\u6027\u80fd", "\u4f18\u5316", "\u9ad8\u6548", "\u52a0\u901f", "\u5feb\u901f",
})
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|[\u4e00-\u9fff]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

_MAX_MUST_HAVE = 8
_MAX_NICE_TO_HAVE = 15
_MAX_AVOID = 5
_MIN_TOKEN_LEN = 2
_MAX_TOKEN_LEN = 40
_MAX_REASONING_LEN = 500
_MAX_LLM_OUTPUT_TOKENS = 500

_MUST_HAVE_WEIGHT = 3.0
_NICE_TO_HAVE_WEIGHT = 1.0
_AVOID_WEIGHT = -2.0
_METADATA_EXACT_MATCH_BONUS = 2.0
_METADATA_FUZZY_MATCH_BONUS = 0.5

_KEYWORD_TASK_DESC_LIMIT = 1_500
_KEYWORD_STUCK_CONTEXT_LIMIT = 1_000
_RANK_CONTENT_PREVIEW_LIMIT = 500
_RAW_RESPONSE_LOG_LIMIT = 200

_KEYWORD_SYSTEM_PROMPT = (
    "You generate keyword bags for local skill retrieval.\n"
    "Return text only. Do not call tools.\n"
    "Return exactly one <keywords> XML block with child tags "
    "must_have, nice_to_have, avoid, reasoning. Tokens inside each "
    "bag tag are comma-separated."
)


@dataclass
class QueryKeywords:
    must_have: List[str] = field(default_factory=list)
    nice_to_have: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)
    reasoning: str = ""

    def is_empty(self) -> bool:
        return not self.must_have and not self.nice_to_have

    def merged_with(self, other: "QueryKeywords") -> "QueryKeywords":
        def _dedup(first: List[str], second: List[str]) -> List[str]:
            seen: Set[str] = set()
            merged: List[str] = []
            for token in list(first) + list(second):
                if token and token not in seen:
                    seen.add(token)
                    merged.append(token)
            return merged

        return QueryKeywords(
            must_have=_dedup(self.must_have, other.must_have),
            nice_to_have=_dedup(self.nice_to_have, other.nice_to_have),
            avoid=_dedup(self.avoid, other.avoid),
            reasoning=self.reasoning or other.reasoning,
        )


@functools.lru_cache(maxsize=1)
def _get_catalog() -> OperatorSkillCatalog:
    """Process-wide OperatorSkillCatalog (construction scans disk once).

    Cached via ``functools.lru_cache`` rather than a mutable module
    global — same lazy-construct semantics, no shared mutable state.
    Tests that need to swap the catalog should ``monkeypatch.setattr``
    the ``_get_catalog`` function itself (see test_runtime_skill_binding).
    To drop the cache in-process, call ``_get_catalog.cache_clear()``.
    """
    return OperatorSkillCatalog()


def _normalize_token(text: str) -> str:
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text.strip())
    return "".join(
        ch.lower() if ch.isascii() and ch.isalpha() else ch
        for ch in collapsed
    )


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    fenced = re.match(r"^```(?:xml|json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def _tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text.replace("_", " ").replace("-", " ")):
        if raw.isascii():
            for piece in _CAMEL_RE.findall(raw) or [raw]:
                token = _normalize_token(piece)
                if token:
                    tokens.append(token)
        else:
            token = _normalize_token(raw)
            if token:
                tokens.append(token)
    return tokens


def _canonical_category(category: str) -> str:
    category = (category or "").strip()
    if category == "method":
        return "guide"
    if category == "implementation":
        return "example"
    return category


_TAG_BAG_PATTERN = {
    "must_have": re.compile(r"<must_have>(.*?)</must_have>", re.DOTALL | re.IGNORECASE),
    "nice_to_have": re.compile(r"<nice_to_have>(.*?)</nice_to_have>", re.DOTALL | re.IGNORECASE),
    "avoid": re.compile(r"<avoid>(.*?)</avoid>", re.DOTALL | re.IGNORECASE),
}
_TAG_REASONING_PATTERN = re.compile(
    r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE
)


def _parse_and_validate_query_keywords(response: str) -> Optional[QueryKeywords]:
    if not response:
        return None
    text = _strip_markdown_fences(response)
    if not text:
        return None

    def _clean_bag(tag: str, cap: int) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for match in _TAG_BAG_PATTERN[tag].findall(text):
            # Each tag body may hold comma- or newline-separated tokens;
            # repeated tags are concatenated so both styles work.
            for raw in re.split(r"[,\n]", match):
                token = _normalize_token(raw)
                if not (_MIN_TOKEN_LEN <= len(token) <= _MAX_TOKEN_LEN):
                    continue
                if token in _BANNED_GENERIC or " " in token:
                    continue
                if token in seen:
                    continue
                seen.add(token)
                out.append(token)
                if len(out) >= cap:
                    return out
        return out

    must = _clean_bag("must_have", _MAX_MUST_HAVE)
    if not must:
        return None

    nice = _clean_bag("nice_to_have", _MAX_NICE_TO_HAVE)
    avoid = _clean_bag("avoid", _MAX_AVOID)

    reasoning_match = _TAG_REASONING_PATTERN.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    return QueryKeywords(
        must_have=must,
        nice_to_have=nice,
        avoid=avoid,
        reasoning=reasoning[:_MAX_REASONING_LEN],
    )


def extract_fallback_keywords(op_name: str, task_desc: str) -> QueryKeywords:
    def _accept(token: str) -> bool:
        if not token:
            return False
        if not (_MIN_TOKEN_LEN <= len(token) <= _MAX_TOKEN_LEN):
            return False
        if token in _FALLBACK_STOPWORDS or token in _BANNED_GENERIC:
            return False
        return True

    must_have: List[str] = []
    seen_must: Set[str] = set()

    def _push_must(token: str) -> None:
        if token and token not in seen_must and _accept(token):
            seen_must.add(token)
            must_have.append(token)

    op_norm = _normalize_token(op_name)
    if _accept(op_norm):
        _push_must(op_norm)
    for token in _tokenize_text(op_name):
        if len(must_have) >= _MAX_MUST_HAVE:
            break
        _push_must(token)

    freq = Counter(
        token for token in _tokenize_text(task_desc or "") if _accept(token)
    )
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))

    idx = 0
    while len(must_have) < _MAX_MUST_HAVE and idx < len(ranked):
        token, _ = ranked[idx]
        idx += 1
        _push_must(token)

    nice_to_have: List[str] = []
    seen_nice: Set[str] = set()
    while len(nice_to_have) < _MAX_NICE_TO_HAVE and idx < len(ranked):
        token, _ = ranked[idx]
        idx += 1
        if token in seen_must or token in seen_nice:
            continue
        seen_nice.add(token)
        nice_to_have.append(token)

    return QueryKeywords(
        must_have=must_have,
        nice_to_have=nice_to_have,
        avoid=[],
        reasoning="",
    )


def _build_keyword_prompt(
    op_name: str,
    task_desc: str,
    stage: str,
    stuck_context: str,
    *,
    dsl: str = "",
    backend: str = "",
    framework: str = "",
    arch: str = "",
    hint: str = "",
) -> str:
    """Build the keyword-generation prompt — dense template, single
    header + key:value lines, runtime invariants stay enforced in
    ``_parse_and_validate_query_keywords`` not here.
    """
    task_text = (task_desc or "")[:_KEYWORD_TASK_DESC_LIMIT]
    stuck_text = (stuck_context or "")[:_KEYWORD_STUCK_CONTEXT_LIMIT]
    target = f"{dsl or '?'} / {backend or '?'} / {arch or '?'} / {framework or '?'}"
    parts = [
        "Generate a keyword bag of single technical tokens for ranking "
        "a local skill library. Respond with EXACTLY one <keywords> "
        "block:",
        "",
        "<keywords>",
        "<must_have>t1, t2, t3</must_have>",
        "<nice_to_have>t4, t5</nice_to_have>",
        "<avoid>t6</avoid>",
        "<reasoning>short rationale</reasoning>",
        "</keywords>",
        "",
        f"op: {op_name or '(unknown)'}",
        f"target: {target}",
        f"stage: {stage or 'optimize'}",
    ]
    if hint:
        parts.append(f"hint: {hint}")
    parts.extend([
        "task:",
        f"```text\n{task_text}\n```",
    ])
    if stuck_text:
        parts.extend([
            "stuck:",
            f"```text\n{stuck_text}\n```",
        ])
    parts.extend([
        "",
        "Rules: single token (no spaces), comma-separated inside each "
        "bag tag, prefer concrete API / kernel / HW terms over generic "
        "words (performance, optimization, \u6027\u80fd, \u4f18\u5316), "
        "must_have <= 8 high-confidence, nice_to_have <= 8, avoid <= 8.",
    ])
    return "\n".join(parts)


async def generate_query_keywords(
    op_name: str,
    task_desc: str,
    *,
    dsl: str = "",
    backend: str = "",
    framework: str = "",
    arch: str = "",
    hint: str = "",
    stage: str = "optimize",
    stuck_context: str = "",
    llm=None,
    timeout: float = 30.0,
) -> QueryKeywords:
    """Generate keyword bags for skill ranking.

    The optional ``dsl`` / ``backend`` / ``framework`` / ``arch``
    kwargs flow into the prompt's ``target:`` line so the LLM can
    distinguish e.g. ``triton_ascend/ascend910b4`` from
    ``triton_cuda/a100`` when the op name and task source are
    otherwise identical.

    The optional ``hint`` is a short natural-language steering string
    supplied by the agent (via ``search_skills``). When non-empty it
    appears as a ``hint:`` line in the prompt; when empty it is
    omitted. The deterministic regex fallback ignores hints — it has
    no semantic capacity to use them.
    """
    fallback = extract_fallback_keywords(op_name, task_desc)
    if llm is None:
        return fallback

    prompt = _build_keyword_prompt(
        op_name, task_desc, stage, stuck_context,
        dsl=dsl, backend=backend, framework=framework, arch=arch,
        hint=hint,
    )
    try:
        response = await asyncio.wait_for(
            llm.call(
                system_prompt=_KEYWORD_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                compact=True,
                max_tokens=_MAX_LLM_OUTPUT_TOKENS,
                max_retries=0,
            ),
            timeout=timeout,
        )
        raw_text = llm.get_response_text(response)
    except Exception as exc:
        logger.warning("[skill-narrow] keyword generation failed: %s", exc)
        return fallback

    parsed = _parse_and_validate_query_keywords(raw_text)
    if parsed is None:
        logger.warning(
            "[skill-narrow] invalid keyword response, falling back. raw=%r",
            (raw_text or "")[:_RAW_RESPONSE_LOG_LIMIT],
        )
        return fallback

    logger.info(
        "[skill-narrow] query keywords: must_have=%s nice_to_have=%s avoid=%s",
        parsed.must_have, parsed.nice_to_have, parsed.avoid,
    )
    return parsed


def rank_skills_by_keywords(
    skills: List,
    keywords: QueryKeywords,
    *,
    stage: str = "optimize",
    op_name: str = "",
) -> List[Tuple[float, Any]]:
    if not skills:
        return []

    if stage not in _CATEGORY_PRIOR:
        logger.warning(
            "rank_skills_by_keywords: unknown stage=%r, falling back to optimize",
            stage,
        )
        stage = "optimize"
    prior = _CATEGORY_PRIOR[stage]

    must = [_normalize_token(token) for token in keywords.must_have if token]
    nice = [_normalize_token(token) for token in keywords.nice_to_have if token]
    avoid = [_normalize_token(token) for token in keywords.avoid if token]
    op_norm = _normalize_token(op_name)
    op_pieces: List[str] = []
    if op_name:
        for token in _tokenize_text(op_name):
            if token and token not in op_pieces:
                op_pieces.append(token)

    scored: List[Tuple[float, Any]] = []
    for skill in skills:
        category = getattr(skill, "category", "") or ""
        family = _canonical_category(category)
        text_parts = [
            getattr(skill, "name", "") or "",
            getattr(skill, "description", "") or "",
        ]
        if category in _CONTENT_MATCH_CATEGORIES:
            text_parts.append(
                (getattr(skill, "content", "") or "")[:_RANK_CONTENT_PREVIEW_LIMIT]
            )
        text = _normalize_token(" ".join(text_parts))

        score = float(prior.get(family, 0.0))
        for token in must:
            if token and token in text:
                score += _MUST_HAVE_WEIGHT
        for token in nice:
            if token and token in text:
                score += _NICE_TO_HAVE_WEIGHT
        for token in avoid:
            if token and token in text:
                score += _AVOID_WEIGHT

        if op_norm:
            meta = getattr(skill, "metadata", {}) or {}
            meta_op = _normalize_token(
                f"{meta.get('operator_type', '')},{meta.get('operator_patterns', '')}"
            )
            if meta_op:
                if op_norm in meta_op:
                    score += _METADATA_EXACT_MATCH_BONUS
                elif any(piece in meta_op for piece in op_pieces):
                    score += _METADATA_FUZZY_MATCH_BONUS

        scored.append((score, skill))

    scored.sort(
        key=lambda item: (
            -item[0],
            _CATEGORY_LAYER.get(getattr(item[1], "category", "") or "", (9, 9)),
            getattr(item[1], "name", "") or "",
        )
    )
    return scored
