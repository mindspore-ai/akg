from __future__ import annotations

import inspect
from functools import lru_cache
from typing import List, Optional

from akg_agents.op.utils.code_checker.base import (
    CheckContext,
    Issue,
    Location,
    TritonDiagnosticChecker,
)

from .ast_utils import TritonUse, extract_triton_info


@lru_cache(maxsize=256)
def _resolve_triton_object(canonical: str):
    import triton
    import triton.language as tl

    if canonical == "triton":
        return triton
    if canonical == "triton.language":
        return tl
    if canonical.startswith("triton.language."):
        cur = tl
        parts = canonical.split(".")[2:]
    elif canonical.startswith("triton."):
        cur = triton
        parts = canonical.split(".")[1:]
    else:
        return None

    for part in parts:
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


@lru_cache(maxsize=256)
def _signature_for(canonical: str):
    obj = _resolve_triton_object(canonical)
    if obj is None:
        return None
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError):
        return None


def _has_var_keyword(sig: inspect.Signature) -> bool:
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def _has_var_positional(sig: inspect.Signature) -> bool:
    return any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in sig.parameters.values())


def _positional_capacity(sig: inspect.Signature) -> int:
    count = 0
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1
    return count


def _unknown_keywords(sig: inspect.Signature, use: TritonUse) -> List[str]:
    if _has_var_keyword(sig):
        return []
    valid = set(sig.parameters)
    return sorted({kw for kw in use.keywords if kw != "**" and kw not in valid})


def _format_signature(canonical: str, sig: inspect.Signature) -> str:
    return f"{canonical}{sig}"


def _bad_kwarg_hint(use: TritonUse, bad_keywords: List[str]) -> Optional[str]:
    if use.canonical == "triton.jit" and {"num_stages", "num_warps"} & set(bad_keywords):
        return (
            "Use @triton.jit without num_stages/num_warps. In this Triton version "
            "those options are launch/autotune metadata, not jit decorator keyword arguments."
        )
    return "Remove unsupported keyword arguments or adjust the call to match the installed Triton API."


CHECKER_NAME = "api_signature"
CHECKER_ID = "triton_api_signature"
RULE_ID = "TRITON_API_SIGNATURE"


class TritonApiSignatureChecker(TritonDiagnosticChecker):
    name = CHECKER_NAME
    checker_id = CHECKER_ID
    rule_id = RULE_ID

    def run(self, code: str, ctx: CheckContext) -> List[Issue]:
        if not (ctx.dsl or "").lower().startswith("triton"):
            return []

        info = extract_triton_info(code)
        issues: List[Issue] = []

        for use in info.uses:
            if use.kind not in {"call", "decorator", "attribute"}:
                continue
            obj = _resolve_triton_object(use.canonical)
            if obj is None:
                issues.append(
                    Issue(
                        severity="ERROR",
                        rule_id="TRITON_API_MISSING",
                        title="Missing Triton API",
                        message=f"{use.raw} resolves to {use.canonical}, which is not available in the installed Triton.",
                        location=Location(lineno=use.lineno, col=use.col),
                        hint=_missing_api_hint(use.canonical),
                        tags={"triton", "api", use.scope},
                    )
                )
                continue

            if use.kind == "attribute":
                continue

            sig = _signature_for(use.canonical)
            if sig is None:
                continue

            bad_keywords = _unknown_keywords(sig, use)
            if bad_keywords:
                issues.append(
                    Issue(
                        severity="ERROR",
                        rule_id="TRITON_API_BAD_KWARG",
                        title="Unsupported Triton API keyword",
                        message=(
                            f"{use.raw} uses unsupported keyword(s) {bad_keywords} for installed API "
                            f"{_format_signature(use.canonical, sig)}."
                        ),
                        location=Location(lineno=use.lineno, col=use.col),
                        hint=_bad_kwarg_hint(use, bad_keywords),
                        tags={"triton", "api", use.scope},
                    )
                )

            if not _has_var_positional(sig) and use.args_count > _positional_capacity(sig):
                issues.append(
                    Issue(
                        severity="ERROR",
                        rule_id="TRITON_API_TOO_MANY_POSITIONAL_ARGS",
                        title="Too many Triton API positional arguments",
                        message=(
                            f"{use.raw} passes {use.args_count} positional arguments, but installed API "
                            f"{_format_signature(use.canonical, sig)} accepts at most {_positional_capacity(sig)}."
                        ),
                        location=Location(lineno=use.lineno, col=use.col),
                        hint="Reduce positional arguments or pass supported parameters by keyword.",
                        tags={"triton", "api", use.scope},
                    )
                )

        return issues


def _missing_api_hint(canonical: str) -> Optional[str]:
    if canonical.endswith(".div") or canonical.endswith(".mod"):
        return "Prefer Python operators: a // b and a % b."
    return "Check spelling and replace the call with an API available in the installed Triton package."
