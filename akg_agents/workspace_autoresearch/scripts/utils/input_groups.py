# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Single-shape vs multi-shape reference input resolver.

The reference module exposes EITHER `get_inputs()` (returns one tuple of
inputs, single-shape) or `get_input_groups()` (returns a list of input
tuples, NPUKernelBench-style multi-shape). This helper duck-types the two
and always hands back a `List[List]` so the rest of autoresearch (verify
script, profile script, batch verifier, validators) can iterate uniformly.

Pure stdlib; imported by the generated verify / profile scripts via
the baked `scripts_dir` sys.path entry.
"""
from __future__ import annotations

from typing import Any, List


def resolve(ref_module: Any) -> List[list]:
    """Return a list-of-input-tuples from `ref_module`.

    Resolution order:
      1. `get_input_groups()` → return as-is, casting each element to list
         (so callers can `*group` and mutate in place).
      2. `get_inputs()` → wrap result in a 1-element list; single
         shape becomes a multi-shape list of size 1.
      3. Neither defined → raise ``AttributeError``.

    The returned tuples MAY contain non-tensor entries (lists for
    `normalized_shape`, ints, etc.). Callers convert to device tensors
    themselves; this function does not touch values.
    """
    fn = getattr(ref_module, "get_input_groups", None)
    if callable(fn):
        groups = fn()
        out: List[list] = []
        for g in groups:
            if isinstance(g, (list, tuple)):
                out.append(list(g))
            else:
                out.append([g])
        return out

    fn = getattr(ref_module, "get_inputs", None)
    if callable(fn):
        single = fn()
        if isinstance(single, (list, tuple)):
            return [list(single)]
        return [[single]]

    raise AttributeError(
        "reference module exposes neither get_input_groups() nor "
        "get_inputs(); cannot enumerate eval inputs"
    )


def num_cases(ref_module: Any) -> int:
    """Convenience: count of input groups (1 for single-shape ref)."""
    return len(resolve(ref_module))


def _value_repr(x: Any) -> str:
    """Render a single argument value (no name)."""
    try:
        shape = tuple(getattr(x, "shape", ()))
        dtype = str(getattr(x, "dtype", ""))
        if shape and dtype:
            return f"tensor{list(shape)}({dtype})"
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}{list(x)[:4]}{'...' if len(x) > 4 else ''}"
    return repr(x)[:24]


def _forward_param_names(model: Any) -> List[str]:
    """Pull positional parameter names off `model.forward` via inspect.

    `self` is auto-excluded for bound methods. *args / **kwargs entries
    are dropped so we don't generate spurious names for variadic slots.
    Returns an empty list on any failure (no model, weird signature,
    inspect raises) so callers fall back cleanly to nameless rendering.
    """
    if model is None:
        return []
    try:
        import inspect
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return []
    out: List[str] = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        out.append(p.name)
    return out


def describe_case(case: list, model: Any = None) -> str:
    """Cheap one-liner describing a case (for logs/reports).

    With `model`: each argument is prefixed by its `forward()` parameter
    name pulled from `inspect.signature(model.forward)`. Without `model`:
    falls back to a nameless positional rendering (kept for tests).

    Example with model (LayerNorm): 'x=tensor[1, 128, 4096](torch.float16)
        + normalized_shape=list[4096] + weight=tensor[4096](...) + ...'
    Example without model: 'tensor[1, 128, 4096](torch.float16) + list + ...'
    """
    names = _forward_param_names(model)
    parts: list[str] = []
    for i, x in enumerate(case):
        value = _value_repr(x)
        name = names[i] if i < len(names) else None
        parts.append(f"{name}={value}" if name else value)
    return " + ".join(parts) if parts else "(empty)"
