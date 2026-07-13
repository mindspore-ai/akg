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


def num_cases_from_ref(ref_path: Any) -> int:
    """SSOT for "how many input groups does this reference define": load the
    reference .py and count its groups; 1 on any failure. Both akg_eval (the
    dispatch budget) and batch verify.py (the parent Tier-2 wall cap) call this
    so the per-shape -> whole-eval timeout expansion shares one num_cases."""
    import os
    import importlib.util
    try:
        if not os.path.isfile(str(ref_path)):
            return 1
        spec = importlib.util.spec_from_file_location("reference", str(ref_path))
        if not (spec and spec.loader):
            return 1
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return max(1, int(num_cases(mod)))
    except Exception:
        return 1
