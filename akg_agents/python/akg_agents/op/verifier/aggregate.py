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

"""Per-shape aggregation — the single owner of how per-shape arrays collapse
to one scalar. Two rules, applied at EVERY call site:

  - LATENCY (gen_time / base_time / roofline_time)  -> arithmetic mean
    (:func:`mean_us`). Wall time is additive; the mean is the per-call cost.

  - SPEEDUP (base/gen, roofline/gen, ...)  -> geometric mean of the
    per-shape ratios (:func:`geomean_ratio`). A single large-magnitude shape
    must not dominate the headline ratio the way it does in mean(slow)/mean(gen):
    geomean weights every shape's *ratio* equally, which is the right notion
    of "typical speedup across shapes".

speedup_vs_ref      = geomean_ratio(per_shape_base_us, per_shape_gen_us)
roofline vs gen     = geomean_ratio(roofline_case_us, per_shape_gen_us)

No I/O, no deps beyond ``math`` — importable from both the core
``akg_agents`` package and ``workspace_autoresearch/scripts`` without cycle
risk."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence


def _finite(x) -> Optional[float]:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def mean_us(values: Sequence[float]) -> Optional[float]:
    """Arithmetic mean of the finite entries; ``None`` when none are finite.
    The canonical latency aggregate."""
    nums = [f for f in (_finite(v) for v in values) if f is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def geomean(values: Sequence[float]) -> Optional[float]:
    """Geometric mean of strictly-positive finite values. ``None`` when the
    input is empty or holds any non-positive / non-finite value — callers
    treat ``None`` as 'no aggregate' and fall back to a scalar."""
    log_sum = 0.0
    n = 0
    for v in values:
        f = _finite(v)
        if f is None or f <= 0:
            return None
        log_sum += math.log(f)
        n += 1
    if n == 0:
        return None
    return math.exp(log_sum / n)


def per_shape_ratio(numer: Sequence[float],
                    denom: Sequence[float]) -> List[float]:
    """Element-wise ``numer[i] / denom[i]`` for storage in metrics.

    Keeps index alignment with ``per_shape_descs``: a shape whose numer/denom
    is non-positive becomes ``0.0`` (a visible 'no ratio' sentinel) rather
    than being dropped. Truncates to the shorter length. Use
    :func:`geomean_ratio` — not ``geomean`` over this list — for the
    aggregate, so those 0.0 sentinels don't void the whole number."""
    out: List[float] = []
    for a, b in zip(numer, denom):
        fa, fb = _finite(a), _finite(b)
        out.append(fa / fb if (fa is not None and fb is not None
                               and fa > 0 and fb > 0) else 0.0)
    return out


def geomean_ratio(numer: Sequence[float],
                  denom: Sequence[float]) -> Optional[float]:
    """THE speedup aggregate: geomean of per-shape ``numer[i] / denom[i]``.

    Only strictly-positive aligned pairs contribute, so one shape that failed
    to time (0 / inf) is skipped rather than voiding the whole ratio.
    Returns ``None`` when no valid pair remains — caller falls back to a
    scalar mean-ratio."""
    ratios = [fa / fb
              for a, b in zip(numer, denom)
              for fa in (_finite(a),) for fb in (_finite(b),)
              if fa is not None and fb is not None and fa > 0 and fb > 0]
    return geomean(ratios)
