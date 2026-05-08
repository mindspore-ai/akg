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

"""Build tune inputs from declarative autotune specs."""

from __future__ import annotations

from typing import Any, Mapping

from akg_agents.op.dynamic_tune.config import Config


def build_configs_from_autotune_spec(
    autotune_spec: Mapping[str, Any],
) -> tuple[tuple[Config, ...], tuple[str, ...]]:
    """Build candidate configs and axis names from a sample autotune spec."""

    if "configs" in autotune_spec:
        configs = [Config(dict(params)) for params in autotune_spec["configs"]]
    elif "sweep_param" in autotune_spec:
        sweep_param = str(autotune_spec["sweep_param"])
        if "candidates" not in autotune_spec:
            raise ValueError(f"sweep autotune 必须显式给 candidates: {autotune_spec}")
        candidates = [int(v) for v in autotune_spec["candidates"]]
        if not candidates:
            raise ValueError(f"sweep autotune candidates 不能为空: {autotune_spec}")
        configs = [Config({sweep_param: int(c)}) for c in candidates]
    else:
        raise ValueError(
            f"autotune spec 必须提供 configs 或 sweep_param+candidates: {autotune_spec}"
        )

    return tuple(configs), tuple(str(name) for name in autotune_spec["key"])


__all__ = ["build_configs_from_autotune_spec"]
