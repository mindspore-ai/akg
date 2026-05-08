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

"""动态调优 config 数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Config:
    """单条 kernel 候选配置。

    用户写法：
        Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3)
    """

    params: Mapping[str, Any] | Iterable[tuple[str, Any]]
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    maxnreg: int | None = None
    pre_hook: Any = None
    config_id: str | None = field(default=None)

    def __post_init__(self) -> None:
        items = _param_items(self.params)
        config_id = str(self.config_id or _default_config_id(items)).strip()
        if not config_id:
            raise ValueError("config_id 不能为空")
        seen: set[str] = set()
        normalized: list[tuple[str, int]] = []
        for raw_name, raw_value in items:
            name = str(raw_name).strip()
            if not name:
                raise ValueError("config param 名称不能为空")
            if name in seen:
                raise ValueError(f"重复的 config param: {name}")
            seen.add(name)
            normalized.append((name, int(raw_value)))
        object.__setattr__(self, "config_id", config_id)
        object.__setattr__(self, "params", tuple(normalized))
        object.__setattr__(self, "num_warps", int(self.num_warps))
        object.__setattr__(self, "num_stages", int(self.num_stages))
        object.__setattr__(self, "num_ctas", int(self.num_ctas))
        if self.maxnreg is not None:
            object.__setattr__(self, "maxnreg", int(self.maxnreg))

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.params)

    def param(self, name: str) -> int:
        for param_name, value in self.params:
            if param_name == name:
                return int(value)
        raise KeyError(name)

    def signature(self) -> tuple[tuple[str, int], ...]:
        return tuple((name, int(value)) for name, value in self.params)

    def runtime_meta(self) -> dict[str, int | None]:
        return {
            "num_warps": int(self.num_warps),
            "num_stages": int(self.num_stages),
            "num_ctas": int(self.num_ctas),
            "maxnreg": None if self.maxnreg is None else int(self.maxnreg),
        }


def dedupe_configs(configs: Iterable[Config]) -> list[Config]:
    """按 (params signature, runtime_meta) 去重，保持原顺序。"""

    ordered: list[Config] = []
    seen: set[tuple[Any, ...]] = set()
    for config in configs:
        key = (config.signature(), tuple(sorted(config.runtime_meta().items())))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(config)
    return ordered


def _param_items(params: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> list[tuple[str, Any]]:
    if isinstance(params, Mapping):
        return list(params.items())
    return list(params)


def _default_config_id(items: Iterable[tuple[str, Any]]) -> str:
    """根据 (name=value) 生成稳定的默认 config_id。"""

    normalized = [(str(name), int(value)) for name, value in items]
    if not normalized:
        return "default"
    items = sorted(normalized)
    return "_".join(f"{name.lower()}{value}" for name, value in items)


__all__ = [
    "Config",
    "dedupe_configs",
]
