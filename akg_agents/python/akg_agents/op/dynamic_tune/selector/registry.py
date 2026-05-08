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

"""Selector 算法注册表。

新算法接入只需：
    1. 写一个 SelectorBase 子类实现 kind/fit/from_artifact/select_index；
    2. 在模块顶层调用 register_selector(cls)；
    3. 在 selector/__init__.py 里 import 一下触发注册。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from akg_agents.op.dynamic_tune.selector.base import SelectorBase


_REGISTRY: dict[str, type["SelectorBase"]] = {}


def register_selector(cls: type["SelectorBase"]) -> type["SelectorBase"]:
    kind = str(getattr(cls, "kind", "")).strip()
    if not kind:
        raise ValueError(f"selector class {cls.__name__} 未设置 kind")
    if kind in _REGISTRY and _REGISTRY[kind] is not cls:
        raise ValueError(f"selector kind={kind!r} 已被 {_REGISTRY[kind].__name__} 占用")
    _REGISTRY[kind] = cls
    return cls


def resolve_selector(kind: str) -> type["SelectorBase"]:
    normalized = str(kind).strip()
    if normalized not in _REGISTRY:
        raise KeyError(
            f"未注册的 selector kind={normalized!r}，已注册：{sorted(_REGISTRY)}"
        )
    return _REGISTRY[normalized]


def list_selectors() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


__all__ = ["register_selector", "resolve_selector", "list_selectors"]
