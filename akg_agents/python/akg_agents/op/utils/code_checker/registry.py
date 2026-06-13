# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Global CodeChecker registry and YAML selection helpers."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from akg_agents.op.utils.code_checker.base import CodeCheckerUnit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckerSpec:
    """A globally registered CodeChecker unit."""

    name: str
    group: str
    factory: Callable[[], CodeCheckerUnit]


class CheckerRegistry:
    """Registry used by YAML config to select CodeChecker pipelines."""

    def __init__(self):
        self._groups: Dict[str, "OrderedDict[str, CheckerSpec]"] = {}

    def register(self, spec: CheckerSpec) -> None:
        group = spec.group.strip().lower()
        name = spec.name.strip().lower()
        if not group or not name:
            raise ValueError("CheckerSpec group/name must be non-empty")
        group_specs = self._groups.setdefault(group, OrderedDict())
        if name in group_specs:
            raise ValueError(f"Duplicate CodeChecker registration: {group}.{name}")
        normalized = CheckerSpec(
            name=name,
            group=group,
            factory=spec.factory,
        )
        group_specs[name] = normalized

    def select(self, group: str, raw_value) -> List[CheckerSpec]:
        group = group.strip().lower()
        specs = self._groups.get(group, OrderedDict())
        default_order = list(specs)
        selected_names = self._normalize_selection(
            raw_value,
            group=group,
            default_order=default_order,
        )
        return [specs[name] for name in selected_names if name in specs]

    @staticmethod
    def _normalize_selection(
        raw_value,
        *,
        group: str,
        default_order: Iterable[str],
    ) -> List[str]:
        default_order = list(default_order)

        if raw_value is None:
            return default_order
        if isinstance(raw_value, bool):
            return default_order if raw_value else []
        if isinstance(raw_value, str):
            raw_items = [raw_value]
        else:
            try:
                raw_items = list(raw_value)
            except TypeError:
                raw_items = [raw_value]

        if not raw_items:
            return []

        normalized = []
        for item in raw_items:
            name = str(item).strip().lower()
            if not name:
                continue
            if name == "all":
                return default_order
            if name in {"none", "off", "false", "disabled"}:
                continue
            normalized.append(name)

        selected = set()
        for name in normalized:
            if name not in default_order:
                logger.warning("Unknown CodeChecker %s entry ignored: %s", group, name)
                continue
            selected.add(name)
        return [name for name in default_order if name in selected]


def checker_group_config(config: Optional[dict], group_name: str):
    code_checker_cfg = (config or {}).get("code_checker", {}) or {}
    if isinstance(code_checker_cfg, dict) and group_name in code_checker_cfg:
        return code_checker_cfg.get(group_name)
    return None


def build_checker_registry(*, backend: str, dsl: str, config: Optional[dict]) -> CheckerRegistry:
    """Build the default global registry for this CodeChecker instance."""
    from akg_agents.op.utils.code_checker.base_checkers import register_base_checkers
    from akg_agents.op.utils.code_checker.triton_checkers import register_triton_checkers

    registry = CheckerRegistry()
    register_base_checkers(registry, backend=backend, dsl=dsl, config=config)
    register_triton_checkers(registry, backend=backend, dsl=dsl, config=config)
    return registry
