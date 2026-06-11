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

"""Shared task.yaml <-> dsl_config mapping for project-backed DSLs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class _FieldSpec:
    flat_key: str
    yaml_keys: tuple[str, ...]
    yaml_key: str
    default_when_block_present: Optional[str] = None


@dataclass(frozen=True)
class _BlockSpec:
    block_name: str
    dsl_names: tuple[str, ...]
    fields: tuple[_FieldSpec, ...]
    project_dir_keys: tuple[str, ...]


_BLOCK_SPECS = (
    _BlockSpec(
        block_name="catlass",
        dsl_names=("ascendc_catlass",),
        fields=(
            _FieldSpec("catlass_root", ("root", "catlass_root"), "root"),
            _FieldSpec(
                "catlass_op_dir",
                ("op_dir", "catlass_op_dir"),
                "op_dir",
                "catlass_op",
            ),
        ),
        project_dir_keys=("catlass_op_dir",),
    ),
    _BlockSpec(
        block_name="ascendc",
        dsl_names=("ascendc",),
        fields=(
            _FieldSpec(
                "ascendc_op_dir",
                (
                    "op_dir",
                    "ascendc_op_dir",
                    "project_dir",
                    "ascendc_project_dir",
                ),
                "op_dir",
                "ascendc_op",
            ),
        ),
        project_dir_keys=("ascendc_op_dir", "ascendc_project_dir"),
    ),
)


def _spec_for_dsl(dsl: Optional[str]) -> Optional[_BlockSpec]:
    if not dsl:
        return None
    for spec in _BLOCK_SPECS:
        if dsl in spec.dsl_names:
            return spec
    return None


def flatten_task_yaml_dsl_blocks(
    raw: Mapping[str, Any],
    *,
    yaml_path: Optional[str] = None,
) -> dict:
    """Flatten explicit per-DSL task.yaml blocks into adapter config keys.

    Blocks are opt-in: no ``catlass:`` / ``ascendc:`` block means no flat
    config key is emitted, leaving adapter defaults as the source of truth.
    """
    dsl_config: dict = {}
    for spec in _BLOCK_SPECS:
        if spec.block_name not in raw:
            continue
        block = raw.get(spec.block_name) or {}
        if not isinstance(block, Mapping):
            prefix = f"{yaml_path}: " if yaml_path else ""
            raise ValueError(
                f"{prefix}'{spec.block_name}' must be a mapping"
            )
        for field in spec.fields:
            value = None
            for yaml_key in field.yaml_keys:
                if block.get(yaml_key) is not None:
                    value = block[yaml_key]
                    break
            if value is None:
                value = field.default_when_block_present
            if value is not None:
                dsl_config[field.flat_key] = value
    return dsl_config


def task_yaml_dsl_blocks(
    dsl: Optional[str],
    dsl_config: Optional[Mapping[str, Any]],
) -> dict:
    """Build explicit task.yaml DSL blocks from flat ``dsl_config`` keys."""
    spec = _spec_for_dsl(dsl)
    if spec is None or not dsl_config:
        return {}
    block: dict = {}
    for field in spec.fields:
        value = dsl_config.get(field.flat_key)
        if value is not None:
            block[field.yaml_key] = value
    return {spec.block_name: block} if block else {}


def project_dir_from_dsl_config(
    dsl: Optional[str],
    dsl_config: Optional[Mapping[str, Any]],
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    """Return the configured project subdirectory for a directory-backed DSL."""
    spec = _spec_for_dsl(dsl)
    if spec is not None and dsl_config:
        for key in spec.project_dir_keys:
            value = dsl_config.get(key)
            if value:
                return str(value)
    return default
