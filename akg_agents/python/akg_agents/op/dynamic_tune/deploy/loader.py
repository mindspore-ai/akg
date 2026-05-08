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

"""部署侧 selector 加载与 shape→config 推理分发。

入口：
    DeployedSelector(manifest)
        .select_config(shape: Mapping | Sequence) -> Config

懒构造 selector instance：第一次 select 时通过 registry 找到对应算法类，
调用 from_artifact 重建。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from akg_agents.op.dynamic_tune.config import Config
from akg_agents.op.dynamic_tune.deploy.locator import manifest_dir_for_caller
from akg_agents.op.dynamic_tune.deploy.manifest import Manifest, load_manifest
from akg_agents.op.dynamic_tune.selector.base import SelectorArtifact, SelectorBase
from akg_agents.op.dynamic_tune.selector.registry import resolve_selector

ShapeInput = Mapping[str, int] | Sequence[int]


@dataclass
class DeployedSelector:
    """绑定一份 manifest 的运行时选择器。"""

    manifest: Manifest
    _selector: SelectorBase | None = None

    def axis_names(self) -> tuple[str, ...]:
        return self.manifest.axis_names

    def select_config(self, shape: ShapeInput) -> Config:
        shape_values = self._normalize_shape(shape)
        selector = self._ensure_selector()
        index = int(selector.select_index(shape_values))
        config_ids = self.manifest.selector.config_ids
        if not config_ids:
            raise RuntimeError("manifest.selector.config_ids 为空，无法定位 config")
        if not 0 <= index < len(config_ids):
            raise RuntimeError(
                f"selector.select_index 返回越界下标={index}，size={len(config_ids)}"
            )
        return self.manifest.find_config_by_id(config_ids[index])

    def _normalize_shape(self, shape: ShapeInput) -> tuple[int, ...]:
        axis_names = self.manifest.axis_names
        if isinstance(shape, Mapping):
            return tuple(int(shape[axis]) for axis in axis_names)
        try:
            values = tuple(int(value) for value in shape)
        except TypeError as exc:
            raise TypeError(
                "shape 必须是按 axis_names 排列的序列，或 {axis: value} 映射"
            ) from exc
        if len(values) != len(axis_names):
            raise ValueError(
                f"shape 维度={len(values)} 与 axis_names={len(axis_names)} 不一致"
            )
        return values

    def _ensure_selector(self) -> SelectorBase:
        if self._selector is not None:
            return self._selector
        selector_cls = resolve_selector(self.manifest.selector.kind)
        artifact = SelectorArtifact(
            kind=self.manifest.selector.kind,
            axis_names=self.manifest.axis_names,
            config_ids=self.manifest.selector.config_ids,
            payload=dict(self.manifest.selector.payload),
            runtime_deps=self.manifest.selector.runtime_deps,
            model_bytes=None,
        )
        instance = selector_cls.from_artifact(artifact)
        self._selector = instance
        return instance


def load_deployed_selector(cache_dir: str | Path | None = None) -> DeployedSelector:
    if cache_dir is None:
        cache_dir = manifest_dir_for_caller()
    return DeployedSelector(manifest=load_manifest(cache_dir))


__all__ = ["DeployedSelector", "ShapeInput", "load_deployed_selector"]
