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

"""Selector 抽象基类与共用数据结构。

设计契约：
    1. 训练侧（调优阶段）调 `Selector.fit(inputs) -> SelectorArtifact`；
    2. 部署侧只看 SelectorArtifact，按 `kind` 字段从 registry 找实现，
       调 `Selector.from_artifact(payload).select_index(shape) -> int`；
    3. 推理路径只能依赖 numpy，不能加载 sklearn 等重依赖（必要时由具体算法
       自行 lazy import 并在 manifest 里声明 runtime_deps）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SelectorTrainingInputs:
    """Selector 训练输入。

    - axis_names: shape 维度名（例如 ('M', 'N')）
    - shape_matrix: shape (n_shapes, n_axes) int 矩阵
    - latencies_us: shape (n_shapes, n_configs) float 矩阵
    - config_ids: 与 latencies_us 列对应的 config_id 列表
    """

    axis_names: tuple[str, ...]
    shape_matrix: np.ndarray
    latencies_us: np.ndarray
    config_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        n_axes = len(self.axis_names)
        n_configs = len(self.config_ids)
        if self.shape_matrix.ndim != 2 or self.shape_matrix.shape[1] != n_axes:
            raise ValueError(
                f"shape_matrix={self.shape_matrix.shape} 与 axis_names({n_axes}) 不一致"
            )
        if self.latencies_us.ndim != 2 or self.latencies_us.shape[1] != n_configs:
            raise ValueError(
                f"latencies_us={self.latencies_us.shape} 与 config_ids({n_configs}) 不一致"
            )
        if self.shape_matrix.shape[0] != self.latencies_us.shape[0]:
            raise ValueError(
                f"shape_matrix 行数={self.shape_matrix.shape[0]} 与 "
                f"latencies_us 行数={self.latencies_us.shape[0]} 不一致"
            )
        if not np.all(np.isfinite(self.latencies_us)):
            raise ValueError("latencies_us 存在非有限值")
        if np.any(self.latencies_us <= 0.0):
            raise ValueError("latencies_us 存在非正值")

    @property
    def n_shapes(self) -> int:
        return int(self.shape_matrix.shape[0])

    @property
    def n_configs(self) -> int:
        return int(self.latencies_us.shape[1])

    def log_latencies(self) -> np.ndarray:
        return np.log(self.latencies_us)


@dataclass(frozen=True)
class SelectorArtifact:
    """训练产物 + 部署需要的全部信息。"""

    kind: str
    axis_names: tuple[str, ...]
    config_ids: tuple[str, ...]
    payload: dict[str, Any]
    runtime_deps: tuple[str, ...] = field(default_factory=lambda: ("numpy",))
    model_bytes: bytes | None = None


class SelectorBase(ABC):
    """所有 selector 算法必须实现的接口。"""

    kind: str = ""

    @classmethod
    @abstractmethod
    def fit(cls, inputs: SelectorTrainingInputs) -> SelectorArtifact:
        """训练并返回部署 artifact。"""

    @classmethod
    @abstractmethod
    def from_artifact(cls, artifact: SelectorArtifact) -> "SelectorBase":
        """部署侧重建。"""

    @abstractmethod
    def select_index(self, shape: tuple[int, ...]) -> int:
        """按 shape 选 config 列下标。"""


__all__ = [
    "SelectorArtifact",
    "SelectorBase",
    "SelectorTrainingInputs",
]
