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

"""把 LatencyMatrix 转成 selector 可吃的训练输入。

为什么单独成文件：
    LatencyMatrix 是 measure 子包的产物（按 measure 自身的需要组织），而
    SelectorTrainingInputs 是 selector 子包的输入（按 ML 训练的需要组织）。
    把"过桥"逻辑放在 tune 层，保持两个子包的独立性。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from akg_agents.op.dynamic_tune.config import Config
from akg_agents.op.dynamic_tune.measure.batch_profiler import LatencyMatrix
from akg_agents.op.dynamic_tune.selector.base import SelectorTrainingInputs


@dataclass(frozen=True)
class PolicyDatasetEntry:
    shape: tuple[int, ...]
    runtime_by_config_us: tuple[float, ...]


@dataclass(frozen=True)
class PolicyDataset:
    """训练侧友好的数据载体（带轴名 + config_id）。"""

    axis_names: tuple[str, ...]
    config_ids: tuple[str, ...]
    entries: tuple[PolicyDatasetEntry, ...]

    def __post_init__(self) -> None:
        if not self.axis_names:
            raise ValueError("axis_names 不能为空")
        if not self.config_ids:
            raise ValueError("config_ids 不能为空")
        if not self.entries:
            raise ValueError("entries 不能为空")
        for entry in self.entries:
            if len(entry.shape) != len(self.axis_names):
                raise ValueError("entry.shape 长度与 axis_names 不一致")
            if len(entry.runtime_by_config_us) != len(self.config_ids):
                raise ValueError("entry.runtime_by_config_us 长度与 config_ids 不一致")

    def to_training_inputs(self) -> SelectorTrainingInputs:
        shape_matrix = np.asarray(
            [entry.shape for entry in self.entries], dtype=np.int64
        )
        latencies = np.asarray(
            [entry.runtime_by_config_us for entry in self.entries], dtype=np.float64
        )
        return SelectorTrainingInputs(
            axis_names=self.axis_names,
            shape_matrix=shape_matrix.astype(np.float64),
            latencies_us=latencies,
            config_ids=self.config_ids,
        )


def build_policy_dataset(
    *,
    axis_names: tuple[str, ...],
    matrix: LatencyMatrix,
) -> PolicyDataset:
    """把 LatencyMatrix 投影成 PolicyDataset。

    matrix.configs 与 matrix.shapes 的顺序必须保持稳定，因为 selector 训练后
    内部记录的是"列下标"，部署侧再把列下标映射回 config_id。
    """

    if len(axis_names) != matrix.shapes[0].__len__():
        raise ValueError("axis_names 与 shapes 维度不一致")
    config_ids = tuple(config.config_id for config in matrix.configs)
    entries = tuple(
        PolicyDatasetEntry(
            shape=shape,
            runtime_by_config_us=tuple(float(value) for value in row),
        )
        for shape, row in zip(matrix.shapes, matrix.latencies_us)
    )
    return PolicyDataset(
        axis_names=tuple(str(name) for name in axis_names),
        config_ids=config_ids,
        entries=entries,
    )


def configs_for_dataset(matrix: LatencyMatrix) -> tuple[Config, ...]:
    return tuple(matrix.configs)


__all__ = [
    "PolicyDataset",
    "PolicyDatasetEntry",
    "build_policy_dataset",
    "configs_for_dataset",
]
