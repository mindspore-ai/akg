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

"""决策树贪心切分 selector。

核心思想：
    - 训练目标：min sum_over_shapes log(latency_at_chosen_config(shape))
    - 每次贪心选一对 (axis, threshold)，把 shapes 分成 left (axis<=threshold) 与
      right (axis>threshold) 两组；
    - 每组在自己组内 argmin 一个共同 config，子代价 = sum log_latency；
    - 选总代价最低的切分，递归 left / right；
    - 终止条件：到达 max_depth、节点 case 太少、或 split 带来的增益小于 min_gain。

推理：沿 (split_axis, threshold) 走到叶子，叶子里直接存 config_index。

实现只用 numpy；序列化为 JSON-ready 的嵌套 dict，部署侧不需要重训。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from akg_agents.op.dynamic_tune.selector.base import (
    SelectorArtifact,
    SelectorBase,
    SelectorTrainingInputs,
)
from akg_agents.op.dynamic_tune.selector.registry import register_selector


DEFAULT_MAX_DEPTH = 8
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_GAIN = 1e-6


@register_selector
class TreeSelector(SelectorBase):
    kind = "tree"

    def __init__(
        self,
        *,
        axis_names: tuple[str, ...],
        config_ids: tuple[str, ...],
        root: Mapping[str, Any],
    ) -> None:
        self._axis_names = tuple(str(name) for name in axis_names)
        self._axis_index = {name: idx for idx, name in enumerate(self._axis_names)}
        self._config_ids = tuple(str(name) for name in config_ids)
        self._root = _normalize_node(root)

    @classmethod
    def fit(cls, inputs: SelectorTrainingInputs) -> SelectorArtifact:
        log_latency = inputs.log_latencies()
        case_indices = tuple(range(inputs.n_shapes))
        root = _build_node(
            case_indices=case_indices,
            shape_matrix=inputs.shape_matrix,
            log_latency=log_latency,
            axis_names=inputs.axis_names,
            depth=0,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
            min_gain=DEFAULT_MIN_GAIN,
        )
        payload = {"root": root}
        return SelectorArtifact(
            kind=cls.kind,
            axis_names=tuple(inputs.axis_names),
            config_ids=tuple(inputs.config_ids),
            payload=payload,
            runtime_deps=("numpy",),
            model_bytes=None,
        )

    @classmethod
    def from_artifact(cls, artifact: SelectorArtifact) -> "TreeSelector":
        if artifact.kind != cls.kind:
            raise ValueError(f"artifact.kind={artifact.kind!r} 不是 {cls.kind!r}")
        root = artifact.payload.get("root")
        if not isinstance(root, Mapping):
            raise ValueError("payload.root 缺失或非 mapping")
        return cls(
            axis_names=artifact.axis_names,
            config_ids=artifact.config_ids,
            root=root,
        )

    def select_index(self, shape: tuple[int, ...]) -> int:
        if len(shape) != len(self._axis_names):
            raise ValueError(
                f"shape 维度={len(shape)} 与 axis_names={len(self._axis_names)} 不一致"
            )
        values_by_axis = {
            name: int(value) for name, value in zip(self._axis_names, shape)
        }
        node: Mapping[str, Any] = self._root
        while "config_index" not in node:
            split_axis = str(node["split_axis"])
            threshold = int(node["threshold"])
            value = values_by_axis[split_axis]
            node = node["left"] if value <= threshold else node["right"]
            if not isinstance(node, Mapping):
                raise RuntimeError("tree 节点结构异常：非叶子节点缺 left/right")
        return int(node["config_index"])


def _normalize_node(node: Mapping[str, Any]) -> dict[str, Any]:
    """递归把节点 dict 标准化（int/str 强制转换）。"""

    if "config_index" in node:
        return {"config_index": int(node["config_index"])}
    return {
        "split_axis": str(node["split_axis"]),
        "threshold": int(node["threshold"]),
        "left": _normalize_node(node["left"]),
        "right": _normalize_node(node["right"]),
    }


def _leaf_cost(
    case_indices: tuple[int, ...], log_latency: np.ndarray
) -> tuple[int, float]:
    """该 case 集合下的最优 config_index 与总 log latency。"""

    if not case_indices:
        return -1, float("inf")
    rows = log_latency[np.asarray(case_indices, dtype=np.int64), :]
    per_config = rows.sum(axis=0)
    best = int(np.argmin(per_config))
    return best, float(per_config[best])


def _build_node(
    *,
    case_indices: tuple[int, ...],
    shape_matrix: np.ndarray,
    log_latency: np.ndarray,
    axis_names: tuple[str, ...],
    depth: int,
    max_depth: int,
    min_samples_split: int,
    min_gain: float,
) -> dict[str, Any]:
    leaf_config_index, leaf_cost = _leaf_cost(case_indices, log_latency)
    if depth >= max_depth or len(case_indices) < min_samples_split:
        return {"config_index": int(leaf_config_index)}
    best = _find_best_split(
        case_indices=case_indices,
        shape_matrix=shape_matrix,
        log_latency=log_latency,
        axis_names=axis_names,
    )
    if best is None:
        return {"config_index": int(leaf_config_index)}
    split_axis, axis_index, threshold, left_indices, right_indices, split_cost = best
    if leaf_cost - split_cost < min_gain:
        return {"config_index": int(leaf_config_index)}
    left_node = _build_node(
        case_indices=left_indices,
        shape_matrix=shape_matrix,
        log_latency=log_latency,
        axis_names=axis_names,
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_gain=min_gain,
    )
    right_node = _build_node(
        case_indices=right_indices,
        shape_matrix=shape_matrix,
        log_latency=log_latency,
        axis_names=axis_names,
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_gain=min_gain,
    )
    return {
        "split_axis": str(split_axis),
        "threshold": int(threshold),
        "left": left_node,
        "right": right_node,
    }


def _find_best_split(
    *,
    case_indices: tuple[int, ...],
    shape_matrix: np.ndarray,
    log_latency: np.ndarray,
    axis_names: tuple[str, ...],
) -> tuple[str, int, int, tuple[int, ...], tuple[int, ...], float] | None:
    """在所有 (axis, threshold) 中找代价最小的切分。"""

    case_array = np.asarray(case_indices, dtype=np.int64)
    if case_array.size < 2:
        return None
    best: tuple[str, int, int, tuple[int, ...], tuple[int, ...], float] | None = None
    for axis_index, axis_name in enumerate(axis_names):
        column = shape_matrix[case_array, axis_index]
        unique_values = np.unique(column)
        if unique_values.size < 2:
            continue
        candidate_thresholds = unique_values[:-1]
        for threshold in candidate_thresholds:
            mask_left = column <= threshold
            if not mask_left.any() or mask_left.all():
                continue
            left_indices = tuple(int(value) for value in case_array[mask_left])
            right_indices = tuple(int(value) for value in case_array[~mask_left])
            _, left_cost = _leaf_cost(left_indices, log_latency)
            _, right_cost = _leaf_cost(right_indices, log_latency)
            split_cost = left_cost + right_cost
            if best is None or split_cost < best[5]:
                best = (
                    str(axis_name),
                    int(axis_index),
                    int(threshold),
                    left_indices,
                    right_indices,
                    float(split_cost),
                )
    return best


__all__ = ["TreeSelector"]
