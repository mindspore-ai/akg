"""Tree selector：训练 + from_artifact 重建后能正确推理。"""

from __future__ import annotations

import numpy as np

from akg_agents.op.dynamic_tune.selector.base import SelectorTrainingInputs
from akg_agents.op.dynamic_tune.selector.registry import resolve_selector


def test_tree_selector_picks_globally_best_when_no_split_helps():
    inputs = SelectorTrainingInputs(
        axis_names=("M",),
        shape_matrix=np.asarray([[64], [128], [256]], dtype=np.float64),
        # cfg0 永远是最快的：tree 应该退化成单叶子选 cfg0。
        latencies_us=np.asarray(
            [
                [1.0, 5.0, 9.0],
                [1.0, 5.0, 9.0],
                [1.0, 5.0, 9.0],
            ],
            dtype=np.float64,
        ),
        config_ids=("cfg0", "cfg1", "cfg2"),
    )
    selector_cls = resolve_selector("tree")
    artifact = selector_cls.fit(inputs)
    selector = selector_cls.from_artifact(artifact)
    assert selector.select_index((64,)) == 0
    assert selector.select_index((128,)) == 0
    assert selector.select_index((256,)) == 0


def test_tree_selector_splits_along_axis_when_better():
    """让 cfg0 在小 M 下最快、cfg1 在大 M 下最快，验证 tree 切出来正确。"""

    inputs = SelectorTrainingInputs(
        axis_names=("M",),
        shape_matrix=np.asarray([[64], [128], [256], [512]], dtype=np.float64),
        latencies_us=np.asarray(
            [
                [1.0, 5.0],
                [1.0, 5.0],
                [9.0, 1.0],
                [9.0, 1.0],
            ],
            dtype=np.float64,
        ),
        config_ids=("cfg_small", "cfg_large"),
    )
    selector_cls = resolve_selector("tree")
    artifact = selector_cls.fit(inputs)
    selector = selector_cls.from_artifact(artifact)
    assert selector.select_index((64,)) == 0
    assert selector.select_index((128,)) == 0
    assert selector.select_index((256,)) == 1
    assert selector.select_index((1024,)) == 1


def test_tree_selector_two_axes_split_picks_best_axis():
    inputs = SelectorTrainingInputs(
        axis_names=("M", "N"),
        shape_matrix=np.asarray(
            [[64, 64], [64, 256], [256, 64], [256, 256]], dtype=np.float64
        ),
        # 让 N 维度最具区分度：N<=64 cfg0 快；N>64 cfg1 快
        latencies_us=np.asarray(
            [
                [1.0, 9.0],
                [9.0, 1.0],
                [1.0, 9.0],
                [9.0, 1.0],
            ],
            dtype=np.float64,
        ),
        config_ids=("cfg_n_small", "cfg_n_large"),
    )
    selector_cls = resolve_selector("tree")
    artifact = selector_cls.fit(inputs)
    selector = selector_cls.from_artifact(artifact)
    # 第一刀必须切在 N 上。
    assert artifact.payload["root"]["split_axis"] == "N"
    assert selector.select_index((128, 32)) == 0
    assert selector.select_index((128, 1024)) == 1


def test_tree_artifact_payload_is_json_serializable():
    import json

    inputs = SelectorTrainingInputs(
        axis_names=("M",),
        shape_matrix=np.asarray([[64], [256]], dtype=np.float64),
        latencies_us=np.asarray([[1.0, 9.0], [9.0, 1.0]], dtype=np.float64),
        config_ids=("a", "b"),
    )
    artifact = resolve_selector("tree").fit(inputs)
    encoded = json.dumps(artifact.payload)
    decoded = json.loads(encoded)
    assert decoded == artifact.payload
