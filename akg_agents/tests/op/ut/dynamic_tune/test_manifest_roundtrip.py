"""manifest.json 写入 → 读出 → DeployedSelector 推理 端到端。"""

from __future__ import annotations

import json
import importlib.util

import numpy as np

from akg_agents.op.dynamic_tune.config import Config
from akg_agents.op.dynamic_tune.deploy import locator
from akg_agents.op.dynamic_tune.deploy.loader import (
    DeployedSelector,
    load_deployed_selector,
)
from akg_agents.op.dynamic_tune.deploy.manifest import (
    Manifest,
    SelectorPayload,
    TuneMeta,
    build_candidates,
    dump_manifest,
    load_manifest,
    manifest_exists,
)
from akg_agents.op.dynamic_tune.selector.base import SelectorTrainingInputs
from akg_agents.op.dynamic_tune.selector.registry import resolve_selector


def _build_manifest():
    cfg_small = Config({"BLOCK_M": 64}, config_id="cfg_small")
    cfg_large = Config({"BLOCK_M": 128}, config_id="cfg_large")
    rejected = Config({"BLOCK_M": 9999}, config_id="cfg_huge")

    inputs = SelectorTrainingInputs(
        axis_names=("M",),
        shape_matrix=np.asarray([[64], [128], [256], [512]], dtype=np.float64),
        latencies_us=np.asarray(
            [[1.0, 9.0], [1.0, 9.0], [9.0, 1.0], [9.0, 1.0]], dtype=np.float64
        ),
        config_ids=("cfg_small", "cfg_large"),
    )
    artifact = resolve_selector("tree").fit(inputs)

    candidates = build_candidates(
        all_configs=[cfg_small, cfg_large, rejected],
        rejected={"cfg_huge": "RuntimeError: local memory exceeds limit"},
    )
    return Manifest(
        axis_names=("M",),
        candidates=candidates,
        selector=SelectorPayload(
            kind=artifact.kind,
            payload=dict(artifact.payload),
            runtime_deps=artifact.runtime_deps,
            config_ids=artifact.config_ids,
        ),
        tune_meta=TuneMeta(path_used="a", warmup=1, repeat=1, notes=("ok",)),
    )


def test_manifest_round_trip_keeps_all_fields(tmp_path):
    manifest = _build_manifest()
    path = dump_manifest(manifest, tmp_path)
    assert manifest_exists(tmp_path)
    assert path.is_file()

    raw = json.loads(path.read_text("utf-8"))
    assert raw["schema_version"] == 1
    assert raw["axis_names"] == ["M"]
    assert raw["selector"]["kind"] == "tree"

    loaded = load_manifest(tmp_path)
    assert loaded.axis_names == manifest.axis_names
    assert loaded.selector.kind == manifest.selector.kind
    assert loaded.selector.config_ids == manifest.selector.config_ids
    assert loaded.tune_meta.path_used == "a"
    rejected = [
        cand for cand in loaded.candidates if cand.status == "rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0].config.config_id == "cfg_huge"
    assert "local memory" in rejected[0].reject_reason


def test_deployed_selector_picks_correct_config_for_shape(tmp_path):
    manifest = _build_manifest()
    dump_manifest(manifest, tmp_path)
    selector = load_deployed_selector(tmp_path)
    assert isinstance(selector, DeployedSelector)
    assert selector.select_config((64,)).config_id == "cfg_small"
    assert selector.select_config((1024,)).config_id == "cfg_large"
    # mapping 形态也支持
    assert selector.select_config({"M": 64}).config_id == "cfg_small"


def test_deployed_selector_can_load_from_source_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(locator, "default_manifest_root", lambda: tmp_path / "manifests")
    module_source = """from akg_agents.op.dynamic_tune import load_deployed_selector

def pick(shape):
    return load_deployed_selector().select_config(shape).config_id
"""
    module_path = tmp_path / "runtime_model.py"
    module_path.write_text(module_source, encoding="utf-8")
    dump_manifest(_build_manifest(), locator.manifest_dir_for_source_text(module_source))

    spec = importlib.util.spec_from_file_location("runtime_model", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.pick((64,)) == "cfg_small"
    assert module.pick((1024,)) == "cfg_large"


def test_manifest_rejects_selector_referencing_unkept_config(tmp_path):
    """selector.config_ids 必须全部在 kept candidates 内，否则构造直接报错。"""

    import pytest

    cfg = Config({"BLOCK_M": 64}, config_id="cfg_kept")
    candidates = build_candidates(all_configs=[cfg], rejected={})
    selector = SelectorPayload(
        kind="tree",
        payload={"root": {"config_index": 0}},
        runtime_deps=("numpy",),
        config_ids=("cfg_not_in_candidates",),
    )
    with pytest.raises(ValueError, match="不在 kept candidates"):
        Manifest(
            axis_names=("M",),
            candidates=candidates,
            selector=selector,
            tune_meta=TuneMeta(path_used="a", warmup=1, repeat=1),
        )
