"""dynamic_tune public API smoke tests."""

from __future__ import annotations

import inspect

import akg_agents.op.dynamic_tune as txa


def test_public_api_exists():
    assert hasattr(txa, "Config")
    assert hasattr(txa, "tune_configs")
    assert hasattr(txa, "load_deployed_selector")
    assert not hasattr(txa, "autotune")
    assert not hasattr(txa, "KernelConfig")


def test_config_constructor_compat():
    cfg = txa.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3)
    assert cfg.param("BLOCK_M") == 64
    assert cfg.param("BLOCK_N") == 64
    assert cfg.num_warps == 8
    assert cfg.num_stages == 3
    assert cfg.config_id == "block_m64_block_n64"


def test_tune_configs_signature_is_explicit_not_decorator_based():
    sig = inspect.signature(txa.tune_configs)
    params = sig.parameters
    for name in [
        "axis_names",
        "shapes",
        "configs",
        "module",
        "inputs_by_shape",
        "cache_dir",
        "selector",
        "warmup",
        "repeat",
    ]:
        assert name in params, f"tune_configs 缺少形参 {name}"
