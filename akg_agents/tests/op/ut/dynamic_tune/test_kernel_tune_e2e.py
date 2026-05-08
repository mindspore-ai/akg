"""端到端：tune_configs（mock backend） → manifest → 模拟新进程加载推理。

验证目标：
    1. 调优阶段 manifest.json 正确写到 cache_dir。
    2. 模拟新进程：从同一 cache_dir 加载 selector，不触发任何 benchmark，
       按 shape 选出对应 config。
"""

from __future__ import annotations

from typing import Sequence

from akg_agents.op.dynamic_tune import Config, load_deployed_selector, tune_configs
from akg_agents.op.dynamic_tune.deploy.manifest import manifest_exists, load_manifest
from akg_agents.op.dynamic_tune.measure.batch_profiler import (
    CaptureRequest,
    CaptureResult,
)


class _ScriptedBackend:
    """按 (shape × config) 顺序返回设定的 latency 序列。

    config 顺序按 configs 列表（cfg0=BLOCK_M=64, cfg1=BLOCK_M=128）。
    构造时给一个 latency_matrix shape=(n_shapes, n_configs)。
    """

    def __init__(self, latency_matrix):
        self._matrix = latency_matrix
        self.captured: list[CaptureRequest] = []

    def capture(self, request: CaptureRequest) -> CaptureResult:
        self.captured.append(request)
        flat = []
        for row in self._matrix:
            flat.extend(float(value) for value in row)
        if len(flat) != len(request.step_launches):
            raise AssertionError(
                f"backend script 长度={len(flat)} 与请求 launches={len(request.step_launches)} 不一致"
            )
        return CaptureResult(step_latencies_us=tuple(flat), note="scripted")


class _Module:
    def __call__(self, *inputs, config):
        return None


def test_e2e_tune_then_load_and_select(tmp_path):
    shapes = [(64,), (128,), (256,), (512,)]
    # 调优阶段：让 cfg0 在 M<=128 更快、cfg1 在 M>=256 更快
    backend = _ScriptedBackend(
        latency_matrix=[
            [1.0, 9.0],
            [1.0, 9.0],
            [9.0, 1.0],
            [9.0, 1.0],
        ]
    )
    outcome = tune_configs(
        axis_names=("M",),
        shapes=shapes,
        configs=[Config({"BLOCK_M": 64}), Config({"BLOCK_M": 128})],
        module=_Module(),
        inputs_by_shape={shape: (object(),) for shape in shapes},
        cache_dir=tmp_path,
        profiler_backend=backend,
        repeat=1,
    )
    assert outcome.matrix.path_used == "a"
    assert manifest_exists(tmp_path)

    manifest = load_manifest(tmp_path)
    assert manifest.selector.kind == "tree"
    assert {cand.config.config_id for cand in manifest.candidates} == {
        "block_m64",
        "block_m128",
    }

    selector = load_deployed_selector(tmp_path)
    assert selector.select_config((64,)).config_id == "block_m64"
    assert selector.select_config((512,)).config_id == "block_m128"


def test_e2e_compile_gate_drops_failing_config(tmp_path):
    shapes = [(64,), (512,)]

    class _CompileGateModule:
        def __call__(self, *inputs, config):
            if config.param("BLOCK_M") == 99999:
                raise RuntimeError("local memory exceeds limit")

    backend = _ScriptedBackend(
        latency_matrix=[[1.0, 5.0], [5.0, 1.0]]  # 只剩 cfg0 / cfg2 两个 config
    )

    outcome = tune_configs(
        axis_names=("M",),
        shapes=shapes,
        configs=[
            Config({"BLOCK_M": 64}),
            Config({"BLOCK_M": 99999}),  # 我们让它 sanity 失败
            Config({"BLOCK_M": 128}),
        ],
        module=_CompileGateModule(),
        inputs_by_shape={shape: (object(),) for shape in shapes},
        cache_dir=tmp_path,
        profiler_backend=backend,
        repeat=1,
    )
    rejected_ids = [
        cand.config.config_id
        for cand in outcome.manifest.candidates
        if cand.status == "rejected"
    ]
    assert rejected_ids == ["block_m99999"]
    assert outcome.manifest.selector.config_ids == ("block_m64", "block_m128")

    selector = load_deployed_selector(tmp_path)
    assert selector.select_config((64,)).config_id == "block_m64"
    assert selector.select_config((512,)).config_id == "block_m128"
