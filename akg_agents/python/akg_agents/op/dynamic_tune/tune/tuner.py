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

"""调优编排：把 compile_gate → batch_measure → selector.fit → manifest.dump 串起来。

核心入口：
    tune_configs(...) -> TuneOutcome

``tune_configs`` 是面向调用方的显式 API；测量层需要的 launch callable
由本模块根据 ModelNew 和 inputs_by_shape 生成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from akg_agents.op.dynamic_tune.config import Config, dedupe_configs
from akg_agents.op.dynamic_tune.deploy.manifest import (
    Manifest,
    SelectorPayload,
    TuneMeta,
    build_candidates,
    dump_manifest,
)
from akg_agents.op.dynamic_tune.measure.batch_profiler import (
    BatchProfiler,
    LatencyMatrix,
    MeasurementBackend,
)
from akg_agents.op.dynamic_tune.measure.compile_gate import (
    CompileGate,
    CompileGateOutcome,
    CompileGateRejection,
)
from akg_agents.op.dynamic_tune.selector.base import SelectorBase
from akg_agents.op.dynamic_tune.selector.registry import resolve_selector
from akg_agents.op.dynamic_tune.tune.runtime_matrix import build_policy_dataset

Shape = tuple[int, ...]
LaunchFactory = Callable[[Shape, Config], Callable[[], None]]
ModuleInputsByShape = Mapping[Shape, Sequence[Any]]


@dataclass(frozen=True)
class _TuneRequest:
    axis_names: tuple[str, ...]
    shapes: tuple[Shape, ...]
    configs: tuple[Config, ...]
    launch_factory: LaunchFactory
    profiler: BatchProfiler
    cache_dir: Path
    selector_kind: str = "tree"
    sanity_shape: Shape | None = None
    compile_gate: CompileGate | None = None
    warmup: int = 1
    repeat: int = 1
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TuneOutcome:
    manifest: Manifest
    manifest_path: Path
    matrix: LatencyMatrix
    rejections: tuple[CompileGateRejection, ...]


def tune_configs(
    *,
    axis_names: Sequence[str],
    shapes: Sequence[Sequence[int]],
    configs: Sequence[Config],
    module: Any,
    inputs_by_shape: ModuleInputsByShape,
    cache_dir: str | Path,
    selector: str = "tree",
    profiler_backend: MeasurementBackend | None = None,
    device: Any = None,
    warmup: int = 1,
    repeat: int = 1,
    sanity_shape: Sequence[int] | None = None,
    compile_gate: CompileGate | None = None,
    oom_detector: Callable[[BaseException], bool] | None = None,
    between_launch_hook: Callable[[], None] | None = None,
    between_shape_hook: Callable[[Shape], None] | None = None,
    extras: dict | None = None,
) -> TuneOutcome:
    """显式执行一次动态 shape 调优并写出 manifest。

    这是 dynamic_tune 的 public 调优入口。调用方需要提供候选配置、代表性
    shape、待调优的 ModelNew 实例，以及每个 shape 对应的一组输入。
    """

    normalized_axis_names = tuple(str(name) for name in axis_names)
    normalized_shapes = tuple(tuple(int(value) for value in shape) for shape in shapes)
    normalized_inputs_by_shape = _normalize_inputs_by_shape(inputs_by_shape)
    normalized_sanity_shape = (
        tuple(int(value) for value in sanity_shape)
        if sanity_shape is not None
        else None
    )
    _validate_inputs_cover_shapes(
        shapes=normalized_shapes,
        sanity_shape=normalized_sanity_shape,
        inputs_by_shape=normalized_inputs_by_shape,
    )

    if profiler_backend is None:
        if device is None:
            raise ValueError("profiler_backend=None 时必须传 device")
        from akg_agents.op.dynamic_tune.measure.batch_profiler import (
            NpuProfilerBackend,
        )

        profiler_backend = NpuProfilerBackend(device=device)

    _bind_current_device(device)
    if compile_gate is None:
        compile_gate = CompileGate(synchronize=_make_device_synchronize(device))

    profiler = BatchProfiler(
        backend=profiler_backend,
        oom_detector=oom_detector,
        between_launch_hook=between_launch_hook,
        between_shape_hook=between_shape_hook,
    )
    launch_factory = _make_module_launch_factory(
        module=module,
        inputs_by_shape=normalized_inputs_by_shape,
        device=device,
    )
    request = _TuneRequest(
        axis_names=normalized_axis_names,
        shapes=normalized_shapes,
        configs=tuple(dedupe_configs(configs)),
        launch_factory=launch_factory,
        profiler=profiler,
        cache_dir=Path(cache_dir).expanduser(),
        selector_kind=str(selector),
        sanity_shape=normalized_sanity_shape,
        compile_gate=compile_gate,
        warmup=int(warmup),
        repeat=int(repeat),
        extras=dict(extras or {}),
    )
    return _run_tuning(request)


def _run_tuning(request: _TuneRequest) -> TuneOutcome:
    if not request.shapes:
        raise ValueError("shapes 不能为空")
    if not request.configs:
        raise ValueError("configs 不能为空")
    if not request.axis_names:
        raise ValueError("axis_names 不能为空")

    sanity_shape: Shape = request.sanity_shape or request.shapes[0]

    gate = request.compile_gate or CompileGate()
    gate_outcome: CompileGateOutcome = gate.filter(
        configs=request.configs,
        sanity_shape=sanity_shape,
        launch_factory=request.launch_factory,
    )

    surviving_configs = gate_outcome.kept_configs
    if not surviving_configs:
        raise RuntimeError(
            "compile_gate 剔除了全部 config；reasons=" + _format_rejections(gate_outcome)
        )

    matrix = request.profiler.measure(
        shapes=request.shapes,
        configs=surviving_configs,
        launch_factory=request.launch_factory,
        repeat=max(1, int(request.repeat)),
    )

    selector_cls = resolve_selector(request.selector_kind)
    selector_artifact = _fit_selector(
        selector_cls=selector_cls,
        axis_names=request.axis_names,
        matrix=matrix,
    )

    rejected_map = {
        rec.config.config_id: rec.reason for rec in gate_outcome.rejections
    }
    candidates = build_candidates(
        all_configs=request.configs, rejected=rejected_map
    )
    selector_payload = SelectorPayload(
        kind=selector_artifact.kind,
        payload=dict(selector_artifact.payload),
        runtime_deps=selector_artifact.runtime_deps,
        config_ids=selector_artifact.config_ids,
    )
    tune_meta = TuneMeta(
        path_used=matrix.path_used,
        warmup=int(request.warmup),
        repeat=int(request.repeat),
        notes=tuple(matrix.notes),
    )
    manifest = Manifest(
        axis_names=request.axis_names,
        candidates=candidates,
        selector=selector_payload,
        tune_meta=tune_meta,
        extras=dict(request.extras),
    )
    manifest_path = dump_manifest(manifest, request.cache_dir)
    return TuneOutcome(
        manifest=manifest,
        manifest_path=manifest_path,
        matrix=matrix,
        rejections=gate_outcome.rejections,
    )


def _fit_selector(
    *,
    selector_cls: type[SelectorBase],
    axis_names: tuple[str, ...],
    matrix: LatencyMatrix,
):
    dataset = build_policy_dataset(axis_names=axis_names, matrix=matrix)
    return selector_cls.fit(dataset.to_training_inputs())


def _format_rejections(outcome: CompileGateOutcome) -> str:
    return "; ".join(
        f"{rec.config.config_id}={rec.reason}" for rec in outcome.rejections
    )


def _normalize_inputs_by_shape(inputs_by_shape: ModuleInputsByShape) -> dict[Shape, tuple[Any, ...]]:
    return {
        tuple(int(value) for value in shape): tuple(inputs)
        for shape, inputs in inputs_by_shape.items()
    }


def _validate_inputs_cover_shapes(
    *,
    shapes: Sequence[Shape],
    sanity_shape: Shape | None,
    inputs_by_shape: Mapping[Shape, tuple[Any, ...]],
) -> None:
    required = set(shapes)
    if sanity_shape is not None:
        required.add(sanity_shape)
    missing = sorted(shape for shape in required if shape not in inputs_by_shape)
    if missing:
        raise ValueError(f"inputs_by_shape 缺少 shape: {missing}")


def _make_module_launch_factory(
    *,
    module: Any,
    inputs_by_shape: Mapping[Shape, tuple[Any, ...]],
    device: Any,
) -> LaunchFactory:
    cached: dict[Shape, tuple[Any, ...]] = {}

    def _inputs_for_shape(shape: Shape) -> tuple[Any, ...]:
        key = tuple(int(value) for value in shape)
        if key not in inputs_by_shape:
            raise KeyError(f"shape={key} 不在 inputs_by_shape 中")
        if key not in cached:
            cached[key] = tuple(_move_to_device(item, device) for item in inputs_by_shape[key])
        return cached[key]

    def _launch_factory(shape: Shape, config: Config) -> Callable[[], None]:
        positional = _inputs_for_shape(shape)

        def _launch() -> None:
            module(*positional, config=config)

        return _launch

    return _launch_factory


def _move_to_device(value: Any, device: Any) -> Any:
    if device is None or not hasattr(value, "to"):
        return value
    return value.to(device)


def _bind_current_device(device: Any) -> None:
    """把 torch 当前 device 绑到 ``device``；NPU/CUDA 都覆盖。"""

    if device is None:
        return
    try:
        import torch  # type: ignore
    except ImportError:
        return
    backend = str(device).split(":", 1)[0].strip().lower()
    setter = None
    if backend == "npu":
        setter = getattr(getattr(torch, "npu", None), "set_device", None)
    elif backend == "cuda":
        setter = getattr(getattr(torch, "cuda", None), "set_device", None)
    if setter is not None:
        setter(device)


def _make_device_synchronize(device: Any) -> Callable[[], None] | None:
    if device is None:
        return None
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    device_str = str(device)
    backend = device_str.split(":", 1)[0].strip().lower()
    if backend == "npu":
        sync = getattr(getattr(torch, "npu", None), "synchronize", None)
    elif backend == "cuda":
        sync = getattr(getattr(torch, "cuda", None), "synchronize", None)
    else:
        sync = None
    if sync is None:
        return None

    def _sync() -> None:
        sync(device_str)  # type: ignore[misc]

    return _sync


__all__ = [
    "TuneOutcome",
    "tune_configs",
]
