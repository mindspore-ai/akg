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

"""批量测时核心：N×M 个 (shape, config) launch 共享一次 NPU profiler 启停。

业务接口：
    BatchProfiler.measure(shapes, configs, launch_factory) -> LatencyMatrix

实现要点：
    - 主路径 a：把所有 (shape, config) launch 拼成一个 step_launches 列表，
      在一次 profile 上下文里跑完，profile 启停成本 = 1。
    - 回退 b：捕获到 OOM-like 异常时，按 shape 分组退化——每个 shape 一次 profile，
      该 shape 下所有 config launch 共享。同 shape 下 payload 不变，显存占用
      固定不会再涨。
    - 每个 launch 之间清 L2 cache + synchronize + prof.step()，避免互相蹭热度。
    - 测时后端通过 MeasurementBackend 协议注入；NPU 真实后端是
      NpuProfilerBackend，单测可以注入 mock backend。

不依赖 triton_perf.* 或 akg_agents.*，所有外部依赖（torch / torch_npu）都是 lazy。
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from akg_agents.op.dynamic_tune.config import Config

Shape = tuple[int, ...]
LaunchFactory = Callable[[Shape, Config], Callable[[], None]]


class BatchProfilerOOMError(RuntimeError):
    """显式标记 a 路径触发 OOM、需要退化到 b 路径。"""


@dataclass(frozen=True)
class CaptureRequest:
    step_launches: tuple[Callable[[], None], ...]
    label: str = ""


@dataclass(frozen=True)
class CaptureResult:
    step_latencies_us: tuple[float, ...]
    note: str = ""


class MeasurementBackend(Protocol):
    """测时后端协议。

    实现需要保证返回 step_latencies_us 长度等于 step_launches 长度（不含
    profiler 自带的 warmup 那一段）。
    """

    def capture(self, request: CaptureRequest) -> CaptureResult:  # pragma: no cover
        ...


@dataclass(frozen=True)
class LatencyMatrix:
    """(shape × config) → latency 矩阵。"""

    shapes: tuple[Shape, ...]
    configs: tuple[Config, ...]
    latencies_us: np.ndarray  # shape: (n_shapes, n_configs)
    path_used: str  # "a" 或 "b"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n_shapes = len(self.shapes)
        n_configs = len(self.configs)
        if self.latencies_us.shape != (n_shapes, n_configs):
            raise ValueError(
                f"latencies_us shape={self.latencies_us.shape} 与 (shapes={n_shapes},configs={n_configs}) 不一致"
            )
        if not np.all(np.isfinite(self.latencies_us)):
            raise ValueError("latencies_us 存在非有限值")
        if np.any(self.latencies_us <= 0.0):
            raise ValueError("latencies_us 存在非正值")


def _default_oom_detector(exc: BaseException) -> bool:
    """识别"显存不足"系列异常。

    NPU 上 OOM 可能以多种形式抛出：torch.npu.OutOfMemoryError、
    torch.cuda.OutOfMemoryError 等；我们用类名宽匹配，避免硬依赖。
    BatchProfilerOOMError 由调用方手工抛出，也算 OOM。

    除了 OOM，path_a 还可能因为单次 profile payload 过大，导致 msprof export-db
    根本拿不到 device 侧 task 数据。这类错误在退化到 path_b（每个 shape 单独
    profile）后通常可以恢复，因此默认也纳入 fallback 条件。
    """

    if isinstance(exc, BatchProfilerOOMError):
        return True
    name = type(exc).__name__.lower()
    if name in {"exportdbnotfounderror", "exportdbparseerror"}:
        return True
    if "outofmemory" in name or name.endswith("oomerror"):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message or "export_db_" in message


class BatchProfiler:
    """批量测时编排器。"""

    def __init__(
        self,
        *,
        backend: MeasurementBackend,
        oom_detector: Callable[[BaseException], bool] | None = None,
        between_launch_hook: Callable[[], None] | None = None,
        between_shape_hook: Callable[[Shape], None] | None = None,
    ) -> None:
        self._backend = backend
        self._oom_detector = oom_detector or _default_oom_detector
        self._between_launch_hook = between_launch_hook
        self._between_shape_hook = between_shape_hook

    def measure(
        self,
        *,
        shapes: Sequence[Shape],
        configs: Sequence[Config],
        launch_factory: LaunchFactory,
        repeat: int = 1,
    ) -> LatencyMatrix:
        """跑批量测时；repeat>1 时每个 (shape, config) 重复 launch repeat 次，
        最后取每条的中位数，显著降低 NPU 测时噪声。"""

        if not shapes:
            raise ValueError("shapes 不能为空")
        if not configs:
            raise ValueError("configs 不能为空")
        normalized_repeat = max(1, int(repeat))
        normalized_shapes = tuple(tuple(int(v) for v in shape) for shape in shapes)
        normalized_configs = tuple(configs)
        try:
            return self._measure_path_a(
                normalized_shapes,
                normalized_configs,
                launch_factory,
                repeat=normalized_repeat,
            )
        except BaseException as exc:
            if not self._oom_detector(exc):
                raise
            return self._measure_path_b(
                normalized_shapes,
                normalized_configs,
                launch_factory,
                repeat=normalized_repeat,
            )

    def _measure_path_a(
        self,
        shapes: tuple[Shape, ...],
        configs: tuple[Config, ...],
        launch_factory: LaunchFactory,
        *,
        repeat: int,
    ) -> LatencyMatrix:
        n_shapes = len(shapes)
        n_configs = len(configs)
        flat_launches: list[Callable[[], None]] = []
        for shape in shapes:
            for config in configs:
                base_launch = launch_factory(shape, config)
                wrapped = self._wrap_launch(base_launch)
                for _ in range(repeat):
                    flat_launches.append(wrapped)
        request = CaptureRequest(
            step_launches=tuple(flat_launches),
            label=f"path_a:n_shapes={n_shapes}:n_configs={n_configs}:repeat={repeat}",
        )
        result = self._backend.capture(request)
        latencies_array = np.asarray(result.step_latencies_us, dtype=np.float64)
        expected = n_shapes * n_configs * repeat
        if latencies_array.size != expected:
            raise RuntimeError(
                f"path_a step_latencies 长度异常: got={latencies_array.size}"
                f" expected={expected}"
            )
        latencies_array = latencies_array.reshape(n_shapes, n_configs, repeat)
        latencies_array = np.median(latencies_array, axis=2)
        return LatencyMatrix(
            shapes=shapes,
            configs=configs,
            latencies_us=latencies_array,
            path_used="a",
            notes=(result.note,) if result.note else (),
        )

    def _measure_path_b(
        self,
        shapes: tuple[Shape, ...],
        configs: tuple[Config, ...],
        launch_factory: LaunchFactory,
        *,
        repeat: int,
    ) -> LatencyMatrix:
        n_shapes = len(shapes)
        n_configs = len(configs)
        latencies_array = np.zeros((n_shapes, n_configs), dtype=np.float64)
        notes: list[str] = []
        for shape_index, shape in enumerate(shapes):
            if self._between_shape_hook is not None:
                self._between_shape_hook(shape)
            shape_launches: list[Callable[[], None]] = []
            for config in configs:
                base_launch = launch_factory(shape, config)
                wrapped = self._wrap_launch(base_launch)
                for _ in range(repeat):
                    shape_launches.append(wrapped)
            request = CaptureRequest(
                step_launches=tuple(shape_launches),
                label=f"path_b:shape_index={shape_index}:repeat={repeat}",
            )
            result = self._backend.capture(request)
            row = np.asarray(result.step_latencies_us, dtype=np.float64)
            expected = n_configs * repeat
            if row.size != expected:
                raise RuntimeError(
                    f"path_b step_latencies 长度异常: shape_index={shape_index}"
                    f" got={row.size} expected={expected}"
                )
            row = row.reshape(n_configs, repeat)
            latencies_array[shape_index, :] = np.median(row, axis=1)
            if result.note:
                notes.append(f"shape_index={shape_index};{result.note}")
        return LatencyMatrix(
            shapes=shapes,
            configs=configs,
            latencies_us=latencies_array,
            path_used="b",
            notes=tuple(notes),
        )

    def _wrap_launch(
        self, base_launch: Callable[[], None]
    ) -> Callable[[], None]:
        hook = self._between_launch_hook

        if hook is None:
            return base_launch

        def _wrapped() -> None:
            hook()
            base_launch()

        return _wrapped


# === 真实 NPU 后端 ===========================================================
#
# NpuProfilerBackend 在没有 torch_npu 的纯逻辑环境下不会被构造，所有真实依赖
# 都在方法体里 lazy import。


class NpuProfilerBackend:
    """基于 torch_npu.profiler + msprof export-db 的真实测时后端。"""

    def __init__(
        self,
        *,
        device: Any,
        keep_raw: bool = False,
        export_db_file_timeout_seconds: float = 30.0,
        enable_l2_cache_clear: bool = True,
    ) -> None:
        self._device = device
        self._keep_raw = bool(keep_raw)
        self._export_db_file_timeout_seconds = float(export_db_file_timeout_seconds)
        # 默认 on：L2 clear 是测时正确性的必选项——同时它产生的 launch 也是
        # parse_export_db 切 step 的边界标记。**不要随便关掉**，关了之后解析层
        # 会拿不到边界，整批 step 会被聚成一个 group。
        self._enable_l2_cache_clear = bool(enable_l2_cache_clear)

    def capture(self, request: CaptureRequest) -> CaptureResult:
        if not request.step_launches:
            raise ValueError("NpuProfilerBackend.capture 收到空 step_launches")
        from torch_npu import profiler  # type: ignore  # noqa: PLC0415

        prof_root = Path(tempfile.mkdtemp(prefix="txa_npu_prof_"))
        active_steps = len(request.step_launches)
        try:
            self._run_profile(profiler, prof_root, active_steps, request.step_launches)
            durations = self._parse_durations(prof_root)
        except Exception:
            if not self._keep_raw:
                shutil.rmtree(prof_root, ignore_errors=True)
            raise
        if not self._keep_raw:
            shutil.rmtree(prof_root, ignore_errors=True)
        if len(durations) != active_steps + 1:
            raise RuntimeError(
                f"export_db_total_count_mismatch:got={len(durations)}"
                f":expected={active_steps + 1}"
            )
        active_durations = durations[1:]
        return CaptureResult(
            step_latencies_us=tuple(float(value) for value in active_durations),
            note=f"npu_profiler_export_db;{request.label}",
        )

    def _run_profile(
        self,
        profiler: Any,
        prof_root: Path,
        active_steps: int,
        step_launches: Sequence[Callable[[], None]],
    ) -> None:
        import torch  # type: ignore

        activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
        sched = profiler.schedule(wait=0, warmup=1, active=active_steps, repeat=1)
        trace_cb = profiler.tensorboard_trace_handler(
            dir_name=str(prof_root),
            analyse_flag=False,
            async_mode=False,
        )
        exp_cfg = profiler._ExperimentalConfig(  # noqa: SLF001
            export_type=profiler.ExportType.Db
        )
        clearer = self._maybe_build_clearer()
        # 进入 profiler 之前就切到目标 device，确保：
        # 1) Triton launch 不会偷偷落到默认卡；
        # 2) L2 clear kernel 与 synchronize 都绑定同一张卡；
        # 3) 导出的 profiling 数据与 request.device 一致。
        with torch.npu.device(self._device):  # type: ignore[attr-defined]
            with profiler.profile(
                activities=activities,
                schedule=sched,
                on_trace_ready=trace_cb,
                experimental_config=exp_cfg,
            ) as prof:
                self._run_step_loop(prof, step_launches, clearer)

    def _maybe_build_clearer(self) -> Any:
        if not self._enable_l2_cache_clear:
            return None
        from akg_agents.op.dynamic_tune.measure.l2_cache import L2CacheClearer

        return L2CacheClearer(device=self._device)

    def _run_step_loop(
        self,
        prof: Any,
        step_launches: Sequence[Callable[[], None]],
        clearer: Any,
    ) -> None:
        import torch  # type: ignore

        if clearer is not None:
            clearer.clear()
        step_launches[0]()
        torch.npu.synchronize(self._device)  # type: ignore[attr-defined]
        prof.step()
        for launch in step_launches:
            if clearer is not None:
                clearer.clear()
            launch()
            torch.npu.synchronize(self._device)  # type: ignore[attr-defined]
            prof.step()

    def _parse_durations(self, prof_root: Path) -> tuple[float, ...]:
        from akg_agents.op.dynamic_tune.measure.parse_export_db import (
            parse_export_db_dir,
            run_msprof_export_db,
        )

        profiler_run_dir = self._resolve_profiler_run_dir(prof_root)
        run_msprof_export_db(profiler_run_dir)
        durations = parse_export_db_dir(
            profiler_run_dir,
            db_file_timeout_seconds=self._export_db_file_timeout_seconds,
        )
        return durations.warmup_and_active_us

    def _resolve_profiler_run_dir(self, prof_root: Path) -> Path:
        if (prof_root / "profiler_info.json").is_file():
            return prof_root
        matches = sorted(
            path.parent for path in prof_root.rglob("profiler_info.json")
        )
        unique = list(dict.fromkeys(matches))
        if not unique:
            raise RuntimeError(f"profiler_run_dir_missing:{prof_root}")
        if len(unique) != 1:
            raise RuntimeError(f"profiler_run_dir_multiple:count={len(unique)}")
        return unique[0]


__all__ = [
    "BatchProfiler",
    "BatchProfilerOOMError",
    "CaptureRequest",
    "CaptureResult",
    "LatencyMatrix",
    "MeasurementBackend",
    "NpuProfilerBackend",
]
