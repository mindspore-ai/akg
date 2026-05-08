"""批量测时：a 主路径正常工作；a 触发 OOM 时退化到 b 路径。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from akg_agents.op.dynamic_tune.config import Config
from akg_agents.op.dynamic_tune.measure.batch_profiler import (
    BatchProfiler,
    BatchProfilerOOMError,
    CaptureRequest,
    CaptureResult,
    LatencyMatrix,
    MeasurementBackend,
    NpuProfilerBackend,
    _default_oom_detector,
)
from akg_agents.op.dynamic_tune.measure.parse_export_db import ExportDbNotFoundError


def _config(name: str):
    return Config({"BLOCK_M": 64}, config_id=name)


class _RecordingBackend:
    """简单 mock backend：根据 launch 数量返回 1.0..N us 的递增 latency。"""

    def __init__(self) -> None:
        self.requests: list[CaptureRequest] = []

    def capture(self, request: CaptureRequest) -> CaptureResult:
        self.requests.append(request)
        latencies = tuple(float(idx + 1) for idx in range(len(request.step_launches)))
        return CaptureResult(step_latencies_us=latencies, note=request.label)


class _FirstCallOOMBackend:
    """第一次调用抛 OOM，第二次起按数量返回 latency。"""

    def __init__(self) -> None:
        self.requests: list[CaptureRequest] = []
        self._has_failed = False

    def capture(self, request: CaptureRequest) -> CaptureResult:
        self.requests.append(request)
        if not self._has_failed:
            self._has_failed = True
            raise BatchProfilerOOMError("simulated NPU OOM")
        latencies = tuple(float(idx + 1) for idx in range(len(request.step_launches)))
        return CaptureResult(step_latencies_us=latencies, note=request.label)


def _launch_factory(shape, config):
    return lambda: None


def test_path_a_happy_path():
    backend = _RecordingBackend()
    profiler = BatchProfiler(backend=backend)
    shapes = [(128,), (256,)]
    configs = [_config("c0"), _config("c1"), _config("c2")]

    matrix = profiler.measure(
        shapes=shapes, configs=configs, launch_factory=_launch_factory
    )
    assert isinstance(matrix, LatencyMatrix)
    assert matrix.path_used == "a"
    assert matrix.shapes == ((128,), (256,))
    assert matrix.configs == tuple(configs)
    assert matrix.latencies_us.shape == (2, 3)
    assert len(backend.requests) == 1
    assert len(backend.requests[0].step_launches) == 6


def test_path_b_fallback_on_oom():
    backend = _FirstCallOOMBackend()
    profiler = BatchProfiler(backend=backend)
    shapes = [(128,), (256,), (512,)]
    configs = [_config("c0"), _config("c1")]

    matrix = profiler.measure(
        shapes=shapes, configs=configs, launch_factory=_launch_factory
    )
    assert matrix.path_used == "b"
    # 一次 a 失败，再加 N 次 b：合计 N + 1 个 capture 请求。
    assert len(backend.requests) == 1 + len(shapes)
    # 后续 b 路径每个 request 大小 = M（configs）。
    for request in backend.requests[1:]:
        assert len(request.step_launches) == len(configs)
    assert matrix.latencies_us.shape == (3, 2)


def test_default_oom_detector_recognizes_common_patterns():
    assert _default_oom_detector(BatchProfilerOOMError("x")) is True
    assert _default_oom_detector(ExportDbNotFoundError("export_db_missing:x")) is True
    assert _default_oom_detector(RuntimeError("device out of memory")) is True
    assert _default_oom_detector(RuntimeError("npu OOM detected")) is True
    assert _default_oom_detector(RuntimeError("export_db_missing:ascend_task.db")) is True
    assert _default_oom_detector(RuntimeError("compile error")) is False


def test_non_oom_exception_propagates():
    class _AlwaysFailBackend:
        def capture(self, request: CaptureRequest) -> CaptureResult:
            raise ValueError("logic bug")

    profiler = BatchProfiler(backend=_AlwaysFailBackend())
    with pytest.raises(ValueError, match="logic bug"):
        profiler.measure(
            shapes=[(1,)], configs=[_config("c")], launch_factory=_launch_factory
        )


def test_between_launch_hook_is_called_per_step():
    counter = {"n": 0}

    def hook() -> None:
        counter["n"] += 1

    backend = _RecordingBackend()
    profiler = BatchProfiler(backend=backend, between_launch_hook=hook)
    profiler.measure(
        shapes=[(1,), (2,)],
        configs=[_config("c0"), _config("c1")],
        launch_factory=_launch_factory,
    )
    # 触发 step_launches 时每个 wrapped launch 调用一次 hook。
    # 注意：BatchProfiler._wrap_launch 把 hook 装到每个 callable 里，
    # 真正调用 hook 的是 backend；mock backend 不会真的 invoke 这些 callable，
    # 因此这里只验证 launches 个数（间接验证 wrap 正确）。
    assert counter["n"] == 0
    # 手工 invoke 一次确认 wrap 行为。
    backend.requests[0].step_launches[0]()
    assert counter["n"] == 1


def test_path_a_with_repeat_takes_median_per_entry():
    """每个 (shape, config) repeat 3 次，backend 返回 [1,2,3]→1, [4,5,6]→5,
    [7,8,9]→8 ……：验证 reshape (n_shapes, n_configs, repeat) 后取 median 正确。"""

    class _IncrementingBackend:
        def capture(self, request: CaptureRequest) -> CaptureResult:
            latencies = tuple(float(idx + 1) for idx in range(len(request.step_launches)))
            return CaptureResult(step_latencies_us=latencies)

    profiler = BatchProfiler(backend=_IncrementingBackend())
    shapes = [(1,), (2,)]
    configs = [_config("c0"), _config("c1")]
    matrix = profiler.measure(
        shapes=shapes, configs=configs, launch_factory=_launch_factory, repeat=3
    )
    # launches: shape0×c0 ×3 (1,2,3 → median 2), shape0×c1 ×3 (4,5,6 → 5),
    #           shape1×c0 ×3 (7,8,9 → 8), shape1×c1 ×3 (10,11,12 → 11)
    expected = np.asarray([[2.0, 5.0], [8.0, 11.0]], dtype=np.float64)
    np.testing.assert_array_equal(matrix.latencies_us, expected)
    assert matrix.path_used == "a"


def test_path_b_with_repeat_per_shape_capture_returns_median():
    """path b 每个 shape 单独 capture，每 shape 内有 M*repeat 个 launch。"""

    class _PerShapeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def capture(self, request: CaptureRequest) -> CaptureResult:
            self.calls += 1
            if self.calls == 1:
                raise BatchProfilerOOMError("force fallback to b")
            base = (self.calls - 2) * 100
            latencies = tuple(float(base + idx + 1) for idx in range(len(request.step_launches)))
            return CaptureResult(step_latencies_us=latencies)

    profiler = BatchProfiler(backend=_PerShapeBackend())
    shapes = [(1,), (2,)]
    configs = [_config("c0"), _config("c1")]
    matrix = profiler.measure(
        shapes=shapes, configs=configs, launch_factory=_launch_factory, repeat=3
    )
    assert matrix.path_used == "b"
    # shape 0: backend 第二次调用 (calls=2)，base=0，latencies=[1..6]
    #   c0: median(1,2,3)=2; c1: median(4,5,6)=5
    # shape 1: 第三次调用 (calls=3)，base=100，latencies=[101..106]
    #   c0: median(101,102,103)=102; c1: median(104,105,106)=105
    expected = np.asarray([[2.0, 5.0], [102.0, 105.0]], dtype=np.float64)
    np.testing.assert_array_equal(matrix.latencies_us, expected)


def test_path_b_runs_between_shape_hook_per_shape():
    backend = _RecordingBackend()
    seen_shapes: list[tuple] = []

    def shape_hook(shape: tuple) -> None:
        seen_shapes.append(shape)

    # 强制走 b 路径：oom_detector 一直返回 True。
    class _AlwaysOOMBackend:
        def __init__(self) -> None:
            self.calls = 0

        def capture(self, request: CaptureRequest) -> CaptureResult:
            self.calls += 1
            if self.calls == 1:
                raise BatchProfilerOOMError("force fallback")
            latencies = tuple(
                float(idx + 1) for idx in range(len(request.step_launches))
            )
            return CaptureResult(step_latencies_us=latencies)

    profiler = BatchProfiler(
        backend=_AlwaysOOMBackend(), between_shape_hook=shape_hook
    )
    profiler.measure(
        shapes=[(1,), (2,)],
        configs=[_config("c0")],
        launch_factory=_launch_factory,
    )
    assert seen_shapes == [(1,), (2,)]


def test_path_b_fallback_on_export_db_missing():
    class _FirstCallExportDbMissingBackend:
        def __init__(self) -> None:
            self.requests: list[CaptureRequest] = []
            self._has_failed = False

        def capture(self, request: CaptureRequest) -> CaptureResult:
            self.requests.append(request)
            if not self._has_failed:
                self._has_failed = True
                raise ExportDbNotFoundError("export_db_missing:ascend_task.db")
            latencies = tuple(float(idx + 1) for idx in range(len(request.step_launches)))
            return CaptureResult(step_latencies_us=latencies, note=request.label)

    backend = _FirstCallExportDbMissingBackend()
    profiler = BatchProfiler(backend=backend)
    shapes = [(128,), (256,)]
    configs = [_config("c0"), _config("c1")]

    matrix = profiler.measure(
        shapes=shapes, configs=configs, launch_factory=_launch_factory
    )
    assert matrix.path_used == "b"
    assert len(backend.requests) == 1 + len(shapes)


def test_npu_profiler_backend_uses_constructor_settings():
    backend = NpuProfilerBackend(
        device="npu:0",
        keep_raw=True,
        export_db_file_timeout_seconds=120,
    )

    assert backend._keep_raw is True
    assert backend._export_db_file_timeout_seconds == pytest.approx(120.0)
