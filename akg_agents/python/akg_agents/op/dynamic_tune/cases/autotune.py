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

from __future__ import annotations

import inspect
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from akg_agents.op.dynamic_tune import tune_configs
from akg_agents.op.dynamic_tune.cases.case import _RuntimeBundle
from akg_agents.op.dynamic_tune.cases.device import _maybe_empty_npu_cache, _set_npu_device
from akg_agents.op.dynamic_tune.tune.diagnostics import (
    print_matrix_table,
    print_selector_decisions,
    summarize_matrix,
    summarize_selector,
)
from akg_agents.op.dynamic_tune.tune.spec import build_configs_from_autotune_spec


DEFAULT_TUNE_REPEAT = 10


@dataclass(frozen=True)
class _AutotuneResult:
    manifest_dir: Path
    log_root: Path
    dynamic_shapes: list[list[int]]
    tune_seconds: float
    matrix_summary: dict[str, Any]


class _AutotuneSession:
    def __init__(
        self,
        *,
        runtime_bundle: _RuntimeBundle,
        npu_device: str,
        cache_dir: Path,
        work_dir: Path,
    ) -> None:
        self._runtime_bundle = runtime_bundle
        self._npu_device = npu_device
        self._cache_dir = cache_dir
        self._work_dir = work_dir

    def run(self) -> _AutotuneResult:
        case_spec = self._runtime_bundle.case_spec
        self._bind_npu_device()
        self._clear_npu_cache()
        base_module = self._runtime_bundle.base_module
        init_params = list(base_module.get_init_inputs())
        train_shapes, cpu_inputs_by_shape = self._collect_cases(
            base_module=base_module,
            shape_expr=case_spec.shape_expr,
        )
        if not train_shapes:
            raise RuntimeError(
                f"base.get_inputs_dyn_list() 未返回任何 case: case={case_spec.name}"
            )
        runtime = self._runtime_bundle.runtime_module.ModelNew(*init_params).to(
            self._npu_device
        )
        configs, axis_names = build_configs_from_autotune_spec(case_spec.autotune_spec)
        log_root = self._work_dir / f"{self._runtime_bundle.case_spec.name}_kv_logs"
        log_root.mkdir(parents=True, exist_ok=True)
        print(
            "[autotune] start "
            f"case={case_spec.name} "
            f"device={self._npu_device} "
            f"repeat={DEFAULT_TUNE_REPEAT} "
            f"configs={len(configs)} "
            f"train_shapes={self._format_shapes(train_shapes)} "
            f"cache_dir={self._cache_dir} "
            f"work_dir={self._work_dir}"
        )
        tune_started = time.perf_counter()
        try:
            outcome = tune_configs(
                axis_names=axis_names,
                shapes=train_shapes,
                configs=configs,
                module=runtime,
                inputs_by_shape=cpu_inputs_by_shape,
                cache_dir=self._cache_dir,
                device=self._npu_device,
                warmup=1,
                repeat=DEFAULT_TUNE_REPEAT,
            )
        finally:
            cpu_inputs_by_shape.clear()
            self._clear_npu_cache()
        tune_seconds = time.perf_counter() - tune_started

        manifest_path = Path(outcome.manifest_path)
        manifest_dir = (
            manifest_path.parent if manifest_path.name == "manifest.json" else manifest_path
        )
        # train==test：动态 shape 直接复用训练集，与 base.get_inputs_dyn_list() 一一对应。
        dynamic_shapes = [list(shape) for shape in train_shapes]

        matrix_summary = summarize_matrix(outcome.matrix)
        selector_summary = summarize_selector(outcome.manifest, outcome.matrix)
        matrix_summary["selector"] = selector_summary
        print_matrix_table(case_spec.name, outcome.matrix, matrix_summary)
        print_selector_decisions(case_spec.name, selector_summary)

        print(
            "[autotune] done "
            f"case={case_spec.name} "
            f"manifest_dir={manifest_dir} "
            f"log_root={log_root} "
            f"tune_seconds={tune_seconds:.3f}s "
            f"dynamic_shapes={self._format_shapes(dynamic_shapes)}"
        )
        return _AutotuneResult(
            manifest_dir=manifest_dir,
            log_root=log_root,
            dynamic_shapes=dynamic_shapes,
            tune_seconds=tune_seconds,
            matrix_summary=matrix_summary,
        )

    def _collect_cases(
        self,
        *,
        base_module: types.ModuleType,
        shape_expr: str,
    ) -> tuple[tuple[tuple[int, ...], ...], dict[tuple[int, ...], tuple]]:
        """从 base.get_inputs_dyn_list() 收集所有 case，按 sample.shape_expr 推 shape key。

        - 张量保持在 CPU，由 `_make_input_cache` 在 launch 前按 shape 懒迁到 NPU，避免一次性 OOM。
        - shape_expr 在 `forward` 形参绑定到对应 tensor 的命名空间下 eval；与 sample.json 中给的字符串保持一致。
        """
        param_names = self._forward_param_names(base_module)
        compiled_shape_expr = compile(shape_expr, f"<shape_expr:{shape_expr}>", "eval")
        eval_globals = {"__builtins__": {"int": int, "len": len, "tuple": tuple}}
        cases: list[tuple[tuple[int, ...], tuple]] = []
        seen: set[tuple[int, ...]] = set()
        for case_idx, inputs in enumerate(base_module.get_inputs_dyn_list()):
            tensors = tuple(inputs)
            if len(tensors) != len(param_names):
                raise RuntimeError(
                    "Model.forward 形参个数与 get_inputs_dyn_list 第 "
                    f"{case_idx} 组不匹配: forward={param_names} inputs={len(tensors)}"
                )
            namespace = dict(zip(param_names, tensors))
            shape_value = eval(compiled_shape_expr, eval_globals, namespace)
            shape = tuple(int(v) for v in shape_value)
            if shape in seen:
                raise RuntimeError(
                    f"shape_expr 在第 {case_idx} 组上与已有 case 重复: shape={shape}"
                )
            seen.add(shape)
            cases.append((shape, tensors))
        train_shapes = tuple(shape for shape, _ in cases)
        cpu_inputs_by_shape = {shape: tensors for shape, tensors in cases}
        return train_shapes, cpu_inputs_by_shape

    @staticmethod
    def _forward_param_names(base_module: types.ModuleType) -> list[str]:
        signature = inspect.signature(base_module.Model.forward)
        return [
            name
            for name, param in signature.parameters.items()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and name != "self"
        ]

    def _bind_npu_device(self) -> None:
        _set_npu_device(self._npu_device)

    def _clear_npu_cache(self) -> None:
        try:
            import torch  # type: ignore
        except ImportError:
            return
        self._bind_npu_device()
        _maybe_empty_npu_cache(torch)

    @staticmethod
    def _format_shapes(shapes) -> str:
        return (
            "[" + ", ".join(str(tuple(int(v) for v in shape)) for shape in shapes) + "]"
        )
