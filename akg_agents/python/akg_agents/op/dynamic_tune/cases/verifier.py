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

import asyncio
import inspect
import re
import time
import types
from pathlib import Path
from typing import Any, Mapping

from akg_agents.op.dynamic_tune.cases.case import _CaseSpec
from akg_agents.op.dynamic_tune.cases.device import _push_default_device


class _KernelVerifierRunner:
    _FUTURE_IMPORT_RE = re.compile(
        r"^\s*from\s+__future__\s+import\s+[^\n]*\n",
        re.MULTILINE,
    )

    def __init__(
        self,
        *,
        backend: str = "ascend",
        arch: str = "ascend910b4",
        dsl: str = "triton_ascend",
        verify_timeout: int = 900,
    ) -> None:
        self._backend = backend
        self._arch = arch
        self._dsl = dsl
        self._verify_timeout = verify_timeout

    def run(
        self,
        *,
        case_spec: _CaseSpec,
        impl_code_path: Path,
        npu_device: str,
        log_root: Path,
        profile: bool = True,
        profile_settings: Mapping[str, Any] | None = None,
        base_module: types.ModuleType | None = None,
        impl_module: types.ModuleType | None = None,
    ) -> dict[str, Any]:
        inline_capable = base_module is not None and impl_module is not None
        # batch profile 需要 base_module / impl_module；具备上下文时默认走本地 BatchProfiler。
        profile_batch = inline_capable

        framework_code = case_spec.base_path.read_text(encoding="utf-8")
        impl_code = impl_code_path.read_text(encoding="utf-8")
        return asyncio.run(
            self._run_async(
                case_spec=case_spec,
                op_name=case_spec.op_name,
                framework_code=framework_code,
                impl_code=impl_code,
                task_id=case_spec.name,
                npu_device=npu_device,
                log_root=log_root,
                profile=profile,
                profile_settings=profile_settings,
                base_module=base_module,
                impl_module=impl_module,
                profile_batch=profile_batch,
            )
        )

    def verifier_impl_source(self, impl_code: str) -> str:
        from akg_agents.op.verifier.adapters.factory import get_dsl_adapter  # type: ignore

        dsl_adapter = get_dsl_adapter(self._dsl)
        return dsl_adapter.get_import_statements("torch") + self._strip_future_imports(impl_code)

    async def _run_async(
        self,
        *,
        case_spec: _CaseSpec,
        op_name: str,
        framework_code: str,
        impl_code: str,
        task_id: str,
        npu_device: str,
        log_root: Path,
        profile: bool,
        profile_settings: Mapping[str, Any] | None,
        base_module: types.ModuleType | None,
        impl_module: types.ModuleType | None,
        profile_batch: bool,
    ) -> dict[str, Any]:
        device_id = self._device_id_from(npu_device)

        summary: dict[str, Any] = {
            "op_name": op_name,
            "task_id": task_id,
            "device": npu_device,
        }

        # ============ 精度比对 ============
        verify_started = time.perf_counter()
        print(f"[verify] case={op_name} mode=akg_kernel_verifier")
        worker = await self._ensure_local_worker(device_id=device_id)
        verifier = self._build_kernel_verifier(
            op_name=op_name,
            framework_code=self._strip_future_imports(framework_code),
            task_id=task_id,
            log_root=log_root,
            worker=worker,
        )
        task_info = {"coder_code": self._strip_future_imports(impl_code)}
        verify_passed, verify_log = await verifier.run(
            task_info,
            current_step=0,
            device_id=device_id,
        )
        if not verify_passed:
            print(
                f"[verify] FAIL case={op_name} verify_failed, continuing to profile"
            )
        else:
            summary["verify_passed"] = True
            summary["verifier_log_excerpt"] = verify_log[-1000:]
        verify_seconds = time.perf_counter() - verify_started
        print(
            f"[verify] done case={op_name} verify_seconds={verify_seconds:.3f}s"
        )

        # ============ 性能 profile ============
        profile_seconds: float | None = None
        if profile:
            kwargs = {"run_times": 3, "warmup_times": 1}
            if profile_settings:
                kwargs.update(dict(profile_settings))
            profile_started = time.perf_counter()
            if profile_batch:
                assert base_module is not None and impl_module is not None
                print(f"[profile] case={op_name} mode=batch_profiler")
                summary["profile"] = self._profile_batch(
                    case_spec=case_spec,
                    base_module=base_module,
                    impl_module=impl_module,
                    npu_device=npu_device,
                    profile_settings=kwargs,
                )
            else:
                print(f"[profile] case={op_name} mode=akg_kernel_verifier")
                worker = await self._ensure_local_worker(device_id=device_id)
                verifier = self._build_kernel_verifier(
                    op_name=op_name,
                    framework_code=self._strip_future_imports(framework_code),
                    task_id=task_id,
                    log_root=log_root,
                    worker=worker,
                )
                task_info = {"coder_code": self._strip_future_imports(impl_code)}
                profile_result = await verifier.run_profile(
                    task_info,
                    current_step=0,
                    device_id=device_id,
                    profile_settings=kwargs,
                )
                summary["profile"] = {
                    "gen_time_us": float(profile_result["gen_time"]),
                    "base_time_us": float(profile_result["base_time"]),
                    "speedup": float(profile_result["speedup"]),
                    "unique_dir": str(profile_result.get("unique_dir", "")),
                }
            profile_seconds = time.perf_counter() - profile_started
            print(
                f"[profile] done case={op_name} "
                f"profile_seconds={profile_seconds:.3f}s "
                f"speedup={summary['profile'].get('speedup', float('nan')):.2f}"
            )
        summary["timings"] = {
            "verify_seconds": verify_seconds,
            "profile_seconds": profile_seconds,
        }
        return summary

    @staticmethod
    def _set_seed(torch_mod: Any) -> None:
        torch_mod.manual_seed(0)
        if hasattr(torch_mod, "npu") and hasattr(torch_mod.npu, "manual_seed"):
            try:
                torch_mod.npu.manual_seed(0)
            except Exception:
                pass

    # === BatchProfiler 一次 prof 启停跑完所有 shape ====================

    def _profile_batch(
        self,
        *,
        case_spec: _CaseSpec,
        base_module: types.ModuleType,
        impl_module: types.ModuleType,
        npu_device: str,
        profile_settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        import torch  # type: ignore

        from akg_agents.op.dynamic_tune.config import Config
        from akg_agents.op.dynamic_tune.measure.batch_profiler import (
            BatchProfiler,
            NpuProfilerBackend,
        )

        self._ensure_npu_set_as_current(npu_device)

        repeat = max(1, int(profile_settings.get("run_times", 3)))

        init_params = list(base_module.get_init_inputs())

        self._set_seed(torch)
        framework_model = base_module.Model(*init_params).to(npu_device).eval()
        self._set_seed(torch)
        impl_model = impl_module.ModelNew(*init_params).to(npu_device).eval()

        # 默认 device 切到 NPU，让 base.py 里裸 torch.randn 直接在 NPU 上落地，
        # 省掉后面那行 `.to(npu_device)` 的 H2D。torch<2.0 时 fallback 到 CPU 构造，
        # 后面的 .to(npu_device) 兜底。
        self._set_seed(torch)
        with _push_default_device(npu_device):
            inputs_list = base_module.get_inputs_dyn_list()
        if not inputs_list:
            raise RuntimeError(
                f"profile_batch: get_inputs_dyn_list 为空 case={case_spec.name}"
            )

        param_names = self._forward_param_names(base_module)
        compiled_shape_expr = compile(
            case_spec.shape_expr, f"<shape_expr:{case_spec.shape_expr}>", "eval"
        )
        eval_globals = {"__builtins__": {"int": int, "len": len, "tuple": tuple}}

        npu_inputs_by_shape: dict[tuple[int, ...], tuple[Any, ...]] = {}
        shapes: list[tuple[int, ...]] = []
        for case_idx, inputs in enumerate(inputs_list):
            # 正常路径下 inputs 已经在 npu (上面 _push_default_device 切默认 device 了)，
            # .to(npu_device) 是 no-op；只有 torch<2 fallback 时这里才会真做 H2D。
            tensors = tuple(
                t.to(npu_device) if hasattr(t, "to") else t for t in inputs
            )
            if len(tensors) != len(param_names):
                raise RuntimeError(
                    f"profile_batch: forward 形参数与 dyn_list[{case_idx}] 不匹配: "
                    f"forward={param_names} inputs={len(tensors)}"
                )
            namespace = dict(zip(param_names, tensors))
            shape = tuple(int(v) for v in eval(compiled_shape_expr, eval_globals, namespace))
            if shape in npu_inputs_by_shape:
                raise RuntimeError(
                    f"profile_batch: shape_expr 在 case={case_idx} 上与已有 case 重复: shape={shape}"
                )
            npu_inputs_by_shape[shape] = tensors
            shapes.append(shape)

        shapes_tuple = tuple(shapes)
        # ModelNew.forward(...) 走自己的 selector, 不读 BatchProfiler 传入的 config,
        # 所以这里给一个 dummy Config 仅用于满足接口约束.
        dummy_cfg = Config({}, config_id="npu_inline_profile")

        def _impl_factory(shape: tuple[int, ...], _config: Any):
            inputs = npu_inputs_by_shape[shape]

            def _launch() -> None:
                with torch.no_grad():
                    impl_model(*inputs)

            return _launch

        def _base_factory(shape: tuple[int, ...], _config: Any):
            inputs = npu_inputs_by_shape[shape]

            def _launch() -> None:
                with torch.no_grad():
                    framework_model(*inputs)

            return _launch

        backend = NpuProfilerBackend(device=npu_device)
        profiler = BatchProfiler(backend=backend)

        impl_matrix = profiler.measure(
            shapes=shapes_tuple,
            configs=(dummy_cfg,),
            launch_factory=_impl_factory,
            repeat=repeat,
        )
        base_matrix = profiler.measure(
            shapes=shapes_tuple,
            configs=(dummy_cfg,),
            launch_factory=_base_factory,
            repeat=repeat,
        )

        impl_per_shape = [float(v) for v in impl_matrix.latencies_us.flatten().tolist()]
        base_per_shape = [float(v) for v in base_matrix.latencies_us.flatten().tolist()]
        import math as _math

        per_shape_speedup = [b / i if i > 0 else 0.0 for i, b in zip(impl_per_shape, base_per_shape)]
        speedup = float(_math.exp(sum(_math.log(s) for s in per_shape_speedup if s > 0) / len(per_shape_speedup))) if any(s > 0 for s in per_shape_speedup) else 0.0
        impl_us = float(sum(impl_per_shape) / len(impl_per_shape))
        base_us = float(sum(base_per_shape) / len(base_per_shape))

        return {
            "gen_time_us": impl_us,
            "base_time_us": base_us,
            "speedup": speedup,
            "unique_dir": "",
            "method": "batch_profiler",
            "path_used_impl": impl_matrix.path_used,
            "path_used_base": base_matrix.path_used,
            "shapes": [list(s) for s in shapes_tuple],
            "per_shape_impl_us": impl_per_shape,
            "per_shape_base_us": base_per_shape,
        }

    async def _ensure_local_worker(self, *, device_id: int):
        from akg_agents.core.worker.manager import (  # type: ignore
            get_worker_manager,
            register_local_worker,
        )

        await register_local_worker([device_id], backend=self._backend, arch=self._arch)
        worker = await get_worker_manager().select(backend=self._backend, arch=self._arch)
        if not worker:
            raise RuntimeError(f"未找到可用 worker: backend={self._backend}, arch={self._arch}")
        return worker

    def _build_kernel_verifier(
        self,
        *,
        op_name: str,
        framework_code: str,
        task_id: str,
        log_root: Path,
        worker: Any,
    ):
        from akg_agents.op.config.config_validator import load_config  # type: ignore
        from akg_agents.op.verifier.kernel_verifier import KernelVerifier  # type: ignore

        config = load_config(dsl=self._dsl, backend=self._backend)
        config["log_dir"] = str(log_root)
        config["verify_timeout"] = self._verify_timeout
        return KernelVerifier(
            op_name=op_name,
            framework_code=framework_code,
            task_id=task_id,
            framework="torch",
            dsl=self._dsl,
            backend=self._backend,
            arch=self._arch,
            impl_func_name="ModelNew",
            config=config,
            worker=worker,
        )

    @classmethod
    def _device_id_from(cls, device: str) -> int:
        return int(str(device).split(":")[-1])

    def _ensure_npu_set_as_current(self, npu_device: str) -> None:
        """Triton / torch_npu 依赖 current device。

        ``_profile_batch`` 里若未先 ``set_device``，在 ``npu:5`` 等非 0 卡上会出现
        数值全错、甚至 MTE DDR OOB。"""
        try:
            import torch_npu  # type: ignore

            torch_npu.npu.set_device(self._device_id_from(npu_device))
        except Exception:
            pass

    @classmethod
    def _strip_future_imports(cls, code: str) -> str:
        return cls._FUTURE_IMPORT_RE.sub("", code)

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
