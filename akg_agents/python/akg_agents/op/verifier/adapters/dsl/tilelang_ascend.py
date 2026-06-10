# Copyright 2025 Huawei Technologies Co., Ltd
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

"""TileLang-Ascend DSL adapter - ModelNew (KernelBench) 格式."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterTilelangAscend(DSLAdapter):
    """Adapter for TileLang-Ascend DSL."""

    profile_via_python_script = True
    impl_func_name_template = "ModelNew"

    def get_import_statements(self, framework: str) -> str:
        code = """import tilelang
tilelang.cache.clear_cache()
try:
    from akg_agents.op.utils.tilelang_compile_patch import apply_tilelang_patches
    apply_tilelang_patches()
except ImportError:
    pass
"""
        if framework == "torch":
            code += "import torch\nimport torch_npu\nimport tilelang.language as T\n"
        return code

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        return f"from {op_name}_tilelang_ascend_impl import ModelNew\n"

    def create_impl_module(self, framework: str,
                          framework_adapter: Any,
                          init_params_var: str = "init_params",
                          device_var: str = "device") -> str:
        code = f"impl_model = ModelNew(*{init_params_var})\n"
        if framework == "torch":
            code += f"impl_model = impl_model.to({device_var})\n"
        return code

    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str,
                  data_dir: Optional[str] = None,
                  framework_output: Optional[str] = None) -> str:
        return f"impl_output = impl_model(*{inputs})\n"

    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      clear_l2_cache: bool = True,
                      framework: str = "torch") -> str:
        """Return code string to benchmark TileLang Ascend implementation.

        对齐 tilelang.profiler.do_bench 的实现：
        - 使用 torch.npu.Event 高精度计时
        - 支持 L2 cache 清除（256MB buffer + zero_）
        - 自动 warmup 和多次测量取最小值
        """
        if framework == "torch":
            code = f"""
        import torch

        def tilelang_benchmark_fn():
            return impl_model(*{inputs})

        # 先执行一次确保编译完成
        tilelang_benchmark_fn()
        torch.npu.synchronize()

        # L2 cache 清除 buffer（对齐 tilelang.profiler.do_bench）
        cache = None
        if {clear_l2_cache}:
            cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="npu")

        # 使用 torch.npu.Event 高精度计时
        start_event = [torch.npu.Event(enable_timing=True) for _ in range({runs})]
        end_event = [torch.npu.Event(enable_timing=True) for _ in range({runs})]

        # warmup
        for _ in range({warmup}):
            if cache is not None:
                cache.zero_()
            tilelang_benchmark_fn()
        torch.npu.synchronize()

        # timing
        for i in range({runs}):
            if cache is not None:
                cache.zero_()
            start_event[i].record()
            tilelang_benchmark_fn()
            end_event[i].record()

        torch.npu.synchronize()
        times = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
            dtype=torch.float,
        )
        execution_time_ms = torch.mean(times).item()
        method = "tilelang_event_timing"
"""
        else:
            raise ValueError(
                f"TileLang Ascend currently only supports framework='torch', "
                f"got framework='{framework}'"
            )
        return code

    def get_special_setup_code(self, framework: str = "torch") -> str:
        return """import tilelang
tilelang.cache.clear_cache()
try:
    from akg_agents.op.utils.tilelang_compile_patch import apply_tilelang_patches
    apply_tilelang_patches()
except ImportError:
    pass
"""
