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

    def needs_binary_io(self) -> bool:
        return False

    def needs_compilation(self) -> bool:
        return False

    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      clear_l2_cache: bool = True) -> str:
        if backend == "ascend":
            code = f"""
        import time
        import torch

        def tilelang_benchmark_fn():
            return impl_model(*{inputs})

        # warmup
        for _ in range({warmup}):
            _ = tilelang_benchmark_fn()
            torch.npu.synchronize()

        # timing
        start_time = time.time()
        for _ in range({runs}):
            _ = tilelang_benchmark_fn()
            torch.npu.synchronize()
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000 / {runs}
        method = "traditional_timing"
"""
        else:
            code = f"""
        import time
        import torch

        def tilelang_benchmark_fn():
            return impl_model(*{inputs})

        for _ in range({warmup}):
            _ = tilelang_benchmark_fn()
            torch.npu.synchronize()

        start_time = time.time()
        for _ in range({runs}):
            _ = tilelang_benchmark_fn()
            torch.npu.synchronize()
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000 / {runs}
        method = "traditional_timing"
"""
        return code

    def get_special_setup_code(self) -> str:
        return """import tilelang
tilelang.cache.clear_cache()
try:
    from akg_agents.op.utils.tilelang_compile_patch import apply_tilelang_patches
    apply_tilelang_patches()
except ImportError:
    pass
"""
