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

"""TileLang CUDA DSL adapter."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterTilelangCuda(DSLAdapter):
    """Adapter for TileLang CUDA DSL."""
    
    def get_import_statements(self, framework: str) -> str:
        """Return TileLang CUDA import statements."""
        code = "import tilelang\nimport tilelang.language as T\n"
        if framework == "torch":
            code = "import torch\n" + code
        return code
    
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation ModelNew import."""
        return f"from {op_name}_tilelang_cuda_impl import ModelNew\n"
    
    def create_impl_module(
        self,
        framework: str,
        framework_adapter: Any,
        init_params_var: str = "init_params",
        device_var: str = "device",
    ) -> str:
        """Instantiate TileLang ModelNew once."""
        code = f"impl_model = ModelNew(*{init_params_var})\n"
        if framework == "torch":
            code += f"impl_model = impl_model.to({device_var})\n"
        return code
    
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call TileLang CUDA implementation function."""
        return f"impl_output = impl_model(*{inputs})\n"
    
    def needs_binary_io(self) -> bool:
        """TileLang CUDA doesn't need binary I/O."""
        return False
    
    def needs_compilation(self) -> bool:
        """TileLang CUDA doesn't need compilation."""
        return False
    
    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None) -> str:
        """Return code string to benchmark TileLang CUDA implementation."""
        # Similar to cuda_c, use traditional timing
        sync_code = "torch.cuda.synchronize()" if backend == "cuda" else ""
        code = f"""        # dsl：tilelang_cuda
        import time
        def tilelang_cuda_benchmark_fn():
            return impl_model(*{inputs})
        for _ in range({warmup}):
            _ = tilelang_cuda_benchmark_fn()
            {sync_code}
        start_time = time.time()
        for _ in range({runs}):
            _ = tilelang_cuda_benchmark_fn()
            {sync_code}
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000 / max({runs}, 1)
        method = "cuda_loop_timer"
"""
        return code

