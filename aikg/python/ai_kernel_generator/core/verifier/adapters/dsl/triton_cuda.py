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

"""Triton CUDA DSL adapter."""

from typing import Any, Optional, Tuple

from .base import DSLAdapter


class DSLAdapterTritonCuda(DSLAdapter):
    """Adapter for Triton CUDA DSL."""
    
    def get_import_statements(self, framework: str) -> str:
        """Return Triton import statements."""
        if framework == "mindspore":
            return "import torch\nimport triton\nimport triton.language as tl\n"
        elif framework == "torch":
            return "import triton\nimport triton.language as tl\n"
        elif framework == "numpy":
            return "import numpy as np\nimport triton\nimport triton.language as tl\n"
        else:
            return "import triton\nimport triton.language as tl\n"
    
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import."""
        return f"from {op_name}_triton_cuda import {impl_func_name}\n"
    
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call Triton CUDA implementation function.
        
        Args:
            impl_func_name: Implementation function name
            inputs: Input variable name (e.g., "inputs_for_impl")
            device_id: Device ID (not used for triton_cuda)
            framework_adapter: Framework adapter (not used for triton_cuda)
            op_name: Operator name
            data_dir: Data directory (not used for triton_cuda)
            framework_output: Framework output (not used for triton_cuda)
            
        Returns:
            str: Code string to call the implementation
        """
        return f"impl_output = {impl_func_name}(*{inputs})\n"
    
    def needs_binary_io(self) -> bool:
        """Triton CUDA doesn't need binary I/O."""
        return False
    
    def needs_compilation(self) -> bool:
        """Triton CUDA doesn't need compilation."""
        return False
    
    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None) -> str:
        """Return code string to benchmark Triton CUDA implementation.
        
        Returns:
            str: Code string for benchmarking
        """
        code = f"""        # 进行最终的性能测试
        def triton_benchmark_fn():
            result = {impl_func_name}(*{inputs})
            return result
        
        import triton.testing
        execution_time_ms = triton.testing.do_bench(
            triton_benchmark_fn,
            warmup={warmup},
            rep={runs},
            return_mode="min"
        )
        method = "triton_do_bench"
"""
        return code

