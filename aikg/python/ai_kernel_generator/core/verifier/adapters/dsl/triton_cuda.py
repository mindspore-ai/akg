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

"""Triton CUDA DSL adapter - 支持 ModelNew (KernelBench) 格式."""

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
        """Return implementation function import.
        
        统一使用 ModelNew 类格式（KernelBench 风格）。
        """
        return f"from {op_name}_triton_cuda import ModelNew\n"
    
    def create_impl_module(self, framework: str,
                          framework_adapter: Any, 
                          init_params_var: str = "init_params",
                          device_var: str = "device") -> str:
        """生成创建 impl_model 的代码（只实例化一次）。
        
        Args:
            framework: Framework name (torch, mindspore, numpy)
            framework_adapter: Framework adapter instance
            init_params_var: Variable name for init_params (default: "init_params")
            device_var: Variable name for device (default: "device")
            
        Returns:
            str: Code string to create impl_model
        """
        code = f"impl_model = ModelNew(*{init_params_var})\n"
        if framework == "torch":
            code += f"impl_model = impl_model.to({device_var})\n"
        
        return code
    
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call Triton CUDA implementation function.
        
        调用已经实例化好的 impl_model（可以多次调用）。
        """
        return f"impl_output = impl_model(*{inputs})\n"
    
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
        
        使用已经实例化好的 impl_model 进行性能测试。
        """
        code = f"""        # 进行最终的性能测试
        def triton_benchmark_fn():
            result = impl_model(*{inputs})
            return result
        
        import triton.testing
        execution_time_ms = triton.testing.do_bench(
            triton_benchmark_fn,
            warmup={warmup},
            rep={runs},
            return_mode="median"
        )
        method = "triton_do_bench"
"""
        return code


