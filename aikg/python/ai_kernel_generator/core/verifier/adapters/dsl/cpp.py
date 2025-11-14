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

"""C++ DSL adapter."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterCpp(DSLAdapter):
    """Adapter for C++ DSL."""
    
    def get_import_statements(self, framework: str) -> str:
        """Return C++ import statements."""
        return "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.cpp_extension import load_inline\n"
    
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import."""
        return f"from {op_name}_cpp import {impl_func_name}\n"
    
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call C++ implementation function."""
        return f"impl_output = {impl_func_name}(*{inputs})\n"
    
    def needs_binary_io(self) -> bool:
        """C++ doesn't need binary I/O."""
        return False
    
    def needs_compilation(self) -> bool:
        """C++ doesn't need compilation (handled by load_inline)."""
        return False
    
    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None) -> str:
        """Return code string to benchmark C++ implementation."""
        code = f"""        # CPU
        import time
        def cpp_benchmark_fn():
            return {impl_func_name}(*{inputs})
        # 执行 warmup
        for _ in range({warmup}):
            _ = cpp_benchmark_fn()
        # 计时 rep 次
        start_t = time.perf_counter()
        for _ in range({runs}):
            _ = cpp_benchmark_fn()
        end_t = time.perf_counter()
        execution_time_ms = (end_t - start_t) * 1000.0 / max({runs}, 1)
        method = "cpu_loop_timer"
"""
        return code

