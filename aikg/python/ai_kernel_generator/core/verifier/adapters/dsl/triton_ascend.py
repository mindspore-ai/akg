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

"""Triton Ascend DSL adapter."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterTritonAscend(DSLAdapter):
    """Adapter for Triton Ascend DSL."""
    
    def get_import_statements(self, framework: str) -> str:
        """Return Triton Ascend import statements."""
        code = ""
        if framework == "mindspore":
            code += "import torch\n"
        code += """try:
    from ai_kernel_generator.utils.triton_autotune_patch import apply_triton_patches
    apply_triton_patches()
except ImportError:
    pass
"""
        if framework == "mindspore":
            code += "import triton\nimport triton.language as tl\n"
        elif framework == "torch":
            code += "import triton\nimport triton.language as tl\n"
        elif framework == "numpy":
            code += "import numpy as np\nimport triton\nimport triton.language as tl\n"
        else:
            code += "import triton\nimport triton.language as tl\n"
        return code
    
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import."""
        return f"from {op_name}_triton_ascend import {impl_func_name}\n"
    
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call Triton Ascend implementation function."""
        return f"impl_output = {impl_func_name}(*{inputs})\n"
    
    def needs_binary_io(self) -> bool:
        """Triton Ascend doesn't need binary I/O."""
        return False
    
    def needs_compilation(self) -> bool:
        """Triton Ascend doesn't need compilation."""
        return False
    
    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None) -> str:
        """Return code string to benchmark Triton Ascend implementation."""
        code = f"""        try:
            from ai_kernel_generator.core.verifier.profiler import profiler_npu
            from ai_kernel_generator.utils.triton_autotune_patch import get_collected_config_timings, clear_collected_config_timings
            # 清除之前的配置信息
            clear_collected_config_timings()
            patch_imported = True
        except ImportError:
            get_collected_config_timings = lambda: {{}}
            clear_collected_config_timings = lambda: None
            patch_imported = False
        
        # 清除缓存确保重新autotune
        if hasattr({impl_func_name}, 'cache'):
            {impl_func_name}.cache.clear()
        
        # 触发autotune
        {impl_func_name}(*{inputs})
        
        # 获取收集的配置信息
        config_timings = get_collected_config_timings()
        
        # 保存autotune信息到当前文件夹
        if config_timings:
            autotune_filename = f"autotune_info_case_{case_idx}.json"
            try:
                with open(autotune_filename, 'w') as f:
                    json.dump(config_timings, f, indent=2, ensure_ascii=False)
                print(f"[{op_name}] Autotune info saved to {{autotune_filename}}")
            except Exception as e:
                print(f"[{op_name}] Warning: Failed to save autotune info: {{e}}")
        
        # 进行最终的性能测试
        def triton_benchmark_fn():
            result = {impl_func_name}(*{inputs})
            return result
        
        if backend == "ascend" and patch_imported:
            execution_time_us = profiler_npu(
                triton_benchmark_fn,
                warmup={warmup},
                active={runs},
                prof_dir_name="prof_generation_output",
                keep_res=False,
                suppress_warnings=True
            )
            execution_time_ms = execution_time_us / 1000
            method = "profiler_npu"
        else:
            # GPU环境或补丁导入失败：使用标准do_bench
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
    
    def get_special_setup_code(self) -> str:
        """Return special setup code for triton_ascend."""
        return """try:
    from ai_kernel_generator.utils.triton_autotune_patch import apply_triton_patches
    apply_triton_patches()
except ImportError:
    pass
"""

