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

"""TileLang NPUIR DSL adapter."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterTilelangNpuir(DSLAdapter):
    """Adapter for TileLang NPUIR DSL."""
    
    def get_import_statements(self, framework: str) -> str:
        """Return TileLang NPUIR import statements."""
        code = """import tilelang
tilelang.cache.clear_cache()
try:
    from ai_kernel_generator.utils.tilelang_compile_patch import apply_tilelang_patches
    apply_tilelang_patches()
except ImportError:
    pass
"""
        if framework == "torch":
            code += "import torch\nimport torch_npu\nimport tilelang.language as T\n"
        return code
    
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import."""
        return f"from {op_name}_tilelang_npuir_impl import {impl_func_name}\n"
    
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call TileLang NPUIR implementation function."""
        return f"impl_output = {impl_func_name}(*{inputs})\n"
    
    def needs_binary_io(self) -> bool:
        """TileLang NPUIR doesn't need binary I/O."""
        return False
    
    def needs_compilation(self) -> bool:
        """TileLang NPUIR doesn't need compilation."""
        return False
    
    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      clear_l2_cache: bool = True) -> str:
        """Return code string to benchmark TileLang NPUIR implementation.
        
        Args:
            impl_func_name: 实现函数名
            inputs: 输入变量名
            warmup: warmup 次数
            runs: 有效运行次数
            backend: 后端类型
            op_name: 算子名称
            case_idx: case 索引
            framework_model: 框架模型变量名（可选）
            framework_adapter: 框架适配器（可选）
            device_id: 设备ID（可选）
            clear_l2_cache: 是否在每次迭代前清除 L2 cache（默认 True）
        """
        if backend == "ascend":
            # 使用 profiler_npu 进行性能测试，支持 L2 cache 清除
            code = f"""        # dsl：tilelang_npuir
        try:
            from ai_kernel_generator.core.verifier.profiler import profiler_npu
            patch_imported = True
        except ImportError:
            patch_imported = False
        
        def tilelang_benchmark_fn():
            return {framework_model}(*{inputs})
        
        if patch_imported:
            # 使用 L2 cache 清除（fallback 方式，使用 zero_()）
            # 注意：tilelang_npuir 没有专用的清除方式，可能有误判风险
            execution_time_us = profiler_npu(
                tilelang_benchmark_fn,
                warmup={warmup},
                active={runs},
                prof_dir_name="prof_generation_output",
                keep_res=False,
                suppress_warnings=True,
                clear_l2_cache={clear_l2_cache},
                dsl="other"
            )
            execution_time_ms = execution_time_us / 1000
            method = "profiler_npu"
        else:
            import time
            start_time = time.time()
            for _ in range({warmup + runs}):
                _ = tilelang_benchmark_fn()
                torch.npu.synchronize()
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000 / {warmup + runs}
            method = "traditional_timing"
"""
        else:
            # 非 ascend 后端，使用传统计时
            sync_code = "torch.npu.synchronize()" if backend == "ascend" else ""
            code = f"""        # dsl：tilelang_npuir
        import time
        start_time = time.time()
        for _ in range({warmup + runs}):
            framework_output = {framework_model}(*{inputs})
            {sync_code}
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000 / {warmup + runs}  # 转换为毫秒
        method = "traditional_timing"
"""
        return code
    
    def get_special_setup_code(self) -> str:
        """Return special setup code for tilelang_npuir."""
        return """import tilelang
tilelang.cache.clear_cache()
try:
    from ai_kernel_generator.utils.tilelang_compile_patch import apply_tilelang_patches
    apply_tilelang_patches()
except ImportError:
    pass
"""

