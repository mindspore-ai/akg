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

"""SWFT DSL adapter."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterSwft(DSLAdapter):
    """Adapter for SWFT DSL."""
    
    def get_import_statements(self, framework: str) -> str:
        """Return SWFT import statements."""
        return "from swft.core import *\nfrom swft.api import *\nimport numpy as np\n"
    
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import."""
        return f"from {op_name}_swft import {impl_func_name}\n"
    
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call SWFT implementation function.
        
        SWFT requires binary I/O, so we need to generate bin files first.
        """
        if data_dir is None:
            data_dir = "os.path.dirname(__file__)"
        if framework_output is None:
            framework_output = "framework_output"
        
        code = f"""        # 运行SWFT实现
        data_dir = os.path.dirname(__file__)
        
        # 生成二进制数据文件
        gen_binary_data({inputs}, {framework_output}, data_dir)
        
        # 运行SWFT实现
        {impl_func_name}(device_id=int({device_id}))
        
        # 加载SWFT输出
        impl_output = load_binary_data(data_dir, {framework_output})
"""
        return code
    
    def needs_binary_io(self) -> bool:
        """SWFT needs binary I/O."""
        return True
    
    def needs_compilation(self) -> bool:
        """SWFT doesn't need compilation."""
        return False
    
    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None) -> str:
        """Return code string to benchmark SWFT implementation."""
        if framework_model is None:
            framework_model = "framework_model"
        if device_id is None:
            device_id = 0
        
        code = f"""        # 运行SWFT实现
        data_dir = os.path.dirname(__file__)
        
        # 生成二进制数据文件
        framework_output = {framework_model}(*{inputs})
        gen_binary_data({inputs}, framework_output, data_dir)
        
        # 运行SWFT实现
        import time
        start_time = time.time()
        for _ in range({warmup + runs}):
            {impl_func_name}(device_id=int({device_id}))
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000 / {warmup + runs}  # 转换为毫秒
        method = "traditional_timing"
"""
        return code
    

