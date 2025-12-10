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

"""
PyTorch DSL adapter - 用于 Triton → PyTorch 转换场景

支持 ModelNew (KernelBench) 格式，生成的代码是纯 PyTorch 实现（不使用 Triton）。
验证时会将生成的 PyTorch 代码与原始 Triton 实现的输出进行对比。
"""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterTorch(DSLAdapter):
    """Adapter for PyTorch DSL (Triton → PyTorch conversion)."""
    
    def get_import_statements(self, framework: str) -> str:
        """Return PyTorch import statements.
        
        注意：这里不需要 import triton，因为生成的代码是纯 PyTorch。
        """
        if framework == "torch":
            return "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
        elif framework == "mindspore":
            # MindSpore 场景下也使用 torch 进行验证（如果需要）
            return "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
        elif framework == "numpy":
            return "import torch\nimport torch.nn as nn\nimport numpy as np\n"
        else:
            return "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
    
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import.
        
        统一使用 ModelNew 类格式（KernelBench 风格）。
        注意：使用 _impl 后缀避免与 framework 文件冲突
        """
        return f"from {op_name}_torch_impl import ModelNew\n"
    
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
        code += "impl_model.eval()\n"
        
        return code
    
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call PyTorch implementation function.
        
        调用已经实例化好的 impl_model（可以多次调用）。
        """
        return f"impl_output = impl_model(*{inputs})\n"
    
    def needs_binary_io(self) -> bool:
        """PyTorch doesn't need binary I/O."""
        return False
    
    def needs_compilation(self) -> bool:
        """PyTorch doesn't need compilation."""
        return False
    
    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None) -> str:
        """Return code string to benchmark PyTorch implementation.
        
        使用已经实例化好的 impl_model 进行性能测试。
        """
        # 根据 backend 选择同步方法
        if backend == "cuda":
            sync_code = "torch.cuda.synchronize()"
        elif backend == "ascend":
            sync_code = "torch.npu.synchronize()"
        else:
            sync_code = "pass  # CPU, no sync needed"
        
        code = f"""        # PyTorch 原生实现性能测试
        import time
        
        def torch_benchmark_fn():
            result = impl_model(*{inputs})
            return result
        
        # 预热
        for _ in range({warmup}):
            _ = torch_benchmark_fn()
            {sync_code}
        
        # 计时
        start_time = time.time()
        for _ in range({runs}):
            _ = torch_benchmark_fn()
            {sync_code}
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000 / {runs}
        method = "pytorch_loop_timer"
"""
        return code
    
    def get_special_setup_code(self) -> str:
        """Return special setup code (not needed for PyTorch)."""
        return ""
