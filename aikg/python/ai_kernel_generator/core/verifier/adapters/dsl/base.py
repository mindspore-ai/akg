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

"""Base class for DSL adapters."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict


class DSLAdapter(ABC):
    """Abstract base class for DSL adapters.
    
    DSL adapters provide a unified interface for different implementation languages
    (Triton, SWFT, AscendC, etc.) to handle function calls, benchmarking, and
    other DSL-specific operations. DSL adapters are unaware of autotune logic.
    """
    
    @abstractmethod
    def get_import_statements(self, framework: str) -> str:
        """Return import statements for the DSL.
        
        Args:
            framework: Framework name (torch, mindspore, numpy)
            
        Returns:
            str: Import statements as a string
        """
        pass
    
    @abstractmethod
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return import statement for implementation function.
        
        Args:
            op_name: Operator name
            impl_func_name: Implementation function name
            
        Returns:
            str: Import statement (e.g., "from {op_name}_triton_cuda import {impl_func_name}\n")
        """
        pass
    
    def create_impl_module(self, framework: str,
                           framework_adapter: Any,
                           init_params_var: str = "init_params",
                           device_var: str = "device") -> str:
        """生成创建 impl_model 的代码（用于 ModelNew 类格式的 DSL）。

        对于使用 ModelNew 类格式的 DSL（如 triton_cuda, triton_ascend, cpp），
        需要先实例化模型。对于函数式 DSL，返回空字符串。

        Args:
            framework: Framework name (torch, mindspore, numpy)
            framework_adapter: Framework adapter instance
            init_params_var: Variable name for init_params (default: "init_params")
            device_var: Variable name for device (default: "device")

        Returns:
            str: Code string to create impl_model, or empty string if not needed
        """
        return ""  # 默认返回空字符串，ModelNew 类格式的 DSL 需要override
    
    @abstractmethod
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call implementation function.
        
        Args:
            impl_func_name: Implementation function name
            inputs: Input variable name (e.g., "inputs_for_impl")
            device_id: Device ID
            framework_adapter: Framework adapter instance (for generating code)
            op_name: Operator name
            data_dir: Data directory variable name (for swft)
            framework_output: Framework output variable name (for swft)
            
        Returns:
            str: Code string to call the implementation
        """
        pass
    
    @abstractmethod
    def needs_binary_io(self) -> bool:
        """Check if DSL needs binary I/O (e.g., SWFT).
        
        Returns:
            bool: True if binary I/O is needed
        """
        pass
    
    @abstractmethod
    def needs_compilation(self) -> bool:
        """Check if DSL needs compilation (e.g., AscendC).
        
        Returns:
            bool: True if compilation is needed
        """
        pass
    
    @abstractmethod
    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None) -> str:
        """Return code string to benchmark implementation function.
        
        Args:
            impl_func_name: Implementation function name
            inputs: Input variable name (e.g., "inputs")
            warmup: Number of warmup iterations
            runs: Number of benchmark iterations
            backend: Backend type
            op_name: Operator name
            case_idx: Case index (for dynamic shape)
            framework_model: Framework model variable name (for swft)
            framework_adapter: Framework adapter (for generating code)
            device_id: Device ID (for swft)
            
        Returns:
            str: Code string for benchmarking
        """
        pass
    
    def get_special_setup_code(self) -> str:
        """Return special setup code (e.g., tilelang cache clear).
        
        Returns:
            str: Setup code as string (empty if not needed)
        """
        return ""
    
    def get_autotune_info(self, case_idx: int) -> Optional[Dict]:
        """Get autotune information (only for triton_ascend in profiling).
        
        Args:
            case_idx: Case index
            
        Returns:
            dict or None: Autotune information
        """
        return None
    
    def get_binary_io_functions(self) -> str:
        """Get binary I/O functions code (only for swft).
        
        Returns:
            str: Function definitions as string (empty if not needed)
        """
        return ""

