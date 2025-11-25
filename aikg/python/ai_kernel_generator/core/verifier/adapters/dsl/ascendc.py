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

"""AscendC DSL adapter."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterAscendC(DSLAdapter):
    """Adapter for AscendC DSL."""
    
    def get_import_statements(self, framework: str) -> str:
        """Return AscendC import statements."""
        return "import sys, os\nimport torch_npu\nimport subprocess\n"
    
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import.
        
        Note: AscendC doesn't import the function directly, it compiles first.
        """
        return ""
    
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call AscendC implementation function.
        
        AscendC requires compilation before calling.
        Note: arch should be passed as a template variable, not from framework_adapter.
        """
        # arch will be provided as a template variable
        code = """        # 处理器
        # 映射架构到 SOC_VERSION
        arch_str = "{{ arch }}"
        ARCH_TO_SOC_VERSION = {{
            "ascend910b1": "Ascend910B1",
            "ascend910b2": "Ascend910B2",
            "ascend910b2c": "Ascend910B2C",
            "ascend910b3": "Ascend910B3",
            "ascend910b4": "Ascend910B4",
            "ascend310p3": "Ascend310P3"
        }}

        SOC_VERSION = ARCH_TO_SOC_VERSION.get(arch_str)
        if SOC_VERSION is None:
            raise ValueError(f"不支持的ascend架构: {{arch_str}}，AscendC DSL仅支持ascend910b1/b2/b2c/b3/b4和ascend310p3")
        try:
            result = subprocess.run(["bash", "run.sh", "-v", SOC_VERSION], check=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[ERROR]：编译失败！")
            else:
                print(f"[INFO]：编译成功！")
        except subprocess.CalledProcessError as e:
            error_msg = f"\\n{{'='*50}}\\n"
            error_msg += f"NPU Compiler Error(exit code {{e.returncode}})\\n"
            if e.stdout:
                error_msg += f"\\nSTDOUT:\\n{{e.stdout}}\\n"
            if e.stderr:
                error_msg += f"\\nSTDERR:\\n{{e.stderr}}\\n"
            raise RuntimeError(error_msg) from e
        sys.path.insert(0, "build")
        import {impl_func_name}
        torch.npu.config.allow_internal_format = False
        impl_output = {impl_func_name}.run_{impl_func_name}(*{inputs})
"""
        return code
    
    def needs_binary_io(self) -> bool:
        """AscendC doesn't need binary I/O."""
        return False
    
    def needs_compilation(self) -> bool:
        """AscendC needs compilation."""
        return True
    
    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None) -> str:
        """Return code string to benchmark AscendC implementation.
        
        AscendC uses msprof for profiling.
        """
        # AscendC profiling is handled by msprof in kernel_verifier
        # This method is not used for ascendc
        return ""

