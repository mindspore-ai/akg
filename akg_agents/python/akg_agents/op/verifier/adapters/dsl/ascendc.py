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

import logging
import os
import shutil
from typing import Any, Dict, Optional

from jinja2 import Template

from akg_agents import get_project_root
from .base import DSLAdapter

logger = logging.getLogger(__name__)

# Templates for the AscendC compile project. The impl_code is exec'd in an
# isolated namespace; it must populate ``host_tiling_src`` /
# ``python_bind_src`` / ``kernel_src`` strings, which are then dropped
# alongside a rendered CMakeLists.txt + copied run.sh.
_CMAKE_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "cmake_template.j2"
)
_RUN_TEMPLATE_PATH = os.path.join(
    get_project_root(), "utils", "compile_tools", "ascend_compile", "run.sh"
)


class DSLAdapterAscendC(DSLAdapter):
    """Adapter for AscendC DSL."""

    impl_func_name_template = "{op_name}_kernel"

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
        Note: The generated code references the ``arch`` Python variable
        which is set by the verify template (kernel_verify_template_refactored.j2).
        """
        code = f"""        # 处理器
        # 映射架构到 SOC_VERSION
        # arch 变量由验证模板在外层设置
        from akg_agents.op.utils.arch_normalize import ascend_soc_version

        SOC_VERSION = ascend_soc_version(arch)
        if SOC_VERSION is None:
            raise ValueError(f"unsupported Ascend arch for AscendC SOC_VERSION: {{arch}}")
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
    
    static_check_via_python_ast = False  # C++ kernel src, no Python AST

    def materialize_impl(self, impl_code: str, verify_dir: str,
                         op_name: str, framework: str,
                         dsl_name: str,
                         task_info: Optional[Dict[str, Any]] = None,
                         config: Optional[Dict[str, Any]] = None) -> None:
        """渲染 CMakeLists.txt + 拷 run.sh，并把 impl_code 在隔离 namespace
        中 exec，取 ``host_tiling_src`` / ``python_bind_src`` / ``kernel_src``
        三段字符串写到对应文件。"""
        try:
            cmake_file = os.path.join(verify_dir, "CMakeLists.txt")
            with open(_CMAKE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                template = Template(f.read())
            cmake_code = template.render(op_name=op_name)
            with open(cmake_file, "w", encoding="utf-8") as f:
                f.write(cmake_code)
            shutil.copy(_RUN_TEMPLATE_PATH, verify_dir)

            ns: Dict[str, Any] = {}
            try:
                compile(impl_code, "<string>", "exec")
                exec(impl_code, ns)
            except Exception as e:
                raise Exception(f"Error in generated code: {e}")

            host_tiling_src = ns.get('host_tiling_src')
            python_binding_src = ns.get('python_bind_src')
            kernel_src = ns.get('kernel_src')

            if host_tiling_src is None:
                raise Exception("host_tiling_src is None - 生成的代码中缺少host侧tiling部分")
            if python_binding_src is None:
                raise Exception("python_bind_src is None - 生成的代码中缺少内核调用python_bind部分")
            if kernel_src is None:
                raise Exception("kernel_src is None - 生成的代码中缺少kernel主函数部分")

            with open(os.path.join(verify_dir, f"{op_name}_tiling.cpp"), "w") as f:
                f.write(host_tiling_src)
            with open(os.path.join(verify_dir, "pybind11.cpp"), "w") as f:
                f.write(python_binding_src)
            with open(os.path.join(verify_dir, f"{op_name}_kernel.cpp"), "w") as f:
                f.write(kernel_src)

            logger.info(f"[{op_name}] AscendC项目文件生成完成")
        except Exception as e:
            logger.error(f"AscendC项目生成失败: {e}")
            raise Exception(f"AscendC项目生成失败: {e}")

    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      framework: str = "torch") -> str:
        """Return code string to benchmark AscendC implementation.
        
        AscendC uses msprof for profiling.
        """
        # AscendC profiling is handled by msprof in kernel_verifier
        # This method is not used for ascendc
        return ""
