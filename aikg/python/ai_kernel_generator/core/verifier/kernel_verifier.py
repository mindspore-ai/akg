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

import os
import re
import shutil
import logging
import subprocess
import json
import sys
import textwrap
from datetime import datetime
from typing import Optional, Literal, Tuple, Dict, Any, List
from jinja2 import Template
import pandas as pd
from pathlib import Path

from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.process_utils import run_command
from ai_kernel_generator.core.utils import normalize_dsl
from ai_kernel_generator.core.verifier.adapters.factory import (
    get_framework_adapter, get_dsl_adapter, get_backend_adapter
)

# 模板路径
TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "kernel_verify_template_refactored.j2")
PROFILE_BASE_TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "prof_base_template_refactored.j2")
PROFILE_GENERATION_TEMPLATE_PATH = os.path.join(
    get_project_root(), "resources", "templates", "prof_generation_template_refactored.j2")
# 生成CMakeLists.txt和运行脚本的路径
CMAKE_TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "cmake_template.j2")
RUN_TEMPLATE_PATH = os.path.join(get_project_root(), "utils", "compile_tools", "ascend_compile", "run.sh")

# 类型定义
FrameworkType = Literal["torch", "mindspore", "numpy"]
ImplType = Literal["triton_cuda", "triton_ascend", "triton-russia", "swft", "cuda_c", "cpp", "tilelang_npuir", "tilelang_cuda", "ascendc"]
BackendType = Literal["cuda", "ascend", "cpu"]
ArchType = Literal["a100", "v100", "h20", "l20", "rtx3090", "ascend910b4", "ascend310p3", "x86_64", "aarch64"]

logger = logging.getLogger(__name__)


class KernelVerifier:
    def __init__(self,
                 op_name: str,
                 framework_code: str,
                 task_id: str = "0",
                 framework: FrameworkType = "torch",
                 dsl: ImplType = "triton_cuda",
                 backend: BackendType = "cuda",
                 arch: ArchType = "a100",
                 impl_func_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化Kernel验证器。

        Args:
            op_name (str): 算子名称
            framework_code (str): 框架实现代码（PyTorch、MindSpore或NumPy）
            log_dir (str): 调试信息目录
            task_id (str, optional): 任务ID，用于生成唯一目录名
            framework (FrameworkType): 深度学习框架，可选值包括 "torch", "mindspore", "numpy"
            dsl (ImplType): 实现类型，可选值包括 "triton_cuda", "triton_ascend", "triton-russia", "swft"
            backend (BackendType): 计算设备后端，可选值包括 "cuda", "ascend"
            arch (ArchType): 硬件架构，可选值包括 "a100", "v100", "h20", "l20", "rtx3090", "ascend910b4", "ascend310p3"
            impl_func_name (str, optional): 实现函数名，默认为op_name_dsl_framework
        """
        self.op_name = op_name
        self.framework_code = framework_code
        self.framework = framework
        # 规范化DSL（自动转换triton为triton_cuda或triton_ascend）
        self.dsl = normalize_dsl(dsl, backend)
        self.backend = backend.lower()
        self.arch = arch.lower()
        self.task_id = task_id
        # 获取AscendC代码
        self.context = {}
        # 从config中获取log_dir
        if config:
            self.config = config
            self.log_dir = config.get("log_dir")
        else:
            raise ValueError("config is required for KernelVerifier")
        if "triton_cuda" in self.dsl or "triton_ascend" in self.dsl:
            if self.dsl == "triton_cuda":
                self.impl_func_name = impl_func_name or f"{op_name}_triton_cuda_{framework}"
            elif self.dsl == "triton_ascend":
                self.impl_func_name = impl_func_name or f"{op_name}_triton_ascend_{framework}"
            else:
                # 兼容旧代码，如果dsl包含triton_cuda或triton_ascend但不是精确匹配
                self.impl_func_name = impl_func_name or f"{op_name}_{self.dsl}_{framework}"
        elif self.dsl == "ascendc":
            self.impl_func_name = impl_func_name or f"{op_name}_kernel"
        else:
            self.impl_func_name = impl_func_name or f"{op_name}_{dsl}_{framework}"

        # 验证backend和arch的组合是否有效
        if self.backend == "cuda" and self.arch not in ["a100", "v100", "h20", "l20", "rtx3090"]:
            raise ValueError(f"cuda后端只支持a100、v100、h20、l20和rtx3090架构，当前架构: {self.arch}")
        if self.backend == "ascend":
            # 支持 ascend910b1, b2, b2c, b3, b4 和 ascend310p3
            supported_ascend_archs = ["ascend910b1", "ascend910b2", "ascend910b2c", "ascend910b3", "ascend910b4", "ascend310p3"]
            if self.arch not in supported_ascend_archs:
                raise ValueError(f"ascend后端只支持ascend910b1/b2/b2c/b3/b4和ascend310p3架构，当前架构: {self.arch}")

    def _create_verify_dir(self, step_counter) -> str:
        """创建验证目录并返回目录路径"""
        expanded_log_dir = os.path.expanduser(self.log_dir)
        unique_dir = f"I{self.task_id}_S{step_counter:02d}_verify"

        target_dir = os.path.join(expanded_log_dir, self.op_name, unique_dir)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def _generate_import_statements(self) -> str:
        """根据framework和dsl生成适当的import语句"""
        import_lines = []

        if "triton_cuda" in self.dsl or "triton_ascend" in self.dsl:
            if self.framework == "mindspore":
                import_lines = [
                    "import torch",
                    "import triton",
                    "import triton.language as tl",
                    "import mindspore as ms"
                ]
            elif self.framework == "torch":
                import_lines = [
                    "import torch",
                    "import triton",
                    "import triton.language as tl"
                ]
            elif self.framework == "numpy":
                import_lines = [
                    "import numpy as np",
                    "import triton",
                    "import triton.language as tl"
                ]
        elif self.dsl == "tilelang_npuir":
            if self.framework == "torch":
                import_lines = [
                    "import torch",
                    "import torch_npu",
                    "import tilelang",
                    "import tilelang.language as T"
                ]
        elif self.dsl == "swft":
            import_lines = [
                "from swft.core import *",
                "from swft.api import *",
                "import numpy as np"
            ]
        elif self.dsl == "cuda_c" or self.dsl == "cpp":
            import_lines = [
                "import torch",
                "import torch.nn as nn",
                "import torch.nn.functional as F",
                "from torch.utils.cpp_extension import load_inline"
            ]
        elif self.framework == "numpy":
            import_lines = [
                "import numpy as np"
            ]
        elif self.framework == "torch":
            import_lines = [
                "import torch"
            ]
        elif self.framework == "mindspore":
            import_lines = [
                "import mindspore as ms"
            ]

        # 添加换行符并连接
        if import_lines:
            return "\n".join(import_lines) + "\n\n"
        return ""
    
    def generate_ascendc_project(self, impl_code: str, verify_dir: str):
        """生成AscendC项目文件"""
        try:
            # 生成CMakeLists.txt
            cmake_file = os.path.join(verify_dir, f"CMakeLists.txt")
            with open(CMAKE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                template = Template(f.read())
            cmake_code = template.render(op_name=self.op_name)
            with open(cmake_file, "w", encoding="utf-8") as f:
                f.write(cmake_code)
            shutil.copy(RUN_TEMPLATE_PATH, verify_dir)
            
            # 填充代码
            try: 
                compile(impl_code, "<string>", "exec")
                exec(impl_code, self.context)
            except Exception as e:
                raise Exception(f"Error in generated code: {e}")

            # 检查并写入三个关键文件
            host_tiling_src = self.context.get('host_tiling_src')
            python_binding_src = self.context.get('python_bind_src')
            kernel_src = self.context.get('kernel_src')
            
            # 检查代码是否为None
            if host_tiling_src is None:
                raise Exception(f"host_tiling_src is None - 生成的代码中缺少host侧tiling部分")
            if python_binding_src is None:
                raise Exception(f"python_bind_src is None - 生成的代码中缺少内核调用python_bind部分")
            if kernel_src is None:
                raise Exception(f"kernel_src is None - 生成的代码中缺少kernel主函数部分")

            # 写入文件
            with open(os.path.join(verify_dir, f"{self.op_name}_tiling.cpp"), "w") as f:
                f.write(host_tiling_src)
            with open(os.path.join(verify_dir, f"pybind11.cpp"), "w") as f:
                f.write(python_binding_src)
            with open(os.path.join(verify_dir, f"{self.op_name}_kernel.cpp"), "w") as f:
                f.write(kernel_src)
                
            logger.info(f"[{self.op_name}] AscendC项目文件生成完成")
            
        except Exception as e:
            logger.error(f"AscendC项目生成失败: {e}")
            raise Exception(f"AscendC项目生成失败: {e}")
        
    def _detect_dynamic_shape(self) -> bool:
        """
        检测框架代码是否包含动态shape函数

        Returns:
            bool: True if contains get_inputs_dyn_list, False otherwise
        """
        return "get_inputs_dyn_list" in self.framework_code

    @staticmethod
    def _prepare_code_lines(code_snippet: Any) -> List[str]:
        """将多行代码片段规范化为按行列表，方便模板渲染时控制缩进。"""
        if not code_snippet:
            return []
        if isinstance(code_snippet, (list, tuple)):
            lines: List[str] = []
            for snippet in code_snippet:
                lines.extend(KernelVerifier._prepare_code_lines(snippet))
            return lines
        if isinstance(code_snippet, str):
            normalized = textwrap.dedent(code_snippet).strip("\n")
            if not normalized:
                return []
            return normalized.split("\n")
        raise TypeError(f"Unsupported code snippet type: {type(code_snippet)}")

    def gen_verify_project(self, impl_code: str, verify_dir: str, device_id: int = 0):
        """生成验证项目文件到指定目录"""
        logger.info(f"[{self.op_name}] 开始生成验证项目，目录: {verify_dir}, device_id={device_id}")
        
        # 创建框架实现文件
        framework_file = os.path.join(verify_dir, f"{self.op_name}_{self.framework}.py")
        try:
            with open(framework_file, "w", encoding="utf-8") as f:
                f.write(self.framework_code)
            logger.debug(f"[{self.op_name}] 框架实现文件已创建: {framework_file}")
        except Exception as e:
            logger.error(f"[{self.op_name}] 框架实现文件创建失败: {framework_file}, 错误: {e}")
            raise

        # 创建具体实现文件
        if "ascendc" in self.dsl:
            logger.info(f"[{self.op_name}] 检测到AscendC DSL，生成编译项目")
            self.generate_ascendc_project(impl_code, verify_dir)
        else:
            file_name = f"{self.op_name}_{self.dsl}.py"
            impl_file = os.path.join(verify_dir, file_name)

            # 使用adapter生成import语句
            try:
                dsl_adapter = get_dsl_adapter(self.dsl)
                import_statements = dsl_adapter.get_import_statements(self.framework)
                logger.debug(f"[{self.op_name}] DSL import语句生成成功")
            except Exception as e:
                logger.error(f"[{self.op_name}] DSL import语句生成失败: {e}")
                raise

            try:
                with open(impl_file, "w", encoding="utf-8") as f:
                    f.write(import_statements + impl_code)
                logger.debug(f"[{self.op_name}] 实现文件已创建: {impl_file}")
            except Exception as e:
                logger.error(f"[{self.op_name}] 实现文件创建失败: {impl_file}, 错误: {e}")
                raise

        # 生成验证脚本
        verify_file = os.path.join(verify_dir, f"verify_{self.op_name}.py")

        # 从文件加载模板
        logger.info(f"[{self.op_name}] 开始生成验证项目，使用模板: {os.path.basename(TEMPLATE_PATH)}")
        try:
            with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                template = Template(f.read())
            logger.debug(f"[{self.op_name}] 模板文件加载成功: {TEMPLATE_PATH}")
        except Exception as e:
            logger.error(f"[{self.op_name}] 模板文件加载失败: {TEMPLATE_PATH}, 错误: {e}")
            raise

        # 检测是否为动态shape
        is_dynamic_shape = self._detect_dynamic_shape()
        logger.info(f"[{self.op_name}] 检测到shape类型: {'动态' if is_dynamic_shape else '静态'}")

        # 获取adapters
        logger.debug(f"[{self.op_name}] 初始化adapters: framework={self.framework}, dsl={self.dsl}, backend={self.backend}")
        try:
            framework_adapter = get_framework_adapter(self.framework)
            dsl_adapter = get_dsl_adapter(self.dsl)
            backend_adapter = get_backend_adapter(self.backend)
            logger.debug(f"[{self.op_name}] Adapters初始化成功")
        except Exception as e:
            logger.error(f"[{self.op_name}] Adapters初始化失败: {e}")
            raise

        # 使用adapter生成代码字符串
        logger.debug(f"[{self.op_name}] 开始生成代码片段...")
        try:
            framework_imports = framework_adapter.get_import_statements()
            logger.debug(f"[{self.op_name}] Framework imports生成成功 (长度: {len(framework_imports)})")
            
            framework_model_import = framework_adapter.get_framework_import(self.op_name, is_dynamic_shape)
            logger.debug(f"[{self.op_name}] Framework model import生成成功 (长度: {len(framework_model_import)})")
            
            dsl_imports = dsl_adapter.get_import_statements(self.framework)
            logger.debug(f"[{self.op_name}] DSL imports生成成功 (长度: {len(dsl_imports)})")
            
            dsl_impl_import = dsl_adapter.get_impl_import(self.op_name, self.impl_func_name)
            logger.debug(f"[{self.op_name}] DSL impl import生成成功 (长度: {len(dsl_impl_import)})")
            
            special_setup_code = dsl_adapter.get_special_setup_code()
            logger.debug(f"[{self.op_name}] Special setup code生成成功 (长度: {len(special_setup_code)})")
            
            # 生成设备设置代码
            backend_adapter.setup_environment(device_id, self.arch)
            logger.debug(f"[{self.op_name}] Backend环境设置完成: device_id={device_id}, arch={self.arch}")
            
            device_setup_code = framework_adapter.get_device_setup_code(self.backend, self.arch, device_id)
            logger.debug(f"[{self.op_name}] Device setup code生成成功 (长度: {len(device_setup_code)})")
            
            # 生成输入处理代码
            process_input_code = framework_adapter.get_process_input_code(self.backend, self.dsl)
            logger.debug(f"[{self.op_name}] Process input code生成成功 (长度: {len(process_input_code)})")
            
            # 生成调用实现代码
            call_impl_code = dsl_adapter.call_impl(
                self.impl_func_name, "inputs_for_impl", device_id,
                framework_adapter, self.op_name, "data_dir", "framework_output"
            )
            logger.debug(f"[{self.op_name}] Call impl code生成成功 (长度: {len(call_impl_code)})")
            
            # 生成set_seed代码
            set_seed_code = framework_adapter.get_set_seed_code(self.backend)
            logger.debug(f"[{self.op_name}] Set seed code生成成功 (长度: {len(set_seed_code)})")
            
            # 生成binary I/O函数（如果需要）
            binary_io_functions = ""
            needs_binary_io = dsl_adapter.needs_binary_io()
            if needs_binary_io:
                binary_io_functions = framework_adapter.get_binary_io_functions(self.op_name)
                logger.info(f"[{self.op_name}] Binary I/O函数生成成功 (长度: {len(binary_io_functions)})")
            else:
                logger.debug(f"[{self.op_name}] 不需要Binary I/O函数")
            
            # 获取TensorType名称（完整路径）
            tensor_type_name = framework_adapter.get_tensor_type_name()
            logger.debug(f"[{self.op_name}] TensorType名称: {tensor_type_name}")
        except Exception as e:
            logger.error(f"[{self.op_name}] 代码片段生成失败: {e}", exc_info=True)
            raise

        # 使用模板变量
        logger.debug(f"[{self.op_name}] 开始渲染模板...")
        try:
            rendered_code = template.render(
                op_name=self.op_name,
                framework=self.framework,
                dsl=self.dsl,
                device_id=device_id,
                impl_func_name=self.impl_func_name,
                backend=self.backend,
                arch=self.arch,
                is_dynamic_shape=is_dynamic_shape,
                timeout=self.config.get('verify_timeout', 300),
                # Adapter生成的代码
                framework_imports=self._prepare_code_lines(framework_imports),
                framework_model_import=self._prepare_code_lines(framework_model_import),
                dsl_imports=self._prepare_code_lines(dsl_imports),
                dsl_impl_import=self._prepare_code_lines(dsl_impl_import),
                special_setup_code=self._prepare_code_lines(special_setup_code),
                device_setup_code=self._prepare_code_lines(device_setup_code),
                process_input_code=self._prepare_code_lines(process_input_code),
                call_impl_code=self._prepare_code_lines(call_impl_code),
                set_seed_code=self._prepare_code_lines(set_seed_code),
                binary_io_functions=self._prepare_code_lines(binary_io_functions),
                needs_binary_io=needs_binary_io,
                tensor_type_name=tensor_type_name,
            )
            logger.info(f"[{self.op_name}] 模板渲染成功，渲染后代码长度: {len(rendered_code)} 字符")
        except Exception as e:
            logger.error(f"[{self.op_name}] 模板渲染失败: {e}", exc_info=True)
            raise

        # 写入文件
        try:
            with open(verify_file, "w", encoding="utf-8") as f:
                f.write(rendered_code)
            logger.info(f"[{self.op_name}] 验证脚本已写入: {verify_file}")
        except Exception as e:
            logger.error(f"[{self.op_name}] 验证脚本写入失败: {verify_file}, 错误: {e}")
            raise
    

    def run_verify(self, verify_dir: str, timeout: int = 300):
        """
        运行验证脚本

        Args:
            verify_dir: 验证目录
            timeout: 超时时间（秒），默认5分钟（传递给模板用于每次计算）
        """
        verify_script = os.path.join(verify_dir, f"verify_{self.op_name}.py")
        logger.info(f"[{self.op_name}] 开始运行验证脚本: {verify_script}, timeout={timeout}秒")
        
        original_cwd = os.getcwd()
        try:
            os.chdir(verify_dir)
            python_cmd = ["python", f"verify_{self.op_name}.py"]
            # 使用run_command但禁用timeout，让验证脚本无限制运行
            result = run_command(python_cmd, f"verify_{self.op_name}", timeout=timeout)
            if result:
                logger.info(f"[{self.op_name}] 验证脚本执行成功")
            else:
                logger.error(f"[{self.op_name}] 验证脚本执行失败")
            return result
        except Exception as e:
            logger.error(f"[{self.op_name}] 验证脚本执行异常: {e}", exc_info=True)
            raise
        finally:
            try:
                os.chdir(original_cwd)
            except Exception:
                pass

    def gen_profile_project(self, verify_dir: str, device_id: int = 0, warmup_times: int = 5, run_times: int = 50):
        """生成profile项目文件到指定目录"""
        # 生成基准性能测试脚本
        profile_file = os.path.join(verify_dir, f"profile_{self.op_name}_base.py")
        self.gen_profile_file_from_template(PROFILE_BASE_TEMPLATE_PATH, profile_file,
                                            device_id, warmup_times, run_times)
        # 生成性能测试脚本
        profile_file = os.path.join(verify_dir, f"profile_{self.op_name}_generation.py")
        self.gen_profile_file_from_template(PROFILE_GENERATION_TEMPLATE_PATH,
                                            profile_file, device_id, warmup_times, run_times)

    def gen_profile_file_from_template(self, template_path: str, profile_file: str, device_id: int, warmup_times: int, run_times: int):
        """从模板生成profile文件"""
        template_name = os.path.basename(template_path)
        logger.info(f"[{self.op_name}] 开始生成性能测试文件，使用模板: {template_name}")
        
        # 从文件加载模板
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template = Template(f.read())
            logger.debug(f"[{self.op_name}] 性能测试模板文件加载成功: {template_path}")
        except Exception as e:
            logger.error(f"[{self.op_name}] 性能测试模板文件加载失败: {template_path}, 错误: {e}")
            raise

        # 检测是否为动态shape
        is_dynamic_shape = self._detect_dynamic_shape()
        logger.debug(f"[{self.op_name}] 性能测试shape类型: {'动态' if is_dynamic_shape else '静态'}")

        # 获取adapters
        try:
            framework_adapter = get_framework_adapter(self.framework)
            dsl_adapter = get_dsl_adapter(self.dsl)
            backend_adapter = get_backend_adapter(self.backend)
            logger.debug(f"[{self.op_name}] 性能测试Adapters初始化成功")
        except Exception as e:
            logger.error(f"[{self.op_name}] 性能测试Adapters初始化失败: {e}")
            raise

        # 使用adapter生成代码字符串
        logger.debug(f"[{self.op_name}] 开始生成性能测试代码片段...")
        try:
            framework_imports = framework_adapter.get_import_statements()
            framework_model_import = framework_adapter.get_framework_import(self.op_name, is_dynamic_shape)
            dsl_imports = dsl_adapter.get_import_statements(self.framework)
            dsl_impl_import = dsl_adapter.get_impl_import(self.op_name, self.impl_func_name)
            special_setup_code = dsl_adapter.get_special_setup_code()
            
            # 生成设备设置代码
            backend_adapter.setup_environment(device_id, self.arch)
            device_setup_code = framework_adapter.get_device_setup_code(self.backend, self.arch, device_id)
            
            # 生成输入处理代码
            process_input_code = framework_adapter.get_process_input_code(self.backend, self.dsl)
            
            # 生成set_seed代码
            set_seed_code = framework_adapter.get_set_seed_code(self.backend)
            
            # 生成binary I/O函数（如果需要）
            binary_io_functions = ""
            needs_binary_io = dsl_adapter.needs_binary_io()
            if needs_binary_io:
                binary_io_functions = framework_adapter.get_binary_io_functions(self.op_name)
                logger.info(f"[{self.op_name}] 性能测试Binary I/O函数生成成功")
            
            # 获取TensorType名称（完整路径）
            tensor_type_name = framework_adapter.get_tensor_type_name()
            
            # 判断是base还是generation模板
            is_base_template = "base" in template_path.lower()
            logger.debug(f"[{self.op_name}] 性能测试模板类型: {'base' if is_base_template else 'generation'}")
            
            # 生成benchmark代码
            if is_base_template:
                # Base模板：benchmark framework model
                benchmark_code = self._generate_base_benchmark_code(framework_adapter, dsl_adapter, 
                                                                     warmup_times, run_times)
                logger.debug(f"[{self.op_name}] Base benchmark代码生成成功 (长度: {len(benchmark_code)})")
            else:
                # Generation模板：benchmark implementation
                benchmark_code = dsl_adapter.benchmark_impl(
                    self.impl_func_name, "inputs", warmup_times, run_times, 
                    self.backend, self.op_name, case_idx=0,
                    framework_model="framework_model" if needs_binary_io else None,
                    framework_adapter=framework_adapter if needs_binary_io else None,
                    device_id=device_id if needs_binary_io else None
                )
                logger.debug(f"[{self.op_name}] Generation benchmark代码生成成功 (长度: {len(benchmark_code)})")
        except Exception as e:
            logger.error(f"[{self.op_name}] 性能测试代码片段生成失败: {e}", exc_info=True)
            raise

        # 使用模板变量
        logger.debug(f"[{self.op_name}] 开始渲染性能测试模板...")
        try:
            rendered_code = template.render(
                op_name=self.op_name,
                framework=self.framework,
                dsl=self.dsl,
                device_id=device_id,
                impl_func_name=self.impl_func_name,
                backend=self.backend,
                arch=self.arch,
                warmup_times=warmup_times,
                run_times=run_times,
                total_count=warmup_times + run_times,
                is_dynamic_shape=is_dynamic_shape,
                # Adapter生成的代码
                framework_imports=self._prepare_code_lines(framework_imports),
                framework_model_import=self._prepare_code_lines(framework_model_import),
                dsl_imports=self._prepare_code_lines(dsl_imports),
                dsl_impl_import=self._prepare_code_lines(dsl_impl_import),
                special_setup_code=self._prepare_code_lines(special_setup_code),
                device_setup_code=self._prepare_code_lines(device_setup_code),
                process_input_code=self._prepare_code_lines(process_input_code),
                set_seed_code=self._prepare_code_lines(set_seed_code),
                binary_io_functions=self._prepare_code_lines(binary_io_functions),
                needs_binary_io=needs_binary_io,
                tensor_type_name=tensor_type_name,
                benchmark_code=self._prepare_code_lines(benchmark_code),
            )
            logger.info(f"[{self.op_name}] 性能测试模板渲染成功，渲染后代码长度: {len(rendered_code)} 字符")
        except Exception as e:
            logger.error(f"[{self.op_name}] 性能测试模板渲染失败: {e}", exc_info=True)
            raise

        # 写入文件
        try:
            with open(profile_file, "w", encoding="utf-8") as f:
                f.write(rendered_code)
            logger.info(f"[{self.op_name}] 性能测试脚本已写入: {profile_file}")
        except Exception as e:
            logger.error(f"[{self.op_name}] 性能测试脚本写入失败: {profile_file}, 错误: {e}")
            raise
    
    def _generate_base_benchmark_code(self, framework_adapter, dsl_adapter, warmup, runs):
        """生成base benchmark代码（benchmark framework model）"""
        if "triton_cuda" in self.dsl or "triton_ascend" in self.dsl:
            if self.backend == "ascend":
                code = f"""        # 导入profiler以支持性能测试
        try:
            from ai_kernel_generator.core.verifier.profiler import profiler_npu
            patch_imported = True
        except ImportError:
            # 如果导入失败，使用标准方法
            patch_imported = False
        # 基准测试函数
        def base_benchmark_fn():
            result = framework_model(*inputs)
            return result
        
        if backend == "ascend" and patch_imported:
            execution_time_us = profiler_npu(
                base_benchmark_fn,
                warmup={warmup},
                active={runs},
                prof_dir_name="prof_base_output",
                keep_res=False,
                suppress_warnings=True
            )
            execution_time_ms = execution_time_us / 1000
            method = "profiler_npu"
        else:
            import triton.testing
            execution_time_ms = triton.testing.do_bench(
                base_benchmark_fn,
                warmup={warmup},
                rep={runs},
                return_mode="min"
            )
            method = "triton_do_bench"
"""
            else:
                code = f"""        import triton.testing
        def base_benchmark_fn():
            result = framework_model(*inputs)
            return result
        
        execution_time_ms = triton.testing.do_bench(
            base_benchmark_fn,
            warmup={warmup},
            rep={runs},
            return_mode="min"
        )
        method = "triton_do_bench"
"""
        elif self.dsl == "cpp":
            code = f"""        # CPU
        import time
        def base_benchmark_fn():
            return framework_model(*inputs)
        # 执行 warmup
        for _ in range({warmup}):
            _ = base_benchmark_fn()
        # 计时 rep 次
        start_t = time.perf_counter()
        for _ in range({runs}):
            _ = base_benchmark_fn()
        end_t = time.perf_counter()
        execution_time_ms = (end_t - start_t) * 1000.0 / max({runs}, 1)
        method = "cpu_loop_timer"
"""
        else:
            sync_code = "torch.cuda.synchronize()" if self.backend == "cuda" else (
                "torch.npu.synchronize()" if self.backend == "ascend" else ""
            )
            code = f"""        # 非triton实现，使用传统循环计时
        import time
        start_time = time.time()
        for _ in range({warmup + runs}):
            framework_output = framework_model(*inputs)
            {sync_code}
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000 / {warmup + runs}  # 转换为毫秒
        method = "traditional_timing"
"""
        return code

    def run_msprof(self, script_path: str) -> Tuple[bool, str, Optional[str]]:
        """运行msprof性能分析"""
        try:
            process = subprocess.run(
                f'msprof --application="python {script_path}"',
                shell=True, capture_output=True, text=True, timeout=600
            )

            for line in process.stdout.split('\n'):
                if "[INFO] Process profiling data complete. Data is saved in" in line:
                    match = re.search(r"Data is saved in (.+)$", line)
                    if match:
                        return True, "", match.group(1).strip()

            return False, "未找到数据保存路径", None
        except Exception as e:
            return False, f"执行错误: {str(e)}", None

    def analyze_prof_data(self, prof_path: str, warmup_times: int, run_times: int) -> Tuple[bool, str, float]:
        """分析PROF数据"""
        try:
            csv_files = list(Path(prof_path).glob("mindstudio_profiler_output/op_summary_*.csv"))
            if not csv_files:
                return False, "未找到CSV文件", 0.0

            df = pd.read_csv(csv_files[0])

            # 移除特定的Op
            df_filtered = df[~df["Op Name"].str.contains("aclnnIsClose_IsCloseAiCpu_IsClose|aclnnAll_ReduceAll_ReduceAll",
                                                         regex=True, na=False)]

            total_count = warmup_times + run_times
            op_counts = df_filtered["Op Name"].value_counts()
            valid_ops = op_counts[op_counts == total_count]

            if len(valid_ops) == 0:
                return False, "没有找到符合预期次数的Op", float('inf')

            # 检查不匹配的Op
            invalid_ops = op_counts[op_counts != total_count]
            if len(invalid_ops) > 0:
                logger.warning(f"[{self.task_id}:{self.op_name}] 发现{len(invalid_ops)}个Op次数不匹配")

            # 计算平均时间
            df_valid = df_filtered[df_filtered["Op Name"].isin(valid_ops.index)]
            total_avg_time = 0.0

            for op_name in valid_ops.index:
                op_data = df_valid[df_valid["Op Name"] == op_name]["Task Duration(us)"].tolist()
                if len(op_data) > warmup_times:
                    valid_data = op_data[warmup_times:]
                    avg_time = sum(valid_data) / len(valid_data)
                    total_avg_time += avg_time

            return True, "", total_avg_time

        except Exception as e:
            return False, f"分析数据时出错: {str(e)}", float('inf')

    def run_nsys(self, script_path: str) -> Tuple[bool, str, Optional[str]]:
        """运行nsys性能分析"""
        try:
            output_name = "nsys_report_" + os.path.basename(script_path).replace(".py", "")
            cmd = f'nsys profile --output={output_name} python {script_path}'
            logger.debug(f"Running nsys profile: {cmd}")
            process = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            report_path = os.path.join(os.path.dirname(script_path), output_name + ".nsys-rep")

            if os.path.exists(report_path):
                return True, "", report_path
            return False, "未找到nsys报告文件", None
        except Exception as e:
            return False, f"执行错误: {str(e)}", None

    def analyze_nsys_data(self, rep_path: str, warmup_times: int, run_times: int, profile_type: str = "") -> Tuple[bool, str, float]:
        """分析nsys生成的rep文件，返回平均耗时(us)，统计方式与analyze_prof_data一致"""

        try:
            dir_plib = Path(rep_path).resolve().parent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 在CSV文件名中添加profile_type标识
            type_suffix = f"_{profile_type}" if profile_type else ""
            csv_base = f"nsys_report_{timestamp}{type_suffix}"
            csv_path = dir_plib / csv_base  # rep_path.replace(".nsys-rep", ".csv")
            # 导出csv
            cmd = f'nsys stats --report gputrace  --timeunit us  --format csv --output {csv_path} {rep_path}'
            logger.debug(f"Running nsys stats: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            csv_path = dir_plib / f"{csv_base}_gputrace.csv"

            if not os.path.exists(csv_path):
                return False, "未生成csv文件", float('inf')
            df = pd.read_csv(csv_path)
            # 兼容不同nsys版本的列名
            name_col = None
            for col in df.columns:
                if col.lower() in ["name", "function name", "kernel name", "Name"]:
                    name_col = col
                    break
            if not name_col:
                # 兜底找包含name的列
                for col in df.columns:
                    if "name" in col.lower():
                        name_col = col
                        break
            time_col = None
            for col in df.columns:
                if "time (ns)" in col.lower() or "average" in col.lower() or "duration" in col.lower():
                    time_col = col
                    break
            if not name_col or not time_col:
                return False, "未找到kernel名或耗时列", float('inf')
            total_count = warmup_times + run_times
            op_counts = df[name_col].value_counts()
            valid_ops = op_counts[op_counts == total_count]
            if len(valid_ops) == 0:
                return False, "没有找到符合预期次数的kernel", float('inf')
            df_valid = df[df[name_col].isin(valid_ops.index)]
            total_avg_time = 0.0
            for op_name in valid_ops.index:
                op_data = df_valid[df_valid[name_col] == op_name][time_col].tolist()
                if len(op_data) > warmup_times:
                    valid_data = op_data[warmup_times:]
                    avg_time = sum(valid_data) / len(valid_data)
                    total_avg_time += avg_time  # timeunit us
            return True, "", total_avg_time
        except Exception as e:
            return False, f"分析nsys数据时出错: {str(e)}", float('inf')

    def save_speedup_result(self, speedup: float, base_time: float, gen_time: float, unique_dir: str):
        """保存加速比结果到txt文件"""
        try:
            profiling_dir = os.path.join(os.path.expanduser(self.log_dir), self.op_name, "profiling")
            os.makedirs(profiling_dir, exist_ok=True)

            filepath = os.path.join(profiling_dir, "speed_up_record.txt")

            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"op_name: {self.op_name}, task_id: {self.task_id}, unique_dir: {unique_dir}, ")
                f.write(f"base_time: {base_time:.6f} us, generation_time: {gen_time:.6f} us, ")
                f.write(f"speedup: {speedup:.6f}x\n")

            logger.debug(f"[{self.task_id}:{self.op_name}] 加速比结果已保存")

        except Exception as e:
            logger.warning(f"[{self.task_id}:{self.op_name}] 保存加速比结果失败: {str(e)}")

    def run_profile(self, current_step: int = 0, device_id: str = "0", profile_settings: dict = {}) -> dict:
        """运行profile分析
        
        Returns:
            dict: 性能分析结果，包含以下字段：
                - gen_time: 生成代码执行时间（微秒）
                - base_time: 基准代码执行时间（微秒）
                - speedup: 加速比
                - autotune_summary: autotune配置详情（仅triton DSL）
        """
        original_cwd = os.getcwd()
        try:
            run_times = profile_settings.get("run_times", 50)
            warmup_times = profile_settings.get("warmup_times", 5)

            # 获取验证目录
            expanded_log_dir = os.path.expanduser(self.log_dir)
            unique_dir_name = f"I{self.task_id}_S{current_step:02d}_verify"
            verify_dir = os.path.join(expanded_log_dir, self.op_name, unique_dir_name)

            os.chdir(verify_dir)

            # 生成profile脚本并运行
            self.gen_profile_project(verify_dir, device_id, warmup_times, run_times)

            # 检查是否为triton DSL，如果是则运行脚本获取性能数据
            if "triton_cuda" in self.dsl or "triton_ascend" in self.dsl:
                base_time, gen_time = self.run_profile_scripts_and_collect_results(verify_dir)
            elif self.backend == "ascend":
                _, _, base_prof_path = self.run_msprof(os.path.join(verify_dir, f"profile_{self.op_name}_base.py"))
                _, _, gen_prof_path = self.run_msprof(os.path.join(
                    verify_dir, f"profile_{self.op_name}_generation.py"))
                _, _, base_time = self.analyze_prof_data(base_prof_path, warmup_times, run_times)
                _, _, gen_time = self.analyze_prof_data(gen_prof_path, warmup_times, run_times)
            elif self.backend == "cuda":
                _, _, base_prof_path = self.run_nsys(os.path.join(verify_dir, f"profile_{self.op_name}_base.py"))
                _, _, base_time = self.analyze_nsys_data(base_prof_path, warmup_times, run_times, "base")
                _, _, gen_prof_path = self.run_nsys(os.path.join(verify_dir, f"profile_{self.op_name}_generation.py"))
                _, _, gen_time = self.analyze_nsys_data(gen_prof_path, warmup_times, run_times, "generation")
            elif self.backend == "cpu":
                # CPU后端使用脚本方式收集性能数据
                base_time, gen_time = self.run_profile_scripts_and_collect_results(verify_dir)
            else:
                logger.warning(f"[{self.task_id}:{self.op_name}] 不支持的backend: {self.backend}")
                return {
                    'gen_time': float('inf'),
                    'base_time': 0.0,
                    'speedup': 0.0
                }

            speedup = base_time / gen_time if gen_time > 0 else 0.0
            speedup_percent = speedup * 100.0
            self.save_speedup_result(speedup, base_time, gen_time, unique_dir_name)
            logger.info(f"orig performance is {base_time:.2f} us")
            logger.info(f"aikg performance is {gen_time:.2f} us")
            logger.info(f"[{self.task_id}:{self.op_name}] 性能分析完成，加速比（基准为100%）: {speedup_percent:.2f} %")
            
            # 构建返回结果
            result = {
                'gen_time': gen_time,
                'base_time': base_time,
                'speedup': speedup,
                'unique_dir': unique_dir_name,
            }
            
            # 只在 triton_ascend 情况下添加 autotune_summary
            if "triton_ascend" in self.dsl and self.backend == "ascend":
                autotune_summary = self.read_autotune_results_from_directory(verify_dir)
                if autotune_summary:
                    result['autotune_summary'] = autotune_summary
                    logger.info(f"[{self.op_name}: {self.task_id}] Autotune配置详情:\n{autotune_summary}")
            
            return result
        except Exception as e:
            logger.warning(f"[{self.task_id}:{self.op_name}] 性能分析失败: {str(e)}")
            return {
                'gen_time': float('inf'),
                'base_time': 0.0,
                'speedup': 0.0
            }
        finally:
            # 恢复原始工作目录
            try:
                os.chdir(original_cwd)
            except Exception:
                pass

    def run_profile_scripts_and_collect_results(self, verify_dir: str) -> Tuple[float, float]:
        """运行性能测试脚本并收集结果

        Args:
            verify_dir: 验证目录，包含性能测试脚本

        Returns:
            (base_time_us, gen_time_us): 基准时间和生成时间（微秒）
        """
        try:
            # 保存当前工作目录
            original_cwd = os.getcwd()

            # 切换到验证目录
            os.chdir(verify_dir)

            try:
                # 步骤1：运行基准性能测试脚本
                base_script = f"profile_{self.op_name}_base.py"
                base_result = run_command(["python", base_script], cmd_msg="base_profile", timeout=300)
                if not base_result[0]:
                    logger.error(f"[{self.op_name}: {self.task_id}] 基准性能脚本执行失败: {base_result[1]}")
                    return float('inf'), float('inf')

                # 步骤2：运行生成代码性能测试脚本
                gen_script = f"profile_{self.op_name}_generation.py"
                gen_result = run_command(["python", gen_script], cmd_msg="generation_profile", timeout=300)
                if not gen_result[0]:
                    logger.error(f"[{self.op_name}: {self.task_id}] 生成代码性能脚本执行失败: {gen_result[1]}")
                    return float('inf'), float('inf')

                # 步骤3：从JSON文件读取性能数据
                base_time_us = self.read_profile_result_from_json(verify_dir, "base_profile_result.json")
                gen_time_us = self.read_profile_result_from_json(verify_dir, "generation_profile_result.json")

                return base_time_us, gen_time_us

            finally:
                # 恢复原始工作目录
                os.chdir(original_cwd)

        except Exception as e:
            logger.error(f"[{self.op_name}: {self.task_id}] 性能脚本执行和结果收集失败: {e}")
            return float('inf'), float('inf')

    def read_autotune_results_from_directory(self, verify_dir: str) -> str:
        """从验证目录读取所有autotune结果并格式化输出
        
        读取指定目录下的所有 autotune_info_case_*.json 文件，
        并以类似 TRITON_PRINT_AUTOTUNING=1 的格式输出。
        
        Args:
            verify_dir: 验证目录路径
            
        Returns:
            格式化的autotune结果字符串，格式如下：
            
            Case 0:
            All config timings for kernel_name:
              Config 1: BLOCK_M=128, BLOCK_N=256 -> 145.2300us (BEST)
              Config 2: BLOCK_M=64, BLOCK_N=128 -> 178.5600us
              ...
        """
        from pathlib import Path
        
        result_lines = []
        
        # 查找所有autotune文件
        verify_path = Path(verify_dir)
        autotune_files = sorted(verify_path.glob("autotune_info_case_*.json"))
        
        if not autotune_files:
            return ""
        
        # 逐个读取并格式化
        for autotune_file in autotune_files:
            # 提取case索引
            case_idx = autotune_file.stem.split('_')[-1]
            
            try:
                with open(autotune_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                result_lines.append(f"Case {case_idx}:")
                
                # 遍历每个kernel
                for kernel_name, configs in data.items():
                    result_lines.append(f"All config timings for {kernel_name}:")
                    
                    # 按rank排序输出
                    sorted_configs = sorted(configs, key=lambda x: x['rank'])
                    
                    for config_info in sorted_configs:
                        config_str = config_info['config']
                        timing_us = config_info['timing_us']
                        is_best = config_info['is_best']
                        rank = config_info['rank']
                        
                        status = " (BEST)" if is_best else ""
                        result_lines.append(f"  Config {rank}: {config_str} -> {timing_us:.4f}us{status}")
                
                result_lines.append("")  # 空行分隔不同case
                
            except Exception as e:
                logger.warning(f"[{self.op_name}: {self.task_id}] 读取autotune文件失败 {autotune_file.name}: {e}")
        
        return "\n".join(result_lines)

    def read_profile_result_from_json(self, verify_dir: str, result_file: str) -> float:
        """从JSON文件读取性能测试结果
        
        该方法读取性能测试脚本生成的JSON结果文件，提取执行时间。
        
        JSON文件格式示例：
        {
            "execution_time_us": 145.23,
            "execution_time_ms": 0.14523,
            "method": "triton_do_bench",
            "warmup_times": 5,
            "run_times": 50
        }

        Args:
            verify_dir: 验证目录
            result_file: 结果文件名（如 "base_profile_result.json"）

        Returns:
            execution_time_us: 执行时间（微秒），失败时返回 float('inf')
        """
        try:
            result_path = os.path.join(verify_dir, result_file)
            if not os.path.exists(result_path):
                logger.error(f"[{self.op_name}: {self.task_id}] 性能结果文件不存在: {result_path}")
                return float('inf')

            with open(result_path, 'r') as f:
                result_data = json.load(f)

            # 获取时间结果（微秒）
            execution_time_us = result_data.get("execution_time_us", float('inf'))
            method = result_data.get("method", "unknown")

            logger.info(f"[{self.op_name}: {self.task_id}] 从 {result_file} 读取性能数据: {execution_time_us:.4f} us (method: {method})")
            return execution_time_us

        except Exception as e:
            logger.error(f"[{self.op_name}: {self.task_id}] 读取性能结果文件失败 {result_file}: {e}")
            return float('inf')

    def _detect_triton_autotune(self, code: str) -> bool:
        """
        检测代码中是否包含@triton.autotune装饰器
        
        Args:
            code: triton代码
            
        Returns:
            bool: 是否包含autotune装饰器
        """
        return '@triton.autotune' in code or '@autotune' in code
    
    def _extract_autotune_configs(self, code: str) -> list:
        """
        从triton代码中提取所有未被注释的autotune config
        
        跳过已经被注释掉的config（通常是之前验证失败的）
        
        Args:
            code: triton代码
            
        Returns:
            list: config列表，每个config是一个字符串（只包含未注释的）
        """
        import re
        
        # 匹配@triton.autotune装饰器块
        pattern = r'@triton\.autotune\s*\(\s*configs\s*=\s*\[(.*?)\]'
        match = re.search(pattern, code, re.DOTALL)
        
        if not match:
            return []
        
        configs_str = match.group(1)
        
        # 匹配所有triton.Config(...)，使用更宽松的模式
        config_pattern = r'triton\.Config\s*\([^)]*\{[^}]+\}[^)]*\)'
        all_matches = re.finditer(config_pattern, configs_str, re.DOTALL)
        
        valid_configs = []
        for match in all_matches:
            # 获取匹配位置之前的内容
            start_pos = match.start()
            # 查找这个config之前最近的换行符位置
            last_newline = configs_str.rfind('\n', 0, start_pos)
            line_start = last_newline + 1 if last_newline != -1 else 0
            # 获取从行首到config开始的内容
            prefix = configs_str[line_start:start_pos]
            
            # 如果prefix中没有#，说明未被注释
            if '#' not in prefix:
                valid_configs.append(match.group(0))
        
        return valid_configs
    
    def _count_all_autotune_configs(self, code: str) -> int:
        """
        统计所有autotune config的数量（包括已注释的）
        
        Args:
            code: triton代码
            
        Returns:
            int: config总数
        """
        import re
        
        # 匹配@triton.autotune装饰器块
        pattern = r'@triton\.autotune\s*\(\s*configs\s*=\s*\[(.*?)\]'
        match = re.search(pattern, code, re.DOTALL)
        
        if not match:
            return 0
        
        configs_str = match.group(1)
        
        # 统计所有包含triton.Config的行（无论是否注释）
        count = configs_str.count('triton.Config')
        
        return count
    
    def _generate_single_config_code(self, original_code: str, config_to_keep: str, config_index: int) -> str:
        """
        生成只包含单个config的代码（其他config被注释掉）
        
        Args:
            original_code: 原始代码
            config_to_keep: 要保留的config字符串
            config_index: config的索引（用于注释）
            
        Returns:
            str: 修改后的代码
        """
        import re
        
        # 找到所有config
        all_configs = self._extract_autotune_configs(original_code)
        
        if not all_configs:
            return original_code
        
        # 构建新的configs列表（只保留一个config）
        new_configs_block = f"configs=[\n        {config_to_keep},\n    ]"
        
        # 替换原来的configs块
        pattern = r'configs\s*=\s*\[(.*?)\]'
        modified_code = re.sub(pattern, new_configs_block, original_code, count=1, flags=re.DOTALL)
        
        return modified_code
    
    def _generate_final_code_with_valid_configs(self, original_code: str, valid_configs: list, all_configs: list) -> str:
        """
        生成最终代码：保留正确的config，注释掉错误的config
        
        Args:
            original_code: 原始代码
            valid_configs: 正确的config列表
            all_configs: 所有config列表
            
        Returns:
            str: 修改后的代码
        """
        import re
        
        if not all_configs:
            return original_code
        
        # 统一逻辑：遍历所有config，在valid_configs中的保留，否则注释掉
        new_configs_lines = []
        
        for config in all_configs:
            if config in valid_configs:
                # 保留正确的config
                new_configs_lines.append(f"        {config},")
            else:
                # 注释掉错误的config并添加失败标注
                config_lines = config.split('\n')
                commented_lines = [f"        # {line}" if line.strip() else line for line in config_lines]
                new_configs_lines.append('\n'.join(commented_lines) + ',  # Failed verification')
        
        new_configs_block = f"configs=[\n" + "\n".join(new_configs_lines) + "\n    ]"
        
        # 替换原来的configs块
        pattern = r'configs\s*=\s*\[(.*?)\]'
        modified_code = re.sub(pattern, new_configs_block, original_code, count=1, flags=re.DOTALL)
        
        return modified_code
    
    def _save_verification_result_to_jsonl(self, verify_dir: str, current_step: int, verification_passed: bool, 
                                          verify_logs: str, all_configs_count: int = 0, valid_configs_count: int = 0):
        """
        保存验证结果到JSONL文件
        
        Args:
            verify_dir: 验证目录
            current_step: 当前步骤
            verification_passed: 验证是否通过
            verify_logs: 验证日志
            all_configs_count: 所有config数量（autotune专用）
            valid_configs_count: 通过的config数量（autotune专用）
        """
        result_jsonl_path = os.path.join(os.path.expanduser(self.log_dir), "verification_results.jsonl")
        result_info = {
            "task_name": self.op_name,
            "task_id": self.task_id,
            "step": current_step,
            "verify_dir": verify_dir,
            "passed": verification_passed,
            "error_log": verify_logs,
            "timestamp": datetime.now().isoformat(),
            "framework": self.framework,
            "dsl": self.dsl,
            "backend": self.backend,
            "arch": self.arch
        }
        
        # 如果是autotune验证，添加config信息
        if all_configs_count > 0:
            result_info["autotune_configs"] = {
                "total": all_configs_count,
                "passed": valid_configs_count
            }
        
        with open(result_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_info, ensure_ascii=False, indent=2) + '\n\n')
    
    def _verify_configs_separately(self, target_code: str, verify_dir: str, device_id: int, verify_timeout: int, current_step: int = 0) -> Tuple[bool, str, str]:
        """
        单独验证每个autotune config
        
        Args:
            target_code: 原始triton代码
            verify_dir: 验证目录
            device_id: 设备ID
            verify_timeout: 验证超时时间
            current_step: 当前步骤（用于记录）
            
        Returns:
            Tuple[bool, str, str]: (是否有config通过, 验证日志, 最终代码)
        """
        logger.info(f"[{self.op_name}] 检测到autotune装饰器，开始单独验证各个config...")
        
        # 统计所有config数量（包括已注释的）
        total_configs_count = self._count_all_autotune_configs(target_code)
        
        # 提取未被注释的config
        all_configs = self._extract_autotune_configs(target_code)
        
        if not all_configs:
            if total_configs_count > 0:
                # 所有config都被注释，生成验证文件但不运行，直接返回False
                logger.info(f"[{self.op_name}] 检测到 {total_configs_count} 个config，但全部已被注释（之前验证失败）")
                
                verify_logs = []
                verify_logs.append(f"=== Autotune Config 验证 ===\n")
                verify_logs.append(f"检测到 {total_configs_count} 个config，全部已被注释（之前验证失败）\n")
                verify_logs.append(f"跳过验证，直接返回失败结果\n")
                
                # 生成最终验证项目（包含所有被注释的config）
                try:
                    self.gen_verify_project(target_code, verify_dir, device_id)
                    logger.info(f"[{self.op_name}] 验证项目已生成（全部config已注释）")
                except Exception as e:
                    error_msg = str(e)
                    verify_logs.append(f"\n生成验证项目失败: {error_msg}\n")
                    logger.error(f"[{self.op_name}] 生成验证项目失败: {error_msg}")
                
                # 保存验证结果到JSONL
                self._save_verification_result_to_jsonl(
                    verify_dir, current_step, False, 
                    "".join(verify_logs), total_configs_count, 0
                )
                
                return False, "".join(verify_logs), target_code
            else:
                logger.warning(f"[{self.op_name}] 未能提取到config，使用正常验证流程")
                return None, "", target_code
        
        skipped_count = total_configs_count - len(all_configs)
        if skipped_count > 0:
            logger.info(f"[{self.op_name}] 检测到 {total_configs_count} 个config，其中 {skipped_count} 个已被注释（跳过），将验证剩余 {len(all_configs)} 个config")
        else:
            logger.info(f"[{self.op_name}] 提取到 {len(all_configs)} 个config，开始逐个验证...")
        
        valid_configs = []
        verify_logs = []
        verify_logs.append(f"=== Autotune Config 单独验证 ===\n")
        if skipped_count > 0:
            verify_logs.append(f"检测到 {total_configs_count} 个config，其中 {skipped_count} 个已被注释（跳过验证）\n")
            verify_logs.append(f"待验证config数量: {len(all_configs)}\n\n")
        else:
            verify_logs.append(f"总共 {len(all_configs)} 个config\n\n")
        
        # 为每个config生成单独的验证文件并验证
        for i, config in enumerate(all_configs):
            config_num = i + 1
            logger.info(f"[{self.op_name}] 验证 Config {config_num}/{len(all_configs)}...")
            verify_logs.append(f"--- Config {config_num} ---\n")
            verify_logs.append(f"{config}\n")
            
            try:
                # 生成只包含当前config的代码
                single_config_code = self._generate_single_config_code(target_code, config, i)
                
                # 生成临时验证项目
                temp_verify_dir = os.path.join(verify_dir, f"config_{config_num}_verify")
                os.makedirs(temp_verify_dir, exist_ok=True)
                
                # 生成验证项目
                self.gen_verify_project(single_config_code, temp_verify_dir, device_id)
                
                # 运行验证
                config_res, config_log = self.run_verify(temp_verify_dir, timeout=verify_timeout)
                
                if config_res:
                    verify_logs.append(f"验证通过\n\n")
                    valid_configs.append(config)
                    logger.info(f"[{self.op_name}] Config {config_num} 验证通过")
                else:
                    verify_logs.append(f"验证失败\n")
                    verify_logs.append(f"错误日志:\n{config_log}\n\n")
                    logger.info(f"[{self.op_name}] Config {config_num} 验证失败")
                
                # 清理临时目录
                shutil.rmtree(temp_verify_dir, ignore_errors=True)
                
            except Exception as e:
                error_msg = str(e)
                verify_logs.append(f"验证异常: {error_msg}\n\n")
                logger.error(f"[{self.op_name}] Config {config_num} 验证异常: {error_msg}")
        
        # 生成验证结果摘要
        verify_logs.append(f"通过的config数量: {len(valid_configs)}/{len(all_configs)}\n")
        
        # 统一生成最终代码（正确的保留，错误的注释掉）
        final_code = self._generate_final_code_with_valid_configs(target_code, valid_configs, all_configs)
        verification_passed = len(valid_configs) > 0
        
        # 记录验证结果
        if verification_passed:
            verify_logs.append(f"验证通过，保留了 {len(valid_configs)} 个正确的config\n")
            logger.info(f"[{self.op_name}] Autotune config验证完成: {len(valid_configs)}/{len(all_configs)} 通过")
        else:
            verify_logs.append(f"所有config都未通过验证\n")
            logger.info(f"[{self.op_name}] 所有config都未通过验证")
        
        verify_logs.append(f"\n=== 生成最终验证项目 ===\n")
        
        # 使用最终代码生成完整的验证项目
        try:
            # 直接在verify_dir下生成验证项目
            self.gen_verify_project(final_code, verify_dir, device_id)
            logger.info(f"[{self.op_name}] 最终验证项目生成成功")
            
            # 只有验证通过时才复制到passed_cases文件夹
            if verification_passed:
                folder_name = os.path.basename(verify_dir)
                dst_dir = Path(self.log_dir) / "passed_cases" / self.op_name / folder_name
                shutil.copytree(verify_dir, dst_dir)
                logger.info(f"[{self.op_name}] 验证文件已保存到: {dst_dir}")
            
        except Exception as e:
            error_msg = str(e)
            verify_logs.append(f"生成最终验证项目失败: {error_msg}\n")
            logger.error(f"[{self.op_name}] 生成最终验证项目失败: {error_msg}")
            verification_passed = False
        
        # 统一保存验证结果到JSONL
        self._save_verification_result_to_jsonl(
            verify_dir, current_step, verification_passed, 
            "".join(verify_logs), len(all_configs), len(valid_configs)
        )
        
        return verification_passed, "".join(verify_logs), final_code

    def run(self, task_info: Dict[str, Any], current_step: int = 0, device_id: int = 0):
        """
        运行内核验证器，验证代码的正确性

        Args:
            task_info: 任务信息字典，包含所有代码和状态
            current_step: 当前步骤
            device_id: 设备ID

        Returns:
            Tuple[bool, str]: (验证结果, 错误日志)
        """
        logger.info(f"Verifier Run - Step: {current_step}, Device: {device_id}")

        # 根据实现类型从task_info获取代码
        target_code = task_info.get('coder_code', '')

        if not target_code:
            logger.error("No target code found for verification")
            return False, "No target code found for verification"

        # 动态创建验证目录
        verify_dir = self._create_verify_dir(current_step)
        
        # 检测是否是triton autotune代码
        is_triton_autotune = (self.dsl in ["triton_cuda", "triton_ascend"] and 
                              self._detect_triton_autotune(target_code))
        
        if is_triton_autotune:
            # 对于autotune的triton代码，单独验证每个config
            config_verify_result, config_verify_log, final_code = self._verify_configs_separately(
                target_code, verify_dir, device_id, self.config.get('verify_timeout', 300), current_step
            )
            
            if config_verify_result is not None:
                # 如果执行了config单独验证，更新代码并返回
                if config_verify_result:
                    # 更新task_info中的代码为只包含正确config的版本
                    task_info['coder_code'] = final_code
                
                return config_verify_result, config_verify_log

        # 在独立目录中生成验证项目
        project_gen_log = ""  # 用于存储项目生成阶段的日志
        try:
            self.gen_verify_project(target_code, verify_dir, device_id)
        except Exception as e:
            # 捕获gen_verify_project中的异常，记录到project_gen_log中
            error_msg = str(e)
            logger.error(f"验证项目生成失败: {error_msg}")
            project_gen_log = f"项目生成失败: {error_msg}\n"

        # 从config获取timeout配置，默认5分钟
        verify_timeout = self.config.get('verify_timeout', 300)

        # 运行验证
        verify_res, verify_log = self.run_verify(verify_dir, timeout=verify_timeout)
        
        # 拼接项目生成日志和验证日志
        verify_log = project_gen_log + verify_log

        # 保存验证结果到JSONL文件
        self._save_verification_result_to_jsonl(verify_dir, current_step, verify_res, verify_log)

        # 保存通过的验证文件
        if verify_res:
            foder_name = os.path.basename(verify_dir)
            dst_dir = Path(self.log_dir) / "passed_cases" / self.op_name / foder_name
            shutil.copytree(verify_dir, dst_dir)

        return verify_res, verify_log
