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
from ai_kernel_generator.core.worker.interface import WorkerInterface
import tarfile
import io
import ast
import asyncio

# 模板路径
TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "kernel_verify_template_refactored.j2")
PROFILE_BASE_TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "prof_base_template_refactored.j2")
PROFILE_GENERATION_TEMPLATE_PATH = os.path.join(
    get_project_root(), "resources", "templates", "prof_generation_template_refactored.j2")
PROFILE_SINGLE_TASK_TEMPLATE_PATH = os.path.join(
    get_project_root(), "resources", "templates", "prof_single_task_template.j2")
# 生成CMakeLists.txt和运行脚本的路径
CMAKE_TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "cmake_template.j2")
RUN_TEMPLATE_PATH = os.path.join(get_project_root(), "utils", "compile_tools", "ascend_compile", "run.sh")

# 类型定义
FrameworkType = Literal["torch", "mindspore", "numpy"]
ImplType = Literal["triton_cuda", "triton_ascend", "triton-russia", "swft", "cuda_c", "cpp", "tilelang_npuir", "tilelang_cuda", "ascendc", "torch"]
BackendType = Literal["cuda", "ascend", "cpu"]
ArchType = Literal["a100", "v100", "h20", "l20", "rtx3090", "ascend910b4", "ascend310p3", "x86_64", "aarch64"]

logger = logging.getLogger(__name__)


def sync_artifacts_to_directory(artifacts: Dict[str, str], target_dir: str, task_id: str = "0") -> None:
    """
    将 artifacts 同步到目标目录。
    
    Args:
        artifacts: 从 Worker 返回的 artifacts 字典，格式为 {relative_path: file_content}
                   例如: {"autotune_info_case_0.json": "{...}", "subdir/result.jsonl": "..."}
        target_dir: 目标目录路径（通常是 verify_dir）
        task_id: 任务ID（用于日志）
    """
    if not artifacts:
        return
        
    logger.info(f"[{task_id}] Syncing {len(artifacts)} artifact files to {target_dir}")
    
    for rel_path, content in artifacts.items():
        # 构建完整路径
        full_path = os.path.join(target_dir, rel_path)
        
        # 确保目录存在
        dir_path = os.path.dirname(full_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"[{task_id}] Created directory: {dir_path}")
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.debug(f"[{task_id}] Synced artifact: {rel_path}")
        except Exception as e:
            logger.warning(f"[{task_id}] Failed to sync artifact {rel_path}: {e}")


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
                 config: Optional[Dict[str, Any]] = None,
                 worker: Optional[WorkerInterface] = None):
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
            worker (WorkerInterface, optional): Worker实例，用于执行验证任务
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
            # 对于 triton_cuda 和 triton_ascend，统一使用 ModelNew 类格式
            self.impl_func_name = impl_func_name or "ModelNew"
        elif self.dsl == "torch":
            # 对于 torch DSL (Triton → PyTorch 转换)，统一使用 ModelNew 类格式
            self.impl_func_name = impl_func_name or "ModelNew"
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

        # 保存Worker实例（可以在运行时动态设置）
        self.worker = worker

    def check_task_desc_static(self, code: str) -> Tuple[bool, str]:
        """
        静态检查 task_desc 代码是否符合规范
        
        Args:
            code: task_desc 代码字符串
            
        Returns:
            Tuple[bool, str]: (是否通过, 错误信息)
        """
        try:
            tree = ast.parse(code)
            
            has_model_class = False
            has_get_inputs = False
            has_get_init_inputs = False
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == 'Model':
                    has_model_class = True
                elif isinstance(node, ast.FunctionDef):
                    if node.name == 'get_inputs':
                        has_get_inputs = True
                    elif node.name == 'get_init_inputs':
                        has_get_init_inputs = True
            
            missing = []
            if not has_model_class:
                missing.append("class Model")
            if not has_get_inputs:
                missing.append("function get_inputs")
            if not has_get_init_inputs:
                missing.append("function get_init_inputs")
                
            if missing:
                return False, f"Missing required components in task_desc: {', '.join(missing)}"
                
            return True, ""
            
        except SyntaxError as e:
            return False, f"Syntax error in task_desc: {e}"
        except Exception as e:
            return False, f"Static check failed: {e}"

    async def check_task_desc_runtime(self, task_desc: str, timeout: int = 60) -> Tuple[bool, str]:
        """
        运行时检查 task_desc 代码是否能正确执行
        
        Args:
            task_desc: task_desc 代码字符串
            timeout: 超时时间
            
        Returns:
            Tuple[bool, str]: (是否通过, 错误信息)
        """
        # 1. 创建临时验证目录
        check_dir = os.path.join(os.path.expanduser(self.log_dir), f"{self.op_name}_check_desc_{self.task_id}")
        os.makedirs(check_dir, exist_ok=True)
        
        try:
            # 2. 写入 task_desc 到 reference.py
            ref_file = os.path.join(check_dir, "reference.py")
            with open(ref_file, "w", encoding="utf-8") as f:
                f.write(task_desc)
                
            # 3. 生成验证脚本 verify_{op_name}.py
            verify_script_content = f"""
import torch
import sys
import os

# Add current directory to sys.path
sys.path.append(os.getcwd())

def run_check():
    print("Starting reference check...")
    try:
        # Import from reference
        try:
            from reference import Model, get_inputs, get_init_inputs
        except ImportError as e:
            print(f"Import failed: {{e}}")
            return False
            
        print("Successfully imported Model and helper functions.")
        
        # Determine device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            device = "npu"
            
        print(f"Using device: {{device}}")
        
        # Instantiate model
        try:
            init_inputs = get_init_inputs()
            model = Model(*init_inputs)
            if device != "cpu":
                model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Model instantiation failed: {{e}}")
            return False
            
        # Get inputs
        try:
            inputs = get_inputs()
            if device != "cpu":
                inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]
        except Exception as e:
            print(f"get_inputs failed: {{e}}")
            return False
            
        # Run forward pass
        try:
            output = model(*inputs)
            print("Forward pass successful.")
        except Exception as e:
            print(f"Forward pass failed: {{e}}")
            return False
            
        return True

    except Exception as e:
        print(f"Unexpected error: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_check()
    if success:
        print("REFERENCE_CHECK_SUCCESS")
        sys.exit(0)
    else:
        print("REFERENCE_CHECK_FAILED")
        sys.exit(1)
"""
            verify_file = os.path.join(check_dir, f"verify_{self.op_name}.py")
            with open(verify_file, "w", encoding="utf-8") as f:
                f.write(verify_script_content)
                
            # 4. 打包目录
            package_data = self._pack_directory(check_dir)
            
            # 5. 使用 Worker 执行
            if not self.worker:
                raise RuntimeError("Worker not set for runtime check")
                
            # 注意：这里我们不需要显式管理 device，因为 reference check 通常只做一个简单的 forward pass
            # 如果是 remote worker，它会自动分发；如果是 local worker，它通常不需要特定的 device lock (除非 OOM)
            # 但为了安全起见，调用方应该已经处理了 resource locking
            
            success, log, _ = await self.worker.verify(package_data, f"{self.task_id}_check", self.op_name, timeout)
            
            if success and "REFERENCE_CHECK_SUCCESS" in log:
                return True, ""
            else:
                return False, f"Runtime check failed:\n{log}"
                
        except Exception as e:
            return False, f"Runtime check exception: {str(e)}"
        finally:
            # 清理临时目录
            shutil.rmtree(check_dir, ignore_errors=True)

    async def generate_reference_data(self, task_desc: str, timeout: int = 120) -> Tuple[bool, str, bytes]:
        """
        在 GPU 上执行 task_desc 并生成参考数据
        
        用于 CUDA-to-Ascend 转换场景：在 GPU Worker 上执行 Triton-CUDA 代码，
        保存输出作为参考数据，供 NPU Worker 验证转换后的代码正确性。
        
        Args:
            task_desc: task_desc 代码字符串（Triton-CUDA 代码）
            timeout: 超时时间
            
        Returns:
            Tuple[bool, str, bytes]: (是否成功, 日志, 参考数据bytes)
            - 成功时 bytes 为 .pt 文件内容
            - 失败时 bytes 为空 b''
        """
        # 1. 创建临时目录
        ref_dir = os.path.join(os.path.expanduser(self.log_dir), f"{self.op_name}_gen_ref_{self.task_id}")
        os.makedirs(ref_dir, exist_ok=True)
        
        try:
            # 2. 写入 task_desc 到 reference.py
            ref_file = os.path.join(ref_dir, "reference.py")
            with open(ref_file, "w", encoding="utf-8") as f:
                f.write(task_desc)
            
            # 3. 生成参考数据脚本
            # 使用固定 seed=0 确保可复现性
            gen_ref_script = f'''
import torch
import sys
import os

# Add current directory to sys.path
sys.path.append(os.getcwd())

def generate_reference():
    print("Starting reference data generation...")
    try:
        # Import from reference
        try:
            from reference import Model, get_inputs, get_init_inputs
        except ImportError as e:
            print(f"Import failed: {{e}}")
            return False
        
        print("Successfully imported Model and helper functions.")
        
        # Determine device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            device = "npu"
        
        print(f"Using device: {{device}}")
        
        # Fixed seed for reproducibility
        torch.manual_seed(0)
        print("[INFO] Random seed: 0")
        
        # Instantiate model
        try:
            init_inputs = get_init_inputs()
            model = Model(*init_inputs)
            if device != "cpu":
                model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Model instantiation failed: {{e}}")
            return False
        
        # Get inputs with fixed seed
        torch.manual_seed(0)
        try:
            inputs = get_inputs()
            if device != "cpu":
                inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]
        except Exception as e:
            print(f"get_inputs failed: {{e}}")
            return False
        
        # Run forward pass
        try:
            with torch.no_grad():
                outputs = model(*inputs)
            print("Forward pass successful.")
        except Exception as e:
            print(f"Forward pass failed: {{e}}")
            return False
        
        # Ensure outputs is a list
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        
        # Move to CPU for saving
        outputs_cpu = [x.cpu() if isinstance(x, torch.Tensor) else x for x in outputs]
        
        # Save reference data
        ref_data = {{
            'op_name': '{self.op_name}',
            'seed': 0,
            'outputs': outputs_cpu,
            'output_shapes': [x.shape if isinstance(x, torch.Tensor) else None for x in outputs_cpu],
        }}
        
        ref_file = os.path.join(os.getcwd(), "{self.op_name}_reference.pt")
        torch.save(ref_data, ref_file)
        print(f"[INFO] Reference data saved to: {{ref_file}}")
        print(f"[INFO] Output count: {{len(outputs_cpu)}}")
        for i, out in enumerate(outputs_cpu):
            if isinstance(out, torch.Tensor):
                print(f"  Output[{{i}}]: shape={{out.shape}}, dtype={{out.dtype}}")
        
        return True
    
    except Exception as e:
        print(f"Unexpected error: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_reference()
    if success:
        print("REFERENCE_GENERATION_SUCCESS")
        sys.exit(0)
    else:
        print("REFERENCE_GENERATION_FAILED")
        sys.exit(1)
'''
            script_file = os.path.join(ref_dir, f"verify_{self.op_name}.py")
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(gen_ref_script)
            
            # 4. 打包目录
            package_data = self._pack_directory(ref_dir)
            
            # 5. 使用 Worker.generate_reference 执行
            if not self.worker:
                raise RuntimeError("Worker not set for reference generation")
            
            # 直接调用 Worker 的 generate_reference 方法
            # 该方法会执行脚本并返回 .pt 文件的 bytes
            success, log, ref_bytes = await self.worker.generate_reference(
                package_data, f"{self.task_id}_gen_ref", self.op_name, timeout
            )
            
            if not success:
                return False, f"Reference generation failed:\n{log}", b''
            
            return True, log, ref_bytes
            
        except Exception as e:
            return False, f"Reference generation exception: {str(e)}", b''
        finally:
            # 清理临时目录
            shutil.rmtree(ref_dir, ignore_errors=True)

    async def profile_single_task(self, task_desc: str, warmup_times: int = 5, 
                                   run_times: int = 50, timeout: int = 300,
                                   device_id: int = 0) -> Dict[str, Any]:
        """
        执行单个任务的性能测试（只测量 task_desc 的性能，不进行 base vs generation 对比）
        
        此功能用于单独测量某段代码（包含 Model 类）的执行性能，会临时创建目录并生成 profile 脚本。
        
        Args:
            task_desc: 包含 Model, get_inputs, get_init_inputs 的代码字符串
            warmup_times: 预热次数
            run_times: 实际运行次数
            timeout: 超时时间
            device_id: 设备ID
            
        Returns:
            Dict[str, Any]: 包含 time_us, success, log 等字段
        """
        # 1. 创建临时目录
        profile_dir = os.path.join(os.path.expanduser(self.log_dir), 
                                    f"{self.op_name}_profile_single_{self.task_id}")
        os.makedirs(profile_dir, exist_ok=True)
        
        try:
            # 2. 写入 task_desc 到 framework_model.py（供模板导入）
            framework_file = os.path.join(profile_dir, "framework_model.py")
            with open(framework_file, "w", encoding="utf-8") as f:
                f.write(task_desc)
            
            # 3. 使用模板生成性能测试脚本
            script_file = os.path.join(profile_dir, f"profile_single_{self.op_name}.py")
            self.gen_profile_single_task_file(script_file, device_id, warmup_times, run_times)
            
            # 4. 打包目录
            package_data = self._pack_directory(profile_dir)
            
            # 5. 使用 Worker.profile_single_task 执行
            if not self.worker:
                raise RuntimeError("Worker not set for profile_single_task")
            
            profile_settings = {
                'warmup_times': warmup_times,
                'run_times': run_times,
                'timeout': timeout
            }
            
            result = await self.worker.profile_single_task(
                package_data, f"{self.task_id}_profile_single", self.op_name, profile_settings
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.op_name}] profile_single_task exception: {e}", exc_info=True)
            return {'time_us': float('inf'), 'success': False, 'log': f"Profile single task exception: {str(e)}"}
        finally:
            # 清理临时目录
            shutil.rmtree(profile_dir, ignore_errors=True)

    def gen_profile_single_task_file(self, profile_file: str, device_id: int, 
                                      warmup_times: int, run_times: int):
        """使用模板生成单任务性能测试脚本"""
        logger.info(f"[{self.op_name}] 开始生成单任务性能测试文件")
        
        # 从文件加载模板
        try:
            with open(PROFILE_SINGLE_TASK_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                template = Template(f.read())
            logger.debug(f"[{self.op_name}] 单任务性能测试模板加载成功")
        except Exception as e:
            logger.error(f"[{self.op_name}] 模板加载失败: {e}")
            raise

        # 检测是否为动态shape
        is_dynamic_shape = self._detect_dynamic_shape()

        # 获取adapters
        try:
            framework_adapter = get_framework_adapter(self.framework)
            backend_adapter = get_backend_adapter(self.backend)
        except Exception as e:
            logger.error(f"[{self.op_name}] Adapters初始化失败: {e}")
            raise

        # 使用adapter生成代码片段
        try:
            framework_imports = framework_adapter.get_import_statements()
            # 注意：不使用 framework_adapter.get_framework_import()，因为模板从固定的 framework_model.py 导入
            
            # 生成设备设置代码
            backend_adapter.setup_environment(device_id, self.arch)
            device_setup_code = framework_adapter.get_device_setup_code(self.backend, self.arch, device_id)
            
            # 生成输入处理代码
            process_input_code = framework_adapter.get_process_input_code(self.backend, self.dsl)
            
            # 生成set_seed代码
            set_seed_code = framework_adapter.get_set_seed_code(self.backend)
            
            # 获取TensorType名称
            tensor_type_name = framework_adapter.get_tensor_type_name()
            
            # 生成benchmark代码（使用base模式，测量framework_model的性能）
            benchmark_code = self._generate_base_benchmark_code(framework_adapter, None, 
                                                                 warmup_times, run_times)
        except Exception as e:
            logger.error(f"[{self.op_name}] 代码片段生成失败: {e}", exc_info=True)
            raise

        # 渲染模板
        try:
            rendered_code = template.render(
                op_name=self.op_name,
                framework=self.framework,
                backend=self.backend,
                arch=self.arch,
                device_id=device_id,
                is_dynamic_shape=is_dynamic_shape,
                warmup_times=warmup_times,
                run_times=run_times,
                # Adapter生成的代码（注意：Model导入由模板固定从framework_model.py获取）
                framework_imports=self._prepare_code_lines(framework_imports),
                device_setup_code=self._prepare_code_lines(device_setup_code),
                process_input_code=self._prepare_code_lines(process_input_code),
                set_seed_code=self._prepare_code_lines(set_seed_code),
                tensor_type_name=tensor_type_name,
                benchmark_code=self._prepare_code_lines(benchmark_code),
            )
            logger.info(f"[{self.op_name}] 模板渲染成功")
        except Exception as e:
            logger.error(f"[{self.op_name}] 模板渲染失败: {e}", exc_info=True)
            raise

        # 写入文件
        try:
            with open(profile_file, "w", encoding="utf-8") as f:
                f.write(rendered_code)
            logger.info(f"[{self.op_name}] 单任务性能测试脚本已写入: {profile_file}")
        except Exception as e:
            logger.error(f"[{self.op_name}] 脚本写入失败: {e}")
            raise

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
        elif self.dsl == "torch":
            # Triton → PyTorch 转换场景
            # 生成的代码是纯 PyTorch，不需要 triton
            import_lines = [
                "import torch",
                "import torch.nn as nn",
                "import torch.nn.functional as F"
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
        
        # ========== 处理参考数据模式 ==========
        use_reference_data = self.config.get('use_reference_data', False)
        reference_file = None
        
        if use_reference_data:
            reference_data_bytes = self.config.get('reference_data')
            if reference_data_bytes:
                # 将参考数据写入验证目录
                # 注意：使用相对路径（只有文件名），这样在 RemoteWorker 场景下
                # 脚本被打包发送到远程服务器后，可以正确从当前工作目录找到参考数据文件
                reference_file_name = f"{self.op_name}_reference.pt"
                reference_file_abs = os.path.join(verify_dir, reference_file_name)
                try:
                    with open(reference_file_abs, 'wb') as f:
                        f.write(reference_data_bytes)
                    logger.info(f"[{self.op_name}] 参考数据已写入: {reference_file_abs} ({len(reference_data_bytes)} bytes)")
                    # 传给模板的是相对路径（只有文件名），脚本执行时从 cwd 查找
                    reference_file = reference_file_name
                except Exception as e:
                    logger.error(f"[{self.op_name}] 参考数据写入失败: {e}")
                    use_reference_data = False
                    reference_file = None
            else:
                logger.warning(f"[{self.op_name}] use_reference_data=True 但未找到 reference_data")
                use_reference_data = False
        
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
            # 统一使用 _impl 后缀，与 framework 文件区分
            file_name = f"{self.op_name}_{self.dsl}_impl.py"
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
            
            # 生成创建 impl_model 的代码（用于 ModelNew 类格式的 DSL）
            create_impl_code = dsl_adapter.create_impl_module(self.framework, framework_adapter)
            logger.debug(f"[{self.op_name}] Create impl module code生成成功 (长度: {len(create_impl_code)})")
            
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
            
            # 生成 compare 函数代码（由 FrameworkAdapter 生成，使用框架原生操作）
            compare_code = framework_adapter.get_compare_code()
            logger.debug(f"[{self.op_name}] Compare code生成成功 (长度: {len(compare_code)})")
            
            # 生成 compare outputs 代码（用于调用 compare 函数）
            compare_outputs_code = framework_adapter.get_compare_outputs_code()
            logger.debug(f"[{self.op_name}] Compare outputs code生成成功 (长度: {len(compare_outputs_code)})")
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
                # 参考数据模式（用于跨后端转换场景）
                use_reference_data=use_reference_data,
                reference_file=reference_file,
                # Adapter生成的代码
                framework_imports=self._prepare_code_lines(framework_imports),
                framework_model_import=self._prepare_code_lines(framework_model_import),
                dsl_imports=self._prepare_code_lines(dsl_imports),
                dsl_impl_import=self._prepare_code_lines(dsl_impl_import),
                special_setup_code=self._prepare_code_lines(special_setup_code),
                device_setup_code=self._prepare_code_lines(device_setup_code),
                process_input_code=self._prepare_code_lines(process_input_code),
                create_impl_code=self._prepare_code_lines(create_impl_code),
                call_impl_code=self._prepare_code_lines(call_impl_code),
                set_seed_code=self._prepare_code_lines(set_seed_code),
                binary_io_functions=self._prepare_code_lines(binary_io_functions),
                needs_binary_io=needs_binary_io,
                tensor_type_name=tensor_type_name,
                compare_code=self._prepare_code_lines(compare_code),
                compare_outputs_code=self._prepare_code_lines(compare_outputs_code),
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
    
    def _pack_directory(self, dir_path: str) -> bytes:
        """将目录打包为tar字节流"""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w') as tar_file:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, dir_path)
                    tar_file.add(file_path, arcname=arcname)
        return tar_buffer.getvalue()

    async def run_verify(self, verify_dir: str, timeout: int = 300, device_id: int = 0):
        """
        运行验证脚本
        
        注意：device 的管理（acquire/release）由调用方（verifier.run()）负责
        这个方法只负责执行已经生成好的脚本

        Args:
            verify_dir: 验证目录
            timeout: 超时时间（秒），默认5分钟（传递给模板用于每次计算）
            device_id: 设备ID（仅用于日志和兼容性，实际设备已在脚本中设置）
        """
        verify_script = os.path.join(verify_dir, f"verify_{self.op_name}.py")
        logger.info(f"[{self.op_name}] 准备运行验证脚本: {verify_script}, timeout={timeout}秒")
        
        try:
            # 调用Worker执行验证
            if not self.worker:
                # 检查 device_id 是否为 -1（表示 RemoteWorker，设备由远程服务器管理）
                if device_id == -1:
                    raise RuntimeError(
                        f"[{self.op_name}] Worker not set and device_id=-1 (RemoteWorker mode). "
                        "Worker must be provided by Task or WorkerManager for RemoteWorker."
                    )
                # 如果没有worker，根据device_id创建LocalWorker（用于测试场景）
                import warnings
                warnings.warn(
                    f"⚠️  [DEPRECATED] KernelVerifier 自动创建 LocalWorker 是旧的兜底逻辑，仅用于测试。\n"
                    f"推荐的新写法：\n"
                    f"  1. 在调用前注册 Worker 到 WorkerManager（一行代码）：\n"
                    f"     from ai_kernel_generator.core.worker.manager import register_local_worker\n"
                    f"     \n"
                    f"     await register_local_worker([{device_id}], backend='{self.backend}', arch='{self.arch}')\n"
                    f"  2. Task 会自动从 WorkerManager 获取 worker\n"
                    f"参考示例：examples/run_torch_npu_triton_single.py",
                    DeprecationWarning,
                    stacklevel=2
                )
                logger.warning(f"⚠️  [{self.op_name}] Worker not set, creating temporary LocalWorker (deprecated)")
                
                from ai_kernel_generator.core.worker.local_worker import LocalWorker
                from ai_kernel_generator.core.async_pool.device_pool import DevicePool
                logger.info(f"[{self.op_name}] Worker not set, creating LocalWorker with device [{device_id}]")
                device_pool = DevicePool([device_id])
                self.worker = LocalWorker(device_pool=device_pool, backend=self.backend)
            
            from ai_kernel_generator.core.worker.local_worker import LocalWorker
            if isinstance(self.worker, LocalWorker):
                if not hasattr(self.worker, 'device_pool') or self.worker.device_pool is None:
                    raise RuntimeError(
                        f"[{self.op_name}] LocalWorker must have device_pool. "
                        "This should be provided by Task when creating _private_worker."
                    )
                package_data = verify_dir
            else:
                package_data = self._pack_directory(verify_dir)
            
            # worker.verify() 只是执行脚本，不需要管理 device
            # device 已经在生成脚本时设置好了
            success, log, artifacts = await self.worker.verify(package_data, self.task_id, self.op_name, timeout)
            
            # 同步 artifacts 到 verify_dir（用于 RemoteWorker 场景）
            if artifacts:
                sync_artifacts_to_directory(artifacts, verify_dir, self.task_id)
            
            if success:
                logger.info(f"[{self.op_name}] 验证执行成功")
            else:
                logger.error(f"[{self.op_name}] 验证执行失败，日志如下：\n{log}")
            return success, log
            
        except Exception as e:
            logger.error(f"[{self.op_name}] 验证执行异常: {e}", exc_info=True)
            return False, str(e)

    def gen_profile_project(self, verify_dir: str, device_id: int = 0, warmup_times: int = 5, 
                            run_times: int = 50, skip_base: bool = False):
        """生成profile项目文件到指定目录
        
        Args:
            verify_dir: 验证目录
            device_id: 设备ID
            warmup_times: 预热次数
            run_times: 运行次数
            skip_base: 是否跳过 base profile（跨后端场景下设为 True）
        """
        # 生成基准性能测试脚本（如果不跳过）
        if not skip_base:
            profile_file = os.path.join(verify_dir, f"profile_{self.op_name}_base.py")
            self.gen_profile_file_from_template(PROFILE_BASE_TEMPLATE_PATH, profile_file,
                                                device_id, warmup_times, run_times)
        else:
            logger.info(f"[{self.op_name}] 跳过 base profile 生成（跨后端场景）")
        
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
            
            # 生成创建 impl_model 的代码（用于 ModelNew 类格式的 DSL）
            create_impl_code = dsl_adapter.create_impl_module(self.framework, framework_adapter)
            logger.debug(f"[{self.op_name}] 性能测试Create impl module code生成成功 (长度: {len(create_impl_code)})")
            
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
                create_impl_code=self._prepare_code_lines(create_impl_code),
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
        if self.dsl == "torch":
            # Triton → PyTorch 转换场景：使用传统计时方法
            sync_code = "torch.cuda.synchronize()" if self.backend == "cuda" else (
                "torch.npu.synchronize()" if self.backend == "ascend" else ""
            )
            code = f"""        # PyTorch 原生实现，使用传统循环计时
        import time
        def base_benchmark_fn():
            result = framework_model(*inputs)
            return result
        
        # 预热
        for _ in range({warmup}):
            _ = base_benchmark_fn()
            {sync_code}
        
        # 计时
        start_time = time.time()
        for _ in range({runs}):
            _ = base_benchmark_fn()
            {sync_code}
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000 / {runs}
        method = "pytorch_loop_timer"
"""
            return code
        elif "triton_cuda" in self.dsl or "triton_ascend" in self.dsl:
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
            return_mode="median"
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

    async def run_profile(self, task_info: Dict[str, Any], current_step: int = 0, device_id: int = -1, profile_settings: dict = {}) -> dict:
        """运行profile分析
        
        注意：与 run() 方法类似，device 的管理在此方法中统一完成
        
        Args:
            device_id: 设备ID（默认-1表示自动管理，LocalWorker会自动从device_pool获取）
        
        Returns:
            dict: 性能分析结果，包含以下字段：
                - gen_time: 生成代码执行时间（微秒）
                - base_time: 基准代码执行时间（微秒）
                - speedup: 加速比
                - autotune_summary: autotune配置详情（仅triton DSL）
        """
        # 【关键】对于 LocalWorker 和 RemoteWorker，在方法开始时就 acquire device
        # 整个 profile 流程（生成脚本 + 执行）都使用这个 device_id
        # 最后在 finally 中统一释放
        actual_device_id = device_id
        acquired_device = None
        from ai_kernel_generator.core.worker.local_worker import LocalWorker
        from ai_kernel_generator.core.worker.remote_worker import RemoteWorker
        
        if self.worker and isinstance(self.worker, LocalWorker):
            # LocalWorker: 从本地 device_pool 获取设备
            acquired_device = await self.worker.device_pool.acquire_device()
            actual_device_id = acquired_device
            logger.info(f"[{self.op_name}] Acquired local device {actual_device_id} for entire profile process")
        elif self.worker and isinstance(self.worker, RemoteWorker):
            # RemoteWorker: 从远程服务器获取设备
            acquired_device = await self.worker.acquire_device(task_id=self.task_id)
            actual_device_id = acquired_device
            logger.info(f"[{self.op_name}] Acquired remote device {actual_device_id} for entire profile process")
        else:
            # 没有 worker（旧流程兼容）
            actual_device_id = device_id if device_id != -1 else 0
            logger.info(f"[{self.op_name}] Using device {actual_device_id} (no worker, deprecated flow)")
        
        try:
            run_times = profile_settings.get("run_times", 50)
            warmup_times = profile_settings.get("warmup_times", 5)

            # 获取验证目录
            expanded_log_dir = os.path.expanduser(self.log_dir)
            unique_dir_name = f"I{self.task_id}_S{current_step:02d}_verify"
            verify_dir = os.path.join(expanded_log_dir, self.op_name, unique_dir_name)

            # 生成profile脚本
            # 对于 RemoteWorker，代码生成时使用 0 作为占位符（实际设备由远程服务器管理）
            # 对于 LocalWorker，使用已经 acquired 的 actual_device_id
            # 跨后端场景（使用参考数据）下，跳过 base profile
            skip_base = self.config.get('use_reference_data', False)
            self.gen_profile_project(verify_dir, actual_device_id, warmup_times, run_times, skip_base=skip_base)

            # 打包并发送给Worker执行
            package_data = self._pack_directory(verify_dir)
            
            if not self.worker:
                # 检查 device_id 是否为 -1（表示自动管理）
                if device_id == -1:
                    raise RuntimeError(
                        f"[{self.op_name}] Worker not set and device_id=-1 (RemoteWorker mode). "
                        "Worker must be provided by Task or WorkerManager for RemoteWorker."
                    )
                # 如果没有worker，根据device_id创建LocalWorker（用于测试场景）
                # 注意：此时 actual_device_id 已在上面设置为 device_id（因为 device_id != -1）
                import warnings
                warnings.warn(
                    f"⚠️  [DEPRECATED] KernelVerifier 自动创建 LocalWorker 是旧的兜底逻辑，仅用于测试。\n"
                    f"推荐的新写法：\n"
                    f"  1. 在调用前注册 Worker 到 WorkerManager（一行代码）：\n"
                    f"     from ai_kernel_generator.core.worker.manager import register_local_worker\n"
                    f"     \n"
                    f"     await register_local_worker([{actual_device_id}], backend='{self.backend}', arch='{self.arch}')\n"
                    f"  2. Task 会自动从 WorkerManager 获取 worker\n"
                    f"参考示例：examples/run_torch_npu_triton_single.py",
                    DeprecationWarning,
                    stacklevel=2
                )
                logger.warning(f"⚠️  [{self.op_name}] Worker not set, creating temporary LocalWorker (deprecated)")
                
                from ai_kernel_generator.core.worker.local_worker import LocalWorker
                from ai_kernel_generator.core.async_pool.device_pool import DevicePool
                logger.info(f"[{self.op_name}] Worker not set, creating LocalWorker with device [{actual_device_id}]")
                device_pool = DevicePool([actual_device_id])
                self.worker = LocalWorker(device_pool=device_pool, backend=self.backend)
            
            # 检查LocalWorker是否有device_pool
            from ai_kernel_generator.core.worker.local_worker import LocalWorker
            if isinstance(self.worker, LocalWorker):
                if not hasattr(self.worker, 'device_pool') or self.worker.device_pool is None:
                    raise RuntimeError(
                        f"[{self.op_name}] LocalWorker must have device_pool. "
                        "This should be provided by Task when creating _private_worker."
                    )
            
            # 传递完整的 profile_settings 给 Worker
            full_settings = {
                **profile_settings,
                'backend': self.backend,
                'dsl': self.dsl,
                'op_name': self.op_name
            }
            
            result = await self.worker.profile(package_data, self.task_id, self.op_name, full_settings)
            
            # 同步 artifacts 到 verify_dir（用于 RemoteWorker 场景）
            artifacts = result.get('artifacts', {})
            if artifacts:
                sync_artifacts_to_directory(artifacts, verify_dir, self.task_id)
            
            # 从 Worker 返回的结果中提取数据
            # 注意：跨后端场景下 base_time 可能是 None（跳过 base profile）
            gen_time = result.get('gen_time')
            base_time = result.get('base_time')
            speedup = result.get('speedup', 0.0)
            
            # 处理 None 值用于日志输出
            gen_time_display = gen_time if gen_time is not None else float('inf')
            base_time_display = base_time if base_time is not None else float('inf')

            # 跨平台场景：使用外部传入的 base_time（如 GPU kernel 时间）
            # 这样 speedup = override_base_time_display / gen_time_display 才有意义
            override_base_time = profile_settings.get('override_base_time_us')
            if override_base_time is not None and override_base_time > 0:
                base_time_display = override_base_time
                # 重新计算 speedup = base / gen
                if gen_time_display > 0 and gen_time_display != float('inf'):
                    speedup = base_time_display / gen_time_display
                else:
                    speedup = 0.0
                logger.info(f"[{self.task_id}:{self.op_name}] 跨平台模式: 使用外部 base_time={base_time_display:.2f}us")

            self.save_speedup_result(speedup, base_time_display, gen_time_display, unique_dir_name)
            
            speedup_percent = speedup * 100.0
            logger.info(f"orig performance is {base_time_display:.2f} us")
            logger.info(f"aikg performance is {gen_time_display:.2f} us")
            logger.info(f"[{self.task_id}:{self.op_name}] 性能分析完成，加速比（基准为100%）: {speedup_percent:.2f} %")
            
            # 构建返回结果
            result = {
                'gen_time': gen_time_display,
                'base_time': base_time_display,
                'speedup': speedup,
                'unique_dir': unique_dir_name  # 添加 unique_dir
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
                'gen_time': None,
                'base_time': None,
                'speedup': 0.0
            }
        finally:
            # 【关键】在方法结束时统一释放设备
            # 设备的整个生命周期由 run_profile() 方法管理
            if acquired_device is not None:
                from ai_kernel_generator.core.worker.local_worker import LocalWorker
                from ai_kernel_generator.core.worker.remote_worker import RemoteWorker
                
                if isinstance(self.worker, LocalWorker):
                    await self.worker.device_pool.release_device(acquired_device)
                    logger.info(f"[{self.op_name}] Released local device {acquired_device}")
                elif isinstance(self.worker, RemoteWorker):
                    await self.worker.release_device(acquired_device, task_id=self.task_id)
                    logger.info(f"[{self.op_name}] Released remote device {acquired_device}")

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
    
    async def _verify_configs_separately(self, target_code: str, verify_dir: str, device_id: int, verify_timeout: int, current_step: int = 0) -> Tuple[bool, str, str]:
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
                config_res, config_log = await self.run_verify(temp_verify_dir, timeout=verify_timeout)
                
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
            
            # 注意：不在这里复制到 passed_cases，等所有验证（包括多case）都通过后再复制
            # 复制操作在 run() 方法的最后统一处理
            
            # 保存验证结果到JSONL文件
            result_jsonl_path = os.path.join(os.path.expanduser(self.log_dir), "verification_results.jsonl")
            result_info = {
                "task_name": self.op_name,
                "task_id": self.task_id,
                "step": current_step,
                "verify_dir": verify_dir,
                "passed": True,
                "error_log": "".join(verify_logs),
                "timestamp": datetime.now().isoformat(),
                "framework": self.framework,
                "dsl": self.dsl,
                "backend": self.backend,
                "arch": self.arch,
                "autotune_configs": {
                    "total": len(all_configs),
                    "passed": len(valid_configs)
                }
            }
            
            with open(result_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_info, ensure_ascii=False, indent=2) + '\n\n')
            
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

    async def run(self, task_info: Dict[str, Any], current_step: int = 0, device_id: int = -1):
        """
        运行内核验证器，验证代码的正确性

        Args:
            task_info: 任务信息字典，包含所有代码和状态
            current_step: 当前步骤
            device_id: 设备ID（默认-1表示自动管理，LocalWorker会自动从device_pool获取）

        Returns:
            Tuple[bool, str]: (验证结果, 错误日志)
        """
        logger.info(f"Verifier Run - Step: {current_step}")

        # 根据实现类型从task_info获取代码
        target_code = task_info.get('coder_code', '')

        if not target_code:
            logger.error("No target code found for verification")
            return False, "No target code found for verification"

        # 动态创建验证目录
        verify_dir = self._create_verify_dir(current_step)
        
        # 【关键】对于 LocalWorker 和 RemoteWorker，在 run() 方法开始时就 acquire device
        # 整个 verify 流程（生成脚本 + 执行）都使用这个 device_id
        # 最后在 finally 中统一释放
        actual_device_id = device_id
        acquired_device = None
        from ai_kernel_generator.core.worker.local_worker import LocalWorker
        from ai_kernel_generator.core.worker.remote_worker import RemoteWorker
        
        if self.worker and isinstance(self.worker, LocalWorker):
            # LocalWorker: 从本地 device_pool 获取设备
            acquired_device = await self.worker.device_pool.acquire_device()
            actual_device_id = acquired_device
            logger.info(f"[{self.op_name}] Acquired local device {actual_device_id} for entire verify process")
        elif self.worker and isinstance(self.worker, RemoteWorker):
            # RemoteWorker: 从远程服务器获取设备
            acquired_device = await self.worker.acquire_device(task_id=self.task_id)
            actual_device_id = acquired_device
            logger.info(f"[{self.op_name}] Acquired remote device {actual_device_id} for entire verify process")
        else:
            # 没有 worker（旧流程兼容）
            actual_device_id = device_id if device_id != -1 else 0
            logger.info(f"[{self.op_name}] Using device {actual_device_id} (no worker, deprecated flow)")
        
        try:
            # 检测是否是triton autotune代码
            is_triton_autotune = (self.dsl in ["triton_cuda", "triton_ascend"] and 
                                  self._detect_triton_autotune(target_code))
            
            if is_triton_autotune:
                # 对于autotune的triton代码，单独验证每个config
                config_verify_result, config_verify_log, final_code = await self._verify_configs_separately(
                    target_code, verify_dir, actual_device_id, self.config.get('verify_timeout', 300), current_step
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
                # 对于 RemoteWorker，代码生成时使用 0 作为占位符（实际设备由远程服务器管理）
                # 对于 LocalWorker，使用已经 acquired 的 actual_device_id
                self.gen_verify_project(target_code, verify_dir, actual_device_id)
            except Exception as e:
                # 捕获gen_verify_project中的异常，记录到project_gen_log中
                error_msg = str(e)
                logger.error(f"验证项目生成失败: {error_msg}")
                project_gen_log = f"项目生成失败: {error_msg}\n"

            # 从config获取timeout配置，默认5分钟
            verify_timeout = self.config.get('verify_timeout', 300)

            # 运行验证
            # worker.verify() 只是执行脚本，不需要管理 device（device 已经在脚本中设置好了）
            verify_res, verify_log = await self.run_verify(
                verify_dir, timeout=verify_timeout, device_id=actual_device_id
            )
        
            # 拼接项目生成日志和验证日志
            verify_log = project_gen_log + verify_log

            # 保存验证结果到JSONL文件
            self._save_verification_result_to_jsonl(verify_dir, current_step, verify_res, verify_log)

            # 注意：不在这里复制到 passed_cases
            # 如果启用了多 case 测试，需要等多 case 验证也通过后才能复制
            # 复制操作由 task.py 统一管理

            return verify_res, verify_log
        finally:
            # 【关键】在 run() 方法结束时统一释放设备
            # 设备的整个生命周期由 run() 方法管理
            if acquired_device is not None:
                from ai_kernel_generator.core.worker.local_worker import LocalWorker
                from ai_kernel_generator.core.worker.remote_worker import RemoteWorker
                
                if isinstance(self.worker, LocalWorker):
                    await self.worker.device_pool.release_device(acquired_device)
                    logger.info(f"[{self.op_name}] Released local device {acquired_device}")
                elif isinstance(self.worker, RemoteWorker):
                    await self.worker.release_device(acquired_device, task_id=self.task_id)
                    logger.info(f"[{self.op_name}] Released remote device {acquired_device}")
