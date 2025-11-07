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
from datetime import datetime
from typing import Optional, Literal, Tuple, Dict, Any
from jinja2 import Template
import pandas as pd
from pathlib import Path

from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.process_utils import run_command

# 模板路径
TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "kernel_verify_template.j2")
PROFILE_BASE_TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "prof_base_template.j2")
PROFILE_GENERATION_TEMPLATE_PATH = os.path.join(
    get_project_root(), "resources", "templates", "prof_generation_template.j2")
# 生成CMakeLists.txt和运行脚本的路径
CMAKE_TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "cmake_template.j2")
RUN_TEMPLATE_PATH = os.path.join(get_project_root(), "utils", "compile_tools", "ascend_compile", "run.sh")

# 类型定义
FrameworkType = Literal["torch", "mindspore", "numpy"]
ImplType = Literal["triton", "triton-russia", "swft", "cuda_c", "cpp", "ascendc"]
BackendType = Literal["cuda", "ascend", "cpu"]
ArchType = Literal["a100", "v100", "h20", "l20", "rtx3090", "ascend910b4", "ascend310p3", "x86_64", "aarch64"]

logger = logging.getLogger(__name__)


class KernelVerifier:
    def __init__(self,
                 op_name: str,
                 framework_code: str,
                 task_id: str = "0",
                 framework: FrameworkType = "torch",
                 dsl: ImplType = "triton",
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
            dsl (ImplType): 实现类型，可选值包括 "triton", "triton-russia", "swft"
            backend (BackendType): 计算设备后端，可选值包括 "cuda", "ascend"
            arch (ArchType): 硬件架构，可选值包括 "a100", "v100", "h20", "l20", "rtx3090", "ascend910b4", "ascend310p3"
            impl_func_name (str, optional): 实现函数名，默认为op_name_dsl_framework
        """
        self.op_name = op_name
        self.framework_code = framework_code
        self.framework = framework
        self.dsl = dsl
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
        if "triton" in self.dsl:
            self.impl_func_name = impl_func_name or f"{op_name}_triton_{framework}"
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

        if "triton" in self.dsl:
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

    def gen_verify_project(self, impl_code: str, verify_dir: str, device_id: int = 0):
        """生成验证项目文件到指定目录"""
        # 创建框架实现文件
        framework_file = os.path.join(verify_dir, f"{self.op_name}_{self.framework}.py")
        with open(framework_file, "w", encoding="utf-8") as f:
            f.write(self.framework_code)

        # 创建具体实现文件
        if "ascendc" in self.dsl:
            self.generate_ascendc_project(impl_code, verify_dir)
        else:
            file_name = f"{self.op_name}_{self.dsl}.py"
            impl_file = os.path.join(verify_dir, file_name)

            # 生成import语句
            import_statements = self._generate_import_statements()

            with open(impl_file, "w", encoding="utf-8") as f:
                # 先写入import语句，再写入原始代码
                f.write(import_statements + impl_code)

        # 生成验证脚本
        verify_file = os.path.join(verify_dir, f"verify_{self.op_name}.py")

        # 从文件加载模板
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template = Template(f.read())

        # 检测是否为动态shape
        is_dynamic_shape = self._detect_dynamic_shape()

        # 使用模板变量
        rendered_code = template.render(
            op_name=self.op_name,
            framework=self.framework,
            dsl=self.dsl,
            device_id=device_id,
            impl_func_name=self.impl_func_name,
            backend=self.backend,
            arch=self.arch,
            is_dynamic_shape=is_dynamic_shape,
            timeout=self.config.get('verify_timeout', 300)
        )

        with open(verify_file, "w", encoding="utf-8") as f:
            f.write(rendered_code)

    def run_verify(self, verify_dir: str, timeout: int = 300):
        """
        运行验证脚本

        Args:
            verify_dir: 验证目录
            timeout: 超时时间（秒），默认5分钟（传递给模板用于每次计算）
        """
        original_cwd = os.getcwd()
        try:
            os.chdir(verify_dir)
            python_cmd = ["python", f"verify_{self.op_name}.py"]
            # 使用run_command但禁用timeout，让验证脚本无限制运行
            return run_command(python_cmd, f"verify_{self.op_name}", timeout=timeout)
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
        # 从文件加载模板
        with open(template_path, "r", encoding="utf-8") as f:
            template = Template(f.read())

        # 检测是否为动态shape
        is_dynamic_shape = self._detect_dynamic_shape()

        # 使用模板变量
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
            is_dynamic_shape=is_dynamic_shape
        )

        with open(profile_file, "w", encoding="utf-8") as f:
            f.write(rendered_code)

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
            print("run_nsys = ", cmd)
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
            print("analyze_nsys_data = ", cmd)
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
            if "triton" in self.dsl:
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
                'speedup': speedup
            }
            
            # 只在 triton + ascend 情况下添加 autotune_summary
            if "triton" in self.dsl and self.backend == "ascend":
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

        # 保存验证结果到JSONL文件（每行一个JSON对象）
        result_jsonl_path = os.path.join(os.path.expanduser(self.log_dir), "verification_results.jsonl")
        result_info = {
            "task_name": self.op_name,
            "task_id": self.task_id,
            "step": current_step,
            "verify_dir": verify_dir,
            "passed": verify_res,
            "error_log": verify_log,
            "timestamp": datetime.now().isoformat(),
            "framework": self.framework,
            "dsl": self.dsl,
            "backend": self.backend,
            "arch": self.arch
        }

        with open(result_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_info, ensure_ascii=False, indent=2) + '\n\n')

        # 保存通过的验证文件
        if verify_res:
            foder_name = os.path.basename(verify_dir)
            dst_dir = Path(self.log_dir) / "passed_cases" / self.op_name / foder_name
            shutil.copytree(verify_dir, dst_dir)

        return verify_res, verify_log
