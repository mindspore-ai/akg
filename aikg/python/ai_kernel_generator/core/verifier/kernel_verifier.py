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
from datetime import datetime
from typing import Optional, Literal, Tuple
from jinja2 import Template
import pandas as pd
from pathlib import Path

from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.process_utils import run_command
from ai_kernel_generator.core.utils import ParsedCode

# 模板路径
TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "kernel_verify_template.j2")
PROFILE_BASE_TEMPLATE_PATH = os.path.join(get_project_root(), "resources", "templates", "msprof_base_template.j2")
PROFILE_GENERATION_TEMPLATE_PATH = os.path.join(
    get_project_root(), "resources", "templates", "msprof_generation_template.j2")

# 类型定义
FrameworkType = Literal["torch", "mindspore", "numpy"]
ImplType = Literal["triton", "triton-russia", "swft"]
BackendType = Literal["cuda", "ascend"]
ArchType = Literal["a100", "v100", "ascend910b4", "ascend310p3"]

logger = logging.getLogger(__name__)


class KernelVerifier:
    def __init__(self,
                 op_name: str,
                 framework_code: str,
                 log_dir: str,
                 task_id: str = "0",
                 framework: FrameworkType = "torch",
                 impl_type: ImplType = "triton",
                 backend: BackendType = "cuda",
                 arch: ArchType = "a100",
                 impl_func_name: Optional[str] = None):
        """
        初始化Kernel验证器。

        Args:
            op_name (str): 算子名称
            framework_code (str): 框架实现代码（PyTorch、MindSpore或NumPy）
            log_dir (str): 调试信息目录
            task_id (str, optional): 任务ID，用于生成唯一目录名
            framework (FrameworkType): 深度学习框架，可选值包括 "torch", "mindspore", "numpy"
            impl_type (ImplType): 实现类型，可选值包括 "triton", "triton-russia", "swft"
            backend (BackendType): 计算设备后端，可选值包括 "cuda", "ascend"
            arch (ArchType): 硬件架构，可选值包括 "a100", "v100", "ascend910b4", "ascend310p3"
            impl_func_name (str, optional): 实现函数名，默认为op_name_impl_type_framework
        """
        self.op_name = op_name
        self.framework_code = framework_code
        self.framework = framework
        self.impl_type = impl_type
        self.backend = backend.lower()
        self.arch = arch.lower()
        self.task_id = task_id
        self.log_dir = log_dir
        if "triton" in self.impl_type:
            self.impl_func_name = impl_func_name or f"{op_name}_triton_{framework}"
        else:
            self.impl_func_name = impl_func_name or f"{op_name}_{impl_type}_{framework}"
            

        # 验证backend和arch的组合是否有效
        if self.backend == "cuda" and self.arch not in ["a100", "v100"]:
            raise ValueError(f"cuda后端只支持a100和v100架构，当前架构: {self.arch}")
        if self.backend == "ascend" and self.arch not in ["ascend910b4", "ascend310p3"]:
            raise ValueError(f"ascend后端只支持ascend910b4和ascend310p3架构，当前架构: {self.arch}")

    def _create_verify_dir(self, step_counter) -> str:
        """创建验证目录并返回目录路径"""
        expanded_log_dir = os.path.expanduser(self.log_dir)
        unique_dir = f"I{self.task_id}_S{step_counter:02d}_verify"

        target_dir = os.path.join(expanded_log_dir, self.op_name, unique_dir)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def gen_verify_project(self, impl_code: str, verify_dir: str, device_id: int = 0):
        """生成验证项目文件到指定目录"""
        # 创建框架实现文件
        framework_file = os.path.join(verify_dir, f"{self.op_name}_{self.framework}.py")
        with open(framework_file, "w", encoding="utf-8") as f:
            f.write(self.framework_code)

        # 创建具体实现文件
        if "triton" in self.impl_type:
            file_name = f"{self.op_name}_triton.py"
        else:
            file_name = f"{self.op_name}_{self.impl_type}.py"
        impl_file = os.path.join(verify_dir, file_name)
        with open(impl_file, "w", encoding="utf-8") as f:
            f.write(impl_code)

        # 生成验证脚本
        verify_file = os.path.join(verify_dir, f"verify_{self.op_name}.py")

        # 从文件加载模板
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template = Template(f.read())

        # 使用模板变量
        rendered_code = template.render(
            op_name=self.op_name,
            framework=self.framework,
            impl_type=self.impl_type,
            device_id=device_id,
            impl_func_name=self.impl_func_name,
            backend=self.backend,
            arch=self.arch
        )

        with open(verify_file, "w", encoding="utf-8") as f:
            f.write(rendered_code)

    def run_verify(self, verify_dir: str):
        """运行验证脚本"""
        os.chdir(verify_dir)
        python_cmd = ["python", f"verify_{self.op_name}.py"]
        return run_command(python_cmd, f"verify_{self.op_name}")

    def gen_profile_project(self, verify_dir: str, device_id: int = 0, warmup_times: int = 5, run_times: int = 50):
        """生成profile项目文件到指定目录"""
        total_count = warmup_times + run_times
        # 生成基准性能测试脚本
        profile_file = os.path.join(verify_dir, f"profile_{self.op_name}_base.py")
        self.gen_profile_file_from_template(PROFILE_BASE_TEMPLATE_PATH, profile_file, device_id, total_count)
        # 生成性能测试脚本
        profile_file = os.path.join(verify_dir, f"profile_{self.op_name}_generation.py")
        self.gen_profile_file_from_template(PROFILE_GENERATION_TEMPLATE_PATH, profile_file, device_id, total_count)

    def gen_profile_file_from_template(self, template_path: str, profile_file: str, device_id: int, total_count: int):
        """从模板生成profile文件"""
        # 从文件加载模板
        with open(template_path, "r", encoding="utf-8") as f:
            template = Template(f.read())

        # 使用模板变量
        rendered_code = template.render(
            op_name=self.op_name,
            framework=self.framework,
            impl_type=self.impl_type,
            device_id=device_id,
            impl_func_name=self.impl_func_name,
            backend=self.backend,
            arch=self.arch,
            total_count=total_count
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
                return False, "没有找到符合预期次数的Op", 0.0

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
            return False, f"分析数据时出错: {str(e)}", 0.0

    def save_speedup_result(self, speedup: float, base_time: float, gen_time: float, unique_dir: str):
        """保存加速比结果到txt文件"""
        try:
            profiling_dir = os.path.join(os.path.expanduser(self.log_dir), self.op_name, "profiling")
            os.makedirs(profiling_dir, exist_ok=True)

            filepath = os.path.join(profiling_dir, "speed_up_record.txt")

            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"op_name: {self.op_name}, task_id: {self.task_id}, unique_dir: {unique_dir}, ")
                f.write(f"base_time: {base_time:.6f} us, generation_time: {gen_time:.6f} us, ")
                f.write(f"speedup: {speedup:.6f}x\n\n")

            logger.debug(f"[{self.task_id}:{self.op_name}] 加速比结果已保存")

        except Exception as e:
            logger.warning(f"[{self.task_id}:{self.op_name}] 保存加速比结果失败: {str(e)}")

    def run_profile(self, current_step: int = 0, device_id: str = "0", profile_settings: dict = {}):
        """运行profile分析"""
        try:
            run_times = profile_settings.get("run_times", 50)
            warmup_times = profile_settings.get("warmup_times", 5)

            # 获取验证目录
            expanded_log_dir = os.path.expanduser(self.log_dir)
            unique_dir_name = f"I{self.task_id}_S{current_step:02d}_verify"
            verify_dir = os.path.join(expanded_log_dir, self.op_name, unique_dir_name)

            # 生成profile脚本并运行
            self.gen_profile_project(verify_dir, device_id, warmup_times, run_times)

            _, _, base_prof_path = self.run_msprof(os.path.join(verify_dir, f"profile_{self.op_name}_base.py"))
            _, _, gen_prof_path = self.run_msprof(os.path.join(verify_dir, f"profile_{self.op_name}_generation.py"))

            _, _, base_time = self.analyze_prof_data(base_prof_path, warmup_times, run_times)
            _, _, gen_time = self.analyze_prof_data(gen_prof_path, warmup_times, run_times)

            speedup = base_time / gen_time if gen_time > 0 else 0.0
            self.save_speedup_result(speedup, base_time, gen_time, unique_dir_name)

            logger.info(f"[{self.task_id}:{self.op_name}] 性能分析完成，加速比: {speedup:.2f}x")
            return speedup
        except Exception as e:
            logger.warning(f"[{self.task_id}:{self.op_name}] 性能分析失败: {str(e)}")
            return 0.0

    def run(self, parsed_code: ParsedCode, current_step: int = 0, device_id: int = 0):
        """完整的验证流程

        Args:
            parsed_code: 解析后的代码
            current_step: 步骤计数器，用于生成唯一目录名
            device_id: 设备ID
        """
        if "triton" in self.impl_type:
            impl_code = parsed_code.triton_code
        elif self.impl_type == "swft":
            impl_code = parsed_code.swft_code

        # 动态创建验证目录
        verify_dir = self._create_verify_dir(current_step)

        # 在独立目录中生成验证项目
        self.gen_verify_project(impl_code, verify_dir, device_id)

        # 运行验证
        verify_res, verify_log = self.run_verify(verify_dir)
        
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
            "impl_type": self.impl_type,
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
