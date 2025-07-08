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

import logging
import platform
import subprocess
from pathlib import Path
from typing import Tuple

from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import ParserFactory
from ai_kernel_generator.utils.markdown_utils import extract_function_details
from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.utils import ParsedCode, ActionType

logger = logging.getLogger(__name__)


def get_aul_base_doc() -> str:
    """加载AUL规范文档"""
    # 按顺序定义要加载的AUL文档文件
    aul_doc_files = [
        "aul_base.md",
        "aul_rules.md",
        "aul_npu.md",
        "aul_npu_special_op.md",
        "aul_npu_templetes.md",
        "aul_suggestions.md"
    ]

    aul_docs_dir = Path(get_project_root()) / "resources" / "docs" / "aul_docs"
    combined_spec = ""

    for doc_file in aul_doc_files:
        doc_path = aul_docs_dir / doc_file
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
                combined_spec += content + "\n\n"
        except Exception as e:
            logger.warning(f"加载AUL文档失败 {doc_file}: {e}")
            continue

    return combined_spec


def get_cpu_info() -> str:
    """动态获取当前CPU配置信息"""
    try:
        system = platform.system()

        if system == "Linux":
            result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                useful_info = []
                useful_keys = [
                    'Architecture', 'CPU(s)', 'Thread(s) per core', 'Core(s) per socket',
                    'Socket(s)', 'Model name', 'CPU MHz', 'CPU max MHz',
                    'L1d cache', 'L1i cache', 'L2 cache', 'L3 cache', 'Flags'
                ]

                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        key = line.split(':', 1)[0].strip()
                        if key in useful_keys:
                            useful_info.append(line)

                if useful_info:
                    return f"# Linux CPU信息 (关键参数)\n" + '\n'.join(useful_info)

        elif system == "Darwin":
            result = subprocess.run(['sysctl', '-a'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                useful_info = []
                useful_keys = [
                    'hw.ncpu', 'hw.physicalcpu', 'hw.logicalcpu', 'hw.cpufrequency',
                    'hw.cpufrequency_max', 'hw.l1dcachesize', 'hw.l1icachesize',
                    'hw.l2cachesize', 'hw.l3cachesize', 'machdep.cpu.brand_string',
                    'machdep.cpu.core_count', 'machdep.cpu.thread_count',
                    'machdep.cpu.features', 'machdep.cpu.leaf7_features'
                ]

                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        key = line.split(':', 1)[0].strip()
                        if key in useful_keys:
                            useful_info.append(line)

                if useful_info:
                    return f"# macOS CPU信息 (关键参数)\n" + '\n'.join(useful_info)

        elif system == "Windows":
            result = subprocess.run(['wmic', 'cpu', 'get', '*', '/format:list'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                useful_info = []
                useful_keys = [
                    'Name', 'NumberOfCores', 'NumberOfLogicalProcessors',
                    'L2CacheSize', 'L3CacheSize', 'MaxClockSpeed',
                    'Architecture', 'Family', 'Manufacturer'
                ]

                for line in lines:
                    line = line.strip()
                    if line and '=' in line:
                        key, value = line.split('=', 1)
                        if key in useful_keys and value and value != 'None' and value != '':
                            useful_info.append(f"{key}={value}")

                if useful_info:
                    return f"# Windows CPU信息 (关键参数)\n" + '\n'.join(useful_info)

        return ""

    except Exception as e:
        logger.error(f"获取CPU信息失败: {e}")
        return ""


class AULDesigner(AgentBase):
    def __init__(self, op_name: str, task_desc: str, model_config: dict, impl_type: str = "", backend: str = "", arch: str = ""):
        self.op_name = op_name
        self.task_desc = task_desc
        self.impl_type = impl_type
        self.arch = arch
        self.backend = backend
        self.model_config = model_config

        agent_name = f"AULDesigner -- [impl_type] {self.impl_type} -- [action] DO_DESIGNER -- [op_name] {self.op_name}"
        super().__init__(agent_name=agent_name)

        # 初始化解析器
        self.code_parser = ParserFactory.get_code_parser()
        self.format_instructions = self.code_parser.get_format_instructions()

        # 初始化模板
        self.aul_gen_prompt = self.load_template("aul/aul_gen_template.j2")
        self.aul_fix_prompt = self.load_template("aul/aul_fix_template.j2")

        # 准备基础文档数据
        self.aul_base_doc = {
            "op_name": self.op_name,
            "task_desc": self.task_desc,
            "aul_spec": get_aul_base_doc(),
            "supported_compute_api": "",
            "aul_tiling": self.load_doc("aul_docs/aul_tiling.md"),
            "hardware_info": self.get_hardware_doc(),
            "format_instructions": self.format_instructions,
        }

        # 为SWFT实现类型添加支持的API
        if self.impl_type == "swft":
            try:
                supported_compute_api_str = extract_function_details()
                self.aul_base_doc["supported_compute_api"] = supported_compute_api_str
            except Exception as e:
                logger.warning(f"获取SWFT支持的API失败: {e}")

        # 初始化输入配置
        self.aul_gen_input = {**self.aul_base_doc}
        self.aul_fix_input = {
            "aul_code": "",
            "suggestions": "",
            **self.aul_base_doc,
        }

    def get_hardware_doc(self) -> str:
        """根据backend和architecture获取硬件信息"""
        hardware_mapping = {
            "ascend": {
                "ascend310p3": "hardware/Ascend310P3.md",
                "ascend910b4": "hardware/Ascend910B4.md"
            },
            "cuda": {
                "a100": "hardware/CUDA_A100.md",
                "v100": "hardware/CUDA_V100.md"
            }
        }

        if self.backend.lower() == "cpu":
            # 对CPU后端使用动态检测
            return get_cpu_info()
        if self.backend.lower() not in hardware_mapping:
            raise ValueError(f"不支持的backend: {self.backend}")

        architecture_mapping = hardware_mapping[self.backend.lower()]
        if self.arch.lower() not in architecture_mapping:
            supported_architectures = list(architecture_mapping.keys())
            raise ValueError(f"不支持的architecture: {self.arch}，支持的architecture: {supported_architectures}")

        return self.load_doc(architecture_mapping[self.arch.lower()])

    def update(self, action_type: ActionType, aul_code: str, suggestions: str):
        """更新代理状态"""
        if action_type != ActionType.DO_DESIGNER:
            self.agent_name = f"AULDesigner -- [impl_type] {self.impl_type} -- [action] {action_type.name} -- [op_name] {self.op_name}"

        if aul_code:
            self.aul_fix_input["aul_code"] = aul_code

        if suggestions:
            self.aul_fix_input["suggestions"] = suggestions

    async def run(self, action_type: ActionType, parsed_code: ParsedCode, suggestions: str) -> Tuple[str, str, str]:
        """执行AUL代码生成或修复

        Args:
            action_type: 操作类型
            parsed_code: conductor传入的解析代码内容
            suggestions: 建议

        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        # 提取AUL代码并更新状态
        aul_code = parsed_code.aul_code if parsed_code else ""
        self.update(action_type, aul_code, suggestions)

        # 根据动作类型选择对应的处理逻辑
        if action_type == ActionType.DO_DESIGNER:
            return await self.run_llm(self.aul_gen_prompt, self.aul_gen_input, self.model_config["aul_designer"])
        elif action_type == ActionType.FIX_DESIGNER:
            return await self.run_llm(self.aul_fix_prompt, self.aul_fix_input, self.model_config["aul_designer_fix"])
        else:
            raise ValueError(f"AULDesigner不支持的动作类型: {action_type}")
