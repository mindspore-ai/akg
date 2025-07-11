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
import json
from pathlib import Path
from typing import Tuple
from jellyfish import jaro_winkler_similarity
import pandas as pd
from python.ai_kernel_generator.core.agent.feature_match import FeatureMatch
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import ParserFactory
from ai_kernel_generator.utils.markdown_utils import extract_function_details
from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.utils import ParsedCode, ActionType
from ai_kernel_generator.core.agent.conductor import Conductor

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

"""database下面子目录的加载"""
def build_map_dir():
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[4]  # 根据项目结构调整回溯层级
    
    target_dir = project_root / "database" / "swft" / "ascend310p3"
    
    # 检查目录是否存在
    if not target_dir.is_dir():
        logger.error(f"警告: 目标目录不存在 - {target_dir}")
        return {
            "reduce": {},
            "elementwise": {}
        }
    
    dir_map = {
        "reduce": {},
        "elementwise": {}
    }
    
    for subdir in target_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        dir_name = subdir.name.lower()
            
        if "reduce" in dir_name or "reduction" in dir_name:
            dir_map["reduce"][subdir.name] = str(subdir)
        else:
            dir_map["elementwise"][subdir.name] = str(subdir)
    return dir_map

"""匹配文件"""
def match_and_load_code(result: str):
    start = result.index('{')
    end = result.rindex('}')
    model_result = result[start:end + 1]
    data = json.loads(model_result)
    op_type = data["op_type"].lower()
    # 检查返回的操作类型
    dir_map = build_map_dir()
    if "reduce" in op_type:
        temp_map = dir_map.get("reduce", {})
    else:
        # 默认为 elementwise
        temp_map = dir_map.get("elementwise", {})
    
    primary_name = data["op_name_primary"].lower()
    secondary_names = [name.lower() for name in data["op_name_secondary"]]
    
    matched_paths = []
    matched_dir_names = []
    
    primary_matches = []
    for dir_name, dir_path in temp_map.items():
        dir_name_lower = dir_name.lower()
        if primary_name in dir_name_lower:
            primary_matches.append((dir_name, dir_path))
    
    if primary_matches:
        for dir_name, dir_path in primary_matches:
            matched_dir_names.append(dir_name)
            matched_paths.append(dir_path)
    else:
        for name in secondary_names:
            for dir_name, dir_path in temp_map.items():
                dir_name_lower = dir_name.lower()
                if name in dir_name_lower:
                    if dir_path not in matched_paths:
                        matched_dir_names.append(dir_name)
                        matched_paths.append(dir_path)
    all_aul_files = []
    for dir_path in matched_paths:
        all_aul_files.extend(Path(dir_path).glob('**/*_aul.py'))
    parts = []
    for i, file in enumerate(all_aul_files, start=1):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                    part = f"请参考示例{i}的代码：\n{content}"
                    parts.append(part)
            except Exception as e:
                print(f"加载文件 {file} 失败: {e}")
    final_merged = "\n\n".join(parts)
    return final_merged

class AULDesigner(AgentBase):
    def __init__(self, op_name: str, task_desc: str, model_config: dict, impl_type: str = "", backend: str = "", arch: str = "", task_id: str = "", log_dir: str = ""):
        self.op_name = op_name
        self.task_desc = task_desc
        self.impl_type = impl_type
        self.arch = arch
        self.backend = backend
        self.model_config = model_config
        self.task_id = task_id
        self.log_dir = log_dir

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
        self.aul_gen_input = {
            "cases": "",
            **self.aul_base_doc,
        }
        self.aul_fix_input = {
            "aul_code": "",
            "suggestions": "",
            **self.aul_base_doc,
        }
        self.feature = FeatureMatch(self.task_desc, self.model_config)
        self.conductor = Conductor(self.op_name, self.task_id, self.log_dir, self.impl_type, self.model_config)
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
            loaded_cases: 生成代码的示例
        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        # 提取AUL代码并更新状态
        aul_code = parsed_code.aul_code if parsed_code else ""
        self.update(action_type, aul_code, suggestions)
        # 根据动作类型选择对应的处理逻辑
        if action_type == ActionType.DO_DESIGNER:
            match_res, match_prompt, match_reasoning = await self.feature.run()
            self.conductor.trace.insert_feature_record(match_res, match_prompt, match_reasoning)
            print("模型返回的结果=============")
            print(match_res)
            # aul_loaded_code = match_and_load_code(match_res)
            # self.aul_gen_input["cases"] = aul_loaded_code
            return await self.run_llm(self.aul_gen_prompt, self.aul_gen_input, self.model_config["aul_designer"])
        elif action_type == ActionType.FIX_DESIGNER:
            return await self.run_llm(self.aul_fix_prompt, self.aul_fix_input, self.model_config["aul_designer_fix"])
        else:
            raise ValueError(f"AULDesigner不支持的动作类型: {action_type}")


if __name__ == "__main__":
    test_str = ["reduce_x", "max_pool", "max", "small_non_reduce_axis"]
    target_str = ["max_reduction_over_a_dimension", "reduce_x_small_non_reduce_axis_larger_32bytes", "hardtanh", "reduce_x_big_non_reduce_axis_small_reduce_axis"]
    df = pd.DataFrame(index=target_str, columns=test_str)
    
    # 计算相似度
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = jaro_winkler_similarity(idx, col)
    print(df)


