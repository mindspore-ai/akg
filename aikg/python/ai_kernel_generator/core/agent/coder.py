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
from typing import Tuple, List
from pathlib import Path

from ai_kernel_generator.utils.common_utils import remove_copyright_from_text
from ai_kernel_generator.utils.parser_registry import create_step_parser
from ai_kernel_generator.utils.hardware_utils import get_hardware_doc
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator import get_project_root

logger = logging.getLogger(__name__)


def get_triton_sample_code(framework: str) -> str:
    """读取triton_docs/examples目录下的所有Python代码文件，并将它们拼接在一起

    Args:
        framework: 框架名称，如'mindspore'、'torch'、'numpy'等

    Returns:
        str: 所有Python代码文件内容拼接后的字符串
    """
    base_dir = Path(get_project_root()) / "resources" / "docs" / "triton_docs" / "examples"

    if not base_dir.exists():
        logger.warning(f"Triton示例目录不存在: {base_dir}, 返回空字符串")
        return ""

    all_code = []
    # 使用glob模式匹配framework开头的Python文件
    for file_path in base_dir.glob(f"{framework}_*.py"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read().strip()
                if code:
                    all_code.append(f"# File: {file_path.name}\n{code}\n")
        except Exception as e:
            logger.warning(f"读取Triton示例文件 {file_path} 时发生错误: {str(e)}")
            continue

    if not all_code:
        logger.warning(f"未找到{framework}相关的示例代码文件")
        return ""

    return "\n".join(all_code)


def get_inspirations(inspirations: List[dict]) -> str:
    """
    将inspirations列表转换为字符串

    Args:
        inspirations: 包含字典的列表，每个字典格式为:
                     {'strategy_mode':xxx, 'impl_code':str, 'profile':float}

    Returns:
        str: 拼接后的字符串，包含所有impl_code和profile信息
    """
    if not inspirations:
        return ""

    result_parts = []

    for i, inspiration in enumerate(inspirations):
        if not isinstance(inspiration, dict):
            logger.warning(f"跳过非字典类型的inspiration: {type(inspiration)}")
            continue

        impl_code = inspiration.get('impl_code', '')
        profile = inspiration.get('profile', float('inf'))

        if impl_code:  # 只有当impl_code不为空时才添加
            inspiration_text = f"## Inspiration {i+1} 代码执行耗时(s): {profile}\n"
            inspiration_text += f"代码：\n```\n{impl_code}\n```\n"
            result_parts.append(inspiration_text)

    return "\n".join(result_parts)


class Coder(AgentBase):
    def __init__(self, op_name: str, task_desc: str, dsl: str, framework: str, backend: str, arch: str = "", workflow_config_path: str = None, config: dict = None):
        self.op_name = op_name
        self.task_desc = remove_copyright_from_text(task_desc)
        self.dsl = dsl
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.workflow_config_path = workflow_config_path
        self.config = config

        # 从config中获取model_config
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for Coder")

        agent_name = f"Coder -- [dsl] {self.dsl} -- [op_name] {self.op_name} -- [framework] {self.framework}"
        super().__init__(agent_name=agent_name, config=config)

        # 直接使用从workflow.yaml获取的coder解析器
        self.code_parser = create_step_parser("coder", self.workflow_config_path)
        if not self.code_parser:
            raise ValueError(
                "Failed to create coder parser from workflow config. Please check your workflow.yaml configuration.")
        self.format_instructions = self.code_parser.get_format_instructions()

        if "triton" in self.dsl:
            self.func_name = f"{self.op_name}_triton_{self.framework}"
        else:
            self.func_name = f"{self.op_name}_{self.dsl}_{self.framework}"

        # 初始化coder生成模板
        self.coder_prompt = self.load_template("coder/codegen.j2")

        # 准备基础文档数据
        self.base_doc = {
            "op_name": self.op_name,
            "task_desc": self.task_desc,
            "framework": self.framework,
            "dsl": self.dsl,
            "func_name": self.func_name,
            "format_instructions": self.format_instructions,

            # API和文档
            "api_docs": self.load_doc("api/api.md"),
            "dsl_basic_docs": self.load_doc("basic_docs.md"),
            "dsl_sample_code": self._load_dsl_sample_code(),
            "expert_suggestion": self.load_doc("suggestion_docs.md"),

            # 可选参数
            "hardware_docs": get_hardware_doc(self.backend, self.arch),
            "arch_name": self.arch,
            "database_examples": "",
        }

    def _load_dsl_sample_code(self) -> str:
        """
        根据framework加载对应的DSL示例代码

        Returns:
            str: 示例代码内容，如果找不到对应示例则返回空字符串
        """
        if not self.framework:
            logger.warning("framework为空，无法加载示例代码")
            return ""

        try:
            # 使用现有的get_triton_sample_code函数
            sample_code = get_triton_sample_code(self.framework)
            if sample_code:
                logger.info(f"成功加载{self.framework}示例代码")
                return sample_code
            else:
                logger.warning(f"未找到{self.framework}的示例代码")
                return ""
        except Exception as e:
            logger.warning(f"加载示例代码失败: {e}")
            return ""

    async def run(self, task_info: dict) -> Tuple[str, str, str]:
        """执行代码生成

        Args:
            task_info: 任务信息字典，包含当前所有代码和状态

        Returns:
            Tuple[str, str, str]: 生成的代码、提示信息和推理过程
        """
        # 从task_info中获取代码信息
        sketch = task_info.get('designer_code', '')

        # 从task_info中获取conductor的建议
        conductor_suggestion = task_info.get('conductor_suggestion', '')

        # 基于base_doc构建输入，只更新变化的部分
        input_data = {
            **self.base_doc,
            "sketch": sketch,  # AUL代码作为sketch
            "llm_suggestions": conductor_suggestion,  # Conductor建议
            "error_log": task_info.get('verifier_error', ''),
            "inspirations": get_inspirations(task_info.get('inspirations', [])),
        }

        # 执行LLM生成
        return await self.run_llm(self.coder_prompt, input_data, self.model_config["coder"])
