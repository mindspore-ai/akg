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
from typing import Tuple
from pathlib import Path

from ai_kernel_generator.utils.common_utils import ParserFactory, remove_copyright_from_text
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.markdown_utils import generate_available_api
from ai_kernel_generator.core.utils import ParsedCode, ActionType

logger = logging.getLogger(__name__)


def get_swft_sample_code(skip_name_list=None) -> str:
    """
    读取 swft_docs/examples 目录下的所有Python代码文件，并将它们拼接在一起。
    如果文件名包含在skip_name_list中，则跳过该文件。

    Args:
        skip_name_list (list): 要跳过的文件名列表，默认为None

    Returns:
        str: 所有Python代码文件内容拼接后的字符串

    Raises:
        FileNotFoundError: 当目录不存在时
    """
    if skip_name_list is None:
        skip_name_list = []

    root_dir = get_project_root()
    sample_dir = Path(root_dir) / "resources" / "docs" / "swft_docs" / "examples"

    if not sample_dir.exists():
        raise FileNotFoundError(f"sample目录不存在: {sample_dir}")

    all_code = []
    for file_path in sample_dir.glob("*.py"):
        if any(skip_name in file_path.name for skip_name in skip_name_list):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read().strip()
                if code:
                    all_code.append(f"# File: {file_path.name}\n{code}\n")
        except Exception as e:
            logger.warning(f"读取文件 {file_path} 时发生错误: {str(e)}")
            continue

    return "\n".join(all_code)


class SWFTCoder(AgentBase):
    def __init__(self, op_name: str, task_desc: str, model_config: dict, impl_type: str, framework: str):
        self.op_name = op_name
        self.task_desc = remove_copyright_from_text(task_desc)
        self.model_config = model_config
        self.impl_type = impl_type
        self.framework = framework
        agent_name = f"SWFTCoder -- [impl_type] {self.impl_type} -- [action] gen -- [op_name] {self.op_name}"
        super().__init__(agent_name=agent_name)

        # 初始化解析器
        self.api_parser = ParserFactory.get_api_parser()
        self.code_parser = ParserFactory.get_code_parser()
        self.format_api_instructions = self.api_parser.get_format_instructions()
        self.format_coder_instructions = self.code_parser.get_format_instructions()

        # 初始化API生成模板
        self.api_prompt = self.load_template("swft/api_gen_template.j2")
        self.api_base_doc = {
            "op_name": self.op_name,
            "framework": self.framework,
            "compute": self.load_doc("swft_docs/compute.md"),
            "composite": self.load_doc("swft_docs/composite.md"),
            "slicedata": self.load_doc("swft_docs/slicedata.md"),
            "move": self.load_doc("swft_docs/move.md"),
            "format_instructions": self.format_api_instructions,
        }
        self.api_input = {
            "aul_code": "",
            **self.api_base_doc,
        }
        self.intermediate_base_doc = {"supported_api": ""}

        # 初始化SWFT生成模板
        self.swft_gen_prompt = self.load_template("swft/swft_gen_template.j2")
        self.swft_fix_prompt = self.load_template("swft/swft_fix_template.j2")
        skip_list = ["mad", "matmul", "att", "encoder", "decoder"]
        self.swft_base_doc = {
            "op_name": self.op_name,
            "framework": self.framework,
            "task_desc": self.task_desc,
            "swft_sample_code": get_swft_sample_code(skip_name_list=skip_list),
            "error_sample": self.load_doc("swft_docs/error_samples.md"),
            "format_instructions": self.format_coder_instructions,
        }
        self.swft_gen_input = {
            "aul_code": "",
            "supported_api": "",
            **self.swft_base_doc,
        }
        self.swft_fix_input = {
            "aul_code": "",
            "supported_api": "",
            "swft_code": "",
            "suggestions": "",
            **self.swft_base_doc,
        }

    def update_api(self, swft_content):
        """
        更新API信息。

        Args:
            swft_content (str): 从LLM获取的SWFT内容。
        """
        parsed_content = self.api_parser.parse(swft_content)
        swft_api = {
            'compute': parsed_content.compute,
            'composite': parsed_content.composite,
            'move': parsed_content.move,
            'slicedata': parsed_content.slicedata
        }
        supported_api_str = generate_available_api(swft_api)
        self.intermediate_base_doc["supported_api"] = supported_api_str
        self.swft_gen_input["supported_api"] = supported_api_str
        self.swft_fix_input["supported_api"] = supported_api_str

    def update(self, action_type: str, aul_code: str, swft_code: str, suggestions: str):
        update_list = []
        if action_type != "":
            self.agent_name = f"SWFTCoder -- [impl_type] {self.impl_type} -- [action] {action_type.name} -- [op_name] {self.op_name}"
            update_list.append("action_type")
        if aul_code != "":
            self.api_input["aul_code"] = aul_code
            self.swft_gen_input["aul_code"] = aul_code
            self.swft_fix_input["aul_code"] = aul_code
            update_list.append("aul_code")
        if swft_code != "":
            self.swft_fix_input["swft_code"] = swft_code
            update_list.append("swft_code")
        if suggestions != "":
            self.swft_fix_input["suggestions"] = suggestions
            update_list.append("suggestions")
        if update_list:
            logger.debug("SWFTCoder update success: [%s] changed", ", ".join(update_list))

    async def run(self, action_type: ActionType, parsed_code: ParsedCode, suggestions: str) -> Tuple[str, str, str]:
        """执行SWFT代码生成或修复

        Args:
            action_type: 执行的动作类型
            parsed_code: conductor传入的解析代码内容
            suggestions: 建议

        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        # 提取代码内容并更新状态
        aul_code = parsed_code.aul_code if parsed_code else ""
        swft_code = parsed_code.swft_code if parsed_code else ""
        self.update(action_type, aul_code, swft_code, suggestions)

        swft_content, _, _ = await self.run_llm(self.api_prompt, self.api_input, self.model_config["swft_coder_api"])
        self.update_api(swft_content)

        # 根据动作类型选择对应的处理逻辑
        if action_type == ActionType.DO_CODER:
            return await self.run_llm(self.swft_gen_prompt, self.swft_gen_input, self.model_config["swft_coder"])
        elif action_type == ActionType.FIX_CODER:
            return await self.run_llm(self.swft_fix_prompt, self.swft_fix_input, self.model_config["swft_coder_fix"])
        else:
            raise ValueError(f"SWFTCoder不支持的动作类型: {action_type}")
