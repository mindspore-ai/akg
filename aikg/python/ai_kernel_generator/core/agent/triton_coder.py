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
from pathlib import Path
from typing import Tuple

from ai_kernel_generator.utils.common_utils import ParserFactory, remove_copyright_from_text
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.utils import ParsedCode, ActionType

logger = logging.getLogger(__name__)


def get_triton_sample_code(framework: str) -> str:
    """读取triton_docs/examples目录下的所有Python代码文件，并将它们拼接在一起

    Returns:
        str: 所有Python代码文件内容拼接后的字符串
    """
    base_dir = Path(get_project_root()) / "resources" / "docs" / "triton_docs" / "examples"
    sample_dir = base_dir / f"{framework}_examples"

    if not sample_dir.exists():
        logger.warning(f"Triton示例目录不存在: {sample_dir}, 返回空字符串")
        return ""

    all_code = []
    for file_path in sample_dir.glob("*.py"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read().strip()
                if code:
                    all_code.append(f"# File: {file_path.name}\n{code}\n")
        except Exception as e:
            logger.warning(f"读取Triton示例文件 {file_path} 时发生错误: {str(e)}")
            continue

    return "\n".join(all_code)


class TritonCoder(AgentBase):
    def __init__(self, op_name: str, task_desc: str, model_config: dict, impl_type: str, framework: str):
        self.op_name = op_name
        self.task_desc = remove_copyright_from_text(task_desc)
        self.model_config = model_config
        self.impl_type = impl_type
        self.framework = framework

        agent_name = f"TritonCoder -- [impl_type] {self.impl_type} -- [action] DO_CODER -- [op_name] {self.op_name} -- [framework] {self.framework}"
        super().__init__(agent_name=agent_name)

        # 初始化解析器
        self.code_parser = ParserFactory.get_code_parser()
        self.format_instructions = self.code_parser.get_format_instructions()

        # 初始化Triton生成模板
        self.triton_gen_prompt = self.load_template("triton/triton_gen_template.j2")
        self.triton_fix_prompt = self.load_template("triton/triton_fix_template.j2")

        # 准备基础文档数据
        self.triton_base_doc = {
            "op_name": self.op_name,
            "task_desc": self.task_desc,
            "framework": self.framework,
            "triton_api_str": self.load_doc("triton_docs/triton_api.md"),
            "triton_tutorial_str": self.load_doc("triton_docs/triton_tutorial.md"),
            "triton_sample_code": get_triton_sample_code(self.framework),
            "format_instructions": self.format_instructions,
        }

        # 初始化输入配置
        self.triton_gen_input = {
            "aul_code": "",
            **self.triton_base_doc,
        }
        self.triton_fix_input = {
            "aul_code": "",
            "triton_code": "",
            "suggestions": "",
            **self.triton_base_doc,
        }

    def update(self, action_type: ActionType, aul_code: str, triton_code: str, suggestions: str):
        """更新代理状态"""
        if action_type != ActionType.DO_CODER:
            self.agent_name = f"TritonCoder -- [impl_type] {self.impl_type} -- [action] {action_type.name} -- [op_name] {self.op_name}"

        if aul_code:
            self.aul_code = aul_code
            self.triton_gen_input["aul_code"] = aul_code
            self.triton_fix_input["aul_code"] = aul_code

        if triton_code:
            self.triton_fix_input["triton_code"] = triton_code

        if suggestions:
            self.triton_fix_input["suggestions"] = suggestions

    async def run(self, action_type: ActionType, parsed_code: ParsedCode, suggestions: str) -> Tuple[str, str, str]:
        """执行Triton代码生成或修复

        Args:
            action_type: 执行的动作类型
            parsed_code: conductor传入的解析代码内容
            suggestions: 基于反馈的改进建议

        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        # 提取代码内容并更新状态
        aul_code = parsed_code.aul_code if parsed_code else ""
        triton_code = parsed_code.triton_code if parsed_code else ""
        self.update(action_type, aul_code, triton_code, suggestions)

        # 根据动作类型选择对应的处理逻辑
        if action_type == ActionType.DO_CODER:
            return await self.run_llm(self.triton_gen_prompt, self.triton_gen_input, self.model_config["triton_coder"])
        elif action_type == ActionType.FIX_CODER:
            return await self.run_llm(self.triton_fix_prompt, self.triton_fix_input, self.model_config["triton_coder_fix"])
        else:
            raise ValueError(f"TritonCoder不支持的动作类型: {action_type}")
