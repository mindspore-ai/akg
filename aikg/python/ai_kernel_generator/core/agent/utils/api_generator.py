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

from ai_kernel_generator.utils.common_utils import ParserFactory, remove_copyright_from_text, load_directory
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.markdown_utils import SWFTDocsProcessor

logger = logging.getLogger(__name__)


def get_swft_path():
    import swft
    return swft.__path__[0]


class APIGenerator(AgentBase):
    def __init__(self, task_desc: str, sketch: str, dsl: str, model_config: dict):
        self.task_desc = remove_copyright_from_text(task_desc)
        self.sketch = sketch
        self.dsl = dsl
        self.model_config = model_config
        self.llm_step_count = 0
        agent_details = {
            "agent_name": "api_generator",
            "dsl": self.dsl,
        }
        super().__init__(agent_details=agent_details)

        # 初始化解析器
        self.api_parser = ParserFactory.get_api_parser()
        self.format_api_instructions = self.api_parser.get_format_instructions()

        if self.dsl == "swft":
            swft_dir = Path(get_swft_path())
            swft_docs_dir = swft_dir / "docs"
            self.api_docs = load_directory(swft_docs_dir, file_extensions=[".md"])
        else:
            root_dir = get_project_root()
            docs_dir = Path(root_dir) / "resources" / "docs" / f"{dsl}_docs"
            self.api_docs = load_directory(docs_dir)

        # 初始化API生成模板
        self.api_prompt = self.load_template("utils/api_gen_template.j2")
        self.api_input = {
            "dsl": self.dsl,
            "sketch": self.sketch,
            "task_desc": self.task_desc,
            "dsl_api_docs": self.api_docs,
            "format_instructions": self.format_api_instructions,
        }

    async def run(self) -> str:
        """
        运行API生成器
        """
        # 执行LLM生成前更新agent_details，确保正确性
        # TODO: 缺少task_id
        self.llm_step_count += 1
        to_update_agent_details = {
            "agent_name": "api_generator",
            "step": self.llm_step_count,
        }
        self.agent_details.update(to_update_agent_details)

        api_json, _, _ = await self.run_llm(self.api_prompt, self.api_input, self.model_config["api_generator"])
        parsed_content = self.api_parser.parse(api_json)

        formatted_str = ""
        for name, desc, impl in zip(parsed_content.api_name, parsed_content.api_desc, parsed_content.api_example):
            formatted_str += f"API name: {name}\nAPI description:{desc}\nAPI implement：\n{impl}\n\n"

        if self.dsl == "swft":
            swft_dir = Path(get_swft_path())
            swft_impl_dir = swft_dir / "api"
            swft_docs_dir = swft_dir / "docs"
            swft_doc = SWFTDocsProcessor()
            supported_api_str = swft_doc.generate_available_api(parsed_content.api_name, swft_docs_dir, swft_impl_dir)
            return supported_api_str.strip()
        return formatted_str.strip()
