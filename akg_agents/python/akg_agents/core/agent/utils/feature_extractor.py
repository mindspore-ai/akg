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
from akg_agents.core_v2.agents import AgentBase, register_agent
from akg_agents.utils.common_utils import ParserFactory, get_md5_hash
logger = logging.getLogger(__name__)


@register_agent
class FeatureExtractor(AgentBase):
    """
    根据算子的初始实现或者调度，分析出当前算子的特征信息。
    特征信息包括：
        - 算子基础信息
        - 调度策略
        - 特征信息总结
    """

    def __init__(self, model_config: dict, impl_code: str = "", framework_code: str = "", dsl: str = "", config: dict = None):
        self.model_config = model_config
        self.impl_code = impl_code
        self.framework_code = framework_code
        self.dsl = dsl
        self.config = config

        context = {
            "agent_name": "feature_extractor",
        }
        # 如果 config 中包含 session_id，添加到 context 中
        # 这样在流式输出启用时，run_llm 方法可以正确获取 session_id
        if config and config.get("session_id"):
            context["session_id"] = config["session_id"]
        super().__init__(context=context, config=config)

        # 初始化解析器
        self.feature_parser = ParserFactory.get_feature_parser()
        self.format_instructions = self.feature_parser.get_format_instructions()

        # 初始化模板
        self.feature_extraction_template = self.load_template("utils/feature_extraction_template.j2")

        self.feature_extraction_input = {
            "impl_code": self.impl_code,
            "framework_code": self.framework_code,
            "dsl": self.dsl,
            "format_instructions": self.format_instructions,
        }

    async def run(self) -> Tuple[str, str, str]:
        # 执行LLM生成前更新context，确保正确性
        hash = get_md5_hash(impl_code=self.impl_code)
        to_update_context = {
            "agent_name": "feature_extractor",
            "hash": hash,
        }
        self.context.update(to_update_context)

        return await self.run_llm(self.feature_extraction_template, self.feature_extraction_input, self.model_config.get("feature_extractor") or "standard")
