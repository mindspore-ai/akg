import logging
from typing import Tuple
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import ParserFactory
logger = logging.getLogger(__name__)

class FeatureExtraction(AgentBase):
    def __init__(self, task_code: str, model_config: dict):
        self.task_code = task_code
        self.model_config = model_config

        agent_name = f"FeatureExtraction"
        super().__init__(agent_name=agent_name)

        # 初始化解析器
        self.feature_parser = ParserFactory.get_feature_parser()
        self.format_instructions = self.feature_parser.get_format_instructions()

        # 初始化模板
        self.feature_extraction_template = self.load_template("feature_extraction/feature_extraction_template.j2")

        self.feature_extraction_input = {
            "task_code": self.task_code,
            "format_instructions": self.format_instructions,
        }

    async def run(self) -> Tuple[str, str, str]:
        return await self.run_llm(self.feature_extraction_template, self.feature_extraction_input, self.model_config["feature_extraction"])
