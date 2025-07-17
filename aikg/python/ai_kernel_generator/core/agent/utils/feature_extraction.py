import logging
from typing import Tuple
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import ParserFactory, get_md5_hash
logger = logging.getLogger(__name__)

class FeatureExtraction(AgentBase):
    def __init__(self, task_desc: str, model_config: dict, impl_type: str = "", backend: str = "", arch: str = ""):
        self.task_desc = task_desc
        self.model_config = model_config
        self.impl_type = impl_type
        self.backend = backend
        self.arch = arch

        agent_name = f"FeatureExtraction"
        super().__init__(agent_name=agent_name)

        # 初始化解析器
        self.feature_parser = ParserFactory.get_feature_parser()
        self.format_instructions = self.feature_parser.get_format_instructions()

        # 初始化模板
        self.feature_extraction_template = self.load_template("feature_extraction/feature_extraction_template.j2")

        self.feature_extraction_input = {
            "task_desc": self.task_desc,
            "impl_type": self.impl_type,
            "backend": self.backend,
            "arch": self.arch,
            "format_instructions": self.format_instructions,
        }

    async def run(self) -> Tuple[str, str, str]:
        return await self.run_llm(self.feature_extraction_template, self.feature_extraction_input, self.model_config["feature_extraction"])
