import logging
import os
from pathlib import Path
from typing import Tuple
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import ParserFactory
from ai_kernel_generator import get_project_root
os.environ['STREAM_OUTPUT_MODE'] = 'on'
logger = logging.getLogger(__name__)
def get_feature_base_doc() -> str:
    doc_files = [
        "feature_suggestions.md"
    ]
    docs_dir = Path(get_project_root()) / "resources" / "docs" / "aul_docs"
    combined_spec = ""

    for doc_file in doc_files:
        doc_path = docs_dir / doc_file
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
                combined_spec += content + "\n\n"
        except Exception as e:
            logger.warning(f"加载AUL文档失败 {doc_file}: {e}")
            continue

    return combined_spec

def get_benchmark_task(op_name, framework="mindspore"):
    # current_file_path = os.path.abspath(__file__)
    commom_path = "/home/zqs/latest-aikg/ai_kernel_generator/"
    task_path = os.path.join(os.path.dirname(commom_path), 'benchmark',
                             'kernelbench', framework,
                             op_name, op_name + f'_{framework}.py')
    with open(task_path, "r", encoding="utf-8") as f:
        benchmark_task_str = f.read()
    return benchmark_task_str

class FeatureExtraction(AgentBase):
    def __init__(self, task_code: str, model_config: dict, impl_type: str = "", backend: str = "", arch: str = ""):
        self.task_code = task_code
        self.model_config = model_config
        self.impl_type = impl_type
        self.backend = backend
        self.arch = arch

        agent_name = f"feature matching -- "
        super().__init__(agent_name=agent_name)

        # 初始化解析器
        self.code_parser = ParserFactory.get_feature_parser()
        self.format_instructions = self.code_parser.get_format_instructions()

        # 初始化模板
        self.feature_gen_prompt = self.load_template("aul/feature_gen_template.j2")

        # 准备基础文档数据
        self.feature_base_doc = {
            "backend": self.backend,
            "arch": self.arch,
            "impl": self.impl_type,
            "task_code": self.task_code,
            "format_instructions": self.format_instructions,
        }

        # 初始化输入配置
        self.feature_gen_input = {**self.feature_base_doc}

    async def run(self) -> Tuple[str, str, str]:
        return await self.run_llm(self.feature_gen_prompt, self.feature_gen_input, self.model_config["feature_match_designer"])
    


    