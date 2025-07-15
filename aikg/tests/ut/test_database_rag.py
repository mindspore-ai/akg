import pytest
import gc
from ai_kernel_generator.core.agent.feature_extraction import FeatureExtraction
from ..utils import get_benchmark_name, get_benchmark_task, add_op_prefix
from ai_kernel_generator.config.config_validator import load_config

@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("framework", ["numpy"])
async def test_database_rag(framework):
    benchmark_name = get_benchmark_name([19], framework=framework)
    config = load_config()
    model_name_dict = config.get("agent_model_config")
    task_desc = get_benchmark_task(benchmark_name[0], framework=framework)
    op_name = add_op_prefix(benchmark_name[0])
    feature = FeatureExtraction(
        task_code=task_desc,
        model_config=model_name_dict
    )
    try:
        feature_res, feature_prompt, feature_reasoning = await feature.run()
        print(f"模型返回的算子{op_name}的特征文本：{feature_res}\n")
    finally:
        if hasattr(feature, "close"):
            await feature.close()
        elif hasattr(feature, "__aexit__"):
            await feature.__aexit__(None, None, None)
        gc.collect()
