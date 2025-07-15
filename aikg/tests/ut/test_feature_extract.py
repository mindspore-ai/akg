import pytest
from ai_kernel_generator.core.agent.feature_extraction import FeatureExtraction
from ..utils import get_benchmark_name, get_benchmark_task, add_op_prefix
from ai_kernel_generator.config.config_validator import load_config

@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("framework,impl_type,backend,arch", [
    # ("mindspore", "swft", "ascend", "ascend310p3"),
    ("numpy", "swft", "ascend", "ascend310p3"),
])
async def test_feature_extract_ascend310p3(framework, impl_type, backend, arch):
    benchmark_name = get_benchmark_name([19], framework=framework)
    config = load_config()
    model_name_dict = config.get("agent_model_config")
    for i in range(len(benchmark_name)):
        task_desc = get_benchmark_task(benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i])
        feature = FeatureExtraction(
            task_code=task_desc,
            model_config=model_name_dict,
            impl_type=impl_type,
            backend=backend,
            arch=arch
        )
        feature_res, feature_prompt, feature_reasoning = await feature.run()

        print(f"模型返回的算子{op_name}的特征文本：{feature_res}\n")
