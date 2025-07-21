import pytest
import gc
from pathlib import Path
from ai_kernel_generator.core.agent.utils.feature_extraction import FeatureExtraction
from ai_kernel_generator.utils.common_utils import load_yaml
from ai_kernel_generator import get_project_root

DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"
DEFAULT_CONFIG_PATH = Path(get_project_root()) / "database" / "rag_config.yaml"

@pytest.mark.level0
@pytest.mark.asyncio
async def test_feature_extract():
    config = load_yaml(DEFAULT_CONFIG_PATH)['agent_model_config']
    op_name = "elu"
    backend = "ascend"
    arch = "ascend310p3"
    impl_type = "swft"
    op_name = "elu"
    impl_code_path = DEFAULT_DATABASE_PATH / impl_type / arch / op_name / "aigen" / "elu_aul.py"

    with open(impl_code_path, "r", encoding="utf-8") as f:
        impl_code = f.read()
    feature = FeatureExtraction(
        model_config=config,
        impl_code=impl_code,
        framework_code="",
        impl_type=impl_type,
        backend=backend,
        arch=arch
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
