import pytest
import gc
from pathlib import Path
from ai_kernel_generator.core.agent.utils.feature_extraction import FeatureExtraction
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator import get_project_root

DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"

@pytest.mark.level0
@pytest.mark.asyncio
async def test_feature_extract():
    config = load_config()
    model_name_dict = config.get("agent_model_config")
    op_name = "elu"
    backend = "ascend"
    arch = "ascend310p3"
    impl_type = "swft"
    op_name = "elu"
    task_desc_path = DEFAULT_DATABASE_PATH / impl_type / arch / op_name / "aigen" / "elu_aul.py"

    with open(task_desc_path, "r", encoding="utf-8") as f:
        task_desc = f.read()
    feature = FeatureExtraction(
        op_name=op_name,
        task_desc=task_desc,
        model_config=model_name_dict,
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
