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

import pytest
import gc
from ai_kernel_generator.core.agent.utils.feature_extractor import FeatureExtractor
from ai_kernel_generator.config.config_validator import load_config


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_feature_extract():
    framework = "torch"
    dsl = "triton"
    
    op_name = "relu"
    
    framework_code_path = f"tests/resources/{op_name}_op/{op_name}_{framework}.py"
    impl_code_path = f"tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(framework_code_path, "r", encoding="utf-8") as f:
        framework_code = f.read()
    with open(impl_code_path, "r", encoding="utf-8") as f:
        impl_code = f.read()
    
    config = load_config(dsl).get("agent_model_config", {})
    feature = FeatureExtractor(
        model_config=config,
        impl_code=impl_code,
        framework_code=framework_code,
        dsl=dsl
    )
    try:
        feature_res, _, _ = await feature.run()
        print(f"模型返回的算子{op_name}的特征文本：{feature_res}\n")
    finally:
        if hasattr(feature, "close"):
            await feature.close()
        elif hasattr(feature, "__aexit__"):
            await feature.__aexit__(None, None, None)
        gc.collect()
