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
from pathlib import Path
from ai_kernel_generator.core.agent.utils.api_generator import APIGenerator
from ai_kernel_generator import get_project_root
from ai_kernel_generator.config.config_validator import load_config

RESOURCES_PATH = Path("tests") / "resources"


@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("framework,dsl", [
    ("numpy", "swft"),
    ("torch", "triton"),
])
async def test_api_gen(framework, dsl):
    config = load_config()['agent_model_config']
    op_name = "relu_op"
    framework_path = RESOURCES_PATH / op_name / f"{op_name}_{framework}.py"
    with open(framework_path, "r", encoding="utf-8") as f:
        task_desc = f.read()

    sketch_path = RESOURCES_PATH / op_name / f"{op_name}_{dsl}.py"
    with open(sketch_path, "r", encoding="utf-8") as f:
        sketch = f.read()

    api_gen = APIGenerator(
        model_config=config,
        dsl=dsl,
        task_desc=task_desc,
        sketch=sketch
    )
    try:
        api_str = await api_gen.run()
        print(f"模型返回的算子的api：{api_str}\n")
    finally:
        if hasattr(api_gen, "close"):
            await api_gen.close()
        elif hasattr(api_gen, "__aexit__"):
            await api_gen.__aexit__(None, None, None)
        gc.collect()
