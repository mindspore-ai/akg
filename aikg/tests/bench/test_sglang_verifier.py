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
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
from ai_kernel_generator.utils.common_utils import create_log_dir
from ai_kernel_generator.config.config_validator import load_config
from ..utils import get_device_id

device_id = get_device_id()


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.asyncio
@pytest.mark.parametrize("op_name", [
    "assign_extend_cache_locs",
    "assign_req_to_token_pool",
    "compute_position",
    "fused_qkvzba_split_reshape_cat",
    "get_last_loc",
    "get_mla_kv_buffer",
    "set_mla_kv_buffer",
    "set_mla_kv_scale_buffer",
    "write_req_to_token_pool",
])
async def test_sglang_verifier_a100(op_name):
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    config = load_config(dsl, backend=backend)

    op_task_file = f"./benchmark/aikgbench/sglang/{op_name}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        framework_code = f.read()

    kernel_code = framework_code.replace("ModelSGLang", "ModelNew")

    log_dir = create_log_dir(f'{op_name}_sglang_{framework}_{backend}_{arch}_{dsl}_test')

    impl_func_name = "ModelNew"

    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=framework_code,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )

    task_info = {}
    task_info["coder_code"] = kernel_code

    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


