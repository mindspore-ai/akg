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

import os
import pytest
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
from ai_kernel_generator.utils.common_utils import create_log_dir
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.worker.manager import register_remote_worker, get_worker_manager


# 从环境变量获取 Worker URL
cuda_worker_url = os.environ.get("CUDA_WORKER_URL", "http://localhost:9001")
ascend_worker_url = os.environ.get("ASCEND_WORKER_URL", "http://localhost:9001")
    
def get_device_id():
    return 3

device_id = get_device_id()


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.asyncio
@pytest.mark.parametrize("op_name", [
    # 激活函数 (11个)
    "silu_and_mul",
    "mul_and_silu",
    "gelu_and_mul",
    "gelu_and_mul_sparse",
    "fatrelu_and_mul",
    "swigluoai_and_mul",
    "gelu_new",
    "gelu_fast",
    "gelu_quick",
    "relu_squared",
    "xielu",
    
    # 层归一化 (4个)
    "rms_norm",
    "fused_add_rms_norm",
    "gemma_rms_norm",
    "layer_norm",
    
    # 旋转位置编码 (2个)
    "rotary_embedding",
    "yarn_scaling_rope",
])
async def test_vllm_torch_verifier_a100(op_name):
    """
    测试vLLM的PyTorch算子精度
    
    验证Model（原生实现）和ModelVLLM（vLLM优化实现）的输出一致性
    """
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    # config = load_config(dsl, backend=backend)
    
    try:
        await register_remote_worker(
            backend="cuda",
            arch="a100",
            worker_url=cuda_worker_url
        )
        print(f"  ✓ CUDA Worker 注册成功")
    except Exception as e:
        print(f"  ✗ CUDA Worker 注册失败: {e}")
        return False

    worker_manager = get_worker_manager()
    print()

    cuda_config = load_config("triton_cuda", backend="cuda")
    
    cuda_worker = await worker_manager.select(backend="cuda", arch="a100")
    if not cuda_worker:
        print("  ✗ 无法获取 CUDA Worker")
        return False

    op_task_file = f"./benchmark/aikgbench/vllm/torch_ops/{op_name}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        framework_code = f.read()

    # 将ModelVLLM替换为ModelNew用于测试
    kernel_code = framework_code.replace("ModelVLLM", "ModelNew")

    log_dir = create_log_dir(f'{op_name}_vllm_{framework}_{backend}_{arch}_{dsl}_test')

    impl_func_name = "ModelNew"
    
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=framework_code,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=cuda_config,
        worker=cuda_worker
    )

    task_info = {}
    task_info["coder_code"] = kernel_code

    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败 [{op_name}]: {error_log}"
