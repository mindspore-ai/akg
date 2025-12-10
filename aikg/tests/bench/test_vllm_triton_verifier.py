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
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.asyncio
@pytest.mark.parametrize("op_name", [
    # Attention操作 (3个)
    "l2norm_fwd",
    "merge_attn_states",
    "reshape_and_cache_flash",
    
    # Lightning Attention (4个)
    "fwd_diag_kernel",
    "fwd_kv_reduce",
    "fwd_none_diag_kernel",
    "linear_attn_decode_kernel",
    
    # Batch Invariant (3个)
    "log_softmax_kernel",
    "mean_kernel",
    "rms_norm_kernel",
    
    # Sample操作 (3个)
    "gumbel_sample_kernel",
    "topk_log_softmax_kernel",
    "ranks_kernel",
    
    # Block Table操作 (2个)
    "append_block_ids_kernel",
    "gather_block_tables_kernel",
    
    # Input Batch操作 (4个)
    "prepare_prefill_inputs_kernel",
    "prepare_pos_seq_lens_kernel",
    "combine_sampled_and_draft_tokens_kernel",
    "post_update_kernel",
    
    # FLA操作 (3个)
    "layernorm_fn",
    "rms_norm_gated_triton",
    "kda_gate_fwd_kernel",
    
    # MoE操作 (3个)
    "count_expert_num_tokens",
    "pack_bitmatrix",
    "compute_identity_kernel",
    
    # 序列工具 (2个)
    "pack_seq_triton",
    "unpack_seq_triton",
    
    # 旋转编码 (1个)
    "triton_mrope",
])
async def test_vllm_triton_verifier_a100(op_name):
    """
    测试vLLM的Triton算子精度
    
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

    op_task_file = f"./benchmark/aikgbench/vllm/triton_ops/{op_name}.py"
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
