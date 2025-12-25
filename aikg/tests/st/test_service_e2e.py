import pytest
import asyncio
import logging
from ai_kernel_generator.server.job_manager import get_job_manager
from ai_kernel_generator.core.worker.manager import get_worker_manager
from ai_kernel_generator.core.worker.local_worker import LocalWorker
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import get_kernelbench_task_desc, add_op_prefix, get_kernelbench_op_name

logger = logging.getLogger(__name__)

@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.asyncio
async def test_e2e_service_flow_cuda_real():
    """
    真实环境下的端到端服务化测试：
    Client(模拟) -> ServerJobManager -> WorkerManager -> LocalWorker -> 真实执行
    
    前置条件：
    1. 需要有 NVIDIA GPU 环境
    2. 需要配置 AIKG_API_KEY (用于 LLM 代码生成)
    """
    
    # 1. 准备环境：注册 LocalWorker
    # 注意：这会直接使用当前机器的 GPU 0
    worker_manager = get_worker_manager()
    
    # 创建并注册 CUDA Worker
    # DevicePool([0]) 假设至少有一张卡
    # 如果是在无卡环境，这里可能不会报错，但在 verify 阶段会失败
    cuda_pool = DevicePool([0]) 
    cuda_worker = LocalWorker(cuda_pool, backend="cuda")
    
    # 注册到 Manager
    await worker_manager.register(cuda_worker, backend="cuda", arch="a100", capacity=1)
    
    # 2. 准备 Job
    job_manager = get_job_manager()
    
    # 获取一个简单的算子任务描述
    try:
        framework = "torch"
        # 获取 KernelBench 中第一个算子
        benchmark_names = get_kernelbench_op_name([0], framework=framework)
        if benchmark_names:
            op_base_name = benchmark_names[0]
            task_desc = get_kernelbench_task_desc(op_base_name, framework=framework)
            op_name = add_op_prefix(op_base_name, benchmark="KernelBench")
        else:
            raise ValueError("No benchmark found")
    except Exception as e:
        logger.warning(f"Failed to load KernelBench task: {e}. Using fallback task.")
        # Fallback 
        op_name = "relu_service_test"
        task_desc = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(16, 1024).cuda()]

def get_init_inputs():
    return []
"""

    # 3. 提交 Job
    logger.info(f"Submitting CUDA Job: {op_name}")
    job_id = await job_manager.submit_job({
        "op_name": op_name,
        "task_desc": task_desc,
        "job_type": "single",
        "backend": "cuda",
        "arch": "a100",
        "dsl": "triton_cuda",
        "framework": "torch",
        "workflow": "coder_only_workflow" 
    })
    
    logger.info(f"Job submitted: {job_id}")
    
    # 4. 等待完成
    # 设置较长的超时时间，因为包含 LLM 生成 + 编译 + 运行
    timeout = 600 # 10分钟
    interval = 5
    
    for _ in range(timeout // interval):
        status = job_manager.get_job_status(job_id)
        state = status.get("status")
        
        if state in ["completed", "failed", "error"]:
            logger.info(f"Job finished with state: {state}")
            break
            
        logger.info(f"Job running... status: {state}")
        await asyncio.sleep(interval)
        
    # 5. 验证结果
    final_status = job_manager.get_job_status(job_id)
    if final_status["status"] != "completed":
        logger.error(f"Job failed details: {final_status}")
        
    assert final_status["status"] == "completed", f"Job failed: {final_status.get('error')}"
    assert final_status["result"] is True
