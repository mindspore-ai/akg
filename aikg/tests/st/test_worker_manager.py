import pytest
import asyncio
from typing import Tuple, Dict, Any
from ai_kernel_generator.core.worker.interface import WorkerInterface
from ai_kernel_generator.core.worker.manager import WorkerManager

# Mock Worker
class MockWorker(WorkerInterface):
    def __init__(self, name):
        self.name = name

    async def verify(self, package_data: bytes, task_id: str, op_name: str, timeout: int = 300) -> Tuple[bool, str]:
        return True, "Success"

    async def profile(self, package_data: bytes, task_id: str, op_name: str, profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        return {}

@pytest.mark.asyncio
async def test_worker_manager_basic_flow():
    """测试基本的注册、选择、释放流程"""
    manager = WorkerManager()
    worker1 = MockWorker("worker1")
    
    # 1. 注册
    await manager.register(worker1, backend="cuda", arch="a100", capacity=2)
    
    # 2. 选择 (第一次)
    selected = await manager.select(backend="cuda", arch="a100")
    assert selected is worker1
    status = await manager.get_status()
    assert status[0]["load"] == 1
    
    # 3. 选择 (第二次，应仍可选中同一个，因为 capacity=2)
    selected2 = await manager.select(backend="cuda")
    assert selected2 is worker1
    status = await manager.get_status()
    assert status[0]["load"] == 2
    
    # 4. 释放
    await manager.release(selected)
    status = await manager.get_status()
    assert status[0]["load"] == 1
    
    await manager.release(selected2)
    status = await manager.get_status()
    assert status[0]["load"] == 0

@pytest.mark.asyncio
async def test_worker_manager_load_balancing():
    """测试负载均衡逻辑"""
    manager = WorkerManager()
    w1 = MockWorker("w1") # capacity=2
    w2 = MockWorker("w2") # capacity=2
    
    await manager.register(w1, "cuda", "a100", capacity=2)
    await manager.register(w2, "cuda", "a100", capacity=2)
    
    # 1. 第一次选，此时都为 0/2，可能会选 w1 (列表序)
    s1 = await manager.select("cuda")
    assert s1 is w1 # w1: 1/2, w2: 0/2
    
    # 2. 第二次选，应选负载更低的 w2
    s2 = await manager.select("cuda")
    assert s2 is w2 # w1: 1/2, w2: 1/2
    
    # 3. 第三次选，此时都为 1/2，可能选 w1
    s3 = await manager.select("cuda")
    assert s3 is w1 # w1: 2/2, w2: 1/2
    
    # 4. 第四次选，应选 w2
    s4 = await manager.select("cuda")
    assert s4 is w2 # w1: 2/2, w2: 2/2
    
    # 释放 w1 的一个任务
    await manager.release(s1) # w1: 1/2, w2: 2/2
    
    # 5. 第五次选，应选 w1 (负载更低)
    s5 = await manager.select("cuda")
    assert s5 is w1

@pytest.mark.asyncio
async def test_worker_manager_filtering():
    """测试筛选逻辑"""
    manager = WorkerManager()
    w_cuda = MockWorker("cuda")
    w_ascend = MockWorker("ascend")
    
    await manager.register(w_cuda, "cuda", "a100")
    await manager.register(w_ascend, "ascend", "910b")
    
    # 匹配 backend
    assert (await manager.select("cuda")) is w_cuda
    assert (await manager.select("ascend")) is w_ascend
    
    # 匹配 arch
    assert (await manager.select("cuda", arch="a100")) is w_cuda
    assert (await manager.select("cuda", arch="v100")) is None # 不存在
    
    # 释放计数
    await manager.release(w_cuda)
    await manager.release(w_cuda) # 此时 load=0
    await manager.release(w_ascend)

@pytest.mark.asyncio
async def test_worker_manager_tags():
    """测试标签筛选"""
    manager = WorkerManager()
    w_remote = MockWorker("remote")
    w_local = MockWorker("local")
    
    await manager.register(w_remote, "cuda", "a100", tags={"remote", "fast"})
    await manager.register(w_local, "cuda", "a100", tags={"local"})
    
    # 筛选 tags
    assert (await manager.select("cuda", tags={"remote"})) is w_remote
    assert (await manager.select("cuda", tags={"local"})) is w_local
    assert (await manager.select("cuda", tags={"fast", "remote"})) is w_remote
    assert (await manager.select("cuda", tags={"fast", "local"})) is None # 没有同时满足的

