from dataclasses import dataclass, field
from typing import List, Optional, Set
import asyncio
import logging
import os
from .interface import WorkerInterface

logger = logging.getLogger(__name__)

@dataclass
class WorkerInfo:
    worker: WorkerInterface
    backend: str
    arch: str
    tags: Set[str] = field(default_factory=set)
    capacity: int = 1
    load: int = 0

class WorkerManager:
    """
    Worker 管理器，负责 Worker 的注册、发现和负载均衡。
    注意：此管理器不负责资源的互斥锁定，只负责基于负载计数的路由选择。
    实际的资源互斥由 Worker 内部的 DevicePool 负责。
    """
    def __init__(self):
        self._workers: List[WorkerInfo] = []
        self._lock = asyncio.Lock()

    async def register(self, worker: WorkerInterface, backend: str, arch: str, tags: Set[str] = None, capacity: int = 1):
        """
        注册一个 Worker。
        
        Args:
            worker: Worker 实例
            backend: 后端类型 (cuda/ascend/cpu)
            arch: 硬件架构
            tags: 标签集合 (例如 {"local", "fast"})
            capacity: 并发能力 (通常对应设备卡数)
        """
        async with self._lock:
            info = WorkerInfo(
                worker=worker,
                backend=backend,
                arch=arch,
                tags=tags or set(),
                capacity=max(1, capacity)
            )
            self._workers.append(info)
            logger.info(f"Registered worker: backend={backend}, arch={arch}, capacity={capacity}")

    async def select(self, backend: str, arch: Optional[str] = None, tags: Set[str] = None) -> Optional[WorkerInterface]:
        """
        选择最佳 Worker。
        策略：
        1. 筛选符合 backend, arch, tags 的 Worker
        2. 在候选者中选择负载率 (load/capacity) 最低的
        3. 增加该 Worker 的 load 计数 (不独占，仅计数)
        
        Returns:
            WorkerInterface or None
        """
        async with self._lock:
            candidates = []
            for info in self._workers:
                if info.backend != backend:
                    continue
                if arch and info.arch != arch:
                    continue
                if tags and not tags.issubset(info.tags):
                    continue
                candidates.append(info)
            
            if not candidates:
                return None

            # 负载均衡策略：选择负载率最低的
            best_info = min(candidates, key=lambda w: w.load / w.capacity)
            
            best_info.load += 1
            logger.debug(f"Selected worker {id(best_info.worker)} (load={best_info.load}/{best_info.capacity})")
            return best_info.worker

    async def has_worker(self, backend: str, arch: Optional[str] = None, tags: Set[str] = None) -> bool:
        """
        判断是否存在匹配条件的 Worker，仅用于前置校验，不会修改负载。
        """
        async with self._lock:
            for info in self._workers:
                if info.backend != backend:
                    continue
                if arch and info.arch != arch:
                    continue
                if tags and not tags.issubset(info.tags):
                    continue
                return True
            return False

    async def release(self, worker: WorkerInterface):
        """
        归还 Worker (减少负载计数)。
        应在任务完成或异常退出时调用。
        """
        async with self._lock:
            for info in self._workers:
                if info.worker is worker:
                    info.load = max(0, info.load - 1)
                    logger.debug(f"Released worker {id(info.worker)} (load={info.load}/{info.capacity})")
                    return
            logger.warning("Released unknown worker")

    async def get_status(self) -> List[dict]:
        """获取所有 Worker 的状态快照"""
        async with self._lock:
            return [
                {
                    "backend": w.backend,
                    "arch": w.arch,
                    "load": w.load,
                    "capacity": w.capacity,
                    "tags": list(w.tags)
                }
                for w in self._workers
            ]

# 全局单例
_GLOBAL_MANAGER = WorkerManager()

def get_worker_manager() -> WorkerManager:
    return _GLOBAL_MANAGER

async def register_local_worker(device_ids: List[int], backend: str, arch: str, tags: Set[str] = None) -> None:
    """
    便捷函数：创建并注册 LocalWorker 到全局 WorkerManager。
    
    Args:
        device_ids: 设备ID列表，例如 [0] 或 [0, 1, 2, 3]
        backend: 后端类型 (cuda/ascend/cpu)
        arch: 硬件架构 (a100/ascend910b4等)
        tags: 可选的标签集合
    
    Example:
        # 单卡
        await register_local_worker([0], backend="cuda", arch="a100")
        
        # 多卡
        await register_local_worker([0, 1, 2, 3], backend="cuda", arch="a100")
    """
    from .local_worker import LocalWorker
    from ..async_pool.device_pool import DevicePool
    
    device_pool = DevicePool(device_ids)
    local_worker = LocalWorker(device_pool, backend=backend)
    await _GLOBAL_MANAGER.register(
        local_worker,
        backend=backend,
        arch=arch,
        tags=tags,
        capacity=len(device_ids)
    )
    logger.info(f"✅ Registered LocalWorker: backend={backend}, arch={arch}, devices={device_ids}")

async def register_remote_worker(backend: str, arch: str, worker_url: Optional[str] = None, capacity: Optional[int] = None, tags: Set[str] = None) -> None:
    """
    便捷函数：创建并注册 RemoteWorker 到全局 WorkerManager。
    如果未提供 worker_url，将从环境变量 AIKG_WORKER_URL 读取。
    如果未提供 capacity，将自动从 remote worker 的 status 接口查询实际设备数量。
    
    Args:
        backend: 后端类型 (cuda/ascend/cpu)
        arch: 硬件架构 (a100/ascend910b4等)
        worker_url: Remote Worker Service 的 URL（可选，默认从环境变量 AIKG_WORKER_URL 读取）
        capacity: 并发能力（通常对应设备卡数）。如果为 None，将自动从 remote worker 查询
        tags: 可选的标签集合
    
    Example:
        # 从环境变量读取 worker_url，自动获取 capacity
        export AIKG_WORKER_URL=http://localhost:9001
        await register_remote_worker(backend="cuda", arch="a100")
        
        # 显式指定 worker_url 和 capacity
        await register_remote_worker(
            backend="cuda", 
            arch="a100", 
            worker_url="http://localhost:9001",
            capacity=2
        )
    """
    import httpx
    from .remote_worker import RemoteWorker
    
    if worker_url is None:
        worker_url = os.getenv("AIKG_WORKER_URL")
        if worker_url is None:
            raise ValueError(
                "worker_url 未提供且环境变量 AIKG_WORKER_URL 未设置。\n"
                "请提供 worker_url 参数或设置环境变量：\n"
                "  export AIKG_WORKER_URL=http://localhost:9001"
            )
    
    # 如果 capacity 未指定，从 remote worker 查询实际设备数量
    if capacity is None:
        try:
            status_url = f"{worker_url.rstrip('/')}/api/v1/status"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(status_url)
                response.raise_for_status()
                status = response.json()
                devices = status.get("devices", [])
                if isinstance(devices, list) and len(devices) > 0:
                    capacity = len(devices)
                    logger.info(f"从 remote worker 查询到 {capacity} 个设备: {devices}")
                else:
                    capacity = 1
                    logger.warning(f"无法从 remote worker 获取设备信息，使用默认 capacity=1")
        except Exception as e:
            capacity = 1
            logger.warning(f"查询 remote worker status 失败: {e}，使用默认 capacity=1")
    
    remote_worker = RemoteWorker(worker_url)
    await _GLOBAL_MANAGER.register(
        remote_worker,
        backend=backend,
        arch=arch,
        tags=tags,
        capacity=max(1, capacity)
    )
    logger.info(f"✅ Registered RemoteWorker: backend={backend}, arch={arch}, url={worker_url}, capacity={capacity}")


async def register_worker(
    backend: str,
    arch: str,
    *,
    device_ids: Optional[List[int]] = None,
    worker_url: Optional[str] = None,
    tags: Optional[Set[str]] = None
) -> None:
    """
    统一的 Worker 注册入口。

    优先级：
      1. 若提供 worker_url 或设置 AIKG_WORKER_URL，则注册 RemoteWorker
      2. 否则若提供 device_ids，则注册 LocalWorker
      3. 若均未提供，则抛出错误提示用户如何注册
    """
    resolved_worker_url = worker_url or os.getenv("AIKG_WORKER_URL")
    if resolved_worker_url:
        await register_remote_worker(
            backend=backend,
            arch=arch,
            worker_url=resolved_worker_url,
            tags=tags
        )
        return

    if device_ids:
        await register_local_worker(device_ids, backend=backend, arch=arch, tags=tags)
        return

    raise RuntimeError(
        "未找到可用的 Worker。请先注册 Worker 再运行 evolve：\n"
        "  方式一：设置远程 Worker URL\n"
        "    export AIKG_WORKER_URL=http://<worker-host>:<port>\n"
        "  方式二：指定本地设备列表调用 register_worker(..., device_ids=[0])"
    )

