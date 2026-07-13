import os
import sys
import logging
from typing import Annotated, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import uvicorn

from akg_agents.core.worker.local_worker import LocalWorker
from akg_agents.core.async_pool.device_pool import DevicePool
from akg_agents.core.worker.eval_config import (
    resolve_eval_timeout,
    resolve_reference_timeout,
)
from akg_agents.cli.service.worker_config import worker_timing
from akg_agents.op.utils.config_utils import check_backend_arch
from akg_agents.op.utils.json_safe import sanitize_floats
from akg_agents.utils.process_utils import reap_orphaned_process_groups

# Configure logging. stream=sys.stdout keeps logs chronological under
# capture frameworks (see akg_agents/__init__.py).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# Global worker instance
worker: Optional[LocalWorker] = None

def get_worker_config():
    """Get worker configuration from environment variables."""
    names = ("WORKER_BACKEND", "WORKER_ARCH", "WORKER_DEVICES")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            f"worker configuration missing: {', '.join(missing)}; "
            "start the daemon through `akg_cli worker --start`")

    backend, arch, devices_str = (os.environ[name].strip() for name in names)
    check_backend_arch(backend, arch)
    try:
        devices = [int(value.strip()) for value in devices_str.split(",")]
        if not devices or any(device < 0 for device in devices):
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            f"WORKER_DEVICES must be comma-separated non-negative integers, "
            f"got {devices_str!r}") from exc

    return backend, arch, devices

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize worker resources on startup."""
    global worker
    backend, arch, devices = get_worker_config()

    reaped = reap_orphaned_process_groups()
    if reaped:
        logger.warning("Reaped orphan eval process groups from predecessor: %s",
                       reaped)

    logger.info(f"Initializing Worker Service: Backend={backend}, Arch={arch}, Devices={devices}")
    
    timing = worker_timing()
    device_pool = DevicePool(devices, lease_ttl_s=timing.lease_ttl)
    device_pool.start_reaper(timing.lease_reap_interval)
    worker = LocalWorker(device_pool, backend=backend)

    yield

    await device_pool.stop_reaper()
    logger.info("Shutting down Worker Service")

app = FastAPI(title="AIKG Worker Service", lifespan=lifespan)


def _require_worker() -> None:
    """503 if the worker isn't initialized yet (startup race)."""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")


@asynccontextmanager
async def _guarded(task_id: str, *, device_id: Optional[int] = None,
                   lease_id: Optional[int] = None):
    """Shared eval-endpoint guard: renew the device lease for the whole
    request, and map any unhandled error to a 500 (HTTPExceptions — e.g. the
    400 from a bad profile_settings parse done before entering — pass
    through). Single owner of the keepalive + error-mapping boilerplate."""
    try:
        async with worker.device_pool.keepalive(
                task_id, device_id=device_id, lease_id=lease_id):
            yield
    except LookupError as e:
        # The script contains a device id acquired under this exact token.
        # If the token is stale, that device may already have a new owner.
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{task_id}] request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/verify")
async def verify(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: Optional[int] = Form(None),
    device_id: Optional[int] = Form(None),
    lease_id: Optional[int] = Form(None),
):
    """
    Execute verification task.
    
    Returns:
        - success: 验证是否成功
        - log: 执行日志
        - artifacts: 执行过程中生成的 JSON 文件内容
    """
    _require_worker()
    logger.info(f"[{task_id}] Received verification request for {op_name}")
    async with _guarded(task_id, device_id=device_id, lease_id=lease_id):
        package_data = await package.read()
        success, log, artifacts = await worker.verify(
            package_data, task_id, op_name, resolve_eval_timeout(timeout))
        return sanitize_floats({
            "success": success,
            "log": log,
            "artifacts": artifacts,
        })

@app.post("/api/v1/profile")
async def profile(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    profile_settings: str = Form("{}"),
    device_id: Optional[int] = Form(None),
    lease_id: Optional[int] = Form(None),
):
    """
    Execute profiling task.
    """
    _require_worker()
    import json
    try:
        settings = json.loads(profile_settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for profile_settings")

    async with _guarded(task_id, device_id=device_id, lease_id=lease_id):
        package_data = await package.read()
        result = await worker.profile(package_data, task_id, op_name, settings)
        return sanitize_floats(result)

@app.post("/api/v1/generate_reference")
async def generate_reference(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: Optional[int] = Form(None),
    device_id: Optional[int] = Form(None),
    lease_id: Optional[int] = Form(None),
):
    """
    Execute task_desc and generate reference data.
    
    用于 CUDA-to-Ascend 转换场景：执行 Triton-CUDA 代码，
    保存输出作为参考数据（.pt 文件），并以 base64 编码返回。
    
    Returns:
        - success: 是否成功生成参考数据
        - log: 执行日志
        - reference_data: base64 编码的 .pt 文件内容
    """
    import base64

    _require_worker()
    logger.info(f"[{task_id}] Received generate_reference request for {op_name}")
    async with _guarded(task_id, device_id=device_id, lease_id=lease_id):
        package_data = await package.read()
        success, log, ref_bytes = await worker.generate_reference(
            package_data, task_id, op_name,
            resolve_reference_timeout(timeout)
        )
        # 成功时 base64 编码返回二进制数据；失败时空串。
        return {
            "success": success,
            "log": log,
            "reference_data": (base64.b64encode(ref_bytes).decode('utf-8')
                               if success else ""),
        }

@app.post("/api/v1/profile_single_task")
async def profile_single_task(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    profile_settings: str = Form("{}"),
    device_id: Optional[int] = Form(None),
    lease_id: Optional[int] = Form(None),
):
    """
    Execute single task profiling (only measure task_desc performance, no base comparison).
    
    单独测量某段代码的执行性能，不进行 base vs generation 对比。
    
    Returns:
        - time_us: 执行时间（微秒）
        - success: 是否成功
        - log: 执行日志
    """
    _require_worker()
    import json
    try:
        settings = json.loads(profile_settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for profile_settings")

    logger.info(f"[{task_id}] Received profile_single_task request for {op_name}")
    async with _guarded(task_id, device_id=device_id, lease_id=lease_id):
        package_data = await package.read()
        result = await worker.profile_single_task(package_data, task_id, op_name, settings)
        return sanitize_floats(result)

@app.get("/api/v1/docs/{doc_name}")
async def get_doc(
    doc_name: str,
):
    """
    获取 worker 当前环境中的文档内容。

    典型场景：
    - server/agent 本地没有 triton_ascend，但远端 worker 有
    - 需要基于远端真实运行时返回过滤后的 API 文档
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        content = await worker.get_doc(doc_name)
        return {
            "doc_name": doc_name,
            "content": content,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Get doc request failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/acquire_device")
async def acquire_device(
    task_id: str = Form(...),
    timeout: Optional[float] = Form(None),
):
    """
    Acquire a device from the device pool.
    Client should call this before generating verification scripts.

    ``timeout`` bounds the server-side wait for a free device. The client
    sends its own wait budget and uses a slightly larger HTTP read timeout, so
    the server gives up first and returns 503 — no orphaned waiter survives to
    grab a freed device for a client that already timed out.
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        wait_timeout = (
            float(timeout) if timeout is not None
            else worker_timing().acquire_timeout
        )
        # renewable: the lease carries a TTL and is kept alive by the
        # subsequent verify/profile requests (renew). If this client dies
        # before /release_device, the reaper reclaims the device.
        device_id, lease_id = await worker.device_pool.acquire_device(
            owner=task_id, timeout=wait_timeout, renewable=True
        )
        logger.info(f"[{task_id}] Acquired device {device_id} (lease {lease_id})")
        return {"device_id": device_id, "lease_id": lease_id}
    except TimeoutError as e:
        logger.warning(f"[{task_id}] No device free within {wait_timeout}s: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[{task_id}] Failed to acquire device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/release_device")
async def release_device(
    task_id: str = Form(...),
    device_id: int = Form(...),
    lease_id: int = Form(...)
):
    """
    Release a device back to the device pool.
    Client should call this after task completion.
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        await worker.device_pool.release_device(device_id, lease_id)
        logger.info(f"[{task_id}] Released device {device_id} (lease {lease_id})")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"[{task_id}] Failed to release device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def status():
    """Daemon liveness + identity. ``log_file`` echoes the daemon's stdout
    log path (set by worker_service via ``AKG_WORKER_LOG_FILE`` env) so
    akg_cli's remote probe tails the actual file instead of guessing."""
    log_file = os.environ.get("AKG_WORKER_LOG_FILE") or ""
    if worker is None:
        return {"status": "initializing", "log_file": log_file}

    backend, arch, devices = get_worker_config()
    return {
        "status": "ready",
        "backend": backend,
        "arch": arch,
        "devices": devices,
        "log_file": log_file,
    }


@app.get("/api/v1/health")
async def health():
    """非阻塞健康探活 —— 验"daemon 接 verify 时的请求路径还活着"，但
    不抢占设备：

      - 用 ``asyncio.Queue.get_nowait()`` 试取一次 device，能取就立刻
        放回；空队列（满载）当作 healthy（"忙不是坏"），不报 degraded
      - 整个 handler 按 worker.health_timeout 超时；超时仅当事件循环本身卡了

    /status 只验证 HTTP server 在线；/health 走一遍真实的 queue 操作
    路径，能抓出"event loop 卡住"或"queue 锁竞争"那类故障。**不会**
    阻塞等设备，所以满载 worker 不会被误判 degraded。"""
    import asyncio
    if worker is None:
        return {"status": "initializing", "healthy": False, "free": 0}

    timing = worker_timing()
    backend, arch, devices = get_worker_config()
    device_pool = worker.device_pool
    pool = device_pool.available_devices
    base = {
        "status": "ready",
        "backend": backend,
        "arch": arch,
        "devices": devices,
        "free": pool.qsize(),
        "healthy": False,
    }

    async def _probe():
        # Exercise the real Queue path (get + immediate put) to catch a
        # wedged event loop / queue. The pool's free set is a plain
        # asyncio.Queue, so put_nowait wakes any pending getter on its own —
        # no Condition to coordinate. A real acquirer racing this only waits
        # the instant between get and put.
        try:
            device_id = pool.get_nowait()
        except asyncio.QueueEmpty:
            # All devices busy — daemon is fine, just at capacity.
            return None
        pool.put_nowait(device_id)
        return device_id

    try:
        device_id = await asyncio.wait_for(_probe(), timeout=timing.health_timeout)
        base["healthy"] = True
        if device_id is not None:
            base["probed_device"] = device_id
        else:
            base["note"] = "all devices busy (healthy, just at capacity)"
        return base
    except asyncio.TimeoutError:
        base["error"] = (
            f"event loop unresponsive (>{timing.health_timeout}s) "
            "—— 事件循环可能阻塞"
        )
        logger.warning(
            "健康探活超时：event loop %s 秒内未响应",
            timing.health_timeout,
        )
        return base
    except Exception as e:
        base["error"] = f"健康探活异常：{type(e).__name__}: {e}"
        logger.warning(f"健康探活异常：{e}")
        return base


def start_server(host: Optional[str] = None, port: Optional[int] = None):
    """
    启动 AIKG Worker Service。
    
    Args:
        host: 监听地址。可从环境变量 WORKER_HOST 设置。
              - IPv4: "0.0.0.0" (所有接口), "127.0.0.1" (本地)
              - IPv6: "::" (所有接口，双栈), "::1" (本地)
              默认: "0.0.0.0"
        port: 监听端口。可从环境变量 WORKER_PORT 设置。
              默认: 9001
    """
    # 从环境变量读取配置，参数优先
    if host is None:
        host = os.environ.get("WORKER_HOST", "0.0.0.0")
    if port is None:
        port = int(os.environ.get("WORKER_PORT", "9001"))
    
    logger.info(f"Starting Worker Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
