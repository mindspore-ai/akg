import os
import logging
from typing import Annotated, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import uvicorn

from ai_kernel_generator.core.worker.local_worker import LocalWorker
from ai_kernel_generator.core.async_pool.device_pool import DevicePool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global worker instance
worker: Optional[LocalWorker] = None

def get_worker_config():
    """Get worker configuration from environment variables."""
    backend = os.environ.get("WORKER_BACKEND", "cuda")
    arch = os.environ.get("WORKER_ARCH", "a100")
    devices_str = os.environ.get("WORKER_DEVICES", "0")
    
    try:
        devices = [int(d.strip()) for d in devices_str.split(",")]
    except ValueError:
        logger.warning(f"Invalid WORKER_DEVICES: {devices_str}, using [0]")
        devices = [0]
        
    return backend, arch, devices

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize worker resources on startup."""
    global worker
    backend, arch, devices = get_worker_config()
    
    logger.info(f"Initializing Worker Service: Backend={backend}, Arch={arch}, Devices={devices}")
    
    device_pool = DevicePool(devices)
    worker = LocalWorker(device_pool, backend=backend)
    
    yield
    
    # Cleanup if needed
    logger.info("Shutting down Worker Service")

app = FastAPI(title="AIKG Worker Service", lifespan=lifespan)

@app.post("/api/v1/verify")
async def verify(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: int = Form(300)
):
    """
    Execute verification task.
    
    Returns:
        - success: 验证是否成功
        - log: 执行日志
        - artifacts: 执行过程中生成的 JSON 文件内容
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    try:
        logger.info(f"[{task_id}] Received verification request for {op_name}")
        
        # Read package data
        package_data = await package.read()
        
        # Execute verification (now returns artifacts)
        success, log, artifacts = await worker.verify(package_data, task_id, op_name, timeout)
        
        return {
            "success": success,
            "log": log,
            "artifacts": artifacts
        }
        
    except Exception as e:
        logger.error(f"[{task_id}] Verification request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/profile")
async def profile(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    profile_settings: str = Form("{}")
):
    """
    Execute profiling task.
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")
        
    import json
    try:
        settings = json.loads(profile_settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for profile_settings")
        
    try:
        package_data = await package.read()
        result = await worker.profile(package_data, task_id, op_name, settings)
        return result
    except Exception as e:
        logger.error(f"[{task_id}] Profiling request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate_reference")
async def generate_reference(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: int = Form(120)
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
    
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    try:
        logger.info(f"[{task_id}] Received generate_reference request for {op_name}")
        
        package_data = await package.read()
        
        success, log, ref_bytes = await worker.generate_reference(
            package_data, task_id, op_name, timeout
        )
        
        if success:
            # 以 base64 编码返回二进制数据
            ref_data_b64 = base64.b64encode(ref_bytes).decode('utf-8')
            return {
                "success": True,
                "log": log,
                "reference_data": ref_data_b64
            }
        else:
            return {
                "success": False,
                "log": log,
                "reference_data": ""
            }
        
    except Exception as e:
        logger.error(f"[{task_id}] Generate reference request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/profile_single_task")
async def profile_single_task(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    profile_settings: str = Form("{}")
):
    """
    Execute single task profiling (only measure task_desc performance, no base comparison).
    
    单独测量某段代码的执行性能，不进行 base vs generation 对比。
    
    Returns:
        - time_us: 执行时间（微秒）
        - success: 是否成功
        - log: 执行日志
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    import json
    try:
        settings = json.loads(profile_settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for profile_settings")
    
    try:
        logger.info(f"[{task_id}] Received profile_single_task request for {op_name}")
        
        package_data = await package.read()
        result = await worker.profile_single_task(package_data, task_id, op_name, settings)
        return result
        
    except Exception as e:
        logger.error(f"[{task_id}] Profile single task request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/acquire_device")
async def acquire_device(
    task_id: str = Form(...)
):
    """
    Acquire a device from the device pool.
    Client should call this before generating verification scripts.
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    try:
        device_id = await worker.device_pool.acquire_device()
        logger.info(f"[{task_id}] Acquired device {device_id}")
        return {"device_id": device_id}
    except Exception as e:
        logger.error(f"[{task_id}] Failed to acquire device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/release_device")
async def release_device(
    task_id: str = Form(...),
    device_id: int = Form(...)
):
    """
    Release a device back to the device pool.
    Client should call this after task completion.
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    try:
        await worker.device_pool.release_device(device_id)
        logger.info(f"[{task_id}] Released device {device_id}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"[{task_id}] Failed to release device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def status():
    """Get worker status."""
    if worker is None:
        return {"status": "initializing"}
    
    backend, arch, devices = get_worker_config()
    return {
        "status": "ready",
        "backend": backend,
        "arch": arch,
        "devices": devices,
        # "available_devices": worker.device_pool.qsize() # DevicePool uses Queue
    }


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
