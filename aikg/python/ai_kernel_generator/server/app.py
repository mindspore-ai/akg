import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from ai_kernel_generator.server.job_manager import get_job_manager
from ai_kernel_generator.core.worker.manager import get_worker_manager
from ai_kernel_generator.core.worker.remote_worker import RemoteWorker

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AIKG Server")

class JobSubmitRequest(BaseModel):
    op_name: str
    task_desc: str
    job_type: str = "single" # single or evolve
    backend: str = "cuda"
    arch: str = "a100"
    dsl: str = "triton"
    framework: str = "torch"
    workflow: Optional[str] = "coder_only_workflow"
    
    # Cross-backend conversion params
    # 当 source_backend 与 backend 不同时，表示跨平台转换场景
    # 例如: source_backend=cuda, backend=ascend 表示 Triton-CUDA 到 Triton-Ascend 转换
    source_backend: Optional[str] = None  # 源后端（如 cuda）
    source_arch: Optional[str] = None     # 源架构（如 a100）
    source_dsl: Optional[str] = None      # 源 DSL（如 triton_cuda），不指定则根据 source_backend 推断
    
    # Evolve params
    max_rounds: int = 1
    parallel_num: int = 1
    num_islands: int = 1
    
    # Test params
    init_code: Optional[str] = None

class WorkerRegisterRequest(BaseModel):
    url: str
    backend: str
    arch: str
    capacity: int = 1
    tags: List[str] = []

@app.post("/api/v1/jobs/submit")
async def submit_job(req: JobSubmitRequest):
    """提交作业"""
    logger.info(f"Received job submission: {req.op_name} ({req.job_type})")
    try:
        job_id = await get_job_manager().submit_job(req.dict())
        return {"job_id": job_id}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=409, detail=str(re))

@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """查询作业状态"""
    status = get_job_manager().get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

def _is_loopback_url(url: str) -> bool:
    """
    检查 URL 是否为本地回环地址。
    支持 IPv4 (localhost, 127.0.0.1) 和 IPv6 ([::1])。
    """
    loopback_patterns = ["localhost", "127.0.0.1", "[::1]"]
    return any(pattern in url for pattern in loopback_patterns)

@app.post("/api/v1/workers/register")
async def register_worker(req: WorkerRegisterRequest):
    """Worker 注册接口"""
    logger.info(f"Registering worker: {req.url} ({req.backend}/{req.arch})")
    
    # 简单的 URL 检查提示 (支持 IPv4 和 IPv6 loopback)
    if _is_loopback_url(req.url):
        logger.warning(f"Worker registered with loopback URL: {req.url}. "
                       "Ensure the Server can access this URL (e.g. they are on the same host).")

    worker = RemoteWorker(req.url)
    await get_worker_manager().register(
        worker, 
        backend=req.backend, 
        arch=req.arch, 
        capacity=req.capacity, 
        tags=set(req.tags)
    )
    return {"status": "registered"}

@app.get("/api/v1/workers/status")
async def get_workers_status():
    """查询所有 Worker 状态"""
    return await get_worker_manager().get_status()


def start_server(host: Optional[str] = None, port: Optional[int] = None):
    """
    启动 AIKG Server。
    
    Args:
        host: 监听地址。可从环境变量 AIKG_SERVER_HOST 设置。
              - IPv4: "0.0.0.0" (所有接口), "127.0.0.1" (本地)
              - IPv6: "::" (所有接口，双栈), "::1" (本地)
              默认: "0.0.0.0"
        port: 监听端口。可从环境变量 AIKG_SERVER_PORT 设置。
              默认: 8000
    """
    import uvicorn
    
    # 从环境变量读取配置，参数优先
    if host is None:
        host = os.environ.get("AIKG_SERVER_HOST", "0.0.0.0")
    if port is None:
        port = int(os.environ.get("AIKG_SERVER_PORT", "8000"))
    
    logger.info(f"Starting AIKG Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
