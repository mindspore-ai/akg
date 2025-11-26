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

@app.post("/api/v1/workers/register")
async def register_worker(req: WorkerRegisterRequest):
    """Worker 注册接口"""
    logger.info(f"Registering worker: {req.url} ({req.backend}/{req.arch})")
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

def start_server(host="0.0.0.0", port=8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
