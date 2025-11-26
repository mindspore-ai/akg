import asyncio
import uuid
import logging
from typing import Dict, Any, Optional
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.evolve import evolve
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.worker.manager import get_worker_manager

logger = logging.getLogger(__name__)

class ServerJobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
    
    async def submit_job(self, request_data: dict) -> str:
        backend = request_data.get("backend")
        arch = request_data.get("arch")

        if not backend:
            raise ValueError("backend is required when submitting a job.")
        if not arch:
            raise ValueError("arch is required when submitting a job.")

        worker_available = await get_worker_manager().has_worker(
            backend=backend,
            arch=arch
        )
        if not worker_available:
            raise RuntimeError(
                f"No available worker for backend={backend}, arch={arch}. "
                "Please register a worker before submitting the job."
            )

        job_id = str(uuid.uuid4())
        job_type = request_data.get("job_type", "single")
        
        # 初始化状态
        self.jobs[job_id] = {
            "status": "pending",
            "job_type": job_type,
            "op_name": request_data.get("op_name"),
            "submit_time": asyncio.get_event_loop().time(),
            "result": None,
            "error": None
        }
        
        # 根据类型启动不同的后台任务
        if job_type == "evolve":
            asyncio.create_task(self._run_evolve_job(job_id, request_data))
        else:
            asyncio.create_task(self._run_single_job(job_id, request_data))
            
        logger.info(f"Job submitted: {job_id} ({job_type})")
        return job_id

    async def _run_single_job(self, job_id: str, data: dict):
        self.jobs[job_id]["status"] = "running"
        try:
            # 加载配置
            # 假设Server运行环境包含必要的配置文件
            try:
                config = load_config(data.get("dsl"), backend=data.get("backend"))
            except Exception:
                config = {}

            # 创建 Core Task
            task = Task(
                op_name=data.get("op_name"),
                task_desc=data.get("task_desc"),
                task_id=job_id, # 复用 job_id 作为 task_id
                backend=data.get("backend"),
                arch=data.get("arch"),
                dsl=data.get("dsl"),
                framework=data.get("framework"),
                config=config,
                workflow=data.get("workflow", "coder_only_workflow")
            )
            
            # 支持注入初始代码 (用于测试或特定场景)
            init_task_info = None
            if data.get("init_code"):
                init_task_info = {"coder_code": data.get("init_code")}

            _, success, task_info = await task.run(init_task_info=init_task_info)
            
            self.jobs[job_id]["status"] = "completed" if success else "failed"
            self.jobs[job_id]["result"] = success
            # 可以在这里保存更多 task_info 信息
        except Exception as e:
            self._handle_error(job_id, e)

    async def _run_evolve_job(self, job_id: str, data: dict):
        self.jobs[job_id]["status"] = "running"
        try:
            config = load_config(data.get("dsl"), backend=data.get("backend"))
            
            task_pool = TaskPool(max_concurrency=data.get("parallel_num", 1))
            
            result = await evolve(
                op_name=data.get("op_name"),
                task_desc=data.get("task_desc"),
                dsl=data.get("dsl"),
                framework=data.get("framework"),
                backend=data.get("backend"),
                arch=data.get("arch"),
                config=config,
                task_pool=task_pool,
                max_rounds=data.get("max_rounds", 1),
                parallel_num=data.get("parallel_num", 1),
                num_islands=data.get("num_islands", 1),
                migration_interval=data.get("migration_interval", 0),
                elite_size=data.get("elite_size", 0),
                parent_selection_prob=data.get("parent_selection_prob", 0.5)
            )
            
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["result"] = result # evolve 返回的是 dict
        except Exception as e:
            self._handle_error(job_id, e)

    def _handle_error(self, job_id, e):
        self.jobs[job_id]["status"] = "error"
        self.jobs[job_id]["error"] = str(e)
        logger.error(f"Job error: {job_id}, error={e}", exc_info=True)

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)

_GLOBAL_JOB_MANAGER = ServerJobManager()

def get_job_manager():
    return _GLOBAL_JOB_MANAGER

