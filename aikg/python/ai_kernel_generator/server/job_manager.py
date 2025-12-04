import asyncio
import uuid
import logging
from typing import Dict, Any, Optional
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.evolve import evolve
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.worker.manager import get_worker_manager
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier

logger = logging.getLogger(__name__)

# task_desc 格式示例（KernelBench 格式）
TASK_DESC_EXAMPLE = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
'''.strip()

def _get_task_desc_format_hint() -> str:
    """返回 task_desc 格式提示信息"""
    return f"""
task_desc must follow the KernelBench format with the following required components:

1. class Model: A model class with __init__ and forward methods
2. def get_inputs(): A function that returns a list of input tensors for the model
3. def get_init_inputs(): A function that returns initialization arguments for the Model class

Example (PyTorch):
{TASK_DESC_EXAMPLE}

Note: The example above uses PyTorch. For other frameworks like MindSpore or NumPy, 
please refer to the corresponding examples in the aikg/examples/ directory.
"""

class ServerJobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
    
    async def submit_job(self, request_data: dict) -> str:
        backend = request_data.get("backend")
        arch = request_data.get("arch")
        task_desc = request_data.get("task_desc", "")

        if not backend:
            raise ValueError("backend is required when submitting a job.")
        if not arch:
            raise ValueError("arch is required when submitting a job.")

        # 静态检查 task_desc
        if task_desc:
            # 临时创建一个 verifier 实例用于检查 (不需要 worker)
            verifier = KernelVerifier(
                op_name="check", 
                framework_code="", 
                config={"log_dir": "/tmp"},
                backend=backend,
                arch=arch,
                dsl=request_data.get("dsl", "triton") # 临时用
            )
            valid, error = verifier.check_task_desc_static(task_desc)
            if not valid:
                hint = _get_task_desc_format_hint()
                raise ValueError(f"Task description static check failed: {error}\n\n{hint}")
        else:
            hint = _get_task_desc_format_hint()
            raise ValueError(f"task_desc is required when submitting a job.\n\n{hint}")

        worker_manager = get_worker_manager()
        
        # 检查目标 backend 的 Worker 是否可用
        worker_available = await worker_manager.has_worker(
            backend=backend,
            arch=arch
        )
        if not worker_available:
            # 获取当前所有已注册的 Worker 状态
            all_workers = await worker_manager.get_status()
            
            error_msg = f"No available worker found for backend='{backend}', arch='{arch}'.\n"
            if not all_workers:
                error_msg += "Reason: No workers are currently registered to the server."
            else:
                worker_list_str = "\n".join([
                    f"- backend='{w['backend']}', arch='{w['arch']}', capacity={w['capacity']}, tags={w['tags']}"
                    for w in all_workers
                ])
                error_msg += f"Reason: Registered workers do not match the requirements.\nCurrently registered workers:\n{worker_list_str}"
            
            error_msg += "\nPlease register a compatible worker before submitting the job."
            
            raise RuntimeError(error_msg)
        
        # 检查是否是 CUDA-to-Ascend 转换场景
        source_backend = request_data.get("source_backend")
        source_arch = request_data.get("source_arch")
        
        if source_backend and source_backend != backend:
            # CUDA-to-Ascend 转换场景，需要检查 source_backend 的 Worker 是否可用
            source_worker_available = await worker_manager.has_worker(
                backend=source_backend,
                arch=source_arch
            )
            if not source_worker_available:
                all_workers = await worker_manager.get_status()
                error_msg = f"No available worker found for source_backend='{source_backend}', source_arch='{source_arch}'.\n"
                if all_workers:
                    worker_list_str = "\n".join([
                        f"- backend='{w['backend']}', arch='{w['arch']}', capacity={w['capacity']}, tags={w['tags']}"
                        for w in all_workers
                    ])
                    error_msg += f"Currently registered workers:\n{worker_list_str}"
                error_msg += "\nPlease register a source backend worker for CUDA-to-Ascend conversion."
                raise RuntimeError(error_msg)

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

    async def _check_task_desc_runtime_wrapper(self, job_id: str, data: dict, config: dict) -> bool:
        """
        在任务开始前执行运行时检查
        
        支持跨后端转换场景：
        - 当 source_backend 与 backend 不同时，在源 Worker 上生成参考数据
        - 将参考数据存入 config['reference_data']，供目标 Worker 验证时使用
        """
        task_desc = data.get("task_desc", "")
        if not task_desc:
            hint = _get_task_desc_format_hint()
            raise ValueError(f"task_desc is required but was not provided.\n\n{hint}")
            
        backend = data.get("backend")
        arch = data.get("arch")
        source_backend = data.get("source_backend")
        source_arch = data.get("source_arch")
        
        worker_manager = get_worker_manager()
        
        # 检查是否需要生成参考数据（只要有 source_backend 就需要）
        need_reference_data = (source_backend is not None and source_backend != backend)
        
        if need_reference_data:
            # ========== 阶段1: 在源 Worker 上生成参考数据 ==========
            logger.info(f"[{job_id}] Cross-backend conversion detected (source={source_backend} -> target={backend}). Generating reference data...")
            
            source_worker = await worker_manager.select(backend=source_backend, arch=source_arch)
            if not source_worker:
                raise RuntimeError(f"No available source worker found for reference generation (backend={source_backend}, arch={source_arch})")
            
            try:
                # 根据 source_backend 决定 dsl
                source_dsl = data.get("source_dsl")
                if not source_dsl:
                    # 默认根据 source_backend 推断
                    if source_backend == "cuda":
                        source_dsl = "triton_cuda"
                    elif source_backend == "ascend":
                        source_dsl = "triton_ascend"
                    else:
                        source_dsl = "triton"
                
                # 创建 verifier 用于生成参考数据
                verifier = KernelVerifier(
                    op_name=data.get("op_name"),
                    framework_code=task_desc,
                    task_id=job_id,
                    framework=data.get("framework"),
                    dsl=source_dsl,
                    backend=source_backend,
                    arch=source_arch,
                    config=config,
                    worker=source_worker
                )
                
                # 生成参考数据
                success, log, ref_bytes = await verifier.generate_reference_data(task_desc, timeout=120)
                
                if not success:
                    raise RuntimeError(f"Reference data generation failed on source worker:\n{log}")
                
                # 将参考数据存入 config
                config['use_reference_data'] = True
                config['reference_data'] = ref_bytes
                logger.info(f"[{job_id}] Reference data generated successfully ({len(ref_bytes)} bytes)")
                
            finally:
                await worker_manager.release(source_worker)
            
            # ========== 阶段2: 不再需要在目标 Worker 上执行运行时检查 ==========
            # 因为我们已经在源 Worker 上验证过 task_desc 可以正常运行
            logger.info(f"[{job_id}] Skipping target runtime check (reference data already generated)")
            return True
        
        else:
            # ========== 普通场景: 在目标 Worker 上执行运行时检查 ==========
            logger.info(f"[{job_id}] Starting runtime check for task description...")
            
            worker = await worker_manager.select(backend=backend, arch=arch)
            if not worker:
                raise RuntimeError(f"No available worker found for runtime check (backend={backend}, arch={arch})")
                
            try:
                # 创建 verifier
                verifier = KernelVerifier(
                    op_name=data.get("op_name"),
                    framework_code=task_desc,
                    task_id=job_id,
                    framework=data.get("framework"),
                    dsl=data.get("dsl"),
                    backend=backend,
                    arch=arch,
                    config=config,
                    worker=worker
                )
                
                # 执行检查
                valid, error = await verifier.check_task_desc_runtime(task_desc, timeout=60)
                
                if not valid:
                    hint = _get_task_desc_format_hint()
                    raise RuntimeError(f"Task description runtime check failed: {error}\n\n{hint}")
                    
                logger.info(f"[{job_id}] Task description runtime check passed.")
                return True
                
            finally:
                # 释放 worker
                await worker_manager.release(worker)

    async def _run_single_job(self, job_id: str, data: dict):
        self.jobs[job_id]["status"] = "running"
        try:
            # 加载配置
            # 假设Server运行环境包含必要的配置文件
            try:
                config = load_config(data.get("dsl"), backend=data.get("backend"))
            except Exception:
                config = {}
            
            # 执行运行时检查
            await self._check_task_desc_runtime_wrapper(job_id, data, config)

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
                workflow=data.get("workflow", "coder_only_workflow"),
                source_backend=data.get("source_backend"),  # 跨后端转换的源后端
                source_arch=data.get("source_arch")         # 跨后端转换的源架构
            )
            
            # 支持注入初始代码 (用于测试或特定场景)
            init_task_info = None
            if data.get("init_code"):
                init_task_info = {"coder_code": data.get("init_code")}

            _, success, task_info = await task.run(init_task_info=init_task_info)
            
            self.jobs[job_id]["status"] = "completed" if success else "failed"
            self.jobs[job_id]["result"] = {
                "success": success,
                "code": task_info.get("coder_code", ""),
            }
        except Exception as e:
            self._handle_error(job_id, e)

    async def _run_evolve_job(self, job_id: str, data: dict):
        self.jobs[job_id]["status"] = "running"
        try:
            config = load_config(data.get("dsl"), backend=data.get("backend"))
            
            # 执行运行时检查
            await self._check_task_desc_runtime_wrapper(job_id, data, config)
            
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
            
            # 提取最优结果
            best_result = {
                "success": result.get("successful_tasks", 0) > 0,
                "code": "",
                "profile": {},
                "op_name": result.get("op_name"),
                "full_result": result  # 保留完整结果以备不时之需
            }

            best_impls = result.get("best_implementations", [])
            if best_impls:
                best_kernel = best_impls[0]
                best_result["code"] = best_kernel.get("impl_code", "")
                best_result["profile"] = best_kernel.get("profile", {})
            
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["result"] = best_result
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

