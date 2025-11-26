import requests
import time
from typing import Optional, Dict, Any

class AIKGClient:
    """
    AIKG Client SDK
    用于与 AIKG Server 交互，提交作业和查询状态。
    """
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def submit_job(self, 
                   op_name: str, 
                   task_desc: str, 
                   job_type: str = "single",
                   backend: str = "cuda", 
                   arch: str = "a100", 
                   dsl: str = "triton_cuda",
                   framework: str = "torch",
                   **kwargs) -> str:
        """
        提交作业
        
        Args:
            op_name: 算子名称
            task_desc: 算子描述
            job_type: "single" or "evolve"
            kwargs: 其他参数 (如 evolve 的 max_rounds, parallel_num 等)
        
        Returns:
            str: job_id
        """
        url = f"{self.server_url}/api/v1/jobs/submit"
        data = {
            "op_name": op_name,
            "task_desc": task_desc,
            "job_type": job_type,
            "backend": backend,
            "arch": arch,
            "dsl": dsl,
            "framework": framework,
            **kwargs
        }
        resp = requests.post(url, json=data)
        resp.raise_for_status()
        return resp.json()["job_id"]

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """查询作业状态"""
        url = f"{self.server_url}/api/v1/jobs/{job_id}/status"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def get_workers_status(self) -> list:
        """查询 Worker 状态"""
        url = f"{self.server_url}/api/v1/workers/status"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def wait_for_completion(self, job_id: str, interval: int = 2, timeout: int = 3600) -> Dict[str, Any]:
        """等待作业完成"""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")
                
            status = self.get_job_status(job_id)
            state = status.get("status")
            
            if state in ["completed", "failed", "error"]:
                return status
                
            time.sleep(interval)
