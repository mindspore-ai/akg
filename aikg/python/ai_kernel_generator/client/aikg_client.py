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

    def _handle_response(self, resp: requests.Response):
        """处理响应，如果出错则尝试提取详细错误信息"""
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # 尝试从 Server 返回的 JSON 中获取 detail 字段
            error_detail = ""
            try:
                error_json = resp.json()
                if "detail" in error_json:
                    error_detail = f"\nServer Error Details:\n{error_json['detail']}"
            except Exception:
                # 如果不是 JSON 格式，尝试获取文本内容
                if resp.text:
                    error_detail = f"\nServer Error Text: {resp.text[:200]}"
            
            # 将详细信息附加到异常消息中
            if error_detail:
                # 修改异常对象的 args，使其打印时包含详情
                new_msg = f"{str(e)}{error_detail}"
                raise requests.exceptions.HTTPError(new_msg, response=resp) from e
            raise e

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
        self._handle_response(resp)
        return resp.json()["job_id"]

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """查询作业状态"""
        url = f"{self.server_url}/api/v1/jobs/{job_id}/status"
        resp = requests.get(url)
        self._handle_response(resp)
        return resp.json()

    def get_workers_status(self) -> list:
        """查询 Worker 状态"""
        url = f"{self.server_url}/api/v1/workers/status"
        resp = requests.get(url)
        self._handle_response(resp)
        return resp.json()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        取消作业
        
        Args:
            job_id: 要取消的作业ID
            
        Returns:
            Dict: {"cancelled": True/False, "reason": "..."}
        """
        url = f"{self.server_url}/api/v1/jobs/{job_id}/cancel"
        resp = requests.post(url)
        self._handle_response(resp)
        return resp.json()

    def wait_for_completion(self, job_id: str, interval: int = 2, timeout: int = 3600, 
                            cancel_on_interrupt: bool = True) -> Dict[str, Any]:
        """
        等待作业完成
        
        Args:
            job_id: 作业ID
            interval: 轮询间隔（秒）
            timeout: 超时时间（秒）
            cancel_on_interrupt: 当用户 Ctrl+C 时是否自动取消 Server 端的 job（默认 True）
            
        Returns:
            Dict: 作业状态
            
        Raises:
            TimeoutError: 超时
            KeyboardInterrupt: 用户中断（如果 cancel_on_interrupt=False）
        """
        start_time = time.time()
        try:
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")
                    
                status = self.get_job_status(job_id)
                state = status.get("status")
                
                if state in ["completed", "failed", "error", "cancelled"]:
                    return status
                    
                time.sleep(interval)
        except KeyboardInterrupt:
            if cancel_on_interrupt:
                print(f"\n⚠️  用户中断，正在取消 Server 端的 job {job_id}...")
                try:
                    result = self.cancel_job(job_id)
                    if result.get("cancelled"):
                        print(f"✅ Job {job_id} 已取消")
                    else:
                        print(f"⚠️  无法取消 job: {result.get('reason', 'unknown')}")
                except Exception as e:
                    print(f"❌ 取消 job 失败: {e}")
            raise  # 重新抛出，让调用方知道是用户中断
