import httpx
import logging
import io
import json
from typing import Tuple, Dict, Any

from .interface import WorkerInterface

logger = logging.getLogger(__name__)

class RemoteWorker(WorkerInterface):
    """
    Remote implementation of WorkerInterface.
    Delegates verification tasks to a remote VerificationService via HTTP.
    
    RemoteWorker 通过 HTTP API 管理远程服务器的设备池：
    - acquire_device(): 向远程服务器请求分配设备
    - release_device(): 归还设备给远程服务器
    - verify()/profile(): 发送任务到远程服务器执行
    """
    def __init__(self, worker_url: str):
        self.worker_url = worker_url.rstrip('/')
    
    async def acquire_device(self, task_id: str = "unknown") -> int:
        """
        从远程服务器获取一个可用设备。
        
        Args:
            task_id: 任务ID（用于日志）
        
        Returns:
            int: 设备ID
        """
        acquire_url = f"{self.worker_url}/api/v1/acquire_device"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                data = {'task_id': task_id}
                response = await client.post(acquire_url, data=data)
                response.raise_for_status()
                
                result = response.json()
                device_id = result.get('device_id')
                logger.info(f"[{task_id}] Acquired remote device {device_id}")
                return device_id
        except Exception as e:
            logger.error(f"[{task_id}] Failed to acquire remote device: {e}")
            raise RuntimeError(f"Failed to acquire remote device: {e}")
    
    async def release_device(self, device_id: int, task_id: str = "unknown"):
        """
        归还设备给远程服务器。
        
        Args:
            device_id: 设备ID
            task_id: 任务ID（用于日志）
        """
        release_url = f"{self.worker_url}/api/v1/release_device"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                data = {'task_id': task_id, 'device_id': device_id}
                response = await client.post(release_url, data=data)
                response.raise_for_status()
                logger.info(f"[{task_id}] Released remote device {device_id}")
        except Exception as e:
            logger.error(f"[{task_id}] Failed to release remote device: {e}")

    async def verify(self, package_data: bytes, task_id: str, op_name: str, timeout: int = 300) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Send verification task to remote worker.
        
        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, log, artifacts)
        """
        verify_url = f"{self.worker_url}/api/v1/verify"
        
        try:
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                # Prepare multipart form data - use .tar extension
                files = {'package': ('package.tar', package_data, 'application/x-tar')}
                data = {
                    'task_id': task_id, 
                    'op_name': op_name,
                    'timeout': str(timeout)
                }
                
                logger.info(f"[{task_id}] Sending verification request to {verify_url}")
                
                response = await client.post(verify_url, files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                success = result.get('success', False)
                log = result.get('log', '')
                artifacts = result.get('artifacts', {})
                
                if artifacts:
                    logger.info(f"[{task_id}] Received {len(artifacts)} artifact files from remote worker")
                
                return success, log, artifacts
                
        except httpx.RequestError as e:
            logger.error(f"[{task_id}] Network error communicating with worker: {e}")
            return False, f"Network error: {e}", {}
        except httpx.HTTPStatusError as e:
            logger.error(f"[{task_id}] Worker returned error status: {e.response.status_code} - {e.response.text}")
            return False, f"Worker error: {e.response.status_code} - {e.response.text}", {}
        except Exception as e:
            logger.error(f"[{task_id}] Remote verification failed: {e}")
            return False, f"Remote verification failed: {e}", {}

    async def profile(self, package_data: bytes, task_id: str, op_name: str, profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send profiling task to remote worker.
        
        Returns:
            Dict[str, Any]: 包含 gen_time, base_time, speedup, artifacts 等字段
        """
        profile_url = f"{self.worker_url}/api/v1/profile"
        
        try:
            async with httpx.AsyncClient(timeout=300) as client: # Default timeout for profile
                files = {'package': ('package.tar', package_data, 'application/x-tar')}
                data = {
                    'task_id': task_id, 
                    'op_name': op_name,
                    'profile_settings': json.dumps(profile_settings)
                }
                
                logger.info(f"[{task_id}] Sending profiling request to {profile_url}")
                
                response = await client.post(profile_url, files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                artifacts = result.get('artifacts', {})
                if artifacts:
                    logger.info(f"[{task_id}] Received {len(artifacts)} artifact files from remote worker")
                
                return result
                
        except Exception as e:
            logger.error(f"[{task_id}] Remote profiling failed: {e}")
            return {'artifacts': {}}
