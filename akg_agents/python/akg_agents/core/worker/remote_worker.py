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
    
    async def acquire_device(self, task_id: str = "unknown", timeout: float = None) -> int:
        """
        从远程服务器获取一个可用设备。
        
        Args:
            task_id: 任务ID（用于日志）
            timeout: 请求超时时间（秒）。默认为 None（无限等待），
                     因为在高并发 evolve 场景下，设备可能被长时间占用，
                     需要等待直到有设备可用。
                     取消等待：由上层 asyncio task 的 cancel 机制处理。
        
        Returns:
            int: 设备ID
        """
        acquire_url = f"{self.worker_url}/api/v1/acquire_device"
        
        try:
            # 使用 timeout=None (无限等待) 作为默认值，因为在并发高时
            # 设备可能被长时间占用，我们需要等待直到有设备可用。
            # httpx 默认 timeout 是 5s，之前硬编码是 10.0s，这在 evolve 流程中是不够的。
            async with httpx.AsyncClient(timeout=timeout) as client:
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
            error_msg = f"Network error communicating with worker at {self.worker_url}: {e}. Please check if the worker service is running and accessible."
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, {}
        except httpx.HTTPStatusError as e:
            error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, {}
        except Exception as e:
            error_msg = f"Remote verification failed: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, {}

    async def profile(self, package_data: bytes, task_id: str, op_name: str, profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send profiling task to remote worker.
        
        Returns:
            Dict[str, Any]: 包含 gen_time, base_time, speedup, artifacts 等字段
        """
        profile_url = f"{self.worker_url}/api/v1/profile"
        
        try:
            # 获取 timeout 设置，默认为 300s (5分钟)
            # 加上 10s 缓冲时间用于网络传输等
            timeout = profile_settings.get('timeout', 300)
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
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

    async def profile_single_task(self, package_data: bytes, task_id: str, op_name: str, 
                                   profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send single task profiling request to remote worker.
        
        单独测量某段代码的执行性能，不进行 base vs generation 对比。
        
        Returns:
            Dict[str, Any]: 包含 time_us, success, log 等字段
        """
        profile_url = f"{self.worker_url}/api/v1/profile_single_task"
        
        try:
            timeout = profile_settings.get('timeout', 300)
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                files = {'package': ('package.tar', package_data, 'application/x-tar')}
                data = {
                    'task_id': task_id, 
                    'op_name': op_name,
                    'profile_settings': json.dumps(profile_settings)
                }
                
                logger.info(f"[{task_id}] Sending profile_single_task request to {profile_url}")
                
                response = await client.post(profile_url, files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                return result
                
        except httpx.RequestError as e:
            error_msg = f"Network error communicating with worker at {self.worker_url}: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return {'time_us': float('inf'), 'success': False, 'log': error_msg}
        except httpx.HTTPStatusError as e:
            error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
            logger.error(f"[{task_id}] {error_msg}")
            return {'time_us': float('inf'), 'success': False, 'log': error_msg}
        except Exception as e:
            error_msg = f"Remote profile_single_task failed: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return {'time_us': float('inf'), 'success': False, 'log': error_msg}

    async def generate_reference(self, package_data: bytes, task_id: str, op_name: str, timeout: int = 120) -> Tuple[bool, str, bytes]:
        """
        Send reference generation task to remote worker.
        
        用于 CUDA-to-Ascend 转换场景：在远程 GPU Worker 上执行 Triton-CUDA 代码，
        生成参考数据（.pt 文件）并返回其二进制内容。
        
        Args:
            package_data: 验证包数据（TAR bytes）
            task_id: 任务ID
            op_name: 算子名称
            timeout: 超时时间
            
        Returns:
            Tuple[bool, str, bytes]: (success, log, reference_data_bytes)
        """
        import base64
        
        generate_ref_url = f"{self.worker_url}/api/v1/generate_reference"
        
        try:
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                files = {'package': ('package.tar', package_data, 'application/x-tar')}
                data = {
                    'task_id': task_id,
                    'op_name': op_name,
                    'timeout': str(timeout)
                }
                
                logger.info(f"[{task_id}] Sending generate_reference request to {generate_ref_url}")
                
                response = await client.post(generate_ref_url, files=files, data=data)
                response.raise_for_status()
                
                result = response.json()
                success = result.get('success', False)
                log = result.get('log', '')
                
                if success:
                    # reference_data 以 base64 编码传输
                    ref_data_b64 = result.get('reference_data', '')
                    if ref_data_b64:
                        ref_bytes = base64.b64decode(ref_data_b64)
                        logger.info(f"[{task_id}] Received reference data: {len(ref_bytes)} bytes")
                        return True, log, ref_bytes
                    else:
                        return False, f"No reference data in response:\n{log}", b''
                else:
                    return False, log, b''
                
        except httpx.RequestError as e:
            error_msg = f"Network error communicating with worker at {self.worker_url}: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, b''
        except httpx.HTTPStatusError as e:
            error_msg = f"Worker returned error status: {e.response.status_code} - {e.response.text}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, b''
        except Exception as e:
            error_msg = f"Remote generate_reference failed: {e}"
            logger.error(f"[{task_id}] {error_msg}")
            return False, error_msg, b''
