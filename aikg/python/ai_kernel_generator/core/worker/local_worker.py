import asyncio
import os
import shutil
import subprocess
import tempfile
import tarfile
import logging
import sys
import io
import json
from typing import Tuple, Dict, Any, Union
from contextlib import ExitStack

from .interface import WorkerInterface
from ..async_pool.device_pool import DevicePool
from ..verifier.profiler_utils import (
    run_profile_scripts_and_collect_results,
    run_msprof,
    analyze_prof_data,
    run_nsys,
    analyze_nsys_data
)

logger = logging.getLogger(__name__)

class LocalWorker(WorkerInterface):
    """
    Local implementation of WorkerInterface.
    Executes verification tasks in a local subprocess, managing devices via DevicePool.
    """
    def __init__(self, device_pool: DevicePool, backend: str = "cuda"):
        self.device_pool = device_pool
        self.backend = backend

    async def verify(self, package_data: Union[bytes, str], task_id: str, op_name: str, timeout: int = 300) -> Tuple[bool, str]:
        """
        Execute verification task locally.
        
        注意：device 的管理（acquire/release）由调用方负责
        这个方法只负责执行已经生成好的脚本（脚本中已包含正确的 device_id）
        
        Args:
            package_data: 验证包数据（bytes 或目录路径）
            task_id: 任务ID
            op_name: 算子名称
            timeout: 超时时间
        """
        try:
            with ExitStack() as stack:
                if isinstance(package_data, (bytes, bytearray)):
                    temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
                    tar_path = os.path.join(temp_dir, "package.tar")
                    with open(tar_path, "wb") as f:
                        f.write(package_data)
                    extract_dir = os.path.join(temp_dir, "extract")
                    os.makedirs(extract_dir, exist_ok=True)
                    try:
                        with tarfile.open(tar_path, 'r') as tar_ref:
                            tar_ref.extractall(extract_dir)
                    except Exception as e:
                        return False, f"Failed to extract package: {e}"
                elif isinstance(package_data, str):
                    extract_dir = package_data
                else:
                    return False, "Unsupported package_data type for LocalWorker.verify"

                script_name = f"verify_{op_name}.py"
                script_path = os.path.join(extract_dir, script_name)
                if not os.path.exists(script_path):
                    return False, f"Verification script {script_name} not found."

                # 注意：脚本中的 device_id 已在生成时设置正确
                # worker 只负责执行脚本，不管理设备分配
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'

                python_exe = sys.executable
                cmd = [python_exe, script_name]
                logger.info(f"[{task_id}] Running verification for {op_name}")
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=extract_dir,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                    returncode = process.returncode
                    
                    output_log = stdout.decode(errors='replace') + "\n" + stderr.decode(errors='replace')
                    success = (returncode == 0)
                    
                    if success:
                        logger.info(f"[{task_id}] Verification passed.")
                    else:
                        logger.error(f"[{task_id}] Verification failed with log:\n{output_log}")
                        
                    return success, output_log
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    logger.error(f"[{task_id}] Verification timed out.")
                    return False, f"Verification timed out after {timeout} seconds."

        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker verification failed: {e}", exc_info=True)
            return False, str(e)

    async def profile(self, package_data: bytes, task_id: str, op_name: str, profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute profiling task locally.
        
        注意：device 的管理（acquire/release）由调用方负责
        这个方法只负责执行已经生成好的 profile 脚本
        """
        try:
            # 2. Create temp directory and extract
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract package
                tar_path = os.path.join(temp_dir, "package.tar")
                with open(tar_path, "wb") as f:
                    f.write(package_data)
                
                extract_dir = os.path.join(temp_dir, "extract")
                os.makedirs(extract_dir, exist_ok=True)
                
                try:
                    with tarfile.open(tar_path, 'r') as tar_ref:
                        tar_ref.extractall(extract_dir)
                except Exception as e:
                    return {'gen_time': float('inf'), 'base_time': 0.0, 'speedup': 0.0, 'error': str(e)}
                
                # 注意：profile 脚本中的 device_id 应该在生成时就已经设置正确
                # （与 verify 类似，通过预先获取设备ID）
                # 3. Get settings
                backend = profile_settings.get('backend', self.backend)
                dsl = profile_settings.get('dsl', '')
                run_times = profile_settings.get('run_times', 50)
                warmup_times = profile_settings.get('warmup_times', 5)
                
                # 4. Execute profiling based on backend/dsl
                try:
                    if "triton_cuda" in dsl or "triton_ascend" in dsl or backend == "cpu":
                        # Triton/CPU: run profile scripts directly (in sync context)
                        loop = asyncio.get_running_loop()
                        base_time, gen_time = await loop.run_in_executor(
                            None,
                            run_profile_scripts_and_collect_results,
                            extract_dir, op_name, task_id
                        )
                        logger.info(f"[{task_id}] Profile results: base={base_time:.2f} us, gen={gen_time:.2f} us")
                    elif backend == "ascend":
                        # Ascend: use msprof
                        loop = asyncio.get_running_loop()
                        base_time, gen_time = await loop.run_in_executor(
                            None,
                            self._run_msprof_profiling,
                            extract_dir, op_name, task_id, warmup_times, run_times
                        )
                    elif backend == "cuda":
                        # CUDA: use nsys
                        loop = asyncio.get_running_loop()
                        base_time, gen_time = await loop.run_in_executor(
                            None,
                            self._run_nsys_profiling,
                            extract_dir, op_name, task_id, warmup_times, run_times
                        )
                    else:
                        logger.warning(f"[{task_id}] Unsupported backend for profiling: {backend}")
                        return {'gen_time': float('inf'), 'base_time': 0.0, 'speedup': 0.0}
                    
                    # 5. Calculate speedup
                    speedup = base_time / gen_time if gen_time > 0 else 0.0
                    
                    return {
                        'gen_time': gen_time,
                        'base_time': base_time,
                        'speedup': speedup
                    }
                    
                except Exception as e:
                    logger.error(f"[{task_id}] Profiling execution failed: {e}", exc_info=True)
                    return {'gen_time': float('inf'), 'base_time': 0.0, 'speedup': 0.0, 'error': str(e)}
        
        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker profiling failed: {e}", exc_info=True)
            return {'gen_time': float('inf'), 'base_time': 0.0, 'speedup': 0.0, 'error': str(e)}
    
    def _run_msprof_profiling(self, extract_dir: str, op_name: str, task_id: str, warmup_times: int, run_times: int) -> Tuple[float, float]:
        """Run msprof profiling for Ascend backend (synchronous)"""
        try:
            # Run msprof for base script
            base_script = os.path.join(extract_dir, f"profile_{op_name}_base.py")
            success, error, base_prof_path = run_msprof(base_script, op_name, task_id)
            if not success or not base_prof_path:
                logger.error(f"[{task_id}] Base msprof failed: {error}")
                return float('inf'), float('inf')
            
            # Run msprof for generation script
            gen_script = os.path.join(extract_dir, f"profile_{op_name}_generation.py")
            success, error, gen_prof_path = run_msprof(gen_script, op_name, task_id)
            if not success or not gen_prof_path:
                logger.error(f"[{task_id}] Generation msprof failed: {error}")
                return float('inf'), float('inf')
            
            # Analyze prof data
            success, error, base_time = analyze_prof_data(base_prof_path, warmup_times, run_times, op_name, task_id)
            if not success:
                logger.error(f"[{task_id}] Base prof analysis failed: {error}")
                return float('inf'), float('inf')
            
            success, error, gen_time = analyze_prof_data(gen_prof_path, warmup_times, run_times, op_name, task_id)
            if not success:
                logger.error(f"[{task_id}] Generation prof analysis failed: {error}")
                return float('inf'), float('inf')
            
            return base_time, gen_time
            
        except Exception as e:
            logger.error(f"[{task_id}] msprof profiling failed: {e}", exc_info=True)
            return float('inf'), float('inf')
    
    def _run_nsys_profiling(self, extract_dir: str, op_name: str, task_id: str, warmup_times: int, run_times: int) -> Tuple[float, float]:
        """Run nsys profiling for CUDA backend (synchronous)"""
        try:
            # Run nsys for base script
            base_script = os.path.join(extract_dir, f"profile_{op_name}_base.py")
            success, error, base_rep_path = run_nsys(base_script, op_name, task_id)
            if not success or not base_rep_path:
                logger.error(f"[{task_id}] Base nsys failed: {error}")
                return float('inf'), float('inf')
            
            # Run nsys for generation script
            gen_script = os.path.join(extract_dir, f"profile_{op_name}_generation.py")
            success, error, gen_rep_path = run_nsys(gen_script, op_name, task_id)
            if not success or not gen_rep_path:
                logger.error(f"[{task_id}] Generation nsys failed: {error}")
                return float('inf'), float('inf')
            
            # Analyze nsys data
            success, error, base_time = analyze_nsys_data(base_rep_path, warmup_times, run_times, "base", op_name, task_id)
            if not success:
                logger.error(f"[{task_id}] Base nsys analysis failed: {error}")
                return float('inf'), float('inf')
            
            success, error, gen_time = analyze_nsys_data(gen_rep_path, warmup_times, run_times, "generation", op_name, task_id)
            if not success:
                logger.error(f"[{task_id}] Generation nsys analysis failed: {error}")
                return float('inf'), float('inf')
            
            return base_time, gen_time
            
        except Exception as e:
            logger.error(f"[{task_id}] nsys profiling failed: {e}", exc_info=True)
            return float('inf'), float('inf')
