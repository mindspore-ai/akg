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
from typing import Tuple, Dict, Any, Union, Optional
from contextlib import ExitStack

from .interface import (
    WorkerInterface,
    DEFAULT_EVAL_TIMEOUT_S,
    DEFAULT_GEN_REF_TIMEOUT_S,
    DEFAULT_WARMUP_TIMES,
    DEFAULT_RUN_TIMES,
)
from ..async_pool.device_pool import DevicePool
from akg_agents.op.utils.triton_ascend_api_docs import load_triton_ascend_api_docs
from akg_agents.op.verifier.profiler_utils import (
    run_profile_scripts_and_collect_results,
    make_profile_section,
    run_msprof,
    analyze_prof_data,
    run_nsys,
    analyze_nsys_data,
)
from akg_agents.op.verifier.roofline_utils import (
    augment_roofline_metrics,
    compute_roofline_profile,
    write_roofline_profile_result,
)

logger = logging.getLogger(__name__)


# 信号编号到名称的映射
_SIGNAL_NAMES = {
    1: "SIGHUP",   # Hangup
    2: "SIGINT",   # Interrupt
    3: "SIGQUIT",  # Quit
    6: "SIGABRT",  # Abort
    9: "SIGKILL",  # Kill
    11: "SIGSEGV", # Segmentation fault
    13: "SIGPIPE", # Broken pipe
    15: "SIGTERM", # Termination
}

def _get_signal_name(signum: int) -> str:
    """将信号编号转换为可读名称"""
    return _SIGNAL_NAMES.get(signum, f"SIG({signum})")


def _empty_profile_result(error: Optional[str] = None) -> Dict[str, Any]:
    """Canonical "no measurement" result shape returned on every dispatch
    failure (unsupported backend, profile subprocess crash, exception).
    Keeps the per_shape_* arrays present-but-empty so consumers don't have
    to special-case ``KeyError`` vs ``None``."""
    out: Dict[str, Any] = {
        "gen_time": None,
        "base_time": None,
        "speedup": 0.0,
        "per_shape_gen_us": [],
        "per_shape_base_us": [],
        "gen_method": None,
        "base_method": None,
        "roofline_time": None,
        "roofline_speedup": 0.0,
        "roofline": None,
        "artifacts": {},
    }
    if error is not None:
        out["error"] = error
    return out


def collect_json_artifacts(directory: str) -> Dict[str, str]:
    """
    收集目录中所有 JSON/JSONL 文件的原始内容。
    
    Args:
        directory: 要扫描的目录路径
        
    Returns:
        Dict[str, str]: 文件相对路径 -> 文件原始内容（字符串）
        例如: {"autotune_info_case_0.json": "{...}", "subdir/result.jsonl": "..."}
    """
    artifacts = {}
    if not os.path.exists(directory):
        return artifacts
        
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json') or filename.endswith('.jsonl'):
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, directory)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        artifacts[rel_path] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read file {rel_path}: {e}")
    return artifacts

class LocalWorker(WorkerInterface):
    """
    Local implementation of WorkerInterface.
    Executes verification tasks in a local subprocess, managing devices via DevicePool.
    """
    def __init__(self, device_pool: DevicePool, backend: str = "cuda"):
        self.device_pool = device_pool
        self.backend = backend

    async def verify(self, package_data: Union[bytes, str], task_id: str, op_name: str, timeout: int = DEFAULT_EVAL_TIMEOUT_S) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute verification task locally.
        
        注意：device 的管理（acquire/release）由调用方负责
        这个方法只负责执行已经生成好的脚本（脚本中已包含正确的 device_id）
        
        Args:
            package_data: 验证包数据（bytes 或目录路径）
            task_id: 任务ID
            op_name: 算子名称
            timeout: 超时时间
            
        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, log, artifacts)
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
                        return False, f"Failed to extract package: {e}", {}
                elif isinstance(package_data, str):
                    extract_dir = package_data
                else:
                    return False, "Unsupported package_data type for LocalWorker.verify", {}

                script_name = f"verify_{op_name}.py"
                script_path = os.path.join(extract_dir, script_name)
                if not os.path.exists(script_path):
                    return False, f"Verification script {script_name} not found.", {}

                # 注意：脚本中的 device_id 已在生成时设置正确
                # worker 只负责执行脚本，不管理设备分配
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'

                python_exe = sys.executable
                cmd = [python_exe, script_name]
                logger.info(f"[{task_id}] Running verification for {op_name}")
                
                # 在 Unix 系统上创建新的进程组，以便超时时能够杀死所有子进程
                preexec_fn = os.setsid if hasattr(os, 'setsid') else None
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=extract_dir,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    preexec_fn=preexec_fn
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                    returncode = process.returncode
                    
                    output_log = stdout.decode(errors='replace') + "\n" + stderr.decode(errors='replace')
                    success = (returncode == 0)

                    # 当 returncode 为负数时，表示进程被信号终止
                    # subprocess pipes 无法捕获 "Segmentation fault" 等 shell 消息
                    # 需要自己生成有意义的错误信息
                    if returncode < 0 and not output_log.strip():
                        signal_name = _get_signal_name(-returncode)
                        output_log = (
                            f"Process terminated by signal {-returncode} ({signal_name}).\n"
                            f"No output captured (process died before writing to stdout/stderr).\n"
                        )

                    # 收集执行过程中生成的 JSON 文件
                    artifacts = collect_json_artifacts(extract_dir)
                    if artifacts:
                        logger.info(f"[{task_id}] Collected {len(artifacts)} artifact files: {list(artifacts.keys())}")
                    
                    if success:
                        logger.info(f"[{task_id}] Verification passed.")
                    else:
                        logger.error(f"[{task_id}] Verification failed with log:\n{output_log}")
                        
                    return success, output_log, artifacts
                except asyncio.TimeoutError:
                    try:
                        # 尝试优雅地终止进程组
                        import signal
                        if hasattr(os, 'killpg'):
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        else:
                            process.terminate()
                        
                        # 给一点时间让进程退出
                        await asyncio.sleep(1)
                        if process.returncode is None:
                            if hasattr(os, 'killpg'):
                                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                            else:
                                process.kill()
                    except Exception as e:
                        logger.warning(f"[{task_id}] Error while killing process: {e}")
                        try:
                            process.kill()
                        except:
                            pass
                    logger.error(f"[{task_id}] Verification timed out.")
                    return False, f"Verification timed out after {timeout} seconds.", {}

        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker verification failed: {e}", exc_info=True)
            return False, str(e), {}

    async def get_doc(self, doc_name: str) -> str:
        """返回 Worker 本地环境可见的文档。"""
        if doc_name == "triton_ascend_api":
            return load_triton_ascend_api_docs()
        raise ValueError(f"Unsupported doc name: {doc_name}")

    async def profile(self, package_data: bytes, task_id: str, op_name: str, profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute profiling task locally.
        
        注意：device 的管理（acquire/release）由调用方负责
        这个方法只负责执行已经生成好的 profile 脚本
        
        Returns:
            Dict[str, Any]: 包含 gen_time, base_time, speedup, roofline,
            roofline_time, roofline_speedup, artifacts 等字段
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
                    return {'gen_time': None, 'base_time': None, 'speedup': 0.0, 'artifacts': {}, 'error': str(e)}
                
                # 注意：profile 脚本中的 device_id 应该在生成时就已经设置正确
                # （与 verify 类似，通过预先获取设备ID）
                # 3. Get settings
                backend = profile_settings.get('backend', self.backend)
                dsl = profile_settings.get('dsl', '')
                run_times = profile_settings.get('run_times', DEFAULT_RUN_TIMES)
                warmup_times = profile_settings.get('warmup_times', DEFAULT_WARMUP_TIMES)
                override_base_time_us = profile_settings.get('override_base_time_us')
                
                # 4. Execute profiling based on backend/dsl. Every dispatched
                # helper returns the canonical ``{"base": Section | None,
                # "gen": Section | None}`` shape (see profiler_utils), so the
                # post-processing below is backend-uniform.
                try:
                    from akg_agents.op.verifier.adapters.factory import get_dsl_adapter
                    if get_dsl_adapter(dsl).profile_via_python_script or backend == "cpu":
                        loop = asyncio.get_running_loop()
                        sections = await loop.run_in_executor(
                            None,
                            run_profile_scripts_and_collect_results,
                            extract_dir, op_name, task_id, override_base_time_us,
                        )
                    elif backend == "ascend":
                        loop = asyncio.get_running_loop()
                        sections = await loop.run_in_executor(
                            None,
                            self._run_msprof_profiling,
                            extract_dir, op_name, task_id, warmup_times, run_times,
                        )
                    elif backend == "cuda":
                        loop = asyncio.get_running_loop()
                        sections = await loop.run_in_executor(
                            None,
                            self._run_nsys_profiling,
                            extract_dir, op_name, task_id, warmup_times, run_times,
                        )
                    else:
                        logger.warning(f"[{task_id}] Unsupported backend for profiling: {backend}")
                        return _empty_profile_result()

                    base_sec = sections.get("base")
                    gen_sec = sections.get("gen")
                    if gen_sec is None:
                        logger.error(f"[{task_id}] Generation profile produced no result")
                        return _empty_profile_result(error="generation profile failed")

                    gen_time = gen_sec["avg_us"]
                    base_time = base_sec["avg_us"] if base_sec else float("inf")
                    per_shape_gen_us = list(gen_sec["per_case_us"])
                    per_shape_base_us = (list(base_sec["per_case_us"])
                                         if base_sec else [])
                    gen_method = gen_sec.get("method")
                    base_method = base_sec.get("method") if base_sec else None

                    if base_sec and gen_time > 0:
                        speedup = base_time / gen_time
                    else:
                        speedup = 0.0

                    roofline_result = compute_roofline_profile(
                        verify_dir=extract_dir,
                        op_name=op_name,
                        task_id=task_id,
                        profile_settings=profile_settings,
                    )
                    roofline_result = augment_roofline_metrics(
                        roofline_result,
                        gen_time_us=gen_time,
                        base_time_us=base_time if base_sec else None,
                    )
                    write_roofline_profile_result(extract_dir, roofline_result)

                    roofline_time = roofline_result.get("time_us") if roofline_result.get("success") else None
                    roofline_speedup = roofline_result.get("speedup_vs_generated", 0.0)

                    artifacts = collect_json_artifacts(extract_dir)
                    if artifacts:
                        logger.info(f"[{task_id}] Collected {len(artifacts)} artifact files: {list(artifacts.keys())}")

                    # ``gen_time`` / ``base_time`` get None-ified on inf so
                    # in-process callers (dynamic_tune etc.) see the
                    # "no-measurement" sentinel uniformly. JSON-safety for
                    # the HTTP / disk boundary lives in
                    # ``worker/server.py`` + ``op/utils/json_safe``.
                    return {
                        "gen_time": gen_time if gen_time < float("inf") else None,
                        "base_time": (base_time if base_sec
                                      and base_time < float("inf") else None),
                        "speedup": speedup,
                        "per_shape_gen_us": per_shape_gen_us,
                        "per_shape_base_us": per_shape_base_us,
                        "gen_method": gen_method,
                        "base_method": base_method,
                        "roofline_time": roofline_time,
                        "roofline_speedup": roofline_speedup,
                        "roofline": roofline_result,
                        "artifacts": artifacts,
                    }

                except Exception as e:
                    logger.error(f"[{task_id}] Profiling execution failed: {e}", exc_info=True)
                    return _empty_profile_result(error=str(e))

        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker profiling failed: {e}", exc_info=True)
            return _empty_profile_result(error=str(e))
    
    def _run_msprof_profiling(self, extract_dir: str, op_name: str, task_id: str,
                              warmup_times: int, run_times: int,
                              timeout: int = 600) -> Dict[str, Optional[Dict[str, Any]]]:
        """Run msprof on the base + generation scripts and return canonical
        ``{"base": Section | None, "gen": Section | None}`` sections.

        msprof aggregates over a single profile run (the script may iterate
        multiple shapes internally, but msprof attributes the kernel time to
        the whole run), so each section's ``per_case_us`` is a single-element
        list — the same as static-shape python-script profiling. Downstream
        consumers iterate per_case_us uniformly regardless of backend."""
        sections: Dict[str, Optional[Dict[str, Any]]] = {"base": None, "gen": None}
        try:
            for kind, json_key in (("base", "base"), ("generation", "gen")):
                script = os.path.join(extract_dir, f"profile_{op_name}_{kind}.py")
                ok, err, prof_path = run_msprof(script, op_name, task_id, timeout=timeout)
                if not ok or not prof_path:
                    logger.error(f"[{task_id}] {kind} msprof failed: {err}")
                    continue
                ok, err, avg_us = analyze_prof_data(
                    prof_path, warmup_times, run_times, op_name, task_id)
                if not ok:
                    logger.error(f"[{task_id}] {kind} prof analysis failed: {err}")
                    continue
                sections[json_key] = make_profile_section(avg_us, method="msprof")
        except Exception as e:
            logger.error(f"[{task_id}] msprof profiling failed: {e}", exc_info=True)
        return sections

    def _run_nsys_profiling(self, extract_dir: str, op_name: str, task_id: str,
                            warmup_times: int, run_times: int,
                            timeout: int = 600) -> Dict[str, Optional[Dict[str, Any]]]:
        """nsys variant of :meth:`_run_msprof_profiling`. Same canonical
        section shape; ``method`` distinguishes the two backends in
        downstream metrics."""
        sections: Dict[str, Optional[Dict[str, Any]]] = {"base": None, "gen": None}
        try:
            for kind, json_key in (("base", "base"), ("generation", "gen")):
                script = os.path.join(extract_dir, f"profile_{op_name}_{kind}.py")
                ok, err, rep_path = run_nsys(script, op_name, task_id, timeout=timeout)
                if not ok or not rep_path:
                    logger.error(f"[{task_id}] {kind} nsys failed: {err}")
                    continue
                ok, err, avg_us = analyze_nsys_data(
                    rep_path, warmup_times, run_times, kind, op_name, task_id)
                if not ok:
                    logger.error(f"[{task_id}] {kind} nsys analysis failed: {err}")
                    continue
                sections[json_key] = make_profile_section(avg_us, method="nsys")
        except Exception as e:
            logger.error(f"[{task_id}] nsys profiling failed: {e}", exc_info=True)
        return sections

    async def profile_single_task(self, package_data: bytes, task_id: str, op_name: str, 
                                   profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute single task profiling locally.
        
        单独测量某段代码的执行性能，不进行 base vs generation 对比。
        
        Returns:
            Dict[str, Any]: 包含 time_us, success, log 等字段
        """
        try:
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
                    return {'time_us': None, 'success': False, 'log': f'Failed to extract package: {e}'}
                
                # Find profile script
                script_name = f"profile_single_{op_name}.py"
                script_path = os.path.join(extract_dir, script_name)
                if not os.path.exists(script_path):
                    return {'time_us': None, 'success': False, 'log': f'Profile script {script_name} not found'}
                
                # Run profile script
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                
                python_exe = sys.executable
                cmd = [python_exe, script_name]
                logger.info(f"[{task_id}] Running single task profiling for {op_name}")
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=extract_dir,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                timeout = profile_settings.get('timeout', DEFAULT_EVAL_TIMEOUT_S)
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                    returncode = process.returncode
                    
                    output_log = stdout.decode(errors='replace') + "\n" + stderr.decode(errors='replace')
                    success = (returncode == 0)
                    
                    if not success:
                        logger.error(f"[{task_id}] Profile single task failed with log:\n{output_log}")
                        return {'time_us': None, 'success': False, 'log': output_log}
                    
                    # Read result from JSON file
                    result_file = os.path.join(extract_dir, "profile_single_result.json")
                    time_us = None
                    if os.path.exists(result_file):
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                            time_us = result_data.get('avg_time_us')
                            # 确保不是 inf
                            if time_us is not None and time_us >= float('inf'):
                                time_us = None
                    else:
                        # Try to parse from output log
                        for line in output_log.split('\n'):
                            if 'avg_time_us' in line or 'PROFILE_RESULT' in line:
                                try:
                                    # Parse "PROFILE_RESULT: 123.45" format
                                    if 'PROFILE_RESULT:' in line:
                                        time_us = float(line.split('PROFILE_RESULT:')[1].strip())
                                except:
                                    pass
                    
                    logger.info(f"[{task_id}] Profile single task result: {time_us} us")
                    return {'time_us': time_us, 'success': time_us is not None, 'log': output_log}
                    
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    logger.error(f"[{task_id}] Profile single task timed out.")
                    return {'time_us': None, 'success': False, 'log': f'Timed out after {timeout} seconds'}
        
        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker profile_single_task failed: {e}", exc_info=True)
            return {'time_us': None, 'success': False, 'log': str(e)}

    async def generate_reference(self, package_data: bytes, task_id: str, op_name: str, timeout: int = DEFAULT_GEN_REF_TIMEOUT_S) -> Tuple[bool, str, bytes]:
        """
        Execute task_desc and generate reference data locally.
        
        用于 CUDA-to-Ascend 转换场景：执行 Triton-CUDA 代码，保存输出作为参考数据。
        
        Args:
            package_data: 验证包数据（bytes）
            task_id: 任务ID
            op_name: 算子名称
            timeout: 超时时间
            
        Returns:
            Tuple[bool, str, bytes]: (success, log, reference_data_bytes)
        """
        try:
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
                    return False, f"Failed to extract package: {e}", b''
                
                # Find and run the verify script
                script_name = f"verify_{op_name}.py"
                script_path = os.path.join(extract_dir, script_name)
                if not os.path.exists(script_path):
                    return False, f"Verification script {script_name} not found.", b''
                
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                
                python_exe = sys.executable
                cmd = [python_exe, script_name]
                logger.info(f"[{task_id}] Running reference generation for {op_name}")
                
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
                    
                    if not success:
                        logger.error(f"[{task_id}] Reference generation failed with log:\n{output_log}")
                        return False, output_log, b''
                    
                    # Check for success marker
                    if "REFERENCE_GENERATION_SUCCESS" not in output_log:
                        return False, f"Reference generation did not complete successfully:\n{output_log}", b''
                    
                    # Read the generated .pt file
                    ref_file = os.path.join(extract_dir, f"{op_name}_reference.pt")
                    if not os.path.exists(ref_file):
                        return False, f"Reference file {ref_file} not found after generation.", b''
                    
                    with open(ref_file, 'rb') as f:
                        ref_bytes = f.read()
                    
                    logger.info(f"[{task_id}] Reference generation succeeded, .pt file size: {len(ref_bytes)} bytes")
                    return True, output_log, ref_bytes
                    
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    logger.error(f"[{task_id}] Reference generation timed out.")
                    return False, f"Reference generation timed out after {timeout} seconds.", b''
        
        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker generate_reference failed: {e}", exc_info=True)
            return False, str(e), b''
