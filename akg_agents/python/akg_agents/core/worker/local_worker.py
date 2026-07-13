# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import math
import os
import tempfile
import tarfile
import logging
import sys
import json
import threading
from typing import Tuple, Dict, Any, Union, Optional
from contextlib import contextmanager

from .interface import WorkerInterface, empty_profile_result as _empty_profile_result
from .eval_config import (
    resolve_eval_timeout,
    resolve_reference_timeout,
    resolve_run_times,
    resolve_warmup_times,
)
from ..async_pool.device_pool import DevicePool
from akg_agents.op.utils.triton_ascend_api_docs import load_triton_ascend_api_docs
from akg_agents.op.verifier import aggregate
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
from akg_agents.utils.process_utils import (
    communicate_or_kill,
    popen_process_group_kwargs,
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


def collect_json_artifacts(directory: str, trace_names: tuple = ()) -> Dict[str, str]:
    """{rel_path: content} for json/jsonl, plus any basename in ``trace_names``
    (the --trace timeline + per-op CSVs); skips other ``*_ascend_pt`` internals."""
    artifacts = {}
    if not os.path.exists(directory):
        return artifacts

    for root, dirs, files in os.walk(directory):
        in_trace = "_ascend_pt" in root
        for filename in files:
            if not (filename in trace_names or
                    (not in_trace and filename.endswith((".json", ".jsonl")))):
                continue
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, directory)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    artifacts[rel_path] = f.read()
            except Exception as e:
                logger.warning(f"Failed to read file {rel_path}: {e}")
    return artifacts


class PackageExtractError(Exception):
    """A worker package could not be materialized into a directory."""


@contextmanager
def _extract_package(package_data: Union[bytes, str]):
    """Yield a directory holding the package contents. ``bytes`` → extracted
    into a TemporaryDirectory (auto-cleaned on exit); ``str`` → an
    already-extracted dir path, used as-is. Raises PackageExtractError on a
    bad type or corrupt archive so callers map it to their own error shape.
    Single owner of the tempdir + untar boilerplate the entry points share."""
    if isinstance(package_data, str):
        yield package_data
        return
    if not isinstance(package_data, (bytes, bytearray)):
        raise PackageExtractError("Unsupported package_data type for LocalWorker")
    with tempfile.TemporaryDirectory() as temp_dir:
        tar_path = os.path.join(temp_dir, "package.tar")
        with open(tar_path, "wb") as f:
            f.write(package_data)
        extract_dir = os.path.join(temp_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with tarfile.open(tar_path, "r") as tar_ref:
                # Worker packages cross an HTTP boundary. Reject path escapes,
                # device nodes and links before extraction; the verifier's
                # packer only emits regular files/directories, so links are
                # unnecessary and would make containment harder to prove.
                extract_root = os.path.realpath(extract_dir)
                for member in tar_ref.getmembers():
                    target = os.path.realpath(
                        os.path.join(extract_root, member.name))
                    try:
                        contained = os.path.commonpath(
                            (extract_root, target)) == extract_root
                    except ValueError:
                        contained = False
                    if (not contained or member.issym() or member.islnk()
                            or member.isdev() or member.isfifo()):
                        raise PackageExtractError(
                            f"Unsafe archive member rejected: {member.name!r}")
                if hasattr(tarfile, "data_filter"):
                    tar_ref.extractall(extract_dir, filter="data")
                else:  # Python 3.10 compatibility after the checks above
                    tar_ref.extractall(extract_dir)
        except Exception as e:
            if isinstance(e, PackageExtractError):
                raise
            raise PackageExtractError(f"Failed to extract package: {e}") from e
        yield extract_dir


class LocalWorker(WorkerInterface):
    """
    Local implementation of WorkerInterface.
    Executes verification tasks in a local subprocess, managing devices via DevicePool.
    """
    def __init__(self, device_pool: DevicePool, backend: str = "cuda"):
        self.device_pool = device_pool
        self.backend = backend

    async def acquire_device(self, task_id: str = "unknown",
                             timeout: Optional[float] = None) -> Tuple[int, int]:
        """Delegate to the in-process DevicePool (non-renewable: release is the
        ``device_lease`` CM's finally, guaranteed even on cancellation)."""
        return await self.device_pool.acquire_device(owner=task_id, timeout=timeout)

    async def release_device(self, device_id: int, lease_id: int,
                             task_id: str = "unknown") -> None:
        await self.device_pool.release_device(device_id, lease_id)

    async def _run_script(self, extract_dir: str, script_name: str,
                          timeout: Optional[int], task_id: str, action: str,
                          keep_res: bool = False
                          ) -> Tuple[Optional[int], bytes, bytes, bool]:
        """Spawn ``python script_name`` in extract_dir as a killable async
        subprocess (own process group, torn down on timeout/cancel). Returns
        ``(returncode, stdout, stderr, timed_out)``; callers own the decode +
        result shaping. Single owner of the worker's subprocess spawn+kill."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if os.name == "posix":
            env["PWD"] = os.path.abspath(extract_dir)
        # --trace: per-subprocess only — never mutate the daemon's global env.
        if keep_res:
            env["AKG_PROF_KEEP_RES"] = "1"
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_name, cwd=extract_dir, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            **popen_process_group_kwargs(),
        )
        stdout, stderr, timed_out = await communicate_or_kill(
            process, timeout, task_id, action)
        return process.returncode, stdout, stderr, timed_out

    async def verify(self, package_data: Union[bytes, str], task_id: str,
                     op_name: str, timeout: Optional[int] = None
                     ) -> Tuple[bool, str, Dict[str, Any]]:
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
        timeout = resolve_eval_timeout(timeout)
        try:
            with _extract_package(package_data) as extract_dir:
                script_name = f"verify_{op_name}.py"
                script_path = os.path.join(extract_dir, script_name)
                if not os.path.exists(script_path):
                    return False, f"Verification script {script_name} not found.", {}

                # 脚本中的 device_id 已在生成时设置正确；worker 只执行脚本。
                logger.info(f"[{task_id}] Running verification for {op_name}")
                returncode, stdout, stderr, timed_out = await self._run_script(
                    extract_dir, script_name, timeout, task_id, "Verification")
                if timed_out:
                    return False, f"Verification timed out after {timeout} seconds.", {}

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
            with _extract_package(package_data) as extract_dir:
                # 注意：profile 脚本中的 device_id 应该在生成时就已经设置正确
                # （与 verify 类似，通过预先获取设备ID）
                # 3. Get settings
                backend = profile_settings.get('backend', self.backend)
                dsl = profile_settings.get('dsl', '')
                run_times = resolve_run_times(profile_settings.get('run_times'))
                warmup_times = resolve_warmup_times(
                    profile_settings.get('warmup_times'))
                profile_timeout = resolve_eval_timeout(
                    profile_settings.get('timeout'))
                # --trace: per-subprocess (see _run_script), not daemon-global.
                keep_res = bool(profile_settings.get('keep_res'))
                # Override IS a Section dict (same shape script path produces).
                override_base_section = profile_settings.get('override_base_section')
                has_valid_base_override = (
                    isinstance(override_base_section, dict)
                    and isinstance(override_base_section.get("avg_us"), (int, float))
                    and 0 < override_base_section["avg_us"] < float("inf")
                )
                base_script = os.path.join(extract_dir, f"profile_{op_name}_base.py")
                gen_script = os.path.join(extract_dir, f"profile_{op_name}_generation.py")
                base_requested = os.path.exists(base_script) or has_valid_base_override
                generation_requested = os.path.exists(gen_script)

                # 4. Execute profiling based on backend/dsl. Every dispatched
                # helper returns the canonical ``{"base": Section | None,
                # "gen": Section | None}`` shape (see profiler_utils), so the
                # post-processing below is backend-uniform.
                try:
                    from akg_agents.op.verifier.adapters.factory import get_dsl_adapter
                    if get_dsl_adapter(dsl).profile_via_python_script or backend == "cpu":
                        # Run the profile scripts in THIS coroutine (not an
                        # executor thread) so a cancelled / timed-out profile
                        # kills the subprocess tree before the device lease is
                        # released — same teardown guarantee as verify().
                        sections = await run_profile_scripts_and_collect_results(
                            extract_dir, op_name,
                            lambda name, label: self._run_profile_script_async(
                                extract_dir, name, profile_timeout, task_id, label,
                                keep_res=keep_res),
                            task_id=task_id,
                            override_base_section=override_base_section,
                        )
                    elif backend == "ascend":
                        sections = await self._run_external_profiler(
                            self._run_msprof_profiling,
                            extract_dir, op_name, task_id,
                            warmup_times, run_times, profile_timeout,
                        )
                    elif backend == "cuda":
                        sections = await self._run_external_profiler(
                            self._run_nsys_profiling,
                            extract_dir, op_name, task_id,
                            warmup_times, run_times, profile_timeout,
                        )
                    else:
                        logger.warning(f"[{task_id}] Unsupported backend for profiling: {backend}")
                        return _empty_profile_result()

                    base_sec = sections.get("base")
                    gen_sec = sections.get("gen")
                    if generation_requested and gen_sec is None:
                        logger.error(f"[{task_id}] Generation profile produced no result")
                        return _empty_profile_result(error="generation profile failed")
                    profile_error = None
                    if base_requested and base_sec is None:
                        logger.error(f"[{task_id}] Base profile produced no result")
                        if gen_sec is None:
                            return _empty_profile_result(error="base profile failed")
                        profile_error = "base profile failed"
                    if not base_requested and not generation_requested:
                        logger.error(f"[{task_id}] No profile scripts or base override found")
                        return _empty_profile_result(error="no profile section requested")

                    gen_time = gen_sec["avg_us"] if gen_sec else float("inf")
                    base_time = base_sec["avg_us"] if base_sec else float("inf")
                    per_shape_gen_us = (list(gen_sec["per_case_us"])
                                        if gen_sec else [])
                    per_shape_base_us = (list(base_sec["per_case_us"])
                                         if base_sec else [])
                    gen_method = gen_sec.get("method") if gen_sec else None
                    base_method = base_sec.get("method") if base_sec else None

                    # Speedup is the geomean of per-shape base/gen ratios
                    # (single owner: aggregate.geomean_ratio), NOT
                    # mean(base)/mean(gen) — a slow shape must not dominate.
                    # None (no valid pair) → 0.0 sentinel like before.
                    if base_sec and gen_sec:
                        speedup = aggregate.geomean_ratio(
                            per_shape_base_us, per_shape_gen_us) or 0.0
                    else:
                        speedup = 0.0

                    if gen_sec:
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
                            gen_per_shape_us=per_shape_gen_us,
                            base_per_shape_us=per_shape_base_us if base_sec else None,
                        )
                        write_roofline_profile_result(extract_dir, roofline_result)

                        roofline_time = (
                            roofline_result.get("time_us")
                            if roofline_result.get("success") else None
                        )
                        roofline_speedup = roofline_result.get("speedup_vs_generated", 0.0)
                    else:
                        roofline_result = None
                        roofline_time = None
                        roofline_speedup = 0.0

                    # --trace ships trace_view.json + per-op CSVs back; bulky
                    # msprof internals dropped. Read here (inside the extract
                    # context) so a remote temp dir can be cleaned right after.
                    artifacts = collect_json_artifacts(
                        extract_dir,
                        ("trace_view.json", "op_statistic.csv", "kernel_details.csv")
                        if keep_res else ())
                    if artifacts:
                        logger.info(f"[{task_id}] Collected {len(artifacts)} artifact files: {list(artifacts.keys())}")

                    # ``gen_time`` / ``base_time`` get None-ified on inf so
                    # in-process callers (dynamic_tune etc.) see the
                    # "no-measurement" sentinel uniformly. JSON-safety for
                    # the HTTP / disk boundary lives in
                    # ``worker/server.py`` + ``op/utils/json_safe``.
                    result = {
                        "gen_time": (gen_time if gen_sec
                                     and gen_time < float("inf") else None),
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
                    if profile_error is not None:
                        result["error"] = profile_error
                    return result

                except Exception as e:
                    logger.error(f"[{task_id}] Profiling execution failed: {e}", exc_info=True)
                    return _empty_profile_result(error=str(e))

        except PackageExtractError as e:
            return _empty_profile_result(error=str(e))
        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker profiling failed: {e}", exc_info=True)
            return _empty_profile_result(error=str(e))
    
    async def _run_profile_script_async(self, verify_dir: str, script_name: str,
                                        timeout: int, task_id: str,
                                        label: str, keep_res: bool = False) -> bool:
        """Run one profile script (killable; self-times + writes its result
        JSON, which the caller reads). True on rc==0."""
        rc, stdout, stderr, timed_out = await self._run_script(
            verify_dir, script_name, timeout, task_id, label, keep_res=keep_res)
        if timed_out:
            logger.error("[%s] %s timed out after %s seconds", task_id, label,
                         timeout)
            return False
        if rc != 0:
            log = (stdout.decode(errors="replace") + "\n"
                   + stderr.decode(errors="replace"))
            logger.error("[%s] %s failed (rc=%s): %s", task_id, label, rc,
                         log.strip())
            return False
        return True

    def _run_trace_profiling(self, extract_dir, op_name, task_id, run, analyze,
                             method) -> Dict[str, Optional[Dict[str, Any]]]:
        """Shared base+generation loop for the external-profiler backends
        (msprof / nsys): run the profiler on each script, analyze the trace,
        build a canonical single-element section (the profiler attributes the
        kernel time to the whole run). ``run(script) -> (ok, err, path)``;
        ``analyze(path, kind) -> (ok, err, avg_us)``; ``method`` tags the
        section + log lines. Single owner of this loop for both backends."""
        sections: Dict[str, Optional[Dict[str, Any]]] = {"base": None, "gen": None}
        try:
            for kind, json_key in (("base", "base"), ("generation", "gen")):
                script = os.path.join(extract_dir, f"profile_{op_name}_{kind}.py")
                if not os.path.exists(script):
                    logger.info(f"[{task_id}] {kind} profile script not found; skipping")
                    continue
                ok, err, path = run(script)
                if not ok or not path:
                    logger.error(f"[{task_id}] {kind} {method} failed: {err}")
                    continue
                ok, err, avg_us = analyze(path, kind)
                if not ok:
                    logger.error(f"[{task_id}] {kind} {method} analysis failed: {err}")
                    continue
                sections[json_key] = make_profile_section(avg_us, method=method)
        except Exception as e:
            logger.error(f"[{task_id}] {method} profiling failed: {e}", exc_info=True)
        return sections

    async def _run_external_profiler(self, func, *args):
        """Run a blocking profiler without leaking it on coroutine cancel.

        Executor futures do not stop their underlying thread.  Propagate a
        cooperative event into every profiler subprocess, then wait until the
        thread has killed/drained that process tree before re-raising cancel.
        This preserves the device-lease teardown ordering used by async script
        profiling and verification.
        """
        cancel_event = threading.Event()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, func, *args, cancel_event)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            cancel_event.set()
            try:
                await asyncio.shield(future)
            except Exception as e:
                logger.warning("External profiler cleanup failed: %s", e)
            raise

    def _run_msprof_profiling(self, extract_dir, op_name, task_id,
                              warmup_times, run_times, timeout=None,
                              cancel_event=None):
        t = resolve_eval_timeout(timeout)
        return self._run_trace_profiling(
            extract_dir, op_name, task_id,
            run=lambda s: run_msprof(
                s, op_name, task_id, timeout=t, cancel_event=cancel_event),
            analyze=lambda path, _k: analyze_prof_data(
                path, warmup_times, run_times, op_name, task_id),
            method="msprof")

    def _run_nsys_profiling(self, extract_dir, op_name, task_id,
                            warmup_times, run_times, timeout=None,
                            cancel_event=None):
        t = resolve_eval_timeout(timeout)
        return self._run_trace_profiling(
            extract_dir, op_name, task_id,
            run=lambda s: run_nsys(
                s, op_name, task_id, timeout=t, cancel_event=cancel_event),
            analyze=lambda path, kind: analyze_nsys_data(
                path, warmup_times, run_times, kind, op_name, task_id,
                cancel_event=cancel_event),
            method="nsys")

    async def profile_single_task(self, package_data: bytes, task_id: str, op_name: str, 
                                   profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute single task profiling locally.
        
        单独测量某段代码的执行性能，不进行 base vs generation 对比。
        
        Returns:
            Dict[str, Any]: 包含 time_us, success, log 等字段
        """
        try:
            with _extract_package(package_data) as extract_dir:
                # Find profile script
                script_name = f"profile_single_{op_name}.py"
                script_path = os.path.join(extract_dir, script_name)
                if not os.path.exists(script_path):
                    return {'time_us': None, 'success': False, 'log': f'Profile script {script_name} not found'}
                
                # Run profile script
                logger.info(f"[{task_id}] Running single task profiling for {op_name}")
                timeout = resolve_eval_timeout(profile_settings.get('timeout'))
                returncode, stdout, stderr, timed_out = await self._run_script(
                    extract_dir, script_name, timeout, task_id, "Profile single task")
                if timed_out:
                    return {'time_us': None, 'success': False, 'log': f'Timed out after {timeout} seconds'}

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
                else:
                    # Try to parse from output log
                    for line in output_log.split('\n'):
                        if 'avg_time_us' in line or 'PROFILE_RESULT' in line:
                            try:
                                # Parse "PROFILE_RESULT: 123.45" format
                                if 'PROFILE_RESULT:' in line:
                                    time_us = float(line.split('PROFILE_RESULT:')[1].strip())
                            except (TypeError, ValueError, IndexError):
                                pass

                try:
                    time_us = float(time_us)
                    if not math.isfinite(time_us) or time_us <= 0:
                        time_us = None
                except (TypeError, ValueError):
                    time_us = None
                
                logger.info(f"[{task_id}] Profile single task result: {time_us} us")
                return {'time_us': time_us, 'success': time_us is not None, 'log': output_log}
        
        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker profile_single_task failed: {e}", exc_info=True)
            return {'time_us': None, 'success': False, 'log': str(e)}

    async def generate_reference(self, package_data: bytes, task_id: str,
                                 op_name: str, timeout: Optional[int] = None
                                 ) -> Tuple[bool, str, bytes]:
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
        timeout = resolve_reference_timeout(timeout)
        try:
            with _extract_package(package_data) as extract_dir:
                # Find and run the verify script
                script_name = f"verify_{op_name}.py"
                script_path = os.path.join(extract_dir, script_name)
                if not os.path.exists(script_path):
                    return False, f"Verification script {script_name} not found.", b''
                
                logger.info(f"[{task_id}] Running reference generation for {op_name}")
                returncode, stdout, stderr, timed_out = await self._run_script(
                    extract_dir, script_name, timeout, task_id, "Reference generation")
                if timed_out:
                    return False, f"Reference generation timed out after {timeout} seconds.", b''

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
        
        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker generate_reference failed: {e}", exc_info=True)
            return False, str(e), b''
