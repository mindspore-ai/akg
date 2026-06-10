# Copyright 2026 Huawei Technologies Co., Ltd
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

"""
Baseline Profiler: 预先测量 baseline 性能

用于 evolve/adaptive_search 场景，在开始前单独 profile baseline 一次，
避免所有任务重复测量。支持 KernelBench、SOL-ExecBench 和 CANN-Bench 三种 bench_type。
"""

import os
import io
import json
import shutil
import tarfile
import logging
from contextlib import AsyncExitStack
from typing import Optional, Dict, Any

from akg_agents.core.worker.interface import (
    DEFAULT_EVAL_TIMEOUT_S,
    DEFAULT_WARMUP_TIMES,
    DEFAULT_RUN_TIMES,
)
from akg_agents.op.verifier.data_cache import (
    build_baseline_cache_key,
    build_baseline_cache_payload,
    build_sol_problem_cache_identity,
    delete_baseline_result_from_cache,
    extract_baseline_time_us,
    get_baseline_cache_file_path,
    get_verifier_data_cache_key_id,
    load_verifier_data_cache_config,
    read_baseline_result_from_cache,
    verifier_data_cache_lock,
    write_baseline_result_to_cache,
)

logger = logging.getLogger(__name__)


async def profile_baseline_once(
    op_name: str,
    task_desc: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int = DEFAULT_WARMUP_TIMES,
    run_times: int = DEFAULT_RUN_TIMES,
    timeout: int = DEFAULT_EVAL_TIMEOUT_S,
) -> Optional[float]:
    """
    预先 profile baseline 一次（只测量框架实现的性能）

    根据 config["bench_type"] 自动选择 KernelBench 或 SOL-ExecBench 的 baseline profiling 流程。

    设备分配通过 worker 的 device_pool acquire/release 管理，
    与正常 verify/profile 流程保持一致。

    Args:
        op_name: 算子名称
        task_desc: 任务描述（KernelBench: 框架代码；SOL: 中文描述文本）
        dsl: DSL 类型
        framework: 框架
        backend: 后端
        arch: 架构
        config: 配置字典
        warmup_times: 预热次数
        run_times: 运行次数
        timeout: 超时时间

    Returns:
        float: baseline 时间（微秒），失败返回 None
    """
    bench_type = config.get("bench_type", "kernelbench")

    if bench_type == "sol":
        return await _profile_sol_baseline(
            op_name, dsl, framework, backend, arch, config,
            warmup_times, run_times, timeout
        )
    elif bench_type == "cann":
        return await _profile_cann_baseline(
            op_name, dsl, framework, backend, arch, config,
            warmup_times, run_times, timeout
        )
    else:
        return await _profile_kernelbench_baseline(
            op_name, task_desc, dsl, framework, backend, arch, config,
            warmup_times, run_times, timeout
        )


async def _profile_kernelbench_baseline(
    op_name: str,
    task_desc: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int,
    run_times: int,
    timeout: int
) -> Optional[float]:
    """KernelBench 模式的 baseline profiling（原有逻辑）"""
    acquired_device = None
    worker = None
    cache_cfg = load_verifier_data_cache_config(config)
    cache_key = None
    cache_file = None
    try:
        from akg_agents.op.verifier.kernel_verifier import KernelVerifier
        from akg_agents.core.worker.manager import get_worker_manager
        from akg_agents.core.worker.local_worker import LocalWorker
        from akg_agents.core.worker.remote_worker import RemoteWorker

        if cache_cfg.enabled and cache_cfg.cache_baseline_result:
            cache_key_id = get_verifier_data_cache_key_id(config, "baseline_profile")
            cache_key = build_baseline_cache_key(
                op_name=op_name,
                framework_code=task_desc,
                framework=framework,
                backend=backend,
                arch=arch,
                bench_type="kernelbench",
                warmup_times=warmup_times,
                run_times=run_times,
                dsl=dsl,
                task_id=cache_key_id,
            )
            cache_file = get_baseline_cache_file_path(
                cache_cfg,
                op_name=op_name,
                cache_key=cache_key,
            )
            cached_entry = read_baseline_result_from_cache(
                cache_cfg,
                op_name=op_name,
                cache_key=cache_key,
            )
            cached_time_us = extract_baseline_time_us(cached_entry)
            if cached_time_us is not None:
                logger.info(
                    f"[{op_name}] ✅ 命中本地 baseline cache: {cached_time_us:.2f}us, "
                    f"cache_file={cache_file}, cache_key={cache_key}"
                )
                return cached_time_us
            if cached_entry:
                logger.warning(
                    f"[{op_name}] baseline cache 内容无效，删除旧缓存并重新测量: "
                    f"cache_file={cache_file}, cache_key={cache_key}"
                )
                delete_baseline_result_from_cache(
                    cache_cfg,
                    op_name=op_name,
                    cache_key=cache_key,
                )

        logger.info(f"[{op_name}] 🚀 开始预先 profile baseline（只测一次）...")

        async with AsyncExitStack() as stack:
            if cache_cfg.enabled and cache_cfg.cache_baseline_result and cache_key:
                await stack.enter_async_context(
                    verifier_data_cache_lock(
                        cache_cfg,
                        namespace="baseline",
                        op_name=op_name,
                        cache_key=cache_key,
                    )
                )
                cached_entry = read_baseline_result_from_cache(
                    cache_cfg,
                    op_name=op_name,
                    cache_key=cache_key,
                )
                cached_time_us = extract_baseline_time_us(cached_entry)
                if cached_time_us is not None:
                    logger.info(
                        f"[{op_name}] ✅ 等待期间命中本地 baseline cache: {cached_time_us:.2f}us, "
                        f"cache_file={cache_file}, cache_key={cache_key}"
                    )
                    return cached_time_us
                if cached_entry:
                    logger.warning(
                        f"[{op_name}] baseline cache 内容无效，删除旧缓存并重新测量: "
                        f"cache_file={cache_file}, cache_key={cache_key}"
                    )
                    delete_baseline_result_from_cache(
                        cache_cfg,
                        op_name=op_name,
                        cache_key=cache_key,
                    )

            worker = await get_worker_manager().select(backend=backend, arch=arch)
            if not worker:
                logger.warning(f"[{op_name}] 无法获取 worker，跳过预先 profile baseline")
                return None

            # 从 device_pool acquire 设备（与 run_profile/run 流程一致）
            device_id = 0
            if isinstance(worker, LocalWorker) and worker.device_pool:
                acquired_device = await worker.device_pool.acquire_device()
                device_id = acquired_device
                logger.info(f"[{op_name}] Acquired device {device_id} from pool for baseline profile")
            elif isinstance(worker, RemoteWorker):
                acquired_device = await worker.acquire_device(task_id="baseline_profile")
                device_id = acquired_device
                logger.info(f"[{op_name}] Acquired remote device {device_id} for baseline profile")

            verifier = KernelVerifier(
                op_name=op_name,
                framework_code=task_desc,
                task_id="baseline_profile",
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                config=config,
                worker=worker
            )

            result = await verifier.profile_single_task(
                task_desc=task_desc,
                warmup_times=warmup_times,
                run_times=run_times,
                timeout=timeout,
                device_id=device_id
            )

            if result.get('success', False):
                baseline_time_us = result.get('time_us')
                if baseline_time_us and baseline_time_us > 0 and baseline_time_us < float('inf'):
                    logger.info(f"[{op_name}] ✅ Baseline profile 完成: {baseline_time_us:.2f}us")
                    _save_baseline_profile_scripts(verifier, op_name, task_desc, warmup_times, run_times, device_id)
                    if cache_cfg.enabled and cache_cfg.cache_baseline_result and cache_key:
                        written_path = write_baseline_result_to_cache(
                            cache_cfg,
                            op_name=op_name,
                            cache_key=cache_key,
                            result_data=build_baseline_cache_payload(
                                base_time_us=baseline_time_us,
                                warmup_times=warmup_times,
                                run_times=run_times,
                                method="profile_single_task",
                            ),
                            metadata={
                                "framework": framework,
                                "dsl": dsl,
                                "cache_key_id": cache_key_id,
                                "backend": backend,
                                "arch": arch,
                                "bench_type": "kernelbench",
                            },
                        )
                        if written_path:
                            logger.info(
                                f"[{op_name}] baseline 结果已写入本地 cache: "
                                f"cache_file={written_path}, cache_key={cache_key}, "
                                f"cache_dir={cache_cfg.cache_dir}"
                            )
                    return baseline_time_us
                else:
                    logger.warning(f"[{op_name}] Baseline profile 结果无效: {baseline_time_us}")
            else:
                error_log = result.get('log', 'Unknown error')
                logger.warning(f"[{op_name}] Baseline profile 失败: {error_log}")

            return None
    except TimeoutError as e:
        logger.warning(f"[{op_name}] 等待 baseline cache lock 超时，跳过预先 profile baseline: {e}")
        return None
    except Exception as e:
        logger.warning(f"[{op_name}] 预先 profile baseline 失败: {e}")
        return None
    finally:
        if acquired_device is not None and worker is not None:
            try:
                from akg_agents.core.worker.local_worker import LocalWorker
                from akg_agents.core.worker.remote_worker import RemoteWorker
                if isinstance(worker, LocalWorker) and worker.device_pool:
                    await worker.device_pool.release_device(acquired_device)
                    logger.info(f"[{op_name}] Released device {acquired_device} after baseline profile")
                elif isinstance(worker, RemoteWorker):
                    await worker.release_device(acquired_device, task_id="baseline_profile")
                    logger.info(f"[{op_name}] Released remote device {acquired_device} after baseline profile")
            except Exception as e:
                logger.warning(f"[{op_name}] Failed to release device {acquired_device}: {e}")

async def _try_read_baseline_cache(
    cache_cfg, op_name: str, cache_key: str, cache_file: str, bench_label: str,
) -> Optional[float]:
    """Try to read baseline time from cache. Returns cached time_us or None."""
    cached_entry = read_baseline_result_from_cache(
        cache_cfg, op_name=op_name, cache_key=cache_key,
    )
    cached_time_us = extract_baseline_time_us(cached_entry)
    if cached_time_us is not None:
        logger.info(
            f"[{op_name}] ✅ 命中本地 {bench_label} baseline cache: {cached_time_us:.2f}us, "
            f"cache_file={cache_file}, cache_key={cache_key}"
        )
        return cached_time_us
    if cached_entry:
        logger.warning(
            f"[{op_name}] {bench_label} baseline cache 内容无效，删除旧缓存并重新测量: "
            f"cache_file={cache_file}, cache_key={cache_key}"
        )
        delete_baseline_result_from_cache(
            cache_cfg, op_name=op_name, cache_key=cache_key,
        )
    return None


def _parse_profile_log_times(output_log: str, op_name: str) -> list:
    """Parse per-item times from profile output log."""
    times_us = []
    for line in output_log.splitlines():
        stripped = line.strip()
        if stripped and op_name in stripped:
            logger.info(f"[{op_name}] {stripped}")
            if "base time:" in stripped and "us" in stripped and "Geometric mean" not in stripped:
                try:
                    time_str = stripped.split("base time:")[1].strip().replace("us", "").strip()
                    times_us.append(float(time_str))
                except (ValueError, IndexError):
                    pass
    return times_us


def _handle_profile_result(
    result, op_name, profile_dir, times_us,
    warmup_times, run_times, backend, framework, dsl,
    cache_cfg, cache_key, cache_key_id, arch,
    bench_type, bench_label, cache_method, times_label,
) -> Optional[float]:
    """Handle profile result: validate, save, cache. Returns baseline_time_us or None."""
    if not result.get('success', False):
        error_log = result.get('log', 'Unknown error')
        logger.warning(f"[{op_name}] {bench_label} Baseline profile 失败: {error_log}")
        return None

    baseline_time_us = result.get('time_us')
    if not baseline_time_us or baseline_time_us <= 0 or baseline_time_us >= float('inf'):
        logger.warning(f"[{op_name}] {bench_label} Baseline profile 结果无效: {baseline_time_us}")
        return None

    logger.info(f"[{op_name}] ✅ {bench_label} Baseline profile 完成（几何平均）: {baseline_time_us:.2f}us")
    _save_baseline_result_json(
        profile_dir, op_name, baseline_time_us,
        times_us, warmup_times, run_times, backend,
        bench_type, times_label,
    )
    if cache_cfg.enabled and cache_cfg.cache_baseline_result and cache_key:
        write_baseline_result_to_cache(
            cache_cfg,
            op_name=op_name,
            cache_key=cache_key,
            result_data=build_baseline_cache_payload(
                base_time_us=baseline_time_us,
                warmup_times=warmup_times,
                run_times=run_times,
                method=cache_method,
                extra={
                    f"{times_label}_count": len(times_us) if times_us else 0,
                    f"{times_label}_times_us": times_us or [],
                },
            ),
            metadata={
                "framework": framework,
                "dsl": dsl,
                "cache_key_id": cache_key_id,
                "backend": backend,
                "arch": arch,
                "bench_type": bench_type,
            },
        )
    logger.info(f"[{op_name}] {bench_label} Baseline profile 脚本及结果已保存到: {profile_dir}")
    return baseline_time_us


def _build_cache_key(cache_cfg, op_name, cache_framework_code, framework, backend, arch, bench_type, warmup_times, run_times, dsl, cache_key_id):
    """Build baseline cache key and file path. Returns (cache_key, cache_file) or (None, None)."""
    if not cache_cfg.enabled or not cache_cfg.cache_baseline_result:
        return None, None
    try:
        cache_key = build_baseline_cache_key(
            op_name=op_name,
            framework_code=cache_framework_code,
            framework=framework,
            backend=backend,
            arch=arch,
            bench_type=bench_type,
            warmup_times=warmup_times,
            run_times=run_times,
            dsl=dsl,
            task_id=cache_key_id,
        )
        cache_file = get_baseline_cache_file_path(cache_cfg, op_name=op_name, cache_key=cache_key)
        return cache_key, cache_file
    except Exception as exc:
        logger.info(f"[{op_name}] baseline cache key 构建失败，跳过 cache: {exc}")
        return None, None


async def _run_cached_baseline_profile(
    op_name: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int,
    run_times: int,
    timeout: int,
    bench_type: str,
    bench_label: str,
    cache_framework_code: str,
    prepare_fn,
    cache_method: str,
    times_label: str,
) -> Optional[float]:
    """Common framework for SOL/CANN baseline profiling with cache support.

    Args:
        prepare_fn: async callable(worker, verifier, profile_dir, warmup_times, run_times, timeout)
                     that builds and executes the profile, returning result dict.
    """
    cache_cfg = load_verifier_data_cache_config(config)
    cache_key_id = get_verifier_data_cache_key_id(config, "baseline_profile")
    try:
        from akg_agents.op.verifier.kernel_verifier import KernelVerifier
        from akg_agents.core.worker.manager import get_worker_manager

        cache_key, cache_file = _build_cache_key(
            cache_cfg, op_name, cache_framework_code, framework, backend, arch,
            bench_type, warmup_times, run_times, dsl, cache_key_id,
        )
        if cache_key:
            cached = await _try_read_baseline_cache(
                cache_cfg, op_name, cache_key, cache_file, bench_label,
            )
            if cached is not None:
                return cached

        logger.info(f"[{op_name}] 🚀 开始预先 {bench_label} baseline profile（只测一次）...")

        async with AsyncExitStack() as stack:
            if cache_cfg.enabled and cache_cfg.cache_baseline_result and cache_key:
                await stack.enter_async_context(
                    verifier_data_cache_lock(
                        cache_cfg, namespace="baseline", op_name=op_name, cache_key=cache_key,
                    )
                )
                cached = await _try_read_baseline_cache(
                    cache_cfg, op_name, cache_key, cache_file, bench_label,
                )
                if cached is not None:
                    return cached

            worker = await get_worker_manager().select(backend=backend, arch=arch)
            if not worker:
                logger.warning(f"[{op_name}] 无法获取 worker，跳过预先 {bench_label} baseline profile")
                return None

            verifier = KernelVerifier(
                op_name=op_name,
                framework_code="",
                task_id="baseline_profile",
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                config=config,
                bench_type=bench_type,
                worker=worker,
            )

            profile_dir = os.path.join(
                os.path.expanduser(verifier.log_dir),
                f"{op_name}_profile_single_baseline_profile",
            )
            os.makedirs(profile_dir, exist_ok=True)

            result = await prepare_fn(worker, verifier, profile_dir, warmup_times, run_times, timeout)

            times_us = _parse_profile_log_times(result.get('log', ''), op_name)

            return _handle_profile_result(
                result, op_name, profile_dir, times_us,
                warmup_times, run_times, backend, framework, dsl,
                cache_cfg, cache_key, cache_key_id, arch,
                bench_type, bench_label, cache_method, times_label,
            )

            return None

    except TimeoutError as e:
        logger.warning(f"[{op_name}] 等待 {bench_label} baseline cache lock 超时，跳过预先 profile baseline: {e}")
        return None
    except Exception as e:
        logger.warning(f"[{op_name}] {bench_label} baseline profile 失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


async def _prepare_sol_profile(worker, verifier, profile_dir, warmup_times, run_times, timeout):
    """Prepare and execute SOL baseline profile project."""
    from akg_agents.op.verifier.sol_verifier import PROF_SOL_BASE_TEMPLATE_PATH
    from akg_agents.op.verifier.adapters.factory import get_framework_adapter, get_backend_adapter
    from akg_agents import get_project_root
    from jinja2 import Template

    config = verifier.config
    sol_problem_dir = config.get("sol_problem_dir")
    if not sol_problem_dir:
        raise ValueError("config['sol_problem_dir'] 未配置")
    sol_problem_dir = os.path.expandvars(os.path.expanduser(str(sol_problem_dir)))
    if not os.path.isdir(sol_problem_dir):
        raise FileNotFoundError(f"SOL case 目录不存在: {sol_problem_dir}")

    for file_name in ["definition.json", "workload.jsonl", "reference.py"]:
        src = os.path.join(sol_problem_dir, file_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing required SOL file: {src}")
        shutil.copy2(src, os.path.join(profile_dir, file_name))

    sol_correctness_src = os.path.join(
        get_project_root(), "op", "resources", "utils", "sol_correctness.py",
    )
    shutil.copy2(sol_correctness_src, os.path.join(profile_dir, "sol_correctness.py"))

    framework_adapter = get_framework_adapter(verifier.framework)
    backend_adapter = get_backend_adapter(verifier.backend)
    backend_adapter.setup_environment(0, verifier.arch)
    device_setup_code = verifier._prepare_code_lines(
        framework_adapter.get_device_setup_code(verifier.backend, verifier.arch, 0),
    )
    sol_execbench_src_dir = os.path.abspath(
        os.path.join(get_project_root(), "..", "..", "thirdparty", "sol-execbench", "src"),
    )

    with open(PROF_SOL_BASE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        base_template = Template(f.read())

    base_script = base_template.render(
        op_name=verifier.op_name,
        backend=verifier.backend,
        arch=verifier.arch,
        device_id=0,
        warmup_times=warmup_times,
        run_times=run_times,
        device_setup_code=device_setup_code,
        sol_execbench_src_dir=sol_execbench_src_dir,
    )

    wrapper = base_script + """

import shutil as _shutil
if os.path.exists("base_profile_result.json"):
    _shutil.copy2("base_profile_result.json", "profile_single_result.json")
"""
    script_path = os.path.join(profile_dir, f"profile_single_{verifier.op_name}.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(wrapper)

    package_data = _pack_directory(profile_dir)
    profile_settings = {
        'warmup_times': warmup_times,
        'run_times': run_times,
        'timeout': timeout,
    }
    return await worker.profile_single_task(
        package_data, "baseline_profile_profile_single", verifier.op_name, profile_settings,
    )


async def _prepare_cann_profile(worker, verifier, profile_dir, warmup_times, run_times, timeout):
    """Prepare and execute CANN baseline profile project."""
    from akg_agents.op.verifier.cann_verifier import PROF_CANN_BASE_TEMPLATE_PATH
    from akg_agents.op.verifier.adapters.factory import get_framework_adapter, get_backend_adapter
    from akg_agents import get_project_root
    from jinja2 import Template
    import yaml

    config = verifier.config
    cann_problem_dir = config.get("cann_problem_dir")
    if not cann_problem_dir:
        raise ValueError("config['cann_problem_dir'] 未配置")
    cann_problem_dir = os.path.expandvars(os.path.expanduser(str(cann_problem_dir)))
    if not os.path.isdir(cann_problem_dir):
        raise FileNotFoundError(f"CANN case 目录不存在: {cann_problem_dir}")

    for file_name in ["proto.yaml", "golden.py", "cases.yaml"]:
        src = os.path.join(cann_problem_dir, file_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing required CANN file: {src}")
        shutil.copy2(src, os.path.join(profile_dir, file_name))

    desc_src = os.path.join(cann_problem_dir, "desc.md")
    if os.path.exists(desc_src):
        shutil.copy2(desc_src, os.path.join(profile_dir, "desc.md"))

    cann_correctness_src = os.path.join(
        get_project_root(), "op", "resources", "utils", "cann_correctness.py",
    )
    shutil.copy2(cann_correctness_src, os.path.join(profile_dir, "cann_correctness.py"))

    framework_adapter = get_framework_adapter(verifier.framework)
    backend_adapter = get_backend_adapter(verifier.backend)
    backend_adapter.setup_environment(0, verifier.arch)
    device_setup_code = verifier._prepare_code_lines(
        framework_adapter.get_device_setup_code(verifier.backend, verifier.arch, 0),
    )

    proto_path = os.path.join(profile_dir, "proto.yaml")
    with open(proto_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)
    schema = proto.get("operator", {}).get("schema", "")

    cann_bench_src_dir = os.path.abspath(
        os.path.join(get_project_root(), "..", "..", "thirdparty", "cann-bench", "src"),
    )

    with open(PROF_CANN_BASE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        base_template = Template(f.read())

    base_script = base_template.render(
        op_name=verifier.op_name,
        backend=verifier.backend,
        arch=verifier.arch,
        dsl=verifier.dsl,
        device_id=0,
        warmup_times=warmup_times,
        run_times=run_times,
        device_setup_code=device_setup_code,
        schema=schema,
        cann_bench_src_dir=cann_bench_src_dir,
    )

    wrapper = base_script + """

import shutil as _shutil
if os.path.exists("base_profile_result.json"):
    _shutil.copy2("base_profile_result.json", "profile_single_result.json")
"""
    script_path = os.path.join(profile_dir, f"profile_single_{verifier.op_name}.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(wrapper)

    package_data = _pack_directory(profile_dir)
    profile_settings = {
        'warmup_times': warmup_times,
        'run_times': run_times,
        'timeout': timeout,
    }
    return await worker.profile_single_task(
        package_data, "baseline_profile_profile_single", verifier.op_name, profile_settings,
    )


async def _profile_sol_baseline(
    op_name: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int,
    run_times: int,
    timeout: int,
) -> Optional[float]:
    """SOL-ExecBench baseline profiling."""
    sol_problem_dir_for_cache = config.get("sol_problem_dir", "")
    if not sol_problem_dir_for_cache:
        logger.warning(f"[{op_name}] config['sol_problem_dir'] 未配置，跳过预先 SOL baseline profile")
        return None
    sol_cache_identity = build_sol_problem_cache_identity(sol_problem_dir_for_cache)
    return await _run_cached_baseline_profile(
        op_name, dsl, framework, backend, arch, config,
        warmup_times, run_times, timeout,
        bench_type="sol",
        bench_label="SOL",
        cache_framework_code=sol_cache_identity,
        prepare_fn=_prepare_sol_profile,
        cache_method="sol_profile_single_task",
        times_label="workload",
    )


async def _profile_cann_baseline(
    op_name: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int,
    run_times: int,
    timeout: int,
) -> Optional[float]:
    """CANN-Bench baseline profiling."""
    return await _run_cached_baseline_profile(
        op_name, dsl, framework, backend, arch, config,
        warmup_times, run_times, timeout,
        bench_type="cann",
        bench_label="CANN",
        cache_framework_code=config.get("cann_problem_dir", ""),
        prepare_fn=_prepare_cann_profile,
        cache_method="cann_profile_single_task",
        times_label="case",
    )


def _save_baseline_result_json(
    profile_dir: str,
    op_name: str,
    baseline_time_us: float,
    times_us: list,
    warmup_times: int,
    run_times: int,
    backend: str,
    bench_type: str,
    times_label: str,
) -> None:
    """将 baseline profile 结果写为 base_profile_result.json，保存到 profile_dir"""
    try:
        method_prefix = f"{bench_type}_base"
        method = f"{method_prefix}_profiler_npu" if backend == "ascend" else f"{method_prefix}_loop_timer"
        if bench_type == "sol":
            method = "sol_base_profiler_npu" if backend == "ascend" else "sol_base_do_bench"
        result_data = {
            "execution_time_ms": baseline_time_us / 1000.0,
            "execution_time_us": baseline_time_us,
            "avg_time_us": baseline_time_us,
            "warmup_times": warmup_times,
            "run_times": run_times,
            f"{times_label}_count": len(times_us) if times_us else 0,
            f"{times_label}_times_us": times_us or [],
            "method": method,
            "bench_type": bench_type,
        }
        result_file = os.path.join(profile_dir, "base_profile_result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
        logger.info(f"[{op_name}] base_profile_result.json 已写入: {result_file}")
    except Exception as e:
        logger.warning(f"[{op_name}] 写入 base_profile_result.json 失败: {e}")


def _pack_directory(dir_path: str) -> bytes:
    """将目录打包为 tar 字节流"""
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar_file:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dir_path)
                tar_file.add(file_path, arcname=arcname)
    return tar_buffer.getvalue()


def _save_baseline_profile_scripts(verifier, op_name: str, task_desc: str,
                                   warmup_times: int, run_times: int,
                                   device_id: int = 0) -> None:
    """
    保存 KernelBench baseline profile 脚本到 log 目录
    """
    try:
        baseline_dir = os.path.join(
            os.path.expanduser(verifier.log_dir),
            f"{op_name}_baseline_profile"
        )
        os.makedirs(baseline_dir, exist_ok=True)

        framework_file = verifier._materialize_framework_bundle(
            baseline_dir, task_desc)

        script_file = os.path.join(baseline_dir, f"profile_baseline_{op_name}.py")
        verifier.gen_profile_single_task_file(script_file, device_id=device_id,
                                              warmup_times=warmup_times,
                                              run_times=run_times)

        logger.info(f"[{op_name}] Baseline profile 脚本已保存到: {baseline_dir}")

    except Exception as e:
        logger.warning(f"[{op_name}] 保存 baseline profile 脚本失败: {e}")


def set_baseline_in_config(config: Dict[str, Any], baseline_time_us: float) -> None:
    """
    将缓存的 baseline 时间设置到 config 中

    Args:
        config: 配置字典
        baseline_time_us: baseline 时间（微秒）
    """
    # 只有当 baseline_time_us 是有效值时才设置
    if baseline_time_us is None or baseline_time_us <= 0 or baseline_time_us >= float('inf'):
        return

    if 'profile_settings' not in config:
        config['profile_settings'] = {}

    config['profile_settings']['override_base_time_us'] = baseline_time_us
    config['profile_settings']['skip_base_profile'] = True
