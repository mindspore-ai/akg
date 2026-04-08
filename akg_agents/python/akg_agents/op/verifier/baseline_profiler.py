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
避免所有任务重复测量。支持 KernelBench 和 SOL-ExecBench 两种 bench_type。
"""

import os
import io
import json
import shutil
import tarfile
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


async def profile_baseline_once(
    op_name: str,
    task_desc: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int = 5,
    run_times: int = 50,
    timeout: int = 300
) -> Optional[float]:
    """
    预先 profile baseline 一次（只测量框架实现的性能）
    
    根据 config["bench_type"] 自动选择 KernelBench 或 SOL-ExecBench 的 baseline profiling 流程。
    
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
    try:
        from akg_agents.op.verifier.kernel_verifier import KernelVerifier
        from akg_agents.core.worker.manager import get_worker_manager
        
        logger.info(f"[{op_name}] 🚀 开始预先 profile baseline（只测一次）...")
        
        worker = await get_worker_manager().select(backend=backend, arch=arch)
        if not worker:
            logger.warning(f"[{op_name}] 无法获取 worker，跳过预先 profile baseline")
            return None
        
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
            device_id=0
        )
        
        if result.get('success', False):
            baseline_time_us = result.get('time_us')
            if baseline_time_us and baseline_time_us > 0 and baseline_time_us < float('inf'):
                logger.info(f"[{op_name}] ✅ Baseline profile 完成: {baseline_time_us:.2f}us")
                _save_baseline_profile_scripts(verifier, op_name, task_desc, warmup_times, run_times)
                return baseline_time_us
            else:
                logger.warning(f"[{op_name}] Baseline profile 结果无效: {baseline_time_us}")
        else:
            error_log = result.get('log', 'Unknown error')
            logger.warning(f"[{op_name}] Baseline profile 失败: {error_log}")
        
        return None
    
    except Exception as e:
        logger.warning(f"[{op_name}] 预先 profile baseline 失败: {e}")
        return None


async def _profile_sol_baseline(
    op_name: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int,
    run_times: int,
    timeout: int
) -> Optional[float]:
    """
    SOL-ExecBench 模式的 baseline profiling
    
    使用 SOL base 模板渲染脚本，测量 reference.run 在所有 workload 上的几何平均时间。
    """
    try:
        from akg_agents.op.verifier.kernel_verifier import KernelVerifier
        from akg_agents.op.verifier.sol_verifier import PROF_SOL_BASE_TEMPLATE_PATH
        from akg_agents.op.verifier.adapters.factory import get_framework_adapter, get_backend_adapter
        from akg_agents.core.worker.manager import get_worker_manager
        from akg_agents import get_project_root
        from jinja2 import Template
        
        sol_problem_dir = config.get("sol_problem_dir")
        if not sol_problem_dir or not os.path.exists(sol_problem_dir):
            logger.warning(f"[{op_name}] sol_problem_dir 不存在，跳过 SOL baseline profile")
            return None
        
        logger.info(f"[{op_name}] 🚀 开始预先 SOL baseline profile（只测一次）...")
        
        worker = await get_worker_manager().select(backend=backend, arch=arch)
        if not worker:
            logger.warning(f"[{op_name}] 无法获取 worker，跳过预先 SOL baseline profile")
            return None
        
        # 创建临时 verifier（只用来获取 log_dir）
        verifier = KernelVerifier(
            op_name=op_name,
            framework_code="",
            task_id="baseline_profile",
            framework=framework,
            dsl=dsl,
            backend=backend,
            arch=arch,
            config=config,
            bench_type="sol",
            worker=worker
        )
        
        # 构建 SOL baseline profile 目录
        profile_dir = os.path.join(
            os.path.expanduser(verifier.log_dir),
            f"{op_name}_profile_single_baseline_profile"
        )
        os.makedirs(profile_dir, exist_ok=True)
        
        # 1. 拷贝 SOL 核心文件
        for file_name in ["definition.json", "workload.jsonl", "reference.py"]:
            src = os.path.join(sol_problem_dir, file_name)
            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing required SOL file: {src}")
            shutil.copy2(src, os.path.join(profile_dir, file_name))
        
        # 2. 拷贝 sol_correctness.py
        sol_correctness_src = os.path.join(
            get_project_root(), "op", "resources", "utils", "sol_correctness.py"
        )
        shutil.copy2(sol_correctness_src, os.path.join(profile_dir, "sol_correctness.py"))
        
        # 3. 渲染 SOL base 模板
        framework_adapter = get_framework_adapter(framework)
        backend_adapter = get_backend_adapter(backend)
        backend_adapter.setup_environment(0, arch)
        device_setup_code = verifier._prepare_code_lines(
            framework_adapter.get_device_setup_code(backend, arch, 0)
        )
        sol_execbench_src_dir = os.path.abspath(
            os.path.join(get_project_root(), "..", "..", "thirdparty", "sol-execbench", "src")
        )
        
        with open(PROF_SOL_BASE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            base_template = Template(f.read())
        
        base_script = base_template.render(
            op_name=op_name,
            backend=backend,
            arch=arch,
            device_id=0,
            warmup_times=warmup_times,
            run_times=run_times,
            device_setup_code=device_setup_code,
            sol_execbench_src_dir=sol_execbench_src_dir,
        )
        
        # 写为 profile_single_{op_name}.py 以兼容 worker.profile_single_task 的查找逻辑
        script_name = f"profile_single_{op_name}.py"
        script_path = os.path.join(profile_dir, script_name)
        
        # 包装：运行 SOL base profile 后将结果复制为 profile_single_result.json
        wrapper = base_script + """

import shutil as _shutil
if os.path.exists("base_profile_result.json"):
    _shutil.copy2("base_profile_result.json", "profile_single_result.json")
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(wrapper)
        
        # 4. 打包目录
        package_data = _pack_directory(profile_dir)
        
        # 5. 使用 worker.profile_single_task 执行
        profile_settings = {
            'warmup_times': warmup_times,
            'run_times': run_times,
            'timeout': timeout
        }
        result = await worker.profile_single_task(
            package_data, "baseline_profile_profile_single", op_name, profile_settings
        )
        
        # 输出每个 workload 的详细时间日志，同时解析各 workload 时间
        output_log = result.get('log', '')
        workload_times_us = []
        for line in output_log.splitlines():
            stripped = line.strip()
            if stripped and op_name in stripped:
                logger.info(f"[{op_name}] {stripped}")
                # 解析 "Workload N/M base time: 1234.5678 us"
                if "base time:" in stripped and "us" in stripped:
                    try:
                        time_str = stripped.split("base time:")[1].strip().replace("us", "").strip()
                        workload_times_us.append(float(time_str))
                    except (ValueError, IndexError):
                        pass
        
        if result.get('success', False):
            baseline_time_us = result.get('time_us')
            if baseline_time_us and baseline_time_us > 0 and baseline_time_us < float('inf'):
                logger.info(f"[{op_name}] ✅ SOL Baseline profile 完成（几何平均）: {baseline_time_us:.2f}us")
                
                # 将结果 JSON 写回 profile_dir，方便后期查看
                _save_sol_baseline_result_json(
                    profile_dir, op_name, baseline_time_us,
                    workload_times_us, warmup_times, run_times, backend
                )
                
                logger.info(f"[{op_name}] SOL Baseline profile 脚本及结果已保存到: {profile_dir}")
                return baseline_time_us
            else:
                logger.warning(f"[{op_name}] SOL Baseline profile 结果无效: {baseline_time_us}")
        else:
            error_log = result.get('log', 'Unknown error')
            logger.warning(f"[{op_name}] SOL Baseline profile 失败: {error_log}")
        
        return None
    
    except Exception as e:
        logger.warning(f"[{op_name}] 预先 SOL baseline profile 失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def _save_sol_baseline_result_json(
    profile_dir: str,
    op_name: str,
    baseline_time_us: float,
    workload_times_us: list,
    warmup_times: int,
    run_times: int,
    backend: str
) -> None:
    """将 SOL baseline profile 结果写为 base_profile_result.json，保存到 profile_dir"""
    try:
        method = "sol_base_profiler_npu" if backend == "ascend" else "sol_base_do_bench"
        result_data = {
            "execution_time_ms": baseline_time_us / 1000.0,
            "execution_time_us": baseline_time_us,
            "avg_time_us": baseline_time_us,
            "warmup_times": warmup_times,
            "run_times": run_times,
            "workload_count": len(workload_times_us) if workload_times_us else 0,
            "workload_times_us": workload_times_us or [],
            "method": method,
            "bench_type": "sol",
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
                                   warmup_times: int, run_times: int) -> None:
    """
    保存 KernelBench baseline profile 脚本到 log 目录
    """
    try:
        baseline_dir = os.path.join(
            os.path.expanduser(verifier.log_dir), 
            f"{op_name}_baseline_profile"
        )
        os.makedirs(baseline_dir, exist_ok=True)
        
        framework_file = os.path.join(baseline_dir, "framework_model.py")
        with open(framework_file, "w", encoding="utf-8") as f:
            f.write(task_desc)
        
        script_file = os.path.join(baseline_dir, f"profile_baseline_{op_name}.py")
        verifier.gen_profile_single_task_file(script_file, device_id=0, 
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
