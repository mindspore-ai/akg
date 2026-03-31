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
避免所有任务重复测量。
"""

import os
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
    
    使用 KernelVerifier.profile_single_task() 只测量 task_desc（框架实现）的性能，
    然后缓存结果，后续所有任务都跳过 baseline profile。
    
    Args:
        op_name: 算子名称
        task_desc: 任务描述（框架代码）
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
    try:
        from akg_agents.op.verifier.kernel_verifier import KernelVerifier
        from akg_agents.core.worker.manager import get_worker_manager
        
        logger.info(f"[{op_name}] 🚀 开始预先 profile baseline（只测一次）...")
        
        # 获取 worker
        worker = await get_worker_manager().select(backend=backend, arch=arch)
        if not worker:
            logger.warning(f"[{op_name}] 无法获取 worker，跳过预先 profile baseline")
            return None
        
        # 创建临时 KernelVerifier（只用于 profile baseline）
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
        
        # 使用 profile_single_task 只测量 task_desc 的性能
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
                
                # 【重要】保存 baseline profile 脚本到 log 目录，方便复现
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


def _save_baseline_profile_scripts(verifier, op_name: str, task_desc: str, 
                                   warmup_times: int, run_times: int) -> None:
    """
    保存 baseline profile 脚本到 log 目录
    
    目录结构：{log_dir}/{op_name}_baseline_profile/
    - framework_model.py: 框架代码
    - profile_baseline_{op_name}.py: profile 脚本
    
    Args:
        verifier: KernelVerifier 实例
        op_name: 算子名称
        task_desc: 任务描述（框架代码）
        warmup_times: 预热次数
        run_times: 运行次数
    """
    try:
        # 创建 baseline profile 目录
        baseline_dir = os.path.join(
            os.path.expanduser(verifier.log_dir), 
            f"{op_name}_baseline_profile"
        )
        os.makedirs(baseline_dir, exist_ok=True)
        
        # 1. 保存 framework_model.py
        framework_file = os.path.join(baseline_dir, "framework_model.py")
        with open(framework_file, "w", encoding="utf-8") as f:
            f.write(task_desc)
        
        # 2. 生成并保存 profile 脚本
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
