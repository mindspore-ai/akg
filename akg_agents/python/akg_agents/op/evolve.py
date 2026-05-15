# Copyright 2025 Huawei Technologies Co., Ltd
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

import os
import logging
from typing import Dict, Any
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.core.async_pool.device_pool import DevicePool
from akg_agents.core.worker.manager import get_worker_manager
from akg_agents.core_v2.config.settings import get_akg_env_var

# 导入处理器和配置
from akg_agents.op.utils.evolve.evolution_processors import (
    create_runtime_config,
    InitializationProcessor,
    TaskCreationProcessor,
    ResultProcessor
)
from akg_agents.op.utils.evolve.evolution_utils import pretty_print_results


logger = logging.getLogger(__name__)


def _send_evolve_message(session_id: str, action: str, data: dict) -> None:
    """发送 evolve 进度消息到 CLI"""
    if not session_id:
        return
    try:
        from akg_agents.cli.runtime.message_sender import send_message
        from akg_agents.cli.messages import PanelDataMessage
        send_message(session_id, PanelDataMessage(action=action, data=data))
    except Exception as e:
        logger.warning(f"Failed to send evolve message ({action}): {e}")


async def evolve(
    op_name: str,
    task_desc: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: dict,
    task_pool: TaskPool,
    device_pool: DevicePool = None,
    max_rounds: int = 1,
    parallel_num: int = 1,
    # 岛屿模型参数（可选）
    num_islands: int = 1,  # 设置为1或更小值可禁用岛屿模型
    migration_interval: int = 0,  # 设置为0可禁用迁移
    elite_size: int = 0,  # 设置为0可禁用精英机制
    parent_selection_prob: float = 0.5,  # 父代选择概率
    handwrite_decay_rate: float = 2.0,  # 手写建议采样的权重衰减率
) -> Dict[str, Any]:
    """统一的进化式算子生成主函数，支持基础模式和岛屿模型

    Args:
        op_name: 算子名称
        task_desc: 任务描述
        dsl: 实现类型（如"triton_cuda", "triton_ascend", "swft"）
        framework: 框架名称（如"mindspore", "torch", "numpy"）
        backend: 后端名称（如"ascend", "cuda"）
        arch: 架构名称（如"ascend910b4", "a100"）
        config: 配置字典
        device_pool: 设备池（可选，用于向后兼容）
        task_pool: 任务池
        max_rounds: 最大进化轮数
        parallel_num: 每轮并行任务数
        num_islands: 岛屿数量（设置为1或更小值可禁用岛屿模型）
        migration_interval: 迁移间隔（设置为0可禁用迁移）
        elite_size: 精英池大小（设置为0可禁用精英机制）
        parent_selection_prob: 父代来源概率
        handwrite_decay_rate: 手写建议采样的权重衰减率，值越大衰减越快（默认2.0）

    Returns:
        进化结果字典
    """
    # ========== 1. 创建运行时配置 ==========
    runtime_config = create_runtime_config(locals())

    # 兼容逻辑：如果有 device_pool，自动注册到全局 WorkerManager
    temp_worker = None
    if device_pool:
        import warnings
        warnings.warn(
            "⚠️  [DEPRECATED] 直接传递 device_pool 给 evolve() 是旧写法，将在未来版本移除。\n"
            "推荐的新写法：\n"
            "  1. 注册 LocalWorker 到 WorkerManager（一行代码）：\n"
            "     from akg_agents.core.worker.manager import register_local_worker\n"
            "     \n"
            "     await register_local_worker([0, 1, 2, 3], backend='cuda', arch='a100')\n"
            "  2. 调用 evolve 时不传 device_pool：\n"
            "     await evolve(\n"
            "         ...,\n"
            "         device_pool=None,  # 不再传递\n"
            "         ...\n"
            "     )\n"
            "参考示例：examples/run_torch_evolve_triton.py",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning("⚠️  检测到使用旧的 device_pool 参数，请参考日志中的警告信息迁移到新写法")
        
        from akg_agents.core.worker.local_worker import LocalWorker
        
        # 创建临时 LocalWorker 并注册
        temp_worker = LocalWorker(device_pool, backend=backend)
        capacity = len(device_pool.devices) if hasattr(device_pool, 'devices') else 1
        await get_worker_manager().register(
            temp_worker, backend=backend, arch=arch, 
            capacity=capacity
        )
    
    manager = get_worker_manager()
    if not await manager.has_worker(backend=backend, arch=arch):
        raise RuntimeError(
            f"未检测到可用的 Worker。请先注册 Worker 后再调用 evolve：\n"
            f"  from akg_agents.core.worker.manager import register_worker\n"
            f"  await register_worker(backend='{backend}', arch='{arch}', device_ids=[0])\n"
            f"或设置环境变量 AKG_AGENTS_WORKER_URL 指向远程 Worker 服务。"
        )
    
    # ========== 2. 初始化阶段 ==========
    init_processor = InitializationProcessor(runtime_config)
    init_data = await init_processor.initialize()
    
    # ========== 3. 创建处理器 ==========
    task_processor = TaskCreationProcessor(runtime_config, init_data)
    result_processor = ResultProcessor(runtime_config, init_data)
    
    # 用于跟踪当前轮次的实现
    round_implementations = []
    
    # 【优化】在开始进化前，先单独 profile baseline 一次
    from akg_agents.op.verifier.baseline_profiler import profile_baseline_once
    baseline_time_us = await profile_baseline_once(
        op_name, task_desc, dsl, framework, backend, arch, config
    )
    if baseline_time_us:
        logger.info(f"[{op_name}] 后续所有任务将跳过 baseline profile")

    # 获取 session_id 用于进度消息
    session_id = str(config.get("session_id") or "").strip()
    # 计算实际的总任务数（考虑岛屿模式）
    if num_islands > 1 and elite_size > 0:
        tasks_per_island = max(1, parallel_num // num_islands)
        actual_tasks_per_round = num_islands * tasks_per_island
    else:
        actual_tasks_per_round = parallel_num
    total_tasks_all_rounds = max_rounds * actual_tasks_per_round
    # 发送 evolve 开始消息（进入静默模式）
    _send_evolve_message(session_id, "evolve_start", {
        "op_name": op_name,
        "max_rounds": max_rounds,
        "parallel_num": parallel_num,
        "total_tasks": total_tasks_all_rounds,
    })

    # 实时进度计数器（用 dict 使闭包可修改）
    counters = {
        "total_completed": 0,
        "total_success": 0,
        "total_failed": 0,
        "round_running": 0,
        "current_round": 0,
    }

    def _send_current_progress(phase: str = "running"):
        _send_evolve_message(session_id, "evolve_progress", {
            "op_name": op_name,
            "current_round": counters["current_round"],
            "max_rounds": max_rounds,
            "parallel_num": parallel_num,
            "total_tasks": total_tasks_all_rounds,
            "total_completed": counters["total_completed"],
            "total_success": counters["total_success"],
            "total_failed": counters["total_failed"],
            "running_count": counters["round_running"],
            "phase": phase,
        })

    # 包装 task_pool.create_task，在每个 task 完成时实时更新计数
    _original_create_task = task_pool.create_task

    def _tracked_create_task(coro_func, *args, task_name=None, **kwargs):
        async def _wrapped(*a, **kw):
            result = await coro_func(*a, **kw)
            # LangGraphTask.run() 返回 (op_name, success, final_state)
            success = isinstance(result, tuple) and len(result) >= 2 and result[1]
            counters["round_running"] -= 1
            counters["total_completed"] += 1
            if success:
                counters["total_success"] += 1
            else:
                counters["total_failed"] += 1
            _send_current_progress()
            return result
        counters["round_running"] += 1
        return _original_create_task(_wrapped, *args, task_name=task_name, **kwargs)

    task_pool.create_task = _tracked_create_task

    # ========== 4. 进化主循环 ==========
    for round_idx in range(1, max_rounds + 1):
        counters["current_round"] = round_idx
        counters["round_running"] = 0
        
        # 【优化】如果已有缓存的 baseline，设置到 config 中
        if baseline_time_us is not None and baseline_time_us > 0 and baseline_time_us < float('inf'):
            from akg_agents.op.verifier.baseline_profiler import set_baseline_in_config
            # 注意：需要同时更新 runtime_config.config，因为 task_processor 使用的是 runtime_config
            set_baseline_in_config(config, baseline_time_us)
            set_baseline_in_config(runtime_config.config, baseline_time_us)

        # 4.1 创建任务（_tracked_create_task 会自动增加 round_running）
        tasks, task_mapping = task_processor.create_tasks_for_round(
            round_idx,
            device_pool,
            task_pool,
            round_implementations
        )

        # 发送轮次开始消息（在 create 之后，此时 round_running 已正确）
        _send_current_progress()

        # 4.2 执行任务（每个 task 完成时 _wrapped 会实时发送进度更新）
        results = await task_pool.wait_all()
        task_pool.tasks.clear()

        # 4.3 处理结果（临时恢复原始 create_task，避免 sketch 任务被计数）
        task_pool.create_task = _original_create_task
        round_data = await result_processor.process_results(
            results,
            round_idx,
            task_pool,
            task_mapping
        )
        task_pool.create_task = _tracked_create_task  # 恢复包装版本
        round_implementations = round_data['round_implementations']

        # 发送轮次完成的进度更新
        _send_current_progress(phase="round_done")
        
        # 4.5 打印轮次结果
        round_result = round_data['round_result']
        pretty_print_results([
            (impl['op_name'], True) for impl in round_implementations
        ] + [
            (f"failed_task_{i}", False) 
            for i in range(round_result['total_tasks'] - round_result['successful_tasks'])
        ])

    # 恢复原始 create_task
    task_pool.create_task = _original_create_task

    # 发送 evolve 结束消息（退出静默模式）
    _send_evolve_message(session_id, "evolve_end", {
        "op_name": op_name,
        "total_success": counters["total_success"],
        "total_failed": counters["total_failed"],
    })

    # ========== 5. 构建最终结果 ==========
    # 按性能排序最佳实现（gen_time越小越好）
    init_data['best_implementations'].sort(
        key=lambda x: x.get('profile', {}).get('gen_time', float('inf'))
    )

    # 计算最终成功率
    final_success_rate = (
        init_data['total_successful_tasks'] / init_data['total_tasks']
        if init_data['total_tasks'] > 0 else 0.0
    )

    # 从config中获取log_dir和task_folder
    log_dir = config.get('log_dir', '')
    task_folder = os.path.basename(log_dir) if log_dir else ''

    # 构建返回结果
    evolution_result = {
        'op_name': op_name,
        'total_rounds': max_rounds,
        'total_tasks': init_data['total_tasks'],
        'successful_tasks': init_data['total_successful_tasks'],
        'final_success_rate': final_success_rate,
        'best_success_rate': init_data['best_success_rate'],
        'implementation_type': dsl,
        'framework': framework,
        'backend': backend,
        'architecture': arch,
        'best_implementations': init_data['best_implementations'][:5],  # 只返回前5个最佳实现
        'round_results': init_data['round_results'],
        'storage_dir': runtime_config.storage_dir,  # 添加存储目录信息
        'task_folder': task_folder,  # 添加Task文件夹名
        'log_dir': str(log_dir),  # 添加完整log_dir路径
    }

    # 如果使用了岛屿模型，添加岛屿信息
    if runtime_config.use_islands:
        evolution_result['island_info'] = {
            'num_islands': num_islands,
            'migration_interval': migration_interval,
            'elite_size': elite_size,
            'parent_selection_prob': parent_selection_prob
        }

    logger.info(f"Evolution completed for {op_name}")
    logger.info(f"Total tasks: {init_data['total_tasks']}, Successful: {init_data['total_successful_tasks']}")
    logger.info(f"Final success rate: {final_success_rate:.2%}")
    logger.info(f"Results stored in: {runtime_config.storage_dir}")

    return evolution_result

