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
import random
from functools import partial
from typing import List, Dict, Any, Tuple
from pathlib import Path
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.sketch import Sketch
from ai_kernel_generator.utils.collector import get_collector

# 导入工具函数
from ai_kernel_generator.utils.evolve_utils import (
    generate_unique_id,
    pretty_print_results,
    load_meta_prompts,
    save_implementation,
    load_best_implementations,
    sample_inspirations,
    migrate_elites,
    select_parent_from_elite
)


os.environ['AIKG_DATA_COLLECT'] = 'on'
logger = logging.getLogger(__name__)


async def evolve(
    op_name: str,
    task_desc: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: dict,
    device_pool: DevicePool,
    task_pool: TaskPool,
    max_rounds: int = 1,
    parallel_num: int = 1,
    # 岛屿模型参数（可选）
    num_islands: int = 1,  # 设置为1或更小值可禁用岛屿模型
    migration_interval: int = 0,  # 设置为0可禁用迁移
    elite_size: int = 0,  # 设置为0可禁用精英机制
    parent_selection_prob: float = 0.5,  # 父代选择概率
) -> Dict[str, Any]:
    """统一的进化式算子生成主函数，支持基础模式和岛屿模型

    Args:
        op_name: 算子名称
        task_desc: 任务描述
        dsl: 实现类型（如"triton", "swft"）
        framework: 框架名称（如"mindspore", "torch", "numpy"）
        backend: 后端名称（如"ascend", "cuda"）
        arch: 架构名称（如"ascend910b4", "a100"）
        config: 配置字典
        device_pool: 设备池
        task_pool: 任务池
        max_rounds: 最大进化轮数
        parallel_num: 每轮并行任务数
        num_islands: 岛屿数量（设置为1或更小值可禁用岛屿模型）
        migration_interval: 迁移间隔（设置为0可禁用迁移）
        elite_size: 精英池大小（设置为0可禁用精英机制）
        parent_selection_prob: 父代来源概率

    Returns:
        进化结果字典
    """
    # 确定是否启用岛屿模型
    use_islands = num_islands > 1 and elite_size > 0

    logger.info(f"Starting evolve process for {op_name}")
    logger.info(f"Configuration: {dsl} on {backend}/{arch} using {framework}")
    if use_islands:
        logger.info(f"Islands: {num_islands}, Migration interval: {migration_interval}, Elite size: {elite_size}")
        logger.info(f"Parent selection probability: {parent_selection_prob}")
    else:
        logger.info("Island model: Disabled (simple evolution mode)")

    # 本地存储路径配置
    import uuid
    random_hash = uuid.uuid4().hex[:8]
    storage_dir = os.path.expanduser(f"~/aikg_evolve/{op_name}_{dsl}_{framework}_{backend}_{arch}/{random_hash}/")
    os.makedirs(storage_dir, exist_ok=True)

    all_results = []
    best_success_rate = 0.0
    round_results = []
    best_implementations = []
    total_tasks = 0
    total_successful_tasks = 0

    # 初始化岛屿（如果启用岛屿模型）
    if use_islands:
        tasks_per_island = max(1, parallel_num // num_islands)
        islands_storage_dirs = []
        for i in range(num_islands):
            island_storage_dir = os.path.join(storage_dir, f"island_{i}")
            os.makedirs(island_storage_dir, exist_ok=True)
            islands_storage_dirs.append(island_storage_dir)

        # 每个岛屿的历史实现
        island_impls = [[] for _ in range(num_islands)]
        # 精英库
        elite_pool = []
        # 当前岛屿索引（用于某些操作的偏好选择）
        current_island = 0
        # 岛屿切换计数器
        current_island_counter = 0
        # 每处理多少任务后切换岛屿
        tasks_per_island_switch = max(1, tasks_per_island)
    else:
        # 简单模式不需要岛屿相关变量
        tasks_per_island = parallel_num
        island_storage_dir = storage_dir

    for round_idx in range(1, max_rounds + 1):
        logger.info(f"Evolve round {round_idx}/{max_rounds} started")

        # 岛屿模型：每隔migration_interval轮进行迁移
        if use_islands and round_idx > 1 and migration_interval > 0 and round_idx % migration_interval == 1 and num_islands > 1:
            logger.info("Performing migration between islands")
            island_impls = migrate_elites(island_impls, elite_size)

        # 为所有岛屿生成任务（确保所有岛屿在每轮都参与）
        if use_islands:
            island_inspirations = [[] for _ in range(num_islands)]
            island_meta_prompts = [[] for _ in range(num_islands)]
        else:
            inspirations: list = list()
            meta_prompts = []

        if round_idx == 1:
            # 第一轮：为所有岛屿初始化空的灵感列表
            if use_islands:
                # 岛屿模型
                for island_idx in range(num_islands):
                    island_inspirations[island_idx] = [[] for _ in range(tasks_per_island)]
                    island_meta_prompts[island_idx] = load_meta_prompts(dsl, tasks_per_island)
            else:
                # 简单模式
                inspirations = [[] for _ in range(parallel_num)]
                # load meta-prompt
                meta_prompts = load_meta_prompts(dsl, parallel_num)
        else:
            # 后续轮次：为所有岛屿生成灵感
            if use_islands:
                # 岛屿模型
                for island_idx in range(num_islands):
                    island_inspirations[island_idx] = []
                    for pid in range(tasks_per_island):
                        # 根据概率p选择父代来源
                        if random.random() < parent_selection_prob:
                            # 在当前岛屿中随机选择父代（保持岛屿隔离）
                            parent_island_idx = island_idx
                            if num_islands == 1:
                                stored_implementations = load_best_implementations(island_storage_dir)
                            else:
                                stored_implementations = load_best_implementations(islands_storage_dirs[parent_island_idx])
                            # 从当前岛屿随机选择一个作为父代
                            parent_implementation = random.choice(stored_implementations) if stored_implementations else None
                        else:
                            # 在精英池中选择父代（直接返回具体的精英个体）
                            parent_implementation, parent_island_idx = select_parent_from_elite(island_idx, elite_pool)
                            # 加载父代所在岛屿的所有实现，用于采样其他灵感
                            if num_islands == 1:
                                stored_implementations = load_best_implementations(island_storage_dir)
                            else:
                                stored_implementations = load_best_implementations(islands_storage_dirs[parent_island_idx])
                            # 如果精英池返回了None（精英池为空），则从当前岛屿随机选一个
                            if parent_implementation is None and stored_implementations:
                                parent_implementation = random.choice(stored_implementations)

                        # 使用分层采样策略来增加多样性，排除当前轮次已生成的实现和父代实现
                        current_round_implementations = [
                            impl for impl in round_implementations if impl.get('round') == round_idx]
                        # 合并当前轮次实现和父代实现以避免重复选择
                        all_excluded_implementations = current_round_implementations
                        if parent_implementation:
                            all_excluded_implementations.append(parent_implementation)

                        sampled = sample_inspirations(stored_implementations, sample_num=min(
                            len(stored_implementations), 3), use_tiered_sampling=True, parent_implementations=all_excluded_implementations)
                        
                        # 将父代加入灵感列表
                        if parent_implementation:
                            parent_inspiration = {
                                'id': parent_implementation.get('id'),
                                'sketch': parent_implementation.get('sketch', ''),
                                'impl_code': parent_implementation.get('impl_code', ''),
                                'profile': parent_implementation.get('profile', (float('inf'), 0.0, 0.0)),
                                'strategy_mode': 'evolution',
                                'is_parent': True  # 标记为父代
                            }
                            sampled.insert(0, parent_inspiration)  # 父代放在第一位
                        
                        island_inspirations[island_idx].append(sampled)

                    island_meta_prompts[island_idx] = load_meta_prompts(dsl, tasks_per_island)
            else:
                # 简单模式：从本地存储加载历史最佳实现作为inspiration
                stored_implementations = load_best_implementations(storage_dir)

                inspirations = []
                for pid in range(parallel_num):
                    # 使用分层采样策略，每个任务采样3个不同层级的inspiration
                    sampled = sample_inspirations(stored_implementations, sample_num=min(len(stored_implementations), 3), use_tiered_sampling=False)
                    inspirations.append(sampled)

                meta_prompts = load_meta_prompts(dsl, parallel_num)

        # 创建所有岛屿的任务
        all_tasks = []
        if use_islands:
            task_mapping = []  # 记录任务到岛屿的映射

            # 为所有岛屿创建任务
            for island_idx in range(num_islands):
                # 确保island_inspirations[island_idx]有足够的元素
                while len(island_inspirations[island_idx]) < tasks_per_island:
                    island_inspirations[island_idx].append([])

                # 确保island_meta_prompts[island_idx]有足够的元素
                while len(island_meta_prompts[island_idx]) < tasks_per_island:
                    island_meta_prompts[island_idx].append("")

                for pid in range(tasks_per_island):
                    task_id = f"{round_idx}_{island_idx}_{pid}"

                    task = Task(
                        op_name=op_name,
                        task_desc=task_desc,
                        task_id=task_id,
                        backend=backend,
                        arch=arch,
                        dsl=dsl,
                        config=config,
                        device_pool=device_pool,
                        framework=framework,
                        task_type="profile",
                        workflow="default_workflow",
                        inspirations=island_inspirations[island_idx][pid],
                        meta_prompts=island_meta_prompts[island_idx][pid] if island_meta_prompts[island_idx] else None,
                    )

                    task_pool.create_task(partial(task.run,))
                    all_tasks.append(task)
                    task_mapping.append(island_idx)
        else:
            # 为简单模式创建任务
            for pid in range(parallel_num):
                task_id = f"{round_idx}_{pid}"

                task = Task(
                    op_name=op_name,
                    task_desc=task_desc,
                    task_id=task_id,
                    backend=backend,
                    arch=arch,
                    dsl=dsl,
                    config=config,
                    device_pool=device_pool,
                    framework=framework,
                    task_type="profile",
                    workflow="default_workflow",
                    inspirations=inspirations[pid],
                    meta_prompts=meta_prompts[pid] if meta_prompts else None,
                )

                task_pool.create_task(partial(task.run,))

        results = await task_pool.wait_all()
        task_pool.tasks.clear()

        # 处理所有岛屿的结果
        round_implementations = []

        if use_islands:
            # 按岛屿分组结果
            island_results = [[] for _ in range(num_islands)]
            for i, result in enumerate(results):
                island_idx = task_mapping[i]
                island_results[island_idx].append(result)

            # 处理每个岛屿的结果
            for island_idx in range(num_islands):
                current_island_results = island_results[island_idx]

                # 统计当前岛屿结果
                island_success_count = 0
                island_impls_list = []

                # 创建sketch agent（复用config）
                sketch_agent = Sketch(
                    op_name=op_name,
                    task_desc=task_desc,
                    dsl=dsl,
                    backend=backend,
                    arch=arch,
                    config=config
                )

                # 收集成功任务信息
                successful_impls = []

                for task_op_name, success, task_info in current_island_results:
                    total_tasks += 1

                    if success:
                        total_successful_tasks += 1
                        island_success_count += 1

                        # 获取性能分析结果字典
                        profile_res = task_info.get("profile_res", {
                            'gen_time': float('inf'),
                            'base_time': 0.0,
                            'speedup': 0.0
                        })

                        # 收集成功的实现信息
                        impl_info = {
                            'id': generate_unique_id(),  # 添加唯一ID
                            'op_name': task_op_name,
                            'round': round_idx,
                            'task_id': task_info.get('task_id', ''),
                            'task_info': task_info,
                            'profile': profile_res,
                            'impl_code': task_info.get("coder_code", ""),
                            'framework_code': task_desc,
                            'backend': backend,
                            'arch': arch,
                            'dsl': dsl,
                            'framework': framework,
                            'sketch': '',
                            'source_island': island_idx  # 明确记录来源岛屿
                        }
                        successful_impls.append(impl_info)

                # 使用task_pool异步执行sketch生成
                if successful_impls:
                    sketch_tasks = []
                    for impl_info in successful_impls:
                        if impl_info['impl_code']:
                            sketch_task = partial(sketch_agent.run, impl_info['task_info'])
                            task_pool.create_task(sketch_task)
                            sketch_tasks.append(impl_info)

                    if sketch_tasks:
                        sketch_results = await task_pool.wait_all()
                        task_pool.tasks.clear()

                        # 处理sketch结果并更新impl_info
                        for i, impl_info in enumerate(sketch_tasks):
                            if impl_info['impl_code'] and i < len(sketch_results):
                                sketch_content = sketch_results[i]
                                impl_info['sketch'] = sketch_content if not isinstance(
                                    sketch_content, Exception) else ""

                            island_impls_list.append(impl_info)
                            round_implementations.append(impl_info)

                            # 保存到对应岛屿的本地文件
                            save_implementation(impl_info, islands_storage_dirs[island_idx])

                            # 添加到全局最佳实现列表
                            best_implementations.append(impl_info)

                            # 添加到当前岛屿实现列表
                            island_impls[island_idx].append(impl_info)

                # 更新精英库
                if island_impls_list:
                    # 添加来源岛屿信息到每个实现
                    for impl in island_impls_list:
                        impl['source_island'] = island_idx

                    # 添加到精英池（新生成的实现都有唯一ID，无需去重）
                    elite_pool.extend(island_impls_list)
                    # 按性能排序精英池
                    elite_pool.sort(key=lambda x: x['profile'][0] if isinstance(
                        x['profile'], (list, tuple)) else x['profile'])
                    # 保持精英库大小限制
                    elite_pool = elite_pool[:elite_size * num_islands]

                # 更新岛屿切换计数器
                if num_islands > 1:
                    current_island_counter += len(current_island_results)
                    if current_island_counter >= tasks_per_island_switch:
                        current_island = (current_island + 1) % num_islands
                        current_island_counter = 0
                        logger.debug(f"Switched to island {current_island}")
        else:
            # 处理简单模式的结果
            # 统计当前轮次结果
            round_success_count = 0
            round_total_count = len(results)
            round_implementations = []

            # 创建sketch agent（复用config）
            sketch_agent = Sketch(
                op_name=op_name,
                task_desc=task_desc,
                dsl=dsl,
                backend=backend,
                arch=arch,
                config=config
            )

            # 收集成功任务信息
            successful_impls = []

            for task_op_name, success, task_info in results:
                total_tasks += 1

                if success:
                    total_successful_tasks += 1
                    round_success_count += 1

                    # 获取性能分析结果字典
                    profile_res = task_info.get("profile_res", {
                        'gen_time': float('inf'),
                        'base_time': 0.0,
                        'speedup': 0.0
                    })

                    # 收集成功的实现信息
                    impl_info = {
                        'id': generate_unique_id(),  # 添加唯一ID
                        'op_name': task_op_name,
                        'round': round_idx,
                        'task_id': task_info.get('task_id', ''),
                        'task_info': task_info,
                        'profile': profile_res,
                        'impl_code': task_info.get("coder_code", ""),
                        'framework_code': task_desc,
                        'backend': backend,
                        'arch': arch,
                        'dsl': dsl,
                        'framework': framework,
                        'sketch': '',
                    }
                    successful_impls.append(impl_info)

            # 使用task_pool异步执行sketch生成
            if successful_impls:
                sketch_tasks = []
                for impl_info in successful_impls:
                    if impl_info['impl_code']:
                        sketch_task = partial(sketch_agent.run, impl_info['task_info'])
                        task_pool.create_task(sketch_task)
                        sketch_tasks.append(impl_info)

                if sketch_tasks:
                    sketch_results = await task_pool.wait_all()
                    task_pool.tasks.clear()

                    # 处理sketch结果并更新impl_info
                    for i, impl_info in enumerate(sketch_tasks):
                        if impl_info['impl_code'] and i < len(sketch_results):
                            sketch_content = sketch_results[i]
                            impl_info['sketch'] = sketch_content if not isinstance(sketch_content, Exception) else ""

                        round_implementations.append(impl_info)
                        best_implementations.append(impl_info)

                        # 保存到本地文件
                        save_implementation(impl_info, storage_dir)

        # 计算当前轮次成功率
        if use_islands:
            round_total_count = len(results)
            round_success_count = sum(
                1 for island_results in island_results for result in island_results if result[1])  # 统计所有成功任务
        round_success_rate = round_success_count / round_total_count if round_total_count > 0 else 0.0
        cumulative_success_rate = total_successful_tasks / total_tasks if total_tasks > 0 else 0.0
        if cumulative_success_rate > best_success_rate:
            best_success_rate = cumulative_success_rate

        # 记录轮次结果
        round_result = {
            'round': round_idx,
            'total_tasks': round_total_count,
            'successful_tasks': round_success_count,  # 当前轮次成功任务数
            'success_rate': round_success_rate,  # 当前轮次成功率
            'implementations': round_implementations
        }
        round_results.append(round_result)
        all_results.extend([(impl['op_name'], True) for impl in round_implementations])

        if os.getenv("AIKG_DATA_COLLECT", "off").lower() == "on":
            try:
                collector = await get_collector()
                collector.set_config(config)
                saved_files = await collector.prepare_and_remove_data()
            except Exception as e:
                logger.error(f"Failed to prepare data for transmission in evolve round {round_idx}: {e}")

        # 打印轮次结果
        pretty_print_results([(impl['op_name'], True) for impl in round_implementations] +
                             [(f"failed_task_{i}", False) for i in range(round_total_count - round_success_count)])

    # 按性能排序最佳实现（gen_time越小越好）
    best_implementations.sort(key=lambda x: x['profile'][0] if isinstance(
        x['profile'], (list, tuple)) else x['profile'])

    # 计算最终成功率
    final_success_rate = total_successful_tasks / total_tasks if total_tasks > 0 else 0.0

    # 构建返回结果
    evolution_result = {
        'op_name': op_name,
        'total_rounds': max_rounds,
        'total_tasks': total_tasks,
        'successful_tasks': total_successful_tasks,
        'final_success_rate': final_success_rate,
        'best_success_rate': best_success_rate,
        'implementation_type': dsl,
        'framework': framework,
        'backend': backend,
        'architecture': arch,
        'best_implementations': best_implementations[:5],  # 只返回前5个最佳实现
        'round_results': round_results,
        'storage_dir': storage_dir,  # 添加存储目录信息
    }

    # 如果使用了岛屿模型，添加岛屿信息
    if use_islands:
        evolution_result['island_info'] = {
            'num_islands': num_islands,
            'migration_interval': migration_interval,
            'elite_size': elite_size,
            'parent_selection_prob': parent_selection_prob
        }

    logger.info(f"Evolution completed for {op_name}")
    logger.info(f"Total tasks: {total_tasks}, Successful: {total_successful_tasks}")
    logger.info(f"Final success rate: {final_success_rate:.2%}")
    logger.info(f"Results stored in: {storage_dir}")

    return evolution_result
