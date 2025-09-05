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
import json
from functools import partial
from typing import List, Dict, Any, Tuple
from pathlib import Path
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.sketch import Sketch
from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.collector import get_collector


os.environ['AIKG_DATA_COLLECT'] = 'on'
logger = logging.getLogger(__name__)


def pretty_print_results(results: List[Tuple[str, bool]]):
    """打印进化结果

    Args:
        results: 任务执行结果列表
    """
    logger.info("=" * 60)
    logger.info("EVOLVE ROUND RESULTS")
    logger.info("=" * 60)

    success_count = 0
    total_count = len(results)

    for op_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{op_name}: {status}")
        if success:
            success_count += 1

    success_rate = success_count / total_count if total_count > 0 else 0
    logger.info("-" * 60)
    logger.info(f"Success Rate: {success_count}/{total_count} ({success_rate:.2%})")
    logger.info("=" * 60)


def load_meta_prompts(parallel_num: int) -> list[str]:
    """
    返回长度为 parallel_num 的 meta prompt 列表。

    Args:
        parallel_num: 并行任务数

    Returns:
        list[str]: meta prompts 字符串列表
        - 当parallel_num <= n时：随机不重复选择
        - 当parallel_num > n时：随机重复选择，保证parallel_num条数据
    """
    try:
        from ai_kernel_generator.resources.docs.triton_docs.meta_prompts import (
            triton_meta_prompts,
        )

        assert triton_meta_prompts
        assert isinstance(
            triton_meta_prompts, list
        ), "triton_meta_prompts should be a list"

        n = len(triton_meta_prompts)

        if parallel_num <= n:
            # 随机不重复选择parallel_num个
            return random.sample(triton_meta_prompts, parallel_num)
        else:
            # 需要重复选择，保证parallel_num条数据
            result = []
            while len(result) < parallel_num:
                # 每轮随机打乱所有prompts
                shuffled_prompts = triton_meta_prompts.copy()
                random.shuffle(shuffled_prompts)

                # 取需要的数量
                remaining = parallel_num - len(result)
                result.extend(shuffled_prompts[:min(remaining, n)])

            return result

    except Exception as e:
        logger.error(f"Failed to load meta prompts: {e}")
        return [""] * parallel_num


def save_implementation(impl_data: Dict[str, Any], storage_dir: str) -> None:
    """保存实现到本地文件

    Args:
        impl_data: 实现数据字典
        storage_dir: 存储目录
    """
    try:
        os.makedirs(storage_dir, exist_ok=True)

        # 生成唯一文件名
        round_idx = impl_data.get('round', 0)
        task_id = impl_data.get('task_id', 'unknown')
        filename = f"impl_{round_idx}_{task_id}.json"
        filepath = os.path.join(storage_dir, filename)

        # 保存数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(impl_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved implementation to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save implementation: {e}")


def load_best_implementations(storage_dir: str) -> List[Dict[str, Any]]:
    """从本地文件加载最佳实现

    Args:
        storage_dir: 存储目录
        max_count: 最大加载数量

    Returns:
        按性能排序的最佳实现列表
    """
    implementations = []

    try:
        if not os.path.exists(storage_dir):
            return implementations

        for filename in os.listdir(storage_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(storage_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        impl_data = json.load(f)
                        implementations.append(impl_data)
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")

        # 按性能排序（gen_time越小越好）
        implementations.sort(key=lambda x: x.get('profile', (float('inf'), 0.0, 0.0))[0])

        logger.info(f"Loaded {len(implementations)} implementations from {storage_dir}")
        return implementations

    except Exception as e:
        logger.error(f"Failed to load implementations from {storage_dir}: {e}")
        return implementations


def classify_implementations_by_performance(implementations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按性能将实现分为三层：差、中等、好

    Args:
        implementations: 实现列表（已按性能排序，gen_time越小越好）

    Returns:
        分层后的实现字典，包含'good', 'medium', 'poor'三个层级
    """
    if not implementations:
        return {'good': [], 'medium': [], 'poor': []}

    # 过滤出有效的加速比数据
    valid_impls = []
    for impl in implementations:
        profile = impl.get('profile', (float('inf'), 0.0, 0.0))
        if len(profile) >= 3 and profile[2] != float('inf') and profile[2] > 0:
            valid_impls.append(impl)

    if not valid_impls:
        return {'good': [], 'medium': [], 'poor': []}

    total_count = len(valid_impls)

    # 按加速比排序（从高到低）
    valid_impls.sort(key=lambda x: x['profile'][2], reverse=True)

    # 分层策略：前30%为好，中间40%为中等，后30%为差
    good_count = max(1, int(total_count * 0.3))
    medium_count = max(1, int(total_count * 0.4))

    classified = {
        'good': valid_impls[:good_count],
        'medium': valid_impls[good_count:good_count + medium_count],
        'poor': valid_impls[good_count + medium_count:]
    }

    logger.info(f"Performance classification: good={len(classified['good'])}, "
                f"medium={len(classified['medium'])}, poor={len(classified['poor'])}")

    return classified


def sample_inspirations(implementations: List[Dict[str, Any]], sample_num: int = 2, use_all: bool = False, use_tiered_sampling: bool = False) -> List[Dict[str, Any]]:
    """从实现列表中采样inspiration格式的数据

    Args:
        implementations: 实现列表
        sample_num: 采样数量（当use_all=False时生效）
        use_all: 是否使用所有数据，如果True则按性能排序返回所有数据
        use_tiered_sampling: 是否使用分层采样策略

    Returns:
        inspiration格式的数据列表
    """
    if not implementations:
        return []

    if use_all:
        # 使用所有数据，按性能排序
        selected = implementations  # implementations已经按性能排序
    else:
        # 检查是否有足够数据进行分层采样
        if use_tiered_sampling and len(implementations) >= 3:  # 至少需要3个实现才进行分层
            # 分层采样：从好、中等、差三个层级各选一个
            classified = classify_implementations_by_performance(implementations)

            selected = []
            # 从每个层级选择一个最佳的
            for tier in ['good', 'medium', 'poor']:
                if classified[tier]:
                    selected.append(classified[tier][0])  # 选择该层级最佳的

            # 如果需要更多样本，从最佳层级补充
            while len(selected) < sample_num and classified['good']:
                remaining_good = [impl for impl in classified['good'] if impl not in selected]
                if remaining_good:
                    selected.append(remaining_good[0])
                else:
                    break

            logger.info(f"Tiered sampling selected {len(selected)} inspirations from different performance tiers")
        else:
            # 传统采样策略
            if len(implementations) <= sample_num:
                selected = implementations
            else:
                # 50%概率选择最佳的，50%概率随机选择
                best_count = max(1, sample_num // 2)
                random_count = sample_num - best_count

                selected = implementations[:best_count]  # 最佳的几个
                if random_count > 0 and len(implementations) > best_count:
                    remaining = implementations[best_count:]
                    selected.extend(random.sample(remaining, min(random_count, len(remaining))))

    # 转换为inspiration格式
    inspirations = []
    for impl in selected:
        profile_tuple = impl.get('profile', (float('inf'), 0.0, 0.0))

        # 优先使用sketch，如果没有sketch则使用原始代码
        sketch = impl.get('sketch', '')
        impl_code = impl.get('impl_code', '')

        inspiration = {
            'sketch': sketch,  # 使用sketch作为inspiration内容
            'impl_code': impl_code,  # 使用原始代码作为inspiration内容
            'profile': profile_tuple,  # 保持完整的三元组
            'strategy_mode': 'evolution'
        }
        inspirations.append(inspiration)

    return inspirations


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
) -> Dict[str, Any]:
    """进化式算子生成主函数

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

    Returns:
        进化结果字典，包含以下字段：
        - op_name: 算子名称
        - total_rounds: 总进化轮数
        - total_tasks: 总任务数
        - successful_tasks: 成功任务数
        - final_success_rate: 最终成功率
        - best_success_rate: 最佳成功率
        - implementation_type: 实现类型
        - framework: 框架名称
        - backend: 后端名称
        - architecture: 架构名称
        - best_implementations: 最佳实现列表
        - round_results: 每轮详细结果
    """
    logger.info(f"Starting evolve process for {op_name}")
    logger.info(f"Configuration: {dsl} on {backend}/{arch} using {framework}")

    # 本地存储路径配置
    import uuid
    random_hash = uuid.uuid4().hex[:8]
    storage_dir = os.path.expanduser(f"~/aikg_evolve/{op_name}_{dsl}_{framework}_{backend}_{arch}/{random_hash}/")
    os.makedirs(storage_dir, exist_ok=True)

    all_results = []
    best_success_rate = 0.0
    meta_prompts = []
    round_results = []
    best_implementations = []
    total_tasks = 0
    total_successful_tasks = 0

    # 新增：跟踪每轮和全局最佳加速比
    round_best_speedups = []  # 每轮最佳加速比
    global_best_speedup = 0.0  # 全局最佳加速比
    global_best_speedup_history = []  # 截至每轮的全局最佳加速比历史

    for round_idx in range(1, max_rounds + 1):
        logger.info(f"Evolve round {round_idx}/{max_rounds} started")
        inspirations: list = list()

        if round_idx == 1:
            inspirations = [[] for _ in range(parallel_num)]
            if dsl == "triton":
                # load meta-prompt
                root_dir = get_project_root()
                # fmt: off
                meta_prompt_path = Path(root_dir) / "resources" / "docs" / f"{dsl}_docs" / "meta_prompts.py"
                # fmt: on

                if meta_prompt_path.exists():
                    meta_prompts = load_meta_prompts(parallel_num)
                else:
                    logger.warning(f"Meta-prompt file not found: {meta_prompt_path}")

                if not meta_prompts or all(not prompt for prompt in meta_prompts):
                    logger.warning(f"No inspirations found in meta-prompts")
        else:
            # 从本地存储加载历史最佳实现作为inspiration
            stored_implementations = load_best_implementations(storage_dir)

            inspirations = []
            for pid in range(parallel_num):
                # 使用分层采样策略，每个任务采样3个不同层级的inspiration
                sampled = sample_inspirations(stored_implementations, sample_num=3, use_tiered_sampling=False)
                inspirations.append(sampled)

            meta_prompts = load_meta_prompts(parallel_num) if meta_prompt_path.exists() else []

        # 创建并行任务
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

                # 获取完整的profile三元组
                profile_res = task_info.get("profile_res", (float('inf'), 0.0, 0.0))

                # 收集成功的实现信息
                impl_info = {
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
                    'sketch': ''
                }
                successful_impls.append(impl_info)

        # 使用task_pool异步执行sketch生成
        if successful_impls:
            for impl_info in successful_impls:
                if impl_info['impl_code']:
                    task_pool.create_task(partial(sketch_agent.run, impl_info['task_info']))

            sketch_results = await task_pool.wait_all()
            task_pool.tasks.clear()

            # 处理sketch结果并更新impl_info
            for i, impl_info in enumerate(successful_impls):
                if impl_info['impl_code'] and i < len(sketch_results):
                    sketch_content = sketch_results[i]
                    impl_info['sketch'] = sketch_content if not isinstance(sketch_content, Exception) else ""

                round_implementations.append(impl_info)
                best_implementations.append(impl_info)

                # 保存到本地文件
                save_implementation(impl_info, storage_dir)

        # 计算当前轮次成功率
        round_success_rate = round_success_count / round_total_count if round_total_count > 0 else 0.0
        if round_success_rate > best_success_rate:
            best_success_rate = round_success_rate

        # 计算当前轮次最佳加速比
        round_best_speedup = 0.0
        if round_implementations:
            round_speedups = []
            for impl in round_implementations:
                profile = impl.get('profile', (float('inf'), 0.0, 0.0))
                if len(profile) >= 3 and profile[2] != float('inf') and profile[2] > 0:
                    round_speedups.append(profile[2])

            if round_speedups:
                round_best_speedup = max(round_speedups)
                # 更新全局最佳加速比
                if round_best_speedup > global_best_speedup:
                    global_best_speedup = round_best_speedup

        round_best_speedups.append(round_best_speedup)
        global_best_speedup_history.append(global_best_speedup)

        logger.info(f"Round {round_idx} best speedup: {round_best_speedup:.2f}x, "
                    f"Global best so far: {global_best_speedup:.2f}x")

        # 记录轮次结果
        round_result = {
            'round': round_idx,
            'total_tasks': round_total_count,
            'successful_tasks': round_success_count,
            'success_rate': round_success_rate,
            'implementations': round_implementations,
            'round_best_speedup': round_best_speedup,  # 新增：当前轮次最佳加速比
            'global_best_speedup': global_best_speedup  # 新增：截至当前轮次的全局最佳加速比
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
        # 新增：加速比统计信息
        'round_best_speedups': round_best_speedups,  # 每轮最佳加速比
        'global_best_speedup_history': global_best_speedup_history,  # 截至每轮的全局最佳加速比历史
        'final_best_speedup': global_best_speedup  # 最终全局最佳加速比
    }

    logger.info(f"Evolution completed for {op_name}")
    logger.info(f"Total tasks: {total_tasks}, Successful: {total_successful_tasks}")
    logger.info(f"Final success rate: {final_success_rate:.2%}")
    logger.info(f"Results stored in: {storage_dir}")

    return evolution_result
