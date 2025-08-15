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
from ai_kernel_generator import get_project_root


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
    如果 meta prompt 数量不足，则自动拼接多个内容，保证每个字符串都不完全重复。

    Args:
        parallel_num: 并行任务数

    Returns:
        list[str]: meta prompts 字符串列表
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
        prompts_pool = triton_meta_prompts.copy()
        random.shuffle(prompts_pool)
        result = []
        idx = 0
        for i in range(parallel_num):
            # 计算每个字符串要拼接的 meta prompt 数量，尽量均匀分配
            base = n // parallel_num
            extra = 1 if i < n % parallel_num else 0
            count = base + extra
            # 随机选择 count 个 meta prompt（可重复）
            if idx + count > n:
                random.shuffle(prompts_pool)
                idx = 0
            prompts = prompts_pool[idx: idx + count]
            idx += count
            result.append("\n\n".join(prompts))
        assert len(result) == parallel_num, "Result length mismatch"
        return result
    except Exception as e:
        logger.error(f"Failed to load meta prompts: {e}")
        return ["" for _ in range(parallel_num)]


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


def load_best_implementations(storage_dir: str, max_count: int = 5) -> List[Dict[str, Any]]:
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
        return implementations[:max_count]
        
    except Exception as e:
        logger.error(f"Failed to load implementations from {storage_dir}: {e}")
        return implementations


def sample_inspirations(implementations: List[Dict[str, Any]], sample_num: int = 2) -> List[Dict[str, Any]]:
    """从实现列表中采样inspiration格式的数据
    
    Args:
        implementations: 实现列表
        sample_num: 采样数量
        
    Returns:
        inspiration格式的数据列表
    """
    if not implementations:
        return []
        
    # 随机采样或选择最佳的几个
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
        inspiration = {
            'impl_code': impl.get('impl_code', ''),
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
            stored_implementations = load_best_implementations(storage_dir, max_count=parallel_num * 2)
            
            inspirations = []
            for pid in range(parallel_num):
                sampled = sample_inspirations(stored_implementations, sample_num=min(parallel_num, 2))
                inspirations.append(sampled)
            
            meta_prompts = None

        # 创建并行任务
        round_tasks = []
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

        for task_op_name, success, task_info in results:
            logger.debug(f"op_name: {task_op_name}, result: {task_info}")
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
                    'profile': profile_res,  # 完整三元组
                    'impl_code': task_info.get("coder_code", ""),
                    'framework_code': task_desc,
                    'backend': backend,
                    'arch': arch,
                    'dsl': dsl,
                    'framework': framework
                }
                round_implementations.append(impl_info)
                best_implementations.append(impl_info)

                # 保存到本地文件
                save_implementation(impl_info, storage_dir)

        # 计算当前轮次成功率
        round_success_rate = round_success_count / round_total_count if round_total_count > 0 else 0.0
        if round_success_rate > best_success_rate:
            best_success_rate = round_success_rate

        # 记录轮次结果
        round_result = {
            'round': round_idx,
            'total_tasks': round_total_count,
            'successful_tasks': round_success_count,
            'success_rate': round_success_rate,
            'implementations': round_implementations
        }
        round_results.append(round_result)
        all_results.extend([(impl['op_name'], True) for impl in round_implementations])

        # 打印轮次结果
        pretty_print_results([(impl['op_name'], True) for impl in round_implementations] +
                             [(f"failed_task_{i}", False) for i in range(round_total_count - round_success_count)])

    # 按性能排序最佳实现（gen_time越小越好）
    best_implementations.sort(key=lambda x: x['profile'][0] if isinstance(x['profile'], (list, tuple)) else x['profile'])

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
        'storage_dir': storage_dir  # 添加存储目录信息
    }

    logger.info(f"Evolution completed for {op_name}")
    logger.info(f"Total tasks: {total_tasks}, Successful: {total_successful_tasks}")
    logger.info(f"Final success rate: {final_success_rate:.2%}")
    logger.info(f"Results stored in: {storage_dir}")

    return evolution_result
