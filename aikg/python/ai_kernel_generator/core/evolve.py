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
from ai_kernel_generator.database.evolve_database import EvolveDatabase
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

    # 临时数据库路径配置
    import os
    import uuid

    random_hash = uuid.uuid4().hex[:8]
    evolve_database_path = os.path.expanduser(f"~/aikg_db/{random_hash}/")
    os.makedirs(evolve_database_path, exist_ok=True)
    evolve_db = EvolveDatabase(database_path=evolve_database_path)

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
            for pid in range(parallel_num):
                task_pool.create_task(partial(
                    evolve_db.samples,
                    output_content=["impl_code", "profile"],
                    sample_num=min(parallel_num, 2),
                    impl_code="",
                    framework_code=task_desc,
                    backend=backend,
                    arch=arch,
                    dsl=dsl,
                    framework=framework,
                ))
            meta_prompts = None
            inspirations = await task_pool.wait_all()
            task_pool.tasks.clear()

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

                # 收集成功的实现信息
                impl_info = {
                    'op_name': task_op_name,
                    'round': round_idx,
                    'task_info': task_info,
                    'profile': task_info.get("profile_res", (float('inf'), 0.0, 0.0))[0]
                }
                round_implementations.append(impl_info)
                best_implementations.append(impl_info)

                task_pool.create_task(partial(
                    evolve_db.insert,
                    impl_code=task_info["coder_code"],
                    framework_code=task_info["task_desc"],
                    backend=backend,
                    arch=arch,
                    dsl=dsl,
                    framework=framework,
                    profile=task_info.get("profile_res", (float('inf'), 0.0, 0.0))[0],
                ))

        await task_pool.wait_all()
        task_pool.tasks.clear()

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

    # 按性能排序最佳实现（越小越好）
    best_implementations.sort(key=lambda x: x['profile'])

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
        'round_results': round_results
    }

    logger.info(f"Evolution completed for {op_name}")
    logger.info(f"Total tasks: {total_tasks}, Successful: {total_successful_tasks}")
    logger.info(f"Final success rate: {final_success_rate:.2%}")

    return evolution_result
