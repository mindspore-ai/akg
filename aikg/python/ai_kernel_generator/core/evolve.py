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

import logging
import json
import random
from functools import partial
from typing import List, Dict, Any, Optional, Tuple, Callable
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.database.evolve_database import EvolveDatabase

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
        evolve_db: 进化数据库实例，需要实现add()和samples()方法
        get_inspirations_from_meta_prompts: 获取初始启发的函数
        max_rounds: 最大进化轮数
        parallel_num: 每轮并行任务数

    Returns:
        进化结果字典
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

    for round_idx in range(1, max_rounds + 1):
        logger.info(f"Evolve round {round_idx}/{max_rounds} started")
        inspirations = list()
        if round_idx == 1:
            inspirations = [[] for _ in range(parallel_num)]
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
                workflow="coder_only_workflow",
                inspirations=inspirations[pid],
            )

            # 使用DO_CODER_DIRECT跳过设计阶段，直接生成代码
            task_pool.create_task(partial(task.run,))

        results = await task_pool.wait_all()
        task_pool.tasks.clear()

        for op_name, success, task_info in results:
            logger.debug(f"op_name: {op_name}, result: {task_info}")
            if success:
                task_pool.create_task(partial(
                    evolve_db.insert,
                    impl_code=task_info["coder_code"],
                    framework_code=task_info["task_desc"],
                    backend=backend,
                    arch=arch,
                    dsl=dsl,
                    framework=framework,
                    profile = task_info.get("profile_res", (float('inf'), 0.0, 0.0))[0],
                ))
        await task_pool.wait_all()
        task_pool.tasks.clear()

    return
