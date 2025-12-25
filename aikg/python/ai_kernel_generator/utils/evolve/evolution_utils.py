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

"""
进化辅助工具模块

包含Meta提示加载、ID生成、结果打印等通用工具函数
"""

import uuid
import random
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# 通用工具函数
# ============================================================================

def generate_unique_id() -> str:
    """生成唯一ID

    Returns:
        str: 唯一ID字符串
    """
    return str(uuid.uuid4())


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


# ============================================================================
# Meta提示加载
# ============================================================================

def load_meta_prompts(dsl: str, parallel_num: int) -> List[str]:
    """
    返回长度为 parallel_num 的 meta prompt 列表。

    Args:
        dsl: DSL类型（如 "triton_cuda", "triton_ascend", "swft" 等）
        parallel_num: 并行任务数

    Returns:
        list[str]: meta prompts 字符串列表
        ▪ 当parallel_num <= n时：随机不重复选择
        ▪ 当parallel_num > n时：随机重复选择，保证parallel_num条数据
    """
    try:
        # 根据DSL类型动态导入对应的meta_prompts
        if dsl == "triton_ascend":
            from ai_kernel_generator.resources.docs.triton_ascend_docs.meta_prompts import (
                triton_meta_prompts,
            )
            meta_prompts = triton_meta_prompts
        elif dsl == "triton_cuda":
            # triton_cuda 目前可能没有单独的 meta_prompts，使用通用的或跳过
            logger.warning(f"DSL '{dsl}' does not support meta prompts yet, using empty prompts")
            return [""] * parallel_num
        elif dsl == "swft":
            # 如果swft有meta_prompts，可以在这里添加
            # from ai_kernel_generator.resources.docs.swft_docs.meta_prompts import swft_meta_prompts
            # meta_prompts = swft_meta_prompts
            logger.warning(f"DSL '{dsl}' does not support meta prompts yet")
            return [""] * parallel_num
        else:
            logger.warning(f"DSL '{dsl}' does not support meta prompts yet")
            return [""] * parallel_num

        assert meta_prompts
        assert isinstance(
            meta_prompts, list
        ), f"{dsl}_meta_prompts should be a list"

        n = len(meta_prompts)

        if parallel_num <= n:
            # 随机不重复选择parallel_num个
            return random.sample(meta_prompts, parallel_num)
        else:
            # 需要重复选择，保证parallel_num条数据
            result = []
            while len(result) < parallel_num:
                # 每轮随机打乱所有prompts
                shuffled_prompts = meta_prompts.copy()
                random.shuffle(shuffled_prompts)

                # 取需要的数量
                remaining = parallel_num - len(result)
                result.extend(shuffled_prompts[:min(remaining, n)])

            return result

    except Exception as e:
        logger.error(f"Failed to load meta prompts for DSL '{dsl}': {e}")
        return [""] * parallel_num

