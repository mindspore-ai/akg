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
Adaptive Search Entry Point

自适应搜索的主入口函数。
"""

import logging
import os
from typing import Dict, Any, Optional

from akg_agents.op.adaptive_search.controller import (
    AdaptiveSearchController,
    SearchConfig
)
from akg_agents.utils.common_utils import load_yaml
from akg_agents import get_project_root

logger = logging.getLogger(__name__)


def load_adaptive_search_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载自适应搜索配置

    Args:
        config_path: 配置文件路径，为空时使用默认路径

    Returns:
        Dict[str, Any]: 配置字典
    """
    if config_path is None:
        config_path = os.path.join(
            get_project_root(),
            "op",
            "config",
            "adaptive_search_config.yaml"
        )

    if os.path.exists(config_path):
        return load_yaml(config_path)

    logger.warning(f"Config file not found: {config_path}, using defaults")
    return {}


def _create_search_config(
    config: Dict[str, Any],
    search_config_dict: Dict[str, Any],
    # 可覆盖的参数
    max_concurrent: Optional[int] = None,
    initial_task_count: Optional[int] = None,
    tasks_per_parent: Optional[int] = None,
    max_total_tasks: Optional[int] = None,
    exploration_coef: Optional[float] = None,
    random_factor: Optional[float] = None,
    use_softmax: Optional[bool] = None,
    softmax_temperature: Optional[float] = None,
    inspiration_sample_num: Optional[int] = None,
    use_tiered_sampling: Optional[bool] = None,
    handwrite_sample_num: Optional[int] = None,
    handwrite_decay_rate: Optional[float] = None,
    use_evolution_controller: Optional[bool] = None,
    storage_dir: Optional[str] = None
) -> SearchConfig:
    """
    创建搜索配置

    优先级：函数参数 > search_config_dict > 默认值
    """
    # 从配置文件提取参数
    concurrency = search_config_dict.get("concurrency", {})
    stopping = search_config_dict.get("stopping", {})
    ucb = search_config_dict.get("ucb_selection", {})
    inspiration = search_config_dict.get("inspiration", {})
    handwrite = search_config_dict.get("handwrite", {})

    return SearchConfig(
        # 并发控制
        max_concurrent=max_concurrent or concurrency.get("max_concurrent", 8),
        initial_task_count=initial_task_count or concurrency.get("initial_task_count", 8),
        tasks_per_parent=tasks_per_parent or concurrency.get("tasks_per_parent", 1),

        # 停止条件（唯一停止条件：达到最大任务数）
        max_total_tasks=max_total_tasks or stopping.get("max_total_tasks", 100),

        # UCB 参数
        exploration_coef=exploration_coef or ucb.get("exploration_coef", 1.414),
        random_factor=random_factor if random_factor is not None else ucb.get("random_factor", 0.1),
        use_softmax=use_softmax if use_softmax is not None else ucb.get("use_softmax", False),
        softmax_temperature=softmax_temperature or ucb.get("softmax_temperature", 1.0),

        # 灵感采样参数
        inspiration_sample_num=inspiration_sample_num or inspiration.get("sample_num", 3),
        use_tiered_sampling=(
            use_tiered_sampling
            if use_tiered_sampling is not None
            else inspiration.get("use_tiered_sampling", True)
        ),
        handwrite_sample_num=handwrite_sample_num or handwrite.get("sample_num", 2),
        handwrite_decay_rate=handwrite_decay_rate or handwrite.get("decay_rate", 2.0),

        # 进化控制器
        use_evolution_controller=(
            use_evolution_controller
            if use_evolution_controller is not None
            else search_config_dict.get("use_evolution_controller", False)
        ),

        # 存储
        storage_dir=storage_dir
    )


async def adaptive_search(
    op_name: str,
    task_desc: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    # 并发控制
    max_concurrent: Optional[int] = None,
    initial_task_count: Optional[int] = None,
    tasks_per_parent: Optional[int] = None,
    # 停止条件
    max_total_tasks: Optional[int] = None,
    # UCB 参数
    exploration_coef: Optional[float] = None,
    random_factor: Optional[float] = None,
    use_softmax: Optional[bool] = None,
    softmax_temperature: Optional[float] = None,
    # 灵感采样参数
    inspiration_sample_num: Optional[int] = None,
    use_tiered_sampling: Optional[bool] = None,
    handwrite_sample_num: Optional[int] = None,
    handwrite_decay_rate: Optional[float] = None,
    # 进化控制器
    use_evolution_controller: Optional[bool] = None,
    # 其他
    storage_dir: Optional[str] = None,
    search_config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    自适应搜索主函数

    使用基于 UCB 选择策略的自适应搜索框架生成优化的 Kernel 实现。

    Args:
        op_name: 算子名称
        task_desc: 任务描述（PyTorch 模型代码）
        dsl: DSL 类型 (triton_cuda, triton_ascend, swft)
        framework: 框架 (torch, mindspore, numpy)
        backend: 后端 (cuda, ascend)
        arch: 架构 (a100, ascend910b4)
        config: 全局配置字典

        # 并发控制
        max_concurrent: 最大并发任务数
        initial_task_count: 初始任务数
        tasks_per_parent: 每次选择父代后生成的任务数

        # 停止条件
        max_total_tasks: 最大总任务数（唯一停止条件）

        # UCB 参数
        exploration_coef: UCB 探索系数
        random_factor: 随机扰动因子
        use_softmax: 是否使用 softmax 采样
        softmax_temperature: softmax 温度

        # 灵感采样参数
        inspiration_sample_num: 灵感采样数量
        use_tiered_sampling: 是否使用层次化采样
        handwrite_sample_num: 手写建议采样数量
        handwrite_decay_rate: 手写建议衰减率

        # 其他
        storage_dir: 存储目录
        search_config_path: 搜索配置文件路径

    Returns:
        Dict[str, Any]: 搜索结果，包含：
            - op_name: 算子名称
            - total_submitted: 总提交任务数
            - total_completed: 总完成任务数
            - total_success: 成功任务数
            - total_failed: 失败任务数
            - success_rate: 成功率
            - elapsed_time: 耗时（秒）
            - stop_reason: 终止原因
            - best_implementations: 最佳实现列表
            - db_statistics: DB 统计信息
            - storage_dir: 存储目录
    """
    logger.info(f"Starting adaptive_search for {op_name}")
    logger.info(f"DSL: {dsl}, Framework: {framework}, Backend: {backend}, Arch: {arch}")

    # 加载搜索配置
    search_config_dict = load_adaptive_search_config(search_config_path)

    # 创建搜索配置
    search_config = _create_search_config(
        config=config,
        search_config_dict=search_config_dict,
        max_concurrent=max_concurrent,
        initial_task_count=initial_task_count,
        tasks_per_parent=tasks_per_parent,
        max_total_tasks=max_total_tasks,
        exploration_coef=exploration_coef,
        random_factor=random_factor,
        use_softmax=use_softmax,
        softmax_temperature=softmax_temperature,
        inspiration_sample_num=inspiration_sample_num,
        use_tiered_sampling=use_tiered_sampling,
        handwrite_sample_num=handwrite_sample_num,
        handwrite_decay_rate=handwrite_decay_rate,
        use_evolution_controller=use_evolution_controller,
        storage_dir=storage_dir
    )

    # 创建控制器并运行
    controller = AdaptiveSearchController(
        op_name=op_name,
        task_desc=task_desc,
        dsl=dsl,
        framework=framework,
        backend=backend,
        arch=arch,
        config=config,
        search_config=search_config
    )

    result = await controller.run()

    logger.info(f"adaptive_search completed for {op_name}")
    return result


async def adaptive_search_from_config(
    op_name: str,
    task_desc: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    search_config_path: str
) -> Dict[str, Any]:
    """
    从配置文件运行自适应搜索

    Args:
        op_name: 算子名称
        task_desc: 任务描述
        dsl: DSL 类型
        framework: 框架
        backend: 后端
        arch: 架构
        config: 全局配置
        search_config_path: 搜索配置文件路径

    Returns:
        Dict[str, Any]: 搜索结果
    """
    return await adaptive_search(
        op_name=op_name,
        task_desc=task_desc,
        dsl=dsl,
        framework=framework,
        backend=backend,
        arch=arch,
        config=config,
        search_config_path=search_config_path
    )
