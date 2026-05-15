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
进化核心功能模块

包含实现的保存/加载、采样策略和岛屿模型逻辑
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# 存储功能：实现的保存和加载
# ============================================================================

def save_implementation(impl_data: Dict[str, Any], storage_dir: str) -> None:
    """保存实现到本地文件

    Args:
        impl_data: 实现数据字典
        storage_dir: 存储目录
    """
    try:
        os.makedirs(storage_dir, exist_ok=True)

        # 确保有唯一ID
        if 'id' not in impl_data:
            from .evolution_utils import generate_unique_id
            impl_data['id'] = generate_unique_id()

        # 生成唯一文件名
        round_idx = impl_data.get('round', 0)
        task_id = impl_data.get('task_id', 'unknown')
        impl_id = impl_data.get('id', 'unknown')[:8]  # 取ID前8位
        filename = f"impl_{round_idx}_{task_id}_{impl_id}.json"
        filepath = os.path.join(storage_dir, filename)

        # 保存数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(impl_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved implementation to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save implementation: {e}")


def load_best_implementations(storage_dir: str, max_count: int = None) -> List[Dict[str, Any]]:
    """从本地文件加载最佳实现

    Args:
        storage_dir: 存储目录
        max_count: 最大加载数量，None表示加载所有实现

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
                        # 确保每个实现都有唯一ID
                        if 'id' not in impl_data:
                            from .evolution_utils import generate_unique_id
                            impl_data['id'] = generate_unique_id()
                        implementations.append(impl_data)
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")

        # 按性能排序（gen_time越小越好）
        implementations.sort(key=lambda x: x.get('profile', {}).get('gen_time', float('inf')))

        logger.info(f"Loaded {len(implementations)} implementations from {storage_dir}")
        
        if max_count is None:
            return implementations
        else:
            return implementations[:max_count]

    except Exception as e:
        logger.error(f"Failed to load implementations from {storage_dir}: {e}")
        return implementations


# ============================================================================
# 采样功能：性能分类和灵感采样
# ============================================================================

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
        profile = impl.get('profile', {})
        
        speedup = profile.get('speedup', 0.0)
        if speedup != float('inf') and speedup > 0:
            valid_impls.append(impl)

    if not valid_impls:
        return {'good': [], 'medium': [], 'poor': []}

    total_count = len(valid_impls)

    # 按生成时间排序（从小到大，越小越好）
    valid_impls.sort(key=lambda x: x.get('profile', {}).get('gen_time', float('inf')))

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


def sample_inspirations(
    implementations: List[Dict[str, Any]], 
    sample_num: int = 2, 
    use_all: bool = False, 
    use_tiered_sampling: bool = False, 
    parent_implementations: List[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """从实现列表中采样inspiration格式的数据

    Args:
        implementations: 实现列表
        sample_num: 采样数量（当 use_all=False 且 use_tiered_sampling=True 时生效）
        use_all: 是否使用所有数据，如果 True 则按性能排序返回所有数据
        use_tiered_sampling: 是否使用分层采样策略
        parent_implementations: 父代实现列表，用于避免重复采样

    Returns:
        inspiration格式的数据列表
    """
    if not implementations:
        return []

    # 收集父代ID，用于排除重复
    parent_ids = set()
    if parent_implementations:
        for parent in parent_implementations:
            parent_id = parent.get('id')
            if parent_id:
                parent_ids.add(parent_id)

    # 排除父代实现
    filtered_implementations = []
    for impl in implementations:
        impl_id = impl.get('id')
        if impl_id and impl_id not in parent_ids:
            filtered_implementations.append(impl)
        elif not impl_id and impl not in parent_implementations:
            # 没有ID的实现也添加进去，但排除父代
            filtered_implementations.append(impl)

    implementations = filtered_implementations

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
        profile_data = impl.get('profile', {
            'gen_time': float('inf'),
            'base_time': 0.0,
            'speedup': 0.0
        })

        # 优先使用sketch，如果没有sketch则使用原始代码
        sketch = impl.get('sketch', '')
        impl_code = impl.get('impl_code', '')

        inspiration = {
            'id': impl.get('id'),  # 保留ID信息
            'sketch': sketch,  # 使用sketch作为inspiration内容
            'impl_code': impl_code,  # 使用原始代码作为inspiration内容
            'profile': profile_data,  # 保持完整的性能数据字典
            'strategy_mode': 'evolution'
        }
        inspirations.append(inspiration)

    return inspirations


# ============================================================================
# 岛屿模型：精英迁移和父代选择
# ============================================================================

def migrate_elites(islands: List[List[Dict[str, Any]]], migration_size: int = 1) -> List[List[Dict[str, Any]]]:
    """在岛屿间迁移精英个体

    Args:
        islands: 所有岛屿的实现列表
        migration_size: 每个岛屿迁移的个体数

    Returns:
        更新后的岛屿列表
    """
    if len(islands) < 2:
        return islands

    updated_islands = [island.copy() for island in islands]

    # 收集所有岛屿的精英
    elites = []
    for island in islands:
        # 每个岛屿选择最好的几个个体（gen_time越小越好）
        sorted_island = sorted(island, key=lambda x: x.get('profile', {}).get('gen_time', float('inf')))
        elites.extend(sorted_island[:migration_size])

    # 随机打乱精英列表
    random.shuffle(elites)

    # 将精英分配给其他岛屿
    for i, island in enumerate(updated_islands):
        # 获取当前岛屿已有的实现ID集合，避免重复
        existing_ids = {impl.get('id') for impl in island if impl.get('id')}

        # 从其他岛屿的精英中选择
        other_elites = [elite for j, elite_list in enumerate(islands)
                        for elite in elite_list[:migration_size] if j != i]

        if other_elites:
            # 顺延选择，跳过重复的实现
            selected_elites = []
            for elite in other_elites:
                if len(selected_elites) >= migration_size:
                    break
                elite_id = elite.get('id')
                if elite_id not in existing_ids:
                    selected_elites.append(elite)
                    existing_ids.add(elite_id)
                # 如果有重复，继续检查下一个

            # 如果选择的数量不够，继续从剩余的精英中选择
            if len(selected_elites) < migration_size:
                for elite in other_elites:
                    if len(selected_elites) >= migration_size:
                        break
                    if elite not in selected_elites:
                        selected_elites.append(elite)

            island.extend(selected_elites)

    return updated_islands


def select_parent_from_elite(current_island_idx: int, elite_pool: List[Dict[str, Any]]) -> Tuple:
    """从精英池中选择父代

    Args:
        current_island_idx: 当前岛屿索引
        elite_pool: 精英池

    Returns:
        (parent_implementation, parent_island_idx): 父代实现和其所在的岛屿索引
        如果精英池为空，返回 (None, current_island_idx)
    """
    if not elite_pool:
        # 如果精英池为空，返回None和当前岛屿
        return None, current_island_idx

    # 从精英池中随机选择一个精英个体作为父代
    selected_elite = random.choice(elite_pool)
    
    # 获取精英个体的来源岛屿
    source_island = selected_elite.get('source_island', current_island_idx)
    
    # 返回选中的精英个体和其来源岛屿
    return selected_elite, source_island

