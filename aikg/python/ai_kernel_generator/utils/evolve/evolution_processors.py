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
进化处理器模块

包含进化流程的三个阶段处理器和运行时配置
"""

import os
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from functools import partial

from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.sketch import Sketch
from ai_kernel_generator.utils.handwrite_loader import HandwriteLoader, HandwriteSampler

from .evolution_core import (
    save_implementation,
    load_best_implementations,
    sample_inspirations,
    migrate_elites,
    select_parent_from_elite
)
from .evolution_utils import (
    load_meta_prompts,
    generate_unique_id
)

logger = logging.getLogger(__name__)


# ============================================================================
# 运行时配置
# ============================================================================

@dataclass
class EvolveRuntimeConfig:
    """进化运行时配置（内部使用，不暴露给用户）"""
    # 基础配置
    op_name: str
    task_desc: str
    dsl: str
    framework: str
    backend: str
    arch: str
    config: dict
    max_rounds: int
    parallel_num: int
    
    # 进化参数
    num_islands: int
    migration_interval: int
    elite_size: int
    parent_selection_prob: float
    handwrite_decay_rate: float
    
    # 采样配置
    handwrite_sample_num: int = 2
    inspiration_sample_num: int = 3
    
    # 运行时计算的属性
    use_islands: bool = field(init=False)
    tasks_per_island: int = field(init=False)
    storage_dir: str = field(init=False)
    islands_storage_dirs: List[str] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """初始化后计算派生属性"""
        self.use_islands = self.num_islands > 1 and self.elite_size > 0
        self.tasks_per_island = max(1, self.parallel_num // self.num_islands) if self.use_islands else self.parallel_num
        
        # 设置存储目录
        random_hash = uuid.uuid4().hex[:8]
        self.storage_dir = os.path.expanduser(
            f"~/aikg_evolve/{self.op_name}_{self.dsl}_{self.framework}_{self.backend}_{self.arch}/{random_hash}/"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # 初始化岛屿存储目录
        if self.use_islands:
            self.islands_storage_dirs = []
            for i in range(self.num_islands):
                island_dir = os.path.join(self.storage_dir, f"island_{i}")
                os.makedirs(island_dir, exist_ok=True)
                self.islands_storage_dirs.append(island_dir)


def create_runtime_config(params: Dict[str, Any]) -> EvolveRuntimeConfig:
    """根据用户参数创建运行时配置
    
    Args:
        params: 包含所有evolve函数参数的字典
        
    Returns:
        EvolveRuntimeConfig: 运行时配置对象
    """
    return EvolveRuntimeConfig(
        op_name=params['op_name'],
        task_desc=params['task_desc'],
        dsl=params['dsl'],
        framework=params['framework'],
        backend=params['backend'],
        arch=params['arch'],
        config=params['config'],
        max_rounds=params['max_rounds'],
        parallel_num=params['parallel_num'],
        num_islands=params['num_islands'],
        migration_interval=params['migration_interval'],
        elite_size=params['elite_size'],
        parent_selection_prob=params['parent_selection_prob'],
        handwrite_decay_rate=params['handwrite_decay_rate']
    )


# ============================================================================
# 初始化处理器
# ============================================================================

class InitializationProcessor:
    """初始化阶段处理器"""
    
    def __init__(self, runtime_config: EvolveRuntimeConfig):
        self.config = runtime_config
        
    async def initialize(self) -> Dict[str, Any]:
        """执行初始化操作
        
        Returns:
            包含初始化数据的字典
        """
        logger.info(f"Starting evolve process for {self.config.op_name}")
        logger.info(f"Configuration: {self.config.dsl} on {self.config.backend}/{self.config.arch} using {self.config.framework}")
        
        if self.config.use_islands:
            logger.info(f"Islands: {self.config.num_islands}, Migration interval: {self.config.migration_interval}, Elite size: {self.config.elite_size}")
            logger.info(f"Parent selection probability: {self.config.parent_selection_prob}")
        else:
            logger.info("Island model: Disabled (simple evolution mode)")
        
        # 创建HandwriteLoader（共享）
        handwrite_loader = HandwriteLoader(
            dsl=self.config.dsl,
            op_name=self.config.op_name,
            task_desc=self.config.task_desc,
            config=self.config.config
        )
        await handwrite_loader.select_relevant_pairs()
        logger.info(f"Shared HandwriteLoader created with {len(handwrite_loader.get_selected_pairs())} selected documents")
        
        # 初始化数据结构
        init_data = {
            'handwrite_loader': handwrite_loader,
            'all_results': [],
            'best_success_rate': 0.0,
            'round_results': [],
            'best_implementations': [],
            'total_tasks': 0,
            'total_successful_tasks': 0
        }
        
        # 岛屿模型特定的初始化
        if self.config.use_islands:
            init_data['island_impls'] = [[] for _ in range(self.config.num_islands)]
            init_data['elite_pool'] = []
            init_data['island_handwrite_samplers'] = [
                HandwriteSampler(
                    loader=handwrite_loader,
                    sample_num=self.config.handwrite_sample_num,
                    decay_rate=self.config.handwrite_decay_rate
                )
                for _ in range(self.config.num_islands)
            ]
            if any(sampler._total_count > 0 for sampler in init_data['island_handwrite_samplers']):
                logger.info(f"Initialized {self.config.num_islands} independent HandwriteSamplers for islands")
        else:
            init_data['individual_handwrite_samplers'] = [
                HandwriteSampler(
                    loader=handwrite_loader,
                    sample_num=self.config.handwrite_sample_num,
                    decay_rate=self.config.handwrite_decay_rate
                )
                for _ in range(self.config.parallel_num)
            ]
            if any(sampler._total_count > 0 for sampler in init_data['individual_handwrite_samplers']):
                logger.info(f"Initialized {self.config.parallel_num} independent HandwriteSamplers for individuals")
        
        return init_data


# ============================================================================
# 任务创建处理器
# ============================================================================

class TaskCreationProcessor:
    """任务创建处理器"""
    
    def __init__(self, runtime_config: EvolveRuntimeConfig, init_data: Dict[str, Any]):
        self.config = runtime_config
        self.init_data = init_data
        
    def create_tasks_for_round(
        self,
        round_idx: int,
        device_pool,
        task_pool,
        round_implementations: List[Dict[str, Any]] = None
    ) -> List[Task]:
        """为当前轮次创建所有任务
        
        Args:
            round_idx: 当前轮次索引
            device_pool: 设备池
            task_pool: 任务池
            round_implementations: 当前轮次的实现列表（用于避免重复）
            
        Returns:
            创建的任务列表
        """
        logger.info(f"Evolve round {round_idx}/{self.config.max_rounds} started")
        
        # 岛屿迁移
        if (self.config.use_islands and round_idx > 1 and 
            self.config.migration_interval > 0 and 
            round_idx % self.config.migration_interval == 1 and 
            self.config.num_islands > 1):
            logger.info("Performing migration between islands")
            self.init_data['island_impls'] = migrate_elites(
                self.init_data['island_impls'],
                self.config.elite_size
            )
        
        # 准备灵感和meta prompts
        if self.config.use_islands:
            inspirations_data = self._prepare_island_inspirations(round_idx, round_implementations)
        else:
            inspirations_data = self._prepare_simple_inspirations(round_idx)
        
        # 创建任务
        tasks = []
        if self.config.use_islands:
            tasks, task_mapping = self._create_island_tasks(
                round_idx,
                inspirations_data,
                device_pool,
                task_pool
            )
            return tasks, task_mapping
        else:
            tasks = self._create_simple_tasks(
                round_idx,
                inspirations_data,
                device_pool,
                task_pool
            )
            return tasks, None
    
    def _prepare_island_inspirations(
        self,
        round_idx: int,
        round_implementations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """为岛屿模式准备灵感数据"""
        island_inspirations = [[] for _ in range(self.config.num_islands)]
        island_meta_prompts = [[] for _ in range(self.config.num_islands)]
        island_handwrite_suggestions = [[] for _ in range(self.config.num_islands)]
        
        if round_idx == 1:
            # 第一轮：初始化空灵感
            for island_idx in range(self.config.num_islands):
                island_inspirations[island_idx] = [[] for _ in range(self.config.tasks_per_island)]
                island_meta_prompts[island_idx] = load_meta_prompts(self.config.dsl, self.config.tasks_per_island)
                island_handwrite_suggestions[island_idx] = []
        else:
            # 后续轮次：生成灵感
            for island_idx in range(self.config.num_islands):
                island_inspirations[island_idx] = []
                for pid in range(self.config.tasks_per_island):
                    # 父代选择
                    if random.random() < self.config.parent_selection_prob:
                        parent_island_idx = island_idx
                        if self.config.num_islands == 1:
                            stored_implementations = load_best_implementations(self.config.storage_dir)
                        else:
                            stored_implementations = load_best_implementations(
                                self.config.islands_storage_dirs[parent_island_idx]
                            )
                        parent_implementation = random.choice(stored_implementations) if stored_implementations else None
                    else:
                        parent_implementation, parent_island_idx = select_parent_from_elite(
                            island_idx,
                            self.init_data['elite_pool']
                        )
                        if self.config.num_islands == 1:
                            stored_implementations = load_best_implementations(self.config.storage_dir)
                        else:
                            stored_implementations = load_best_implementations(
                                self.config.islands_storage_dirs[parent_island_idx]
                            )
                        if parent_implementation is None and stored_implementations:
                            parent_implementation = random.choice(stored_implementations)
                    
                    # 采样其他灵感
                    current_round_implementations = [
                        impl for impl in (round_implementations or []) if impl.get('round') == round_idx
                    ]
                    all_excluded_implementations = current_round_implementations.copy()
                    if parent_implementation:
                        all_excluded_implementations.append(parent_implementation)
                    
                    sampled = sample_inspirations(
                        stored_implementations,
                        sample_num=min(len(stored_implementations), self.config.inspiration_sample_num),
                        use_tiered_sampling=True,
                        parent_implementations=all_excluded_implementations
                    )
                    
                    # 将父代加入灵感列表
                    if parent_implementation:
                        parent_inspiration = {
                            'id': parent_implementation.get('id'),
                            'sketch': parent_implementation.get('sketch', ''),
                            'impl_code': parent_implementation.get('impl_code', ''),
                            'profile': parent_implementation.get('profile', {
                                'gen_time': float('inf'),
                                'base_time': 0.0,
                                'speedup': 0.0
                            }),
                            'strategy_mode': 'evolution',
                            'is_parent': True
                        }
                        sampled.insert(0, parent_inspiration)
                    
                    island_inspirations[island_idx].append(sampled)
                
                island_meta_prompts[island_idx] = load_meta_prompts(self.config.dsl, self.config.tasks_per_island)
                island_handwrite_suggestions[island_idx] = self.init_data['island_handwrite_samplers'][island_idx].sample()
        
        return {
            'inspirations': island_inspirations,
            'meta_prompts': island_meta_prompts,
            'handwrite_suggestions': island_handwrite_suggestions
        }
    
    def _prepare_simple_inspirations(self, round_idx: int) -> Dict[str, Any]:
        """为简单模式准备灵感数据"""
        if round_idx == 1:
            inspirations = [[] for _ in range(self.config.parallel_num)]
            meta_prompts = load_meta_prompts(self.config.dsl, self.config.parallel_num)
            handwrite_suggestions_list = [[] for _ in range(self.config.parallel_num)]
        else:
            stored_implementations = load_best_implementations(self.config.storage_dir)
            inspirations = []
            for pid in range(self.config.parallel_num):
                sampled = sample_inspirations(
                    stored_implementations,
                    sample_num=min(len(stored_implementations), self.config.inspiration_sample_num),
                    use_tiered_sampling=True
                )
                inspirations.append(sampled)
            
            meta_prompts = load_meta_prompts(self.config.dsl, self.config.parallel_num)
            handwrite_suggestions_list = [
                self.init_data['individual_handwrite_samplers'][pid].sample()
                for pid in range(self.config.parallel_num)
            ]
        
        return {
            'inspirations': inspirations,
            'meta_prompts': meta_prompts,
            'handwrite_suggestions': handwrite_suggestions_list
        }
    
    def _create_island_tasks(
        self,
        round_idx: int,
        inspirations_data: Dict[str, Any],
        device_pool,
        task_pool
    ) -> tuple:
        """创建岛屿模式的任务"""
        all_tasks = []
        task_mapping = []
        
        island_inspirations = inspirations_data['inspirations']
        island_meta_prompts = inspirations_data['meta_prompts']
        island_handwrite_suggestions = inspirations_data['handwrite_suggestions']
        
        for island_idx in range(self.config.num_islands):
            # 确保有足够的元素
            while len(island_inspirations[island_idx]) < self.config.tasks_per_island:
                island_inspirations[island_idx].append([])
            while len(island_meta_prompts[island_idx]) < self.config.tasks_per_island:
                island_meta_prompts[island_idx].append("")
            
            for pid in range(self.config.tasks_per_island):
                task_id = f"{round_idx}_{island_idx}_{pid}"
                
                task = Task(
                    op_name=self.config.op_name,
                    task_desc=self.config.task_desc,
                    task_id=task_id,
                    backend=self.config.backend,
                    arch=self.config.arch,
                    dsl=self.config.dsl,
                    config=self.config.config,
                    device_pool=device_pool,
                    framework=self.config.framework,
                    task_type="profile",
                    workflow="default_workflow",
                    inspirations=island_inspirations[island_idx][pid],
                    meta_prompts=island_meta_prompts[island_idx][pid] if island_meta_prompts[island_idx] else None,
                    handwrite_suggestions=island_handwrite_suggestions[island_idx],
                )
                
                task_pool.create_task(partial(task.run,))
                all_tasks.append(task)
                task_mapping.append(island_idx)
        
        return all_tasks, task_mapping
    
    def _create_simple_tasks(
        self,
        round_idx: int,
        inspirations_data: Dict[str, Any],
        device_pool,
        task_pool
    ) -> List[Task]:
        """创建简单模式的任务"""
        tasks = []
        inspirations = inspirations_data['inspirations']
        meta_prompts = inspirations_data['meta_prompts']
        handwrite_suggestions_list = inspirations_data['handwrite_suggestions']
        
        for pid in range(self.config.parallel_num):
            task_id = f"{round_idx}_{pid}"
            
            task = Task(
                op_name=self.config.op_name,
                task_desc=self.config.task_desc,
                task_id=task_id,
                backend=self.config.backend,
                arch=self.config.arch,
                dsl=self.config.dsl,
                config=self.config.config,
                device_pool=device_pool,
                framework=self.config.framework,
                task_type="profile",
                workflow="default_workflow",
                inspirations=inspirations[pid],
                meta_prompts=meta_prompts[pid] if meta_prompts else None,
                handwrite_suggestions=handwrite_suggestions_list[pid] if handwrite_suggestions_list else [],
            )
            
            task_pool.create_task(partial(task.run,))
            tasks.append(task)
        
        return tasks


# ============================================================================
# 结果处理器
# ============================================================================

class ResultProcessor:
    """结果处理器"""
    
    def __init__(self, runtime_config: EvolveRuntimeConfig, init_data: Dict[str, Any]):
        self.config = runtime_config
        self.init_data = init_data
    
    async def process_results(
        self,
        results: List,
        round_idx: int,
        task_pool,
        task_mapping: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """处理轮次结果
        
        Args:
            results: 任务执行结果列表
            round_idx: 当前轮次索引
            task_pool: 任务池
            task_mapping: 任务到岛屿的映射（岛屿模式）
            
        Returns:
            包含轮次统计信息的字典
        """
        round_implementations = []
        
        if self.config.use_islands:
            round_data = await self._process_island_results(
                results,
                round_idx,
                task_pool,
                task_mapping
            )
            round_implementations = round_data['implementations']
        else:
            round_data = await self._process_simple_results(
                results,
                round_idx,
                task_pool
            )
            round_implementations = round_data['implementations']
        
        # 更新全局统计
        self.init_data['all_results'].extend([
            (impl['op_name'], True) for impl in round_implementations
        ])
        
        # 记录轮次结果
        round_result = {
            'round': round_idx,
            'total_tasks': round_data['total_tasks'],
            'successful_tasks': round_data['successful_tasks'],
            'success_rate': round_data['success_rate'],
            'implementations': round_implementations
        }
        self.init_data['round_results'].append(round_result)
        
        # 更新累计统计
        self.init_data['total_tasks'] += round_data['total_tasks']
        self.init_data['total_successful_tasks'] += round_data['successful_tasks']
        cumulative_success_rate = (
            self.init_data['total_successful_tasks'] / self.init_data['total_tasks']
            if self.init_data['total_tasks'] > 0 else 0.0
        )
        if cumulative_success_rate > self.init_data['best_success_rate']:
            self.init_data['best_success_rate'] = cumulative_success_rate
        
        return {
            'round_implementations': round_implementations,
            'round_result': round_result
        }
    
    async def _process_island_results(
        self,
        results: List,
        round_idx: int,
        task_pool,
        task_mapping: List[int]
    ) -> Dict[str, Any]:
        """处理岛屿模式的结果"""
        # 按岛屿分组结果
        island_results = [[] for _ in range(self.config.num_islands)]
        for i, result in enumerate(results):
            island_idx = task_mapping[i]
            island_results[island_idx].append(result)
        
        all_implementations = []
        total_success_count = 0
        
        # 处理每个岛屿的结果
        for island_idx in range(self.config.num_islands):
            current_island_results = island_results[island_idx]
            
            # 创建sketch agent
            sketch_agent = Sketch(
                op_name=self.config.op_name,
                task_desc=self.config.task_desc,
                dsl=self.config.dsl,
                backend=self.config.backend,
                arch=self.config.arch,
                config=self.config.config
            )
            
            # 收集成功任务信息
            successful_impls = []
            for task_op_name, success, task_info in current_island_results:
                if success:
                    total_success_count += 1
                    profile_res = task_info.get("profile_res", {
                        'gen_time': float('inf'),
                        'base_time': 0.0,
                        'speedup': 0.0
                    })
                    
                    impl_info = {
                        'id': generate_unique_id(),
                        'op_name': task_op_name,
                        'round': round_idx,
                        'task_id': task_info.get('task_id', ''),
                        'unique_dir': profile_res.get('unique_dir', ''),
                        'task_info': task_info,
                        'profile': profile_res,
                        'impl_code': task_info.get("coder_code", ""),
                        'framework_code': self.config.task_desc,
                        'backend': self.config.backend,
                        'arch': self.config.arch,
                        'dsl': self.config.dsl,
                        'framework': self.config.framework,
                        'sketch': '',
                        'source_island': island_idx
                    }
                    successful_impls.append(impl_info)
            
            # 异步生成sketch
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
                    
                    for i, impl_info in enumerate(sketch_tasks):
                        if impl_info['impl_code'] and i < len(sketch_results):
                            sketch_content = sketch_results[i]
                            impl_info['sketch'] = sketch_content if not isinstance(sketch_content, Exception) else ""
                        
                        all_implementations.append(impl_info)
                        
                        # 保存到岛屿存储
                        save_implementation(impl_info, self.config.islands_storage_dirs[island_idx])
                        
                        # 添加到全局最佳实现列表
                        self.init_data['best_implementations'].append(impl_info)
                        
                        # 添加到岛屿实现列表
                        self.init_data['island_impls'][island_idx].append(impl_info)
            
            # 更新精英库
            if successful_impls:
                for impl in successful_impls:
                    impl['source_island'] = island_idx
                self.init_data['elite_pool'].extend(successful_impls)
                self.init_data['elite_pool'].sort(key=lambda x: x.get('profile', {}).get('gen_time', float('inf')))
                self.init_data['elite_pool'] = self.init_data['elite_pool'][:self.config.elite_size * self.config.num_islands]
        
        round_total_count = len(results)
        round_success_rate = total_success_count / round_total_count if round_total_count > 0 else 0.0
        
        return {
            'implementations': all_implementations,
            'total_tasks': round_total_count,
            'successful_tasks': total_success_count,
            'success_rate': round_success_rate
        }
    
    async def _process_simple_results(
        self,
        results: List,
        round_idx: int,
        task_pool
    ) -> Dict[str, Any]:
        """处理简单模式的结果"""
        round_success_count = 0
        round_total_count = len(results)
        round_implementations = []
        
        # 创建sketch agent
        sketch_agent = Sketch(
            op_name=self.config.op_name,
            task_desc=self.config.task_desc,
            dsl=self.config.dsl,
            backend=self.config.backend,
            arch=self.config.arch,
            config=self.config.config
        )
        
        # 收集成功任务信息
        successful_impls = []
        for task_op_name, success, task_info in results:
            if success:
                round_success_count += 1
                profile_res = task_info.get("profile_res", {
                    'gen_time': float('inf'),
                    'base_time': 0.0,
                    'speedup': 0.0
                })
                
                impl_info = {
                    'id': generate_unique_id(),
                    'op_name': task_op_name,
                    'round': round_idx,
                    'task_id': task_info.get('task_id', ''),
                    'unique_dir': profile_res.get('unique_dir', ''),
                    'task_info': task_info,
                    'profile': profile_res,
                    'impl_code': task_info.get("coder_code", ""),
                    'framework_code': self.config.task_desc,
                    'backend': self.config.backend,
                    'arch': self.config.arch,
                    'dsl': self.config.dsl,
                    'framework': self.config.framework,
                    'sketch': '',
                }
                successful_impls.append(impl_info)
        
        # 异步生成sketch
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
                
                for i, impl_info in enumerate(sketch_tasks):
                    if impl_info['impl_code'] and i < len(sketch_results):
                        sketch_content = sketch_results[i]
                        impl_info['sketch'] = sketch_content if not isinstance(sketch_content, Exception) else ""
                    
                    round_implementations.append(impl_info)
                    self.init_data['best_implementations'].append(impl_info)
                    
                    # 保存到本地文件
                    save_implementation(impl_info, self.config.storage_dir)
        
        round_success_rate = round_success_count / round_total_count if round_total_count > 0 else 0.0
        
        return {
            'implementations': round_implementations,
            'total_tasks': round_total_count,
            'successful_tasks': round_success_count,
            'success_rate': round_success_rate
        }

