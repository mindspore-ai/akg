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
Adaptive Search Controller

自适应搜索的主控制器，协调任务池、DB、选择器和生成器。
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

from ai_kernel_generator.core.adaptive_search.success_db import SuccessDB
from ai_kernel_generator.core.adaptive_search.task_pool import AsyncTaskPool, TaskResult
from ai_kernel_generator.core.adaptive_search.ucb_selector import UCBParentSelector
from ai_kernel_generator.core.adaptive_search.task_generator import TaskGenerator, TaskGeneratorConfig
from ai_kernel_generator.core.langgraph_task import LangGraphTask
from ai_kernel_generator.core.sketch import Sketch

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """搜索配置"""
    # 并发控制
    max_concurrent: int = 8
    initial_task_count: int = 8
    tasks_per_parent: int = 1  # 每次选择父代后生成的任务数
    
    # 停止条件
    max_total_tasks: int = 100
    target_success_count: int = 10
    target_speedup: float = 2.0
    
    # UCB 选择参数
    exploration_coef: float = 1.414
    random_factor: float = 0.1
    use_softmax: bool = False
    softmax_temperature: float = 1.0
    
    # 灵感采样参数
    inspiration_sample_num: int = 3
    use_tiered_sampling: bool = True
    handwrite_sample_num: int = 2
    handwrite_decay_rate: float = 2.0
    
    # 其他
    poll_interval: float = 0.5  # 轮询间隔（秒）
    storage_dir: Optional[str] = None  # 存储目录


class AdaptiveSearchController:
    """
    自适应搜索控制器
    
    协调以下组件：
    - AsyncTaskPool: 管理并发任务执行
    - SuccessDB: 存储成功任务
    - UCBParentSelector: 选择父代
    - TaskGenerator: 生成新任务
    """
    
    def __init__(self,
                 op_name: str,
                 task_desc: str,
                 dsl: str,
                 framework: str,
                 backend: str,
                 arch: str,
                 config: Dict[str, Any],
                 search_config: Optional[SearchConfig] = None):
        """
        初始化控制器
        
        Args:
            op_name: 算子名称
            task_desc: 任务描述
            dsl: DSL 类型
            framework: 框架
            backend: 后端
            arch: 架构
            config: 全局配置
            search_config: 搜索配置
        """
        self.op_name = op_name
        self.task_desc = task_desc
        self.dsl = dsl
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.config = config
        self.search_config = search_config or SearchConfig()
        
        # 设置存储目录
        if not self.search_config.storage_dir:
            home = os.path.expanduser("~")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.search_config.storage_dir = os.path.join(
                home, "aikg_adaptive_search", op_name, timestamp
            )
        os.makedirs(self.search_config.storage_dir, exist_ok=True)
        
        # 初始化组件
        self.db = SuccessDB(storage_dir=self.search_config.storage_dir)
        self.task_pool = AsyncTaskPool(max_concurrent=self.search_config.max_concurrent)
        self.selector = UCBParentSelector(
            db=self.db,
            exploration_coef=self.search_config.exploration_coef,
            random_factor=self.search_config.random_factor,
            use_softmax=self.search_config.use_softmax,
            softmax_temperature=self.search_config.softmax_temperature
        )
        
        generator_config = TaskGeneratorConfig(
            inspiration_sample_num=self.search_config.inspiration_sample_num,
            use_tiered_sampling=self.search_config.use_tiered_sampling,
            handwrite_sample_num=self.search_config.handwrite_sample_num,
            handwrite_decay_rate=self.search_config.handwrite_decay_rate
        )
        self.generator = TaskGenerator(
            op_name=op_name,
            task_desc=task_desc,
            dsl=dsl,
            framework=framework,
            backend=backend,
            arch=arch,
            config=config,
            db=self.db,
            generator_config=generator_config
        )
        
        # 统计信息
        self._total_submitted = 0
        self._total_completed = 0
        self._total_success = 0
        self._total_failed = 0
        self._start_time: Optional[datetime] = None
        self._stop_reason: Optional[str] = None
        
        # 任务代数追踪
        self._task_generations: Dict[str, int] = {}  # task_id -> generation
        self._task_parents: Dict[str, Optional[str]] = {}  # task_id -> parent_id
        
        # Sketch agent 用于根据最终代码重新生成 sketch
        self._sketch_agent: Optional[Sketch] = None
        
        logger.info(f"AdaptiveSearchController initialized for {op_name}")
        logger.info(f"Storage dir: {self.search_config.storage_dir}")
    
    def _get_sketch_agent(self) -> Sketch:
        """获取或创建 Sketch agent"""
        if self._sketch_agent is None:
            self._sketch_agent = Sketch(
                op_name=self.op_name,
                task_desc=self.task_desc,
                dsl=self.dsl,
                backend=self.backend,
                arch=self.arch,
                config=self.config
            )
        return self._sketch_agent
    
    async def _submit_initial_task(self) -> str:
        """提交一个初始任务"""
        task = await self.generator.generate_initial_task()
        task_id = task.task_id
        
        self._task_generations[task_id] = 0
        self._task_parents[task_id] = None
        
        await self.task_pool.submit(
            task_id=task_id,
            coroutine_factory=task.run,
            generation=0,
            parent_id=None
        )
        
        self._total_submitted += 1
        return task_id
    
    async def _submit_evolved_task(self) -> Optional[str]:
        """
        提交一个进化任务（自动选择父代）
        
        Returns:
            str: 任务 ID，如果无法生成则返回 None
        """
        parent = self.selector.select()
        if not parent:
            logger.warning("No parent selected, cannot generate evolved task")
            return None
        return await self._submit_evolved_task_with_parent(parent)
    
    async def _submit_evolved_task_with_parent(self, parent) -> Optional[str]:
        """
        使用指定父代提交一个进化任务
        
        Args:
            parent: 父代记录
            
        Returns:
            str: 任务 ID，如果无法生成则返回 None
        """
        # 计算新任务的代数
        generation = parent.generation + 1
        
        # 生成任务
        task = await self.generator.generate_evolved_task(parent, generation)
        task_id = task.task_id
        
        self._task_generations[task_id] = generation
        self._task_parents[task_id] = parent.id
        
        await self.task_pool.submit(
            task_id=task_id,
            coroutine_factory=task.run,
            generation=generation,
            parent_id=parent.id
        )
        
        self._total_submitted += 1
        return task_id
    
    async def _process_results(self) -> None:
        """处理已完成的任务结果"""
        results = self.task_pool.pop_completed_results()
        
        for result in results:
            self._total_completed += 1
            
            if result.success:
                self._total_success += 1
                
                # 添加到 DB（sketch 暂时为空）
                record = self.db.add_from_result(
                    task_id=result.task_id,
                    final_state=result.final_state,
                    generation=result.generation,
                    parent_id=result.parent_id
                )
                
                if record:
                    logger.info(
                        f"Task {result.task_id} succeeded: "
                        f"gen_time={record.gen_time:.4f}ms, speedup={record.speedup:.2f}x"
                    )
            else:
                self._total_failed += 1
                error = result.error or result.final_state.get("error", "Unknown error")
                logger.warning(f"Task {result.task_id} failed: {error}")
        
        # 异步生成 sketch（根据最终代码重新生成）
        await self._generate_pending_sketches()
    
    async def _generate_pending_sketches(self) -> None:
        """为待处理的成功任务生成 sketch"""
        pending_records = self.db.get_pending_sketch_records()
        if not pending_records:
            return
        
        sketch_agent = self._get_sketch_agent()
        
        for record in pending_records:
            try:
                # 从 meta_info 中获取完整的 task_info
                task_info = record.meta_info.get("task_info", {})
                if not task_info:
                    # 如果没有 task_info，构造一个基本的
                    task_info = {"coder_code": record.impl_code}
                
                # 调用 Sketch agent 生成 sketch
                sketch = await sketch_agent.run(task_info)
                
                if sketch and not isinstance(sketch, Exception):
                    self.db.update_sketch(record.id, sketch)
                    logger.debug(f"Generated sketch for record {record.id}")
                else:
                    # 生成失败，使用空 sketch
                    self.db.update_sketch(record.id, "")
                    logger.warning(f"Failed to generate sketch for record {record.id}")
                    
            except Exception as e:
                logger.warning(f"Exception generating sketch for {record.id}: {e}")
                self.db.update_sketch(record.id, "")
    
    def _check_stop_conditions(self) -> str:
        """
        检查停止条件
        
        Returns:
            str: 停止原因类型（空字符串表示不停止）
                 - "max_total_tasks": 达到最大任务数，需要等待剩余任务
                 - "target_success_count": 达到目标成功数，不需要等待
                 - "target_speedup": 达到目标加速比，不需要等待
        """
        # 达到目标成功数（不需要等待剩余任务）
        if self._total_success >= self.search_config.target_success_count:
            self._stop_reason = f"Reached target_success_count ({self.search_config.target_success_count})"
            return "target_success_count"
        
        # 达到目标加速比（不需要等待剩余任务）
        best_speedup = self.db.get_best_speedup()
        if best_speedup >= self.search_config.target_speedup:
            self._stop_reason = f"Reached target_speedup ({self.search_config.target_speedup}x, actual={best_speedup:.2f}x)"
            return "target_speedup"
        
        # 达到最大任务数（需要等待剩余任务）
        if self._total_submitted >= self.search_config.max_total_tasks:
            self._stop_reason = f"Reached max_total_tasks ({self.search_config.max_total_tasks})"
            return "max_total_tasks"
        
        return ""
    
    async def _refill_task_pool(self) -> int:
        """
        填充任务池
        
        逻辑：
        1. 先从等待队列取任务填充运行池
        2. 如果等待队列空且有空位且未达到最大任务数，选父代生成 n 个新任务
        3. 新任务有空位就运行，其余放入等待队列
        
        Returns:
            int: 提交的新任务数量
        """
        submitted = 0
        
        # 1. 从等待队列填充运行池
        filled = await self.task_pool.refill_from_queue()
        
        # 2. 只有等待队列空了才生成新任务
        if self.task_pool.get_waiting_count() == 0:
            # 检查是否需要生成新任务
            if (self.task_pool.has_capacity() and 
                self._total_submitted < self.search_config.max_total_tasks):
                
                if self.db.is_empty():
                    # DB 为空，生成初始任务
                    await self._submit_initial_task()
                    submitted += 1
                else:
                    # DB 非空，选择一个父代，生成 tasks_per_parent 个进化任务
                    parent = self.selector.select()
                    if parent:
                        # 生成 n 个任务（有空位运行，其余放等待队列）
                        for _ in range(self.search_config.tasks_per_parent):
                            if self._total_submitted >= self.search_config.max_total_tasks:
                                break
                            task_id = await self._submit_evolved_task_with_parent(parent)
                            if task_id:
                                submitted += 1
                    else:
                        # 无法选择父代，生成初始任务
                        await self._submit_initial_task()
                        submitted += 1
        
        return submitted
    
    async def run(self) -> Dict[str, Any]:
        """
        运行自适应搜索
        
        Returns:
            Dict[str, Any]: 搜索结果
        """
        self._start_time = datetime.now()
        logger.info(f"Starting adaptive search for {self.op_name}")
        logger.info(f"Config: max_concurrent={self.search_config.max_concurrent}, "
                   f"max_total_tasks={self.search_config.max_total_tasks}, "
                   f"target_success={self.search_config.target_success_count}, "
                   f"target_speedup={self.search_config.target_speedup}x")
        
        stop_type = ""
        try:
            # 1. 提交初始任务
            logger.info(f"Submitting {self.search_config.initial_task_count} initial tasks...")
            for _ in range(self.search_config.initial_task_count):
                if self._total_submitted >= self.search_config.max_total_tasks:
                    break
                await self._submit_initial_task()
            
            # 2. 主搜索循环
            while True:
                # 等待任务完成
                if self.task_pool.get_running_count() > 0:
                    await self.task_pool.wait_for_any(timeout=self.search_config.poll_interval)
                
                # 处理完成的结果（更新 _total_success 等计数器）
                await self._process_results()
                
                # 检查停止条件（在处理结果之后，确保计数器已更新）
                stop_type = self._check_stop_conditions()
                if stop_type:
                    logger.info(f"Stop condition met: {self._stop_reason}")
                    break
                
                # 如果任务池空闲且已达到最大任务数，退出
                if self.task_pool.is_idle() and self._total_submitted >= self.search_config.max_total_tasks:
                    logger.info("All tasks completed")
                    break
                
                # 填充任务池
                await self._refill_task_pool()
                
                # 如果任务池空闲但还没达到最大任务数，继续生成
                if self.task_pool.is_idle() and self._total_submitted < self.search_config.max_total_tasks:
                    await self._refill_task_pool()
                
                # 避免空转
                await asyncio.sleep(0.1)
            
            # 3. 处理剩余任务
            remaining = self.task_pool.get_running_count()
            waiting = self.task_pool.get_waiting_count()
            
            if remaining > 0 or waiting > 0:
                # max_total_tasks 触发时需要等待剩余任务
                # target_success_count 或 target_speedup 触发时取消剩余任务
                should_wait = (stop_type == "max_total_tasks")
                
                if should_wait:
                    logger.info(f"Waiting for {remaining} remaining tasks to complete...")
                    while self.task_pool.get_running_count() > 0:
                        await self.task_pool.wait_for_any(timeout=1.0)
                        await self._process_results()
                else:
                    # 目标达成，取消所有正在运行和等待的任务
                    logger.info(f"Target reached, cancelling {remaining} running and {waiting} waiting tasks")
                    
                    # 清空等待队列
                    self.task_pool.clear_waiting_queue()
                    
                    # 取消正在运行的任务
                    cancelled = await self.task_pool.cancel_all_running()
                    logger.info(f"Cancelled {cancelled} tasks, processing completed results...")
                    
                    # 处理已完成的结果（取消前可能有任务刚完成）
                    await self._process_results()
            
        except Exception as e:
            logger.error(f"Search failed with exception: {e}", exc_info=True)
            self._stop_reason = f"Exception: {e}"
        
        # 4. 收集结果
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        result = self._collect_results(elapsed_time)
        
        logger.info(f"Adaptive search completed for {self.op_name}")
        logger.info(f"Results: {self._total_success} successful, {self._total_failed} failed, "
                   f"{elapsed_time:.1f}s elapsed")
        logger.info(f"Stop reason: {self._stop_reason}")
        
        return result
    
    def _collect_results(self, elapsed_time: float) -> Dict[str, Any]:
        """收集搜索结果"""
        # 获取最佳实现
        best_implementations = []
        log_dir = self.config.get('log_dir', '')
        
        for record in self.db.get_all_sorted_by_performance()[:10]:
            # 从 profile 中获取 unique_dir（格式如 Iinit_1_S02_verify）
            unique_dir = record.profile.get('unique_dir', '')
            
            # 构建完整验证文件夹路径
            if unique_dir and log_dir:
                verify_dir = os.path.join(os.path.expanduser(log_dir), self.op_name, unique_dir)
            else:
                verify_dir = ""
            
            best_implementations.append({
                'id': record.id,  # 完整任务 ID
                'impl_code': record.impl_code,
                'sketch': record.sketch,
                'profile': record.profile,
                'gen_time': record.gen_time,
                'speedup': record.speedup,
                'generation': record.generation,  # 进化代数：0=初始，1=第一代进化...
                'parent_id': record.parent_id,  # 父代任务 ID
                'selection_count': record.selection_count,  # 被选为父代的次数
                'verify_dir': unique_dir,  # 验证文件夹名（如 Iinit_1_S02_verify）
            })
        
        # 从 config 中获取 log_dir 和 task_folder
        log_dir = self.config.get('log_dir', '')
        task_folder = os.path.basename(log_dir) if log_dir else ''
        
        return {
            'op_name': self.op_name,
            'total_submitted': self._total_submitted,
            'total_completed': self._total_completed,
            'total_success': self._total_success,
            'total_failed': self._total_failed,
            'success_rate': self._total_success / max(1, self._total_completed),
            'elapsed_time': elapsed_time,
            'stop_reason': self._stop_reason,
            'best_implementations': best_implementations,
            'db_statistics': self.db.get_statistics(),
            'selection_statistics': self.selector.get_selection_stats(),
            'storage_dir': self.search_config.storage_dir,
            'task_folder': task_folder,  # Task 文件夹名
            'log_dir': str(log_dir),  # 完整 log_dir 路径
            'config': {
                'max_concurrent': self.search_config.max_concurrent,
                'initial_task_count': self.search_config.initial_task_count,
                'tasks_per_parent': self.search_config.tasks_per_parent,
                'max_total_tasks': self.search_config.max_total_tasks,
                'target_success_count': self.search_config.target_success_count,
                'target_speedup': self.search_config.target_speedup,
                'exploration_coef': self.search_config.exploration_coef,
                'random_factor': self.search_config.random_factor
            }
        }