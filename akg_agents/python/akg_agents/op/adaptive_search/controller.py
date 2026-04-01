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
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from akg_agents.op.adaptive_search.success_db import SuccessDB, SuccessRecord
from akg_agents.op.adaptive_search.task_pool import AsyncTaskPool
from akg_agents.op.adaptive_search.ucb_selector import UCBParentSelector
from akg_agents.op.adaptive_search.task_generator import TaskGenerator, TaskGeneratorConfig
from akg_agents.op.sketch import Sketch
from akg_agents.cli.runtime.message_sender import send_message
from akg_agents.cli.messages import PanelDataMessage

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """搜索配置"""
    # 并发控制
    max_concurrent: int = 8
    initial_task_count: int = 8
    tasks_per_parent: int = 1  # 每次选择父代后生成的任务数
    
    # 停止条件（唯一停止条件：达到最大任务数）
    max_total_tasks: int = 100
    
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
    
    # 进化控制器（外挂增强模块）
    use_evolution_controller: bool = False

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
                home, "akg_agents_adaptive_search", op_name, timestamp
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
        
        # 选择批次追踪（用于失败时回滚 selection_count）
        # task_id -> batch_id
        self._task_to_batch: Dict[str, int] = {}
        # batch_id -> {parent_id, child_ids, completed, success_count}
        self._selection_batches: Dict[int, Dict[str, Any]] = {}
        self._next_batch_id: int = 0
        
        # Sketch agent 用于根据最终代码重新生成 sketch
        self._sketch_agent: Optional[Sketch] = None
        
        # 【优化】Baseline 性能缓存（避免重复测量）
        # 第一个成功任务的 base_time 会被缓存，后续任务复用
        self._baseline_time_us: Optional[float] = None

        # 进化控制器（可选的外挂增强模块）
        self.evo_controller = None
        if self.search_config.use_evolution_controller:
            try:
                from akg_agents.op.adaptive_search.evolution_controller import (
                    EvolutionController,
                )
                from akg_agents.op.adaptive_search.evolution_controller.config import (
                    load_evolution_controller_config,
                )
                evo_config = load_evolution_controller_config(
                    max_total_tasks=self.search_config.max_total_tasks
                )
                self.evo_controller = EvolutionController(evo_config)
                logger.info("EvolutionController enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize EvolutionController: {e}")
                self.evo_controller = None

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
    
    def _update_profile_settings_in_config(self) -> None:
        """【优化】更新 config 中的 profile_settings，传递缓存的 baseline"""
        if self._baseline_time_us is not None and self._baseline_time_us > 0 and self._baseline_time_us < float('inf'):
            from akg_agents.op.verifier.baseline_profiler import set_baseline_in_config
            set_baseline_in_config(self.config, self._baseline_time_us)
    
    def _send_profile_to_history(self, record) -> None:
        """
        立即发送性能结果到历史记录
        
        Args:
            record: SuccessRecord 对象
        """
        if not record:
            return
        
        profile = record.profile
        if not isinstance(profile, dict):
            return
        
        try:
            gen_time = float(profile.get("gen_time") or 0.0)
            base_time = float(profile.get("base_time") or 0.0)
            speedup = float(record.speedup or 0.0)
        except Exception:
            gen_time = base_time = speedup = 0.0
        
        # 检查是否有有效的性能数据
        if gen_time > 0.0 or base_time > 0.0 or speedup > 0.0:
            session_id = str(self.config.get("session_id") or "").strip()
            if session_id:
                # 构造 log_dir: {base_log_dir}/{op_name}/{unique_dir}
                unique_dir = profile.get('unique_dir', '')
                base_log_dir = self.config.get('log_dir', '')
                log_dir = ""
                
                if base_log_dir and unique_dir:
                    log_dir = os.path.join(os.path.expanduser(base_log_dir), self.op_name, unique_dir)
                elif unique_dir:
                    # 如果没有 base_log_dir，尝试使用 unique_dir 作为相对路径
                    log_dir = unique_dir
                
                try:
                    send_message(
                        session_id,
                        PanelDataMessage(
                            action="move_to_history",
                            data={
                                "speedup": speedup,
                                "gen_time": gen_time,
                                "base_time": base_time,
                                "log_dir": log_dir,
                            },
                        ),
                    )
                    logger.debug(f"Sent profile result to history immediately: op_name={self.op_name}, speedup={speedup:.2f}x")
                except Exception as e:
                    logger.warning(f"Failed to send profile result to history: {e}")
    
    def _send_progress_update(self) -> None:
        """
        发送进度更新到 CLI
        
        发送当前任务进度信息，包括已提交、已完成、成功/失败数量。
        """
        session_id = str(self.config.get("session_id") or "").strip()
        if not session_id:
            return
        
        try:
            send_message(
                session_id,
                PanelDataMessage(
                    action="adaptive_search_progress",
                    data={
                        "op_name": self.op_name,
                        "total_submitted": self._total_submitted,
                        "total_completed": self._total_completed,
                        "total_success": self._total_success,
                        "total_failed": self._total_failed,
                        "max_total_tasks": self.search_config.max_total_tasks,
                        "running_count": self.task_pool.get_running_count(),
                        "waiting_count": self.task_pool.get_waiting_count(),
                    },
                ),
            )
            logger.debug(
                f"Sent progress update: submitted={self._total_submitted}, "
                f"completed={self._total_completed}, success={self._total_success}"
            )
        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")
    
    def _send_search_start(self) -> None:
        """发送自适应搜索开始消息，进入静默模式"""
        session_id = str(self.config.get("session_id") or "").strip()
        if not session_id:
            return
        
        try:
            send_message(
                session_id,
                PanelDataMessage(
                    action="adaptive_search_start",
                    data={
                        "op_name": self.op_name,
                        "max_total_tasks": self.search_config.max_total_tasks,
                    },
                ),
            )
            logger.debug(f"Sent adaptive_search_start message for {self.op_name}")
        except Exception as e:
            logger.warning(f"Failed to send search start: {e}")
    
    def _send_search_end(self) -> None:
        """发送自适应搜索结束消息，退出静默模式"""
        session_id = str(self.config.get("session_id") or "").strip()
        if not session_id:
            return
        
        try:
            send_message(
                session_id,
                PanelDataMessage(
                    action="adaptive_search_end",
                    data={
                        "op_name": self.op_name,
                        "total_success": self._total_success,
                        "total_failed": self._total_failed,
                    },
                ),
            )
            logger.debug(f"Sent adaptive_search_end message for {self.op_name}")
        except Exception as e:
            logger.warning(f"Failed to send search end: {e}")
    
    def _select_parent(self) -> Optional[SuccessRecord]:
        """选择父代（薄代理：如果启用了进化控制器则委托，否则走原始 UCB）"""
        if self.evo_controller:
            progress = self._total_submitted / max(self.search_config.max_total_tasks, 1)
            return self.evo_controller.select_parent(self.db, progress)
        return self.selector.select()

    async def _submit_initial_task(self) -> str:
        """提交一个初始任务"""
        # 【优化】为任务设置 profile_settings（传递缓存的 baseline）
        self._update_profile_settings_in_config()
        
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
        parent = self._select_parent()
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

        # 如果启用了进化控制器，由其选择灵感
        inspirations = None
        if self.evo_controller:
            progress = self._total_submitted / max(self.search_config.max_total_tasks, 1)
            inspirations = self.evo_controller.select_inspirations(parent, self.db, progress)

        # 【优化】为任务设置 profile_settings（传递缓存的 baseline）
        self._update_profile_settings_in_config()
        
        # 生成任务
        task = await self.generator.generate_evolved_task(parent, generation, inspirations=inspirations)
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
            task_success = False
            
            if result.success:
                self._total_success += 1
                task_success = True
                
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
                    
                    # 立即发送性能结果到历史记录
                    self._send_profile_to_history(record)
            else:
                self._total_failed += 1
                error = result.error or result.final_state.get("error", "Unknown error")
                logger.warning(f"Task {result.task_id} failed: {error}")
            
            # 更新选择批次状态
            self._update_selection_batch(result.task_id, task_success)

            # 通知进化控制器
            if self.evo_controller:
                self.evo_controller.on_result(self.db, success=task_success)

        # 异步生成 sketch（根据最终代码重新生成）
        await self._generate_pending_sketches()
    
    def _update_selection_batch(self, task_id: str, success: bool) -> None:
        """
        更新选择批次状态，如果批次完成且全部失败则回滚父代的 selection_count
        
        Args:
            task_id: 完成的任务 ID
            success: 任务是否成功
        """
        if task_id not in self._task_to_batch:
            # 初始任务不属于任何批次
            return
        
        batch_id = self._task_to_batch[task_id]
        if batch_id not in self._selection_batches:
            return
        
        batch = self._selection_batches[batch_id]
        batch['completed'] += 1
        if success:
            batch['success_count'] += 1
        
        # 检查批次是否完成
        total_children = len(batch['child_ids'])
        if batch['completed'] >= total_children:
            # 批次完成，检查是否全部失败
            if batch['success_count'] == 0:
                # 全部失败，回滚父代的 selection_count
                parent_id = batch['parent_id']
                self.db.decrement_selection(parent_id)
                logger.info(
                    f"Selection batch {batch_id} all failed ({total_children} tasks), "
                    f"decremented selection_count for parent {parent_id}"
                )
            else:
                logger.debug(
                    f"Selection batch {batch_id} completed: "
                    f"{batch['success_count']}/{total_children} succeeded"
                )
            
            # 清理批次数据
            for child_id in batch['child_ids']:
                self._task_to_batch.pop(child_id, None)
            del self._selection_batches[batch_id]
    
    async def _generate_pending_sketches(self) -> None:
        """为待处理的成功任务生成 sketch"""
        # 检查 sketch 生成开关
        if not self.config.get("enable_sketch_generation", True):
            logger.debug("Sketch generation disabled by config")
            return
        
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
    
    def _check_stop_conditions(self) -> bool:
        """
        检查停止条件

        Returns:
            bool: 是否应该停止
        """
        # 进化控制器的多维收敛检测
        if self.evo_controller:
            should_stop, reason = self.evo_controller.should_stop(
                self._total_submitted, self._total_success
            )
            if should_stop:
                self._stop_reason = reason
                return True
            return False

        # 原始逻辑：唯一的停止条件：达到最大任务数
        if self._total_submitted >= self.search_config.max_total_tasks:
            self._stop_reason = f"Reached max_total_tasks ({self.search_config.max_total_tasks})"
            return True

        return False
    
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
                    parent = self._select_parent()
                    if parent:
                        # 创建选择批次，用于追踪子任务成功/失败
                        batch_id = self._next_batch_id
                        self._next_batch_id += 1
                        self._selection_batches[batch_id] = {
                            'parent_id': parent.id,
                            'child_ids': [],
                            'completed': 0,
                            'success_count': 0
                        }
                        
                        # 生成 n 个任务（有空位运行，其余放等待队列）
                        for _ in range(self.search_config.tasks_per_parent):
                            if self._total_submitted >= self.search_config.max_total_tasks:
                                break
                            task_id = await self._submit_evolved_task_with_parent(parent)
                            if task_id:
                                self._selection_batches[batch_id]['child_ids'].append(task_id)
                                self._task_to_batch[task_id] = batch_id
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
                   f"max_total_tasks={self.search_config.max_total_tasks}, ")

        # 通知进化控制器
        if self.evo_controller:
            self.evo_controller.on_search_start()

        # 发送搜索开始消息（进入静默模式）
        self._send_search_start()
        
        # 【优化】在开始搜索前，先单独 profile baseline 一次
        from akg_agents.op.verifier.baseline_profiler import profile_baseline_once
        self._baseline_time_us = await profile_baseline_once(
            self.op_name, self.task_desc, self.dsl, self.framework,
            self.backend, self.arch, self.config
        )
        if self._baseline_time_us:
            logger.info(f"[{self.op_name}] 后续 {self.search_config.max_total_tasks} 个任务将跳过 baseline profile")
        
        # 使用上下文管理器，自动清理所有子任务（正常退出或异常都会清理）
        async with self.task_pool:
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
                    
                    # 发送进度更新到 CLI
                    self._send_progress_update()
                    # 检查停止条件：达到最大任务数
                    if self._check_stop_conditions():
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
                
                # 3. 等待剩余任务完成
                remaining = self.task_pool.get_running_count()
                if remaining > 0:
                    logger.info(f"Waiting for {remaining} remaining tasks to complete...")
                    while self.task_pool.get_running_count() > 0:
                        await self.task_pool.wait_for_any(timeout=1.0)
                        await self._process_results()
                        # 发送进度更新到 CLI
                        self._send_progress_update()
            
            except asyncio.CancelledError:
                # 用户取消操作，记录日志并向上传播
                # 清理工作由上下文管理器自动完成
                logger.info(f"Adaptive search for {self.op_name} was cancelled by user")
                self._stop_reason = "Cancelled by user"
                raise  # 重新抛出，__aexit__ 会自动清理
                
            except Exception as e:
                # 其他异常，记录日志
                # 清理工作由上下文管理器自动完成
                logger.error(f"Search failed with exception: {e}", exc_info=True)
                self._stop_reason = f"Exception: {e}"
        
        # 退出 async with 时，上下文管理器已自动清理所有子任务
        
        # 发送搜索结束消息（退出静默模式）
        self._send_search_end()
        
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
            # 从 profile 中获取 unique_dir（格式如 Iteration_Init_Task1_Step02_verify）
            unique_dir = record.profile.get('unique_dir', '')
            
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
                'verify_dir': unique_dir,  # 验证文件夹名（如 Iteration_Init_Task1_Step02_verify）
            })
        
        # 从 config 中获取 log_dir 和 task_folder
        log_dir = self.config.get('log_dir', '')
        task_folder = os.path.basename(log_dir) if log_dir else ''
        
        # 生成谱系图
        lineage_graph_path = self._generate_lineage_graph(log_dir)
        
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
            'lineage_graph': lineage_graph_path,  # 谱系图路径
            'config': {
                'max_concurrent': self.search_config.max_concurrent,
                'initial_task_count': self.search_config.initial_task_count,
                'tasks_per_parent': self.search_config.tasks_per_parent,
                'max_total_tasks': self.search_config.max_total_tasks,
                'exploration_coef': self.search_config.exploration_coef,
                'random_factor': self.search_config.random_factor
            }
        }
    
    def _generate_lineage_graph(self, log_dir: str) -> Optional[str]:
        """
        生成任务谱系图（父子关系图）- Mermaid 格式
        
        Args:
            log_dir: 日志目录路径
            
        Returns:
            str: Mermaid 文件保存路径，失败返回 None
        """
        if not log_dir:
            logger.warning("log_dir not specified, skipping lineage graph generation")
            return None
        
        records = self.db.get_all()
        if not records:
            logger.warning("No successful tasks, skipping lineage graph generation")
            return None
        
        # 构建节点数据
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Tuple[str, str]] = []
        record_ids = {r.id for r in records}
        
        for record in records:
            nodes[record.id] = {
                'gen_time': record.gen_time,
                'speedup': record.speedup,
                'generation': record.generation,
                'parent_id': record.parent_id,
                'selection_count': record.selection_count
            }
            if record.parent_id and record.parent_id in record_ids:
                edges.append((record.parent_id, record.id))
        
        # 按代数分组
        generations: Dict[int, List[str]] = {}
        for node_id, data in nodes.items():
            gen = data['generation']
            if gen not in generations:
                generations[gen] = []
            generations[gen].append(node_id)
        
        # 对每代内按 gen_time 排序
        for gen in generations:
            generations[gen] = sorted(generations[gen], key=lambda x: nodes[x]['gen_time'])
        
        max_gen = max(generations.keys()) if generations else 0
        
        # 获取性能范围用于着色
        valid_times = [data['gen_time'] for data in nodes.values() if data['gen_time'] != float('inf')]
        best_gen_time = min(valid_times) if valid_times else 0
        worst_gen_time = max(valid_times) if valid_times else 1
        
        # 生成 Mermaid 代码
        mermaid_lines = []
        mermaid_lines.append("```mermaid")
        mermaid_lines.append("flowchart TB")
        mermaid_lines.append("")
        
        # 添加子图（按代数分组）
        for gen in sorted(generations.keys()):
            node_ids = generations[gen]
            gen_label = "初始任务" if gen == 0 else f"Gen{gen}"
            
            mermaid_lines.append(f"    subgraph {gen_label}")
            for node_id in node_ids:
                data = nodes[node_id]
                gen_time = data['gen_time']
                speedup = data['speedup']
                sel_count = data['selection_count']
                
                # 节点显示内容
                if gen_time != float('inf'):
                    node_label = f"{node_id}<br/>{gen_time:.2f}us | {speedup:.2f}x"
                    if sel_count > 0:
                        node_label += f"<br/>选{sel_count}次"
                else:
                    node_label = f"{node_id}<br/>∞"
                
                # 节点 ID 需要转换为合法的 Mermaid ID
                safe_id = node_id.replace('-', '_')
                mermaid_lines.append(f'        {safe_id}["{node_label}"]')
            
            mermaid_lines.append("    end")
            mermaid_lines.append("")
        
        # 添加边（父子关系）
        mermaid_lines.append("    %% 父子关系")
        for parent_id, child_id in edges:
            safe_parent = parent_id.replace('-', '_')
            safe_child = child_id.replace('-', '_')
            mermaid_lines.append(f"    {safe_parent} --> {safe_child}")
        
        mermaid_lines.append("")
        
        # 添加样式（根据性能着色）
        mermaid_lines.append("    %% 样式：绿色=性能好，红色=性能差")
        for node_id, data in nodes.items():
            gen_time = data['gen_time']
            safe_id = node_id.replace('-', '_')
            
            if gen_time != float('inf') and worst_gen_time > best_gen_time:
                ratio = (gen_time - best_gen_time) / (worst_gen_time - best_gen_time)
                # 从绿色(#90EE90)到红色(#FFB6C1)
                if ratio < 0.33:
                    color = "#90EE90"  # 浅绿
                elif ratio < 0.67:
                    color = "#FFE4B5"  # 浅橙
                else:
                    color = "#FFB6C1"  # 浅红
            else:
                color = "#D3D3D3"  # 浅灰
            
            mermaid_lines.append(f"    style {safe_id} fill:{color}")
        
        mermaid_lines.append("```")
        
        # 构建完整的 Markdown 内容
        md_content = []
        md_content.append(f"# Task Lineage Graph - {self.op_name}")
        md_content.append("")
        md_content.append(f"**总任务数**: {len(nodes)} | **代数**: {max_gen + 1} | **最佳性能**: {best_gen_time:.2f}us")
        md_content.append("")
        md_content.append("## 谱系图")
        md_content.append("")
        md_content.extend(mermaid_lines)
        md_content.append("")
        md_content.append("## 图例")
        md_content.append("")
        md_content.append("- 🟢 浅绿色：性能好（前 33%）")
        md_content.append("- 🟡 浅橙色：性能中等（中间 34%）")
        md_content.append("- 🔴 浅红色：性能差（后 33%）")
        md_content.append("- 箭头：父代 → 子代")
        md_content.append("")
        md_content.append("## 任务详情")
        md_content.append("")
        md_content.append("| 任务 ID | 代数 | gen_time | speedup | 父代 | 被选次数 |")
        md_content.append("|---------|------|----------|---------|------|----------|")
        
        # 按 gen_time 排序输出详情
        sorted_nodes = sorted(nodes.items(), key=lambda x: x[1]['gen_time'])
        for node_id, data in sorted_nodes:
            gen = "初始" if data['generation'] == 0 else f"G{data['generation']}"
            parent = data['parent_id'] if data['parent_id'] else "-"
            if data['gen_time'] != float('inf'):
                md_content.append(f"| {node_id} | {gen} | {data['gen_time']:.2f}us | {data['speedup']:.2f}x | {parent} | {data['selection_count']} |")
            else:
                md_content.append(f"| {node_id} | {gen} | ∞ | - | {parent} | {data['selection_count']} |")
        
        # 追加停止原因和进化控制器诊断信息
        md_content.append("")
        md_content.append("## 搜索终止信息")
        md_content.append("")
        md_content.append(f"- **停止原因**: {self._stop_reason or 'N/A'}")
        md_content.append(f"- **总提交**: {self._total_submitted} | **总成功**: {self._total_success} | **总失败**: {self._total_failed}")
        md_content.append(f"- **使用进化控制器**: {'是' if self.evo_controller else '否'}")

        if self.evo_controller:
            diag = self.evo_controller.get_diagnostics()
            conv = diag.get("convergence", {})

            md_content.append("")
            md_content.append("### 收敛检测器状态")
            md_content.append("")
            md_content.append(f"- **最终状态**: {conv.get('state', 'N/A')}")
            md_content.append(f"- **停止原因**: {conv.get('stop_reason', 'N/A')}")
            md_content.append("")
            md_content.append("### 收敛信号详情")
            md_content.append("")
            md_content.append("| 信号 | 当前值 | 阈值 | 判定 |")
            md_content.append("|------|--------|------|------|")

            # S1: 性能停滞
            s1_imp = conv.get("s1_perf_improvement")
            s1_thr = conv.get("s1_perf_threshold", "N/A")
            s1_plateau = conv.get("s1_plateau", False)
            s1_imp_str = f"{s1_imp:.4%}" if s1_imp is not None else "N/A (数据不足)"
            md_content.append(
                f"| S1 性能改善率 | {s1_imp_str} | < {s1_thr} 则停滞 | "
                f"{'**停滞**' if s1_plateau else '改善中'} |"
            )

            # S1 count
            plateau_count = conv.get("plateau_count", 0)
            patience = conv.get("patience", "N/A")
            md_content.append(
                f"| S1 连续停滞窗口数 | {plateau_count} | ≥ {patience} 则触发 | "
                f"{'**达到**' if plateau_count >= (patience if isinstance(patience, int) else 999) else '未达到'} |"
            )

            # S2: 多样性趋势
            s2_change = conv.get("s2_diversity_change")
            s2_thr = conv.get("s2_diversity_threshold", "N/A")
            s2_trend = conv.get("s2_trend", "N/A")
            s2_change_str = f"{s2_change:+.4f}" if s2_change is not None else "N/A"
            md_content.append(
                f"| S2 多样性变化 | {s2_change_str} | < -{s2_thr} 则下降 | "
                f"**{s2_trend}** |"
            )

            # S3: 谱系活跃度
            s3_val = conv.get("s3_activity")
            s3_thr = conv.get("s3_activity_threshold", "N/A")
            s3_val_str = f"{s3_val:.4f}" if s3_val is not None else "N/A"
            md_content.append(
                f"| S3 谱系活跃度 | {s3_val_str} | ≥ {s3_thr} 则充分探索 | "
                f"{'**充分**' if s3_val is not None and s3_val >= (s3_thr if isinstance(s3_thr, float) else 999) else '未充分'} |"
            )

            # 收敛判定总结
            md_content.append("")
            md_content.append("### 收敛判定逻辑")
            md_content.append("")
            md_content.append("WATCHING → STOPPED 需同时满足：S1 连续停滞 ≥ patience **AND** S2 = declining **AND** S3 ≥ activity_threshold")
            md_content.append("")

            all_met = (s1_plateau
                       and plateau_count >= (patience if isinstance(patience, int) else 999)
                       and s2_trend == "declining"
                       and s3_val is not None and s3_val >= (s3_thr if isinstance(s3_thr, float) else 999))
            if all_met:
                md_content.append("**判定结果：三个条件全部满足 → 收敛停止**")
            else:
                conditions = []
                if not s1_plateau or plateau_count < (patience if isinstance(patience, int) else 999):
                    conditions.append("S1 未持续停滞")
                if s2_trend != "declining":
                    conditions.append(f"S2 = {s2_trend}（非 declining）")
                if s3_val is None or s3_val < (s3_thr if isinstance(s3_thr, float) else 999):
                    conditions.append(f"S3 = {s3_val_str}（未达阈值 {s3_thr}）")
                md_content.append(f"**判定结果：未满足收敛条件**（原因：{'; '.join(conditions)}）")

            # 谱系统计
            lineage_stats = diag.get("lineage_stats", {})
            if lineage_stats:
                md_content.append("")
                md_content.append("### 谱系统计")
                md_content.append("")
                md_content.append(f"- **谱系数量**: {diag.get('num_lineages', 'N/A')}")
                md_content.append(f"- **多样性指数 D**: {diag.get('diversity_index', 'N/A')}")
                md_content.append("")
                md_content.append("| 谱系根 | 大小 | 占比 | 最优 speedup | 最大深度 | 总选择次数 |")
                md_content.append("|--------|------|------|-------------|---------|-----------|")
                for root_id, stats in lineage_stats.items():
                    md_content.append(
                        f"| {root_id} | {stats['size']} | {stats['share']:.1%} | "
                        f"{stats['best_speedup']:.2f}x | {stats['max_depth']} | {stats['total_selections']} |"
                    )

        # 保存文件
        save_dir = os.path.expanduser(log_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.op_name}_lineage_graph.md")

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))

        logger.info(f"Lineage graph saved to: {save_path}")
        return save_path

