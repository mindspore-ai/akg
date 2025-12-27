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

from ai_kernel_generator.core.adaptive_search.success_db import SuccessDB
from ai_kernel_generator.core.adaptive_search.task_pool import AsyncTaskPool
from ai_kernel_generator.core.adaptive_search.ucb_selector import UCBParentSelector
from ai_kernel_generator.core.adaptive_search.task_generator import TaskGenerator, TaskGeneratorConfig
from ai_kernel_generator.core.sketch import Sketch

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
        
        # 选择批次追踪（用于失败时回滚 selection_count）
        # task_id -> batch_id
        self._task_to_batch: Dict[str, int] = {}
        # batch_id -> {parent_id, child_ids, completed, success_count}
        self._selection_batches: Dict[int, Dict[str, Any]] = {}
        self._next_batch_id: int = 0
        
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
            else:
                self._total_failed += 1
                error = result.error or result.final_state.get("error", "Unknown error")
                logger.warning(f"Task {result.task_id} failed: {error}")
            
            # 更新选择批次状态
            self._update_selection_batch(result.task_id, task_success)
        
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
        # 唯一的停止条件：达到最大任务数
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
                    parent = self.selector.select()
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
        
        # 保存文件
        save_dir = os.path.expanduser(log_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.op_name}_lineage_graph.md")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        
        logger.info(f"Lineage graph saved to: {save_path}")
        return save_path