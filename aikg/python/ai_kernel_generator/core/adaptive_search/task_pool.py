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
Async Task Pool for Adaptive Search

管理并发任务执行和等待队列。
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Deque, Optional, Callable, Awaitable, List
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"       # 等待队列中
    RUNNING = "running"       # 正在执行
    COMPLETED = "completed"   # 执行完成（成功或失败）


@dataclass
class PendingTask:
    """等待执行的任务"""
    task_id: str
    coroutine_factory: Callable[[], Awaitable[Any]]
    generation: int = 0
    parent_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    success: bool
    final_state: Dict[str, Any]
    generation: int = 0
    parent_id: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TaskWrapper:
    """包装 asyncio.Task，携带额外信息"""
    
    def __init__(self, 
                 task_id: str, 
                 async_task: asyncio.Task,
                 generation: int = 0,
                 parent_id: Optional[str] = None):
        self.task_id = task_id
        self.async_task = async_task
        self.generation = generation
        self.parent_id = parent_id
        self.started_at = datetime.now().isoformat()


class AsyncTaskPool:
    """
    异步任务池
    
    管理并发执行的任务和等待队列，支持"及时替补"机制。
    """
    
    def __init__(self, max_concurrent: int = 8):
        """
        初始化任务池
        
        Args:
            max_concurrent: 最大并发任务数
        """
        self.max_concurrent = max_concurrent
        self._running: Dict[str, TaskWrapper] = {}  # 正在运行的任务
        self._waiting: Deque[PendingTask] = deque()  # 等待队列
        self._results: Dict[str, TaskResult] = {}    # 已完成的结果
        self._task_counter = 0
        
        logger.info(f"AsyncTaskPool initialized with max_concurrent={max_concurrent}")
    
    def generate_task_id(self, prefix: str = "task") -> str:
        """生成唯一任务 ID"""
        self._task_counter += 1
        return f"{prefix}_{self._task_counter:05d}_{datetime.now().strftime('%H%M%S')}"
    
    async def submit(self, 
                     task_id: str,
                     coroutine_factory: Callable[[], Awaitable[Any]],
                     generation: int = 0,
                     parent_id: Optional[str] = None) -> str:
        """
        提交任务
        
        如果有空闲槽位，立即执行；否则加入等待队列。
        
        Args:
            task_id: 任务 ID
            coroutine_factory: 协程工厂函数（返回协程的函数）
            generation: 代数
            parent_id: 父代 ID
            
        Returns:
            str: 任务 ID
        """
        if len(self._running) < self.max_concurrent:
            await self._start_task(task_id, coroutine_factory, generation, parent_id)
            logger.info(f"Task {task_id} started immediately. Running: {len(self._running)}/{self.max_concurrent}")
        else:
            pending = PendingTask(
                task_id=task_id,
                coroutine_factory=coroutine_factory,
                generation=generation,
                parent_id=parent_id
            )
            self._waiting.append(pending)
            logger.info(f"Task {task_id} added to waiting queue. Queue size: {len(self._waiting)}")
        
        return task_id
    
    async def _start_task(self,
                          task_id: str,
                          coroutine_factory: Callable[[], Awaitable[Any]],
                          generation: int,
                          parent_id: Optional[str]) -> None:
        """启动任务"""
        async_task = asyncio.create_task(
            self._run_and_collect(task_id, coroutine_factory, generation, parent_id)
        )
        wrapper = TaskWrapper(
            task_id=task_id,
            async_task=async_task,
            generation=generation,
            parent_id=parent_id
        )
        self._running[task_id] = wrapper
    
    async def _run_and_collect(self,
                               task_id: str,
                               coroutine_factory: Callable[[], Awaitable[Any]],
                               generation: int,
                               parent_id: Optional[str]) -> None:
        """执行任务并收集结果"""
        started_at = datetime.now().isoformat()
        task_result: Optional[TaskResult] = None  # 预先初始化，防止 CancelledError 导致未定义
        try:
            # 调用协程工厂获取协程并执行
            result = await coroutine_factory()
            
            # 解析结果（假设返回 (op_name, success, final_state) 元组）
            if isinstance(result, tuple) and len(result) >= 3:
                _, success, final_state = result[:3]
            else:
                success = True
                final_state = result if isinstance(result, dict) else {"result": result}
            
            task_result = TaskResult(
                task_id=task_id,
                success=success,
                final_state=final_state,
                generation=generation,
                parent_id=parent_id,
                started_at=started_at
            )
            
            if success:
                logger.info(f"Task {task_id} completed successfully")
            else:
                logger.warning(f"Task {task_id} completed with failure")
        
        except asyncio.CancelledError:
            # 处理任务被取消的情况（Ctrl+C 等）
            logger.info(f"Task {task_id} was cancelled")
            task_result = TaskResult(
                task_id=task_id,
                success=False,
                final_state={"error": "Task was cancelled"},
                generation=generation,
                parent_id=parent_id,
                error="Task was cancelled",
                started_at=started_at
            )
            # 重新抛出以保持取消语义
            raise
                
        except Exception as e:
            logger.error(f"Task {task_id} raised exception: {e}", exc_info=True)
            task_result = TaskResult(
                task_id=task_id,
                success=False,
                final_state={"error": str(e)},
                generation=generation,
                parent_id=parent_id,
                error=str(e),
                started_at=started_at
            )
        
        finally:
            # 存储结果（确保 task_result 已定义）
            if task_result is not None:
                self._results[task_id] = task_result
            # 从运行列表中移除
            if task_id in self._running:
                del self._running[task_id]
    
    async def refill_from_queue(self) -> int:
        """
        从等待队列填充任务池
        
        Returns:
            int: 填充的任务数量
        """
        filled = 0
        while len(self._running) < self.max_concurrent and self._waiting:
            pending = self._waiting.popleft()
            await self._start_task(
                pending.task_id,
                pending.coroutine_factory,
                pending.generation,
                pending.parent_id
            )
            filled += 1
            logger.info(f"Refilled task {pending.task_id} from queue. Running: {len(self._running)}/{self.max_concurrent}")
        
        return filled
    
    def pop_completed_results(self) -> List[TaskResult]:
        """
        获取并清除所有已完成的结果
        
        Returns:
            List[TaskResult]: 已完成的任务结果列表
        """
        results = list(self._results.values())
        self._results.clear()
        return results
    
    def has_capacity(self) -> bool:
        """检查是否有空闲槽位"""
        return len(self._running) < self.max_concurrent
    
    def get_available_slots(self) -> int:
        """获取空闲槽位数量"""
        return self.max_concurrent - len(self._running)
    
    def is_idle(self) -> bool:
        """检查任务池是否空闲（无运行任务和等待任务）"""
        return len(self._running) == 0 and len(self._waiting) == 0
    
    def get_running_count(self) -> int:
        """获取正在运行的任务数"""
        return len(self._running)
    
    def get_waiting_count(self) -> int:
        """获取等待队列长度"""
        return len(self._waiting)
    
    def get_pending_result_count(self) -> int:
        """获取待处理结果数量"""
        return len(self._results)
    
    async def wait_for_any(self, timeout: Optional[float] = None) -> bool:
        """
        等待任意任务完成
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否有任务完成
        """
        if not self._running:
            return False
        
        tasks = [w.async_task for w in self._running.values()]
        try:
            done, _ = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED
            )
            return len(done) > 0
        except Exception as e:
            logger.error(f"Error waiting for tasks: {e}")
            return False
    
    async def wait_for_all(self, timeout: Optional[float] = None) -> None:
        """等待所有运行中的任务完成"""
        if not self._running:
            return
        
        tasks = [w.async_task for w in self._running.values()]
        try:
            await asyncio.wait(tasks, timeout=timeout)
        except Exception as e:
            logger.error(f"Error waiting for all tasks: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取任务池状态"""
        return {
            "max_concurrent": self.max_concurrent,
            "running": len(self._running),
            "waiting": len(self._waiting),
            "pending_results": len(self._results),
            "total_submitted": self._task_counter,
            "running_task_ids": list(self._running.keys()),
            "waiting_task_ids": [p.task_id for p in self._waiting]
        }
    
    async def cancel_all_running(self) -> int:
        """
        取消所有正在运行的任务
        
        Returns:
            int: 取消的任务数量
        """
        if not self._running:
            return 0
        
        cancelled_count = 0
        task_ids = list(self._running.keys())
        
        for task_id in task_ids:
            wrapper = self._running.get(task_id)
            if wrapper and not wrapper.async_task.done():
                wrapper.async_task.cancel()
                cancelled_count += 1
                logger.info(f"Task {task_id} cancelled")
        
        # 等待取消的任务结束
        if cancelled_count > 0:
            tasks = [w.async_task for w in self._running.values()]
            try:
                await asyncio.wait(tasks, timeout=5.0)
            except Exception as e:
                logger.warning(f"Error waiting for cancelled tasks: {e}")
        
        # 清理正在运行的任务列表
        self._running.clear()
        
        logger.info(f"Cancelled {cancelled_count} running tasks")
        return cancelled_count
    
    def clear_waiting_queue(self) -> int:
        """
        清空等待队列
        
        Returns:
            int: 清除的任务数量
        """
        count = len(self._waiting)
        if count > 0:
            self._waiting.clear()
            logger.info(f"Cleared {count} tasks from waiting queue")
        return count

