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

import asyncio
import logging
from typing import Any, List
from collections.abc import Callable

logger = logging.getLogger(__name__)

class TaskPool:
    """
    异步任务池，用于管理和控制异步任务的执行。

    Args:
        max_concurrency (int): 最大并发任务数，默认为4。

    Example:
        >>> pool = TaskPool(max_concurrency=5)  # 创建一个最大并发数为5的任务池
        >>> task1 = pool.create_task(my_coro1, arg1, kwarg1=value)  # 创建并启动任务
        >>> task2 = pool.create_task(my_coro2, arg2)  # 创建并启动任务
        >>> await pool.wait_all()  # 等待所有任务完成
    """

    def __init__(self, max_concurrency: int = 4):
        # 通过信号量控制最大并发量
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.tasks: List[asyncio.Task] = []  # 跟踪所有活动任务

    def create_task(self, coro_func: Callable, *args: Any, **kwargs: Any) -> asyncio.Task:
        """
        创建并跟踪异步任务

        Args:
            coro_func: 需要执行的协程函数
            *args: 传递给协程函数的位置参数
            **kwargs: 传递给协程函数的关键字参数

        Returns:
            asyncio.Task: 已创建的任务对象

        Example:
            >>> pool = TaskPool()
            >>> task = pool.create_task(my_coro, arg1, kwarg1=value)
        """
        task = asyncio.create_task(self.run(coro_func, *args, **kwargs))
        self.tasks.append(task)
        # 添加自动清理回调
        task.add_done_callback(
            lambda t: (
                self.tasks.remove(t),
                logger.debug(f'Task {t.get_name()} completed')
            )
        )
        return task

    
    async def run(self, coro_func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        在任务池中执行协程函数

        Args:
            coro_func: 需要执行的协程函数引用
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            协程函数的执行结果

        Raises:
            Exception: 执行过程中发生的异常
        """
        async with self.semaphore:
            try:
                return await coro_func(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f'Task execution failed: {str(e)}')
                raise
    
    async def wait_all(self) -> List[Any]:
        """
        等待所有任务完成并返回结果

        Returns:
            List[Any]: 所有任务的执行结果列表

        Example:
            >>> pool = TaskPool()
            >>> task1 = pool.create_task(my_coro1)
            >>> task2 = pool.create_task(my_coro2)
            >>> results = await pool.wait_all()
        """
        if self.tasks:
            return await asyncio.gather(*self.tasks)
        return []
