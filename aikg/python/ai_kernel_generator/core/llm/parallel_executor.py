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


class ParallelExecutor:
    """并行执行器工具类，提供多种并行执行方法"""

    @staticmethod
    async def _run_task_async(chain, inputs):
        """异步运行单个任务"""
        return await chain.ainvoke(inputs)

    @staticmethod
    async def _run_all_tasks_async(tasks):
        """异步并行执行所有任务"""
        coroutines = [ParallelExecutor._run_task_async(chain, inputs) for chain, inputs in tasks]
        return await asyncio.gather(*coroutines)

    @staticmethod
    def dynamic_dispatcher(task_pair):
        """动态分发器，用于RunnableLambda并行执行"""
        chain, inputs = task_pair
        return chain.invoke(inputs)

    @staticmethod
    def create_wrapper_function(chain, task_idx):
        """创建包装函数，用于从大的输入字典中提取所需输入"""
        def wrapper(all_inputs):
            task_input = all_inputs[f"inputs_{task_idx}"]
            return chain.invoke(task_input)
        return wrapper
