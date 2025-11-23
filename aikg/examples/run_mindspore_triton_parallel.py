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
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.worker.manager import register_local_worker
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task


def get_op_name_and_task_desc():
    pair1 = ('relu', '''
import mindspore as ms
from mindspore import nn


class Model(nn.Cell):
    """
    ReLU激活函数模型
    """
    def __init__(self):
        super(Model, self).__init__()

    def construct(self, x: ms.Tensor) -> ms.Tensor:

        return ms.ops.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = ms.ops.randn(batch_size, dim, dtype=ms.float16)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
    ''')

    pair2 = ('tanh', '''
import mindspore as ms
from mindspore import nn


class Model(nn.Cell):
    """
    Tanh激活函数模型
    """
    def __init__(self):
        super(Model, self).__init__()

    def construct(self, x: ms.Tensor) -> ms.Tensor:

        return ms.ops.tanh(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = ms.ops.randn(batch_size, dim, dtype=ms.float16)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
    ''')
    return [pair1, pair2]


async def run_mindspore_triton_parallel():
    op_name_and_task_desc = get_op_name_and_task_desc()

    task_pool = TaskPool()
    
    # 新写法：一行代码注册 LocalWorker（多卡）
    await register_local_worker([0, 1], backend="ascend", arch="ascend910b4")
    
    config = load_config("triton_ascend", backend="ascend")  # or load_config("/your-path-to-config/xxx_config.yaml")

    check_env_for_task("mindspore", "ascend", "triton_ascend", config)

    for i, (op_name, task_desc) in enumerate(op_name_and_task_desc):
        task = Task(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            dsl="triton_ascend",
            backend="ascend",
            arch="ascend910b4",
            config=config,
            framework="mindspore",
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()
    for op_name, result, _ in results:
        if result:
            print(f"Task {op_name} passed")
        else:
            print(f"Task {op_name} failed")

if __name__ == "__main__":
    asyncio.run(run_mindspore_triton_parallel())
