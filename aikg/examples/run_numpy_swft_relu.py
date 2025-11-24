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

from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.utils.environment_check import check_env_for_task
import asyncio
import os
os.environ['AIKG_STREAM_OUTPUT'] = 'on'


def get_op_name():
    return 'relu'


def get_task_desc():
    return '''
import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)


batch_size = 16
dim = 16384


def get_inputs():
    x = np.random.randn(batch_size, dim).astype(np.float32)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed 
'''


async def run_numpy_swft_single():
    op_name = get_op_name()
    task_desc = get_task_desc()

    task_pool = TaskPool()
    device_pool = DevicePool([0])
    config = load_config("swft")  # or load_config("/your-path-to-config/xxx_config.yaml")
    # config = load_config(config_path="./python/ai_kernel_generator/config/default_swft_config.yaml")

    check_env_for_task("numpy", "ascend", "swft", config)

    task = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="swft",
        backend="ascend",
        arch="ascend310p3",
        config=config,
        device_pool=device_pool,
        framework="numpy",
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
    asyncio.run(run_numpy_swft_single())
