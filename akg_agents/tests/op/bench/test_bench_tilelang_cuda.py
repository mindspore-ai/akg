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

import os
import pytest
from pathlib import Path
from collections import defaultdict
from akg_agents.core.async_pool.task_pool import TaskPool

# 自动选择 Task 实现：优先使用 LangGraphTask，否则使用原 Task
try:
    import langgraph
    from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
    _USE_LANGGRAPH = True
except ImportError:
    from akg_agents.core.task import Task as AIKGTask
    _USE_LANGGRAPH = False
from akg_agents.core.worker.manager import register_local_worker
from ..utils import (
    get_kernelbench_op_name, get_multikernelbench_op_name,
    get_kernelbench_task_desc, get_multikernelbench_task_desc,
    get_akg_kernels_bench_op_name, get_akg_kernels_bench_task_desc,
    get_evokernel_mhc_op_name, get_evokernel_task_desc,
    add_op_prefix, generate_beautiful_test_report, get_device_id
)
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task

os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'
device_id = get_device_id()


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.tilelang
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_tilelang_cuda_a100():
    """测试 KernelBench - PyTorch TileLang CUDA A100"""
    framework = "torch"
    dsl = "tilelang_cuda"
    backend = "cuda"
    arch = "a100"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    config = load_config(dsl=dsl, backend=backend)

    check_env_for_task(framework, backend, dsl, config)

    await register_local_worker([device_id], backend=backend, arch=arch)

    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19, ], framework=framework)

    if benchmark_name is None:
        raise RuntimeError("在 KernelBench 中未找到指定序号的任务文件，请检查 task_index_list 参数是否正确")

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(
            benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark=benchmark)

        task = AIKGTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.tilelang
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_akg_kernels_bench_torch_tilelang_cuda_a100():
    """测试 AIKGBench - PyTorch TileLang CUDA A100"""
    framework = "torch"
    dsl = "tilelang_cuda"
    backend = "cuda"
    arch = "a100"
    benchmark = "AIKGBench"
    category = "dynamic"
    subcategory = "elemwise"

    task_pool = TaskPool()
    config = load_config(dsl=dsl, backend=backend)

    check_env_for_task(framework, backend, dsl, config)

    await register_local_worker([device_id], backend=backend, arch=arch)

    benchmark_name = get_akg_kernels_bench_op_name(
        op_name="elemwise_add_001_var",
        category=category,
        subcategory=subcategory,
        framework=framework
    )

    if benchmark_name is None:
        raise RuntimeError("在 AIKGBench 中未找到指定的任务文件")

    for i in range(len(benchmark_name)):
        task_desc = get_akg_kernels_bench_task_desc(
            benchmark_name[i], category=category, framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark=benchmark)

        task = AIKGTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.tilelang
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_evokernel_mhc_torch_tilelang_cuda_a100():
    """测试 EvoKernel MHC - PyTorch TileLang CUDA A100"""
    framework = "torch"
    dsl = "tilelang_cuda"
    backend = "cuda"
    arch = "a100"
    benchmark = "EvoKernel"
    category = "MHC"

    task_pool = TaskPool()
    config = load_config(dsl=dsl, backend=backend)

    check_env_for_task(framework, backend, dsl, config)

    await register_local_worker([device_id], backend=backend, arch=arch)

    benchmark_name = get_evokernel_mhc_op_name(op_name="05_StreamWrite")

    if benchmark_name is None:
        raise RuntimeError(f"在 EvoKernel {category} 中未找到指定的操作")

    for i in range(len(benchmark_name)):
        task_desc = get_evokernel_task_desc(
            benchmark_name[i], category=category)
        op_name = add_op_prefix(f"{category}_{benchmark_name[i]}", benchmark=benchmark)

        task = AIKGTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )