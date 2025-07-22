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

import pytest
import asyncio
from pathlib import Path
from ai_kernel_generator import get_project_root
from ai_kernel_generator.database.database import Database, DEFAULT_DATABASE_PATH
from ..utils import get_benchmark_name

DEFAULT_BENCHMARK_PATH = Path(get_project_root()).parent.parent / "benchmark"


def gen_task_code(framework_path: str = "", impl_path: str = ""):
    framework_code = ""
    impl_code = ""
    if framework_path:
        framework_code = Path(framework_path).read_text()
    if impl_path:
        impl_code = Path(impl_path).read_text()
    return impl_code, framework_code


@pytest.mark.level0
def test_insert():
    # 初始化系统
    db_system = Database()

    arch = "ascend910b4"
    backend = "ascend"
    framework = "torch"
    impl_type = "triton"

    # 插入示例
    id_list = [2, 3, 4, 5]
    benchmark_name = get_benchmark_name(id_list)
    for op_name in benchmark_name:
        impl_code, framework_code = gen_task_code(
            framework_path=DEFAULT_BENCHMARK_PATH / "kernelbench" / framework / op_name / f"{op_name}_{framework}.py",
            impl_path=DEFAULT_DATABASE_PATH / "triton" / arch / f"{op_name}_{impl_type}.py"
        )
        db_system.insert(impl_code, framework_code, backend, arch, impl_type, framework)


@pytest.mark.level0
def test_sample():
    # 初始化系统
    db_system = Database()

    arch = "ascend910b4"
    backend = "ascend"
    framework = "torch"
    impl_type = "triton"

    # 查询示例
    op_name = get_benchmark_name([1])[0]
    query_code_path = DEFAULT_DATABASE_PATH / "triton" / arch / f"{op_name}_{impl_type}.py"
    query_code = Path(query_code_path).read_text()
    db_system.samples(query_code, backend=backend, arch=arch, impl_type=impl_type, top_k=3)


@pytest.mark.level0
def test_delete():
    # 初始化系统
    db_system = Database()

    arch = "ascend910b4"
    backend = "ascend"
    framework = "torch"
    impl_type = "triton"

    # 删除示例
    op_name = get_benchmark_name([3])[0]
    impl_path = DEFAULT_DATABASE_PATH / "triton" / arch / f"{op_name}_{impl_type}.py"
    impl_code = Path(impl_path).read_text()
    db_system.delete(impl_code, backend, arch, impl_type)


@pytest.mark.asyncio
@pytest.mark.level0
async def test_async_database():
    # 初始化系统
    db_system = Database()

    arch = "ascend910b4"
    backend = "ascend"
    framework = "torch"
    impl_type = "triton"

    # 定义并发任务: 2个添加任务、2个查询任务、2个删除任务
    async def insert_task(benchmark_id):
        op_name = get_benchmark_name([benchmark_id])[0]
        impl_code, framework_code = gen_task_code(
            framework_path=DEFAULT_BENCHMARK_PATH / "kernelbench" / framework / op_name / f"{op_name}_{framework}.py",
            impl_path=DEFAULT_DATABASE_PATH / "triton" / arch / f"{op_name}_{impl_type}.py"
        )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, db_system.insert, impl_code, framework_code, backend, arch, impl_type, framework)

    async def sample_task(benchmark_id):
        op_name = get_benchmark_name([benchmark_id])[0]
        query_code_path = DEFAULT_DATABASE_PATH / "triton" / arch / f"{op_name}_{impl_type}.py"
        query_code = Path(query_code_path).read_text()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: db_system.samples(query_code, backend=backend, arch=arch, impl_type=impl_type))

    async def delete_task(benchmark_id):
        op_name = get_benchmark_name([benchmark_id])[0]
        impl_path = DEFAULT_DATABASE_PATH / "triton" / arch / f"{op_name}_{impl_type}.py"
        impl_code = Path(impl_path).read_text()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, db_system.delete, impl_code, backend, arch, impl_type)

    # 创建多个并发任务
    tasks = [
        insert_task(6),
        insert_task(7),
        sample_task(1),
        sample_task(1),
        delete_task(6),
        delete_task(7)
    ]

    # 执行所有并发任务
    return await asyncio.gather(*tasks)
