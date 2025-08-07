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
from ai_kernel_generator.database.database import Database, DEFAULT_DATABASE_PATH, RetrievalStrategy
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
@pytest.mark.asyncio
async def test_insert():
    # 初始化系统
    db_system = Database()

    arch = "ascend310p3"
    backend = "ascend"
    framework = "numpy"
    dsl = "swft"

    # 插入示例
    id_list = [49, 50]
    benchmark_name = get_benchmark_name(id_list)
    for i, op_name in enumerate(benchmark_name):
        name = op_name.strip(f"{str(id_list[i])}_")
        impl_code, framework_code = gen_task_code(
            framework_path=DEFAULT_BENCHMARK_PATH / "kernelbench" / framework / op_name / f"{op_name}_{framework}.py",
            impl_path=DEFAULT_DATABASE_PATH / dsl / arch / name / "aigen" / f"{name}_{dsl}.py"
        )
        await db_system.insert(impl_code, framework_code, backend, arch, dsl, framework)


@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_mode", [
    (RetrievalStrategy.NAIVETY),
    (RetrievalStrategy.MMR),
    # (RetrievalStrategy.OPTIMALITY),
    (RetrievalStrategy.RULE)
])
async def test_samples(strategy_mode):
    # 初始化系统
    db_system = Database()

    arch = "ascend310p3"
    backend = "ascend"
    framework = "numpy"
    dsl = "swft"

    # 查询示例
    benchmark_id = 53
    op_name = get_benchmark_name([benchmark_id])[0]
    name = op_name.strip(f"{str(benchmark_id)}_")
    impl_path = DEFAULT_DATABASE_PATH / dsl / arch / name / "aigen" / f"{name}_{dsl}.py"
    impl_code = Path(impl_path).read_text()
    results = await db_system.samples(
        output_content=["impl_code"],
        strategy_mode=strategy_mode,
        impl_code=impl_code,
        backend=backend,
        arch=arch,
        dsl=dsl,
        sample_num=3
    )
    print('\n' + '-' * 80)
    for result in results:
        print('\n'.join(result["impl_code"].split("\n")[:10]))
        print('-' * 80)


@pytest.mark.level0
@pytest.mark.asyncio
async def test_delete():
    # 初始化系统
    db_system = Database()

    arch = "ascend310p3"
    backend = "ascend"
    framework = "numpy"
    dsl = "swft"

    # 删除示例
    benchmark_id = 50
    op_name = get_benchmark_name([benchmark_id])[0]
    name = op_name.strip(f"{str(benchmark_id)}_")
    impl_path = DEFAULT_DATABASE_PATH / dsl / arch / name / "aigen" / f"{name}_{dsl}.py"
    impl_code = Path(impl_path).read_text()
    await db_system.delete(impl_code, backend, arch, dsl)


@pytest.mark.asyncio
@pytest.mark.level0
async def test_async_database():
    # 初始化系统
    db_system = Database()

    arch = "ascend310p3"
    backend = "ascend"
    framework = "numpy"
    dsl = "swft"

    # 定义并发任务: 2个添加任务、2个查询任务、2个删除任务
    async def insert_task(benchmark_id):
        op_name = get_benchmark_name([benchmark_id])[0]
        name = op_name.strip(f"{str(benchmark_id)}_")
        impl_code, framework_code = gen_task_code(
            framework_path=DEFAULT_BENCHMARK_PATH / "kernelbench" / framework / op_name / f"{op_name}_{framework}.py",
            impl_path=DEFAULT_DATABASE_PATH / dsl / arch / name / "aigen" / f"{name}_{dsl}.py"
        )
        return await db_system.insert(impl_code, framework_code, backend, arch, dsl, framework)

    async def sample_task(benchmark_id):
        op_name = get_benchmark_name([benchmark_id])[0]
        name = op_name.strip(f"{str(benchmark_id)}_")
        impl_path = DEFAULT_DATABASE_PATH / dsl / arch / name / "aigen" / f"{name}_{dsl}.py"
        impl_code = Path(impl_path).read_text()
        return await db_system.samples(
            output_content=["impl_code"],
            impl_code=impl_code,
            backend=backend,
            arch=arch,
            dsl=dsl,
            sample_num=3
        )

    async def delete_task(benchmark_id):
        op_name = get_benchmark_name([benchmark_id])[0]
        name = op_name.strip(f"{str(benchmark_id)}_")
        impl_path = DEFAULT_DATABASE_PATH / dsl / arch / name / "aigen" / f"{name}_{dsl}.py"
        impl_code = Path(impl_path).read_text()
        return await db_system.delete(impl_code, backend, arch, dsl)

    # 创建多个并发任务
    tasks = [
        insert_task(47),
        insert_task(48),
        sample_task(53),
        sample_task(53),
        delete_task(47),
        delete_task(48)
    ]

    # 执行所有并发任务
    return await asyncio.gather(*tasks)


@pytest.mark.level0
@pytest.mark.asyncio
async def test_random_database():
    # 初始化系统
    db_system = Database(random_mode=True)

    arch = "ascend310p3"
    backend = "ascend"
    framework = "numpy"
    dsl = "swft"

    # 插入示例
    id_list = [49, 50]
    benchmark_name = get_benchmark_name(id_list)
    for i, op_name in enumerate(benchmark_name):
        name = op_name.strip(f"{str(id_list[i])}_")
        impl_code, framework_code = gen_task_code(
            framework_path=DEFAULT_BENCHMARK_PATH / "kernelbench" / framework / op_name / f"{op_name}_{framework}.py",
            impl_path=DEFAULT_DATABASE_PATH / dsl / arch / name / "aigen" / f"{name}_{dsl}.py"
        )
        await db_system.insert(impl_code, framework_code, backend, arch, dsl, framework)

    # 查询示例
    benchmark_id = 49
    op_name = get_benchmark_name([benchmark_id])[0]
    name = op_name.strip(f"{str(benchmark_id)}_")
    impl_path = DEFAULT_DATABASE_PATH / dsl / arch / name / "aigen" / f"{name}_{dsl}.py"
    impl_code = Path(impl_path).read_text()
    results = await db_system.samples(
        output_content=["impl_code"],
        impl_code=impl_code,
        backend=backend,
        arch=arch,
        dsl=dsl,
        sample_num=3
    )
    print('\n' + '-' * 80)
    for result in results:
        print('\n'.join(result["impl_code"].split("\n")[:10]))
        print('-' * 80)

    # 删除示例
    benchmark_id = 49
    op_name = get_benchmark_name([benchmark_id])[0]
    name = op_name.strip(f"{str(benchmark_id)}_")
    impl_path = DEFAULT_DATABASE_PATH / dsl / arch / name / "aigen" / f"{name}_{dsl}.py"
    impl_code = Path(impl_path).read_text()
    await db_system.delete(impl_code, backend, arch, dsl)
