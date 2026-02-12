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
CoderDatabase 测试用例

测试内容：
1. 数据库创建与 auto_update
2. 算子的增删操作（包含 skip/overwrite 模式）
3. samples 采样功能
4. 异步并发操作
"""

import pytest
import asyncio
import shutil
from pathlib import Path
from akg_agents import get_project_root
from akg_agents.op.database.coder_database import CoderDatabase, DEFAULT_BENCHMARK_PATH
from akg_agents.op.database.coder_vector_store import CoderVectorStore
from akg_agents.op.config.config_validator import load_config
from akg_agents.op.config.config_validator import load_config

# 测试配置
TEST_DSL = "triton_ascend"
TEST_FRAMEWORK = "torch"
TEST_BACKEND = "ascend"
TEST_ARCH = "ascend910b4"
MAX_TEST_FILES = 3  # 限制测试文件数量，避免测试时间过长

# 测试临时目录（在当前项目目录下）
TEST_DB_DIR = Path(get_project_root()).parent.parent / "test_temp_db"


def clear_all_instances():
    """清除所有单例实例缓存"""
    CoderDatabase._instances.clear()
    CoderVectorStore._instances.clear()


def get_test_op_files(dsl: str = TEST_DSL, max_files: int = MAX_TEST_FILES):
    """获取测试用的算子文件信息"""
    benchmark_path = Path(DEFAULT_BENCHMARK_PATH)
    impl_dir = benchmark_path / dsl / "impl" / "static_shape" / "elemwise"
    
    if not impl_dir.exists():
        pytest.skip(f"Benchmark directory not found: {impl_dir}")
    
    impl_files = list(impl_dir.glob("*.py"))[:max_files]
    test_ops = []
    
    for impl_file in impl_files:
        op_name = impl_file.stem
        framework_file = benchmark_path / "static_shape" / "elemwise" / f"{op_name}.py"
        
        if framework_file.exists():
            with open(impl_file, 'r', encoding='utf-8') as f:
                impl_code = f.read()
            with open(framework_file, 'r', encoding='utf-8') as f:
                framework_code = f.read()
            test_ops.append({
                "op_name": op_name,
                "impl_code": impl_code,
                "framework_code": framework_code
            })
    
    return test_ops


@pytest.fixture(scope="module")
def temp_database_path():
    """创建临时数据库目录（module级别共享）"""
    TEST_DB_DIR.mkdir(parents=True, exist_ok=True)
    yield str(TEST_DB_DIR)
    # 清理
    if TEST_DB_DIR.exists():
        shutil.rmtree(TEST_DB_DIR)


@pytest.fixture(scope="module")
def config():
    """加载测试配置"""
    return load_config(TEST_DSL)


@pytest.fixture(autouse=True)
def setup_test():
    """每个测试前清理单例缓存"""
    clear_all_instances()
    yield


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_auto_update(temp_database_path, config):
    """测试：auto_update 功能（新建与 skip 模式）"""
    db = CoderDatabase(
        database_path=temp_database_path,
        config=config
    )
    
    # 第一次 auto_update - 新建数据库
    await db.auto_update(
        dsl=TEST_DSL,
        framework=TEST_FRAMEWORK,
        backend=TEST_BACKEND,
        arch=TEST_ARCH,
        ref_type="impl",
        max_files=MAX_TEST_FILES,
        update_mode="skip"
    )
    
    # 验证数据库目录已创建
    db_dir = Path(temp_database_path) / TEST_ARCH / TEST_DSL
    assert db_dir.exists(), f"Database directory not created: {db_dir}"
    
    # 验证有数据写入
    cases = list(db_dir.iterdir())
    initial_count = len(cases)
    assert initial_count > 0, "No cases were inserted"
    assert initial_count <= MAX_TEST_FILES, f"More files than expected: {initial_count}"
    print(f"✓ auto_update 成功创建 {initial_count} 个算子记录")
    
    # 清除 _auto_update_completed 以允许再次执行
    db._auto_update_completed.clear()
    
    # 第二次 auto_update（skip 模式，不应重复插入）
    await db.auto_update(
        dsl=TEST_DSL,
        framework=TEST_FRAMEWORK,
        backend=TEST_BACKEND,
        arch=TEST_ARCH,
        ref_type="impl",
        max_files=MAX_TEST_FILES,
        update_mode="skip"
    )
    
    final_count = len(list(db_dir.iterdir()))
    assert final_count == initial_count, "Skip mode should not duplicate entries"
    print(f"✓ auto_update skip 模式正常工作，记录数: {final_count}")


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_insert_delete_and_modes(temp_database_path, config):
    """测试：算子的插入、删除以及 skip/overwrite 模式
    
    逻辑：
    - test_auto_update 已经创建了数据库并插入了 test_op
    - 直接用 skip 模式插入同一个算子，验证 skip 功能
    - 用 overwrite 模式插入，验证 overwrite 功能
    - 最后删除，验证 delete 功能
    """
    db = CoderDatabase(
        database_path=temp_database_path,
        config=config
    )
    
    test_ops = get_test_op_files(max_files=1)
    if not test_ops:
        pytest.skip("No test operators available")
    
    test_op = test_ops[0]
    db_dir = Path(temp_database_path) / TEST_ARCH / TEST_DSL
    
    # 找到对应的 case 目录
    from akg_agents.utils.common_utils import get_md5_hash
    md5_hash = get_md5_hash(impl_code=test_op["impl_code"], backend=TEST_BACKEND, arch=TEST_ARCH, dsl=TEST_DSL)
    case_dir = db_dir / md5_hash
    
    assert case_dir.exists(), f"Case directory should exist from test_auto_update: {case_dir}"
    initial_mtime = (case_dir / "metadata.json").stat().st_mtime
    print(f"✓ 验证数据已存在: {test_op['op_name']}")
    
    # ===== 测试 skip 模式（数据已存在，应跳过）=====
    await asyncio.sleep(0.1)  # 确保时间戳不同
    
    await db.insert(
        impl_code=test_op["impl_code"],
        framework_code=test_op["framework_code"],
        backend=TEST_BACKEND,
        arch=TEST_ARCH,
        dsl=TEST_DSL,
        framework=TEST_FRAMEWORK,
        mode="skip"
    )
    
    skip_mtime = (case_dir / "metadata.json").stat().st_mtime
    assert skip_mtime == initial_mtime, "Skip mode should not modify existing files"
    print("✓ skip 模式正常工作")
    
    # ===== 测试 overwrite 模式 =====
    await asyncio.sleep(0.1)
    
    await db.insert(
        impl_code=test_op["impl_code"],
        framework_code=test_op["framework_code"],
        backend=TEST_BACKEND,
        arch=TEST_ARCH,
        dsl=TEST_DSL,
        framework=TEST_FRAMEWORK,
        mode="overwrite"
    )
    
    overwrite_mtime = (case_dir / "metadata.json").stat().st_mtime
    assert overwrite_mtime > initial_mtime, "Overwrite mode should update files"
    print("✓ overwrite 模式正常工作")
    
    # ===== 测试删除 =====
    db.delete(
        impl_code=test_op["impl_code"],
        backend=TEST_BACKEND,
        arch=TEST_ARCH,
        dsl=TEST_DSL
    )
    
    assert not case_dir.exists(), f"Case directory should be deleted: {case_dir}"
    print(f"✓ 成功删除算子: {test_op['op_name']}")


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_concurrent_operations(temp_database_path, config):
    """测试：并发操作（多个插入任务同时执行）"""
    # 使用独立的子目录
    test_path = str(Path(temp_database_path) / "concurrent_test")
    Path(test_path).mkdir(parents=True, exist_ok=True)
    
    db = CoderDatabase(
        database_path=test_path,
        config=config
    )
    
    test_ops = get_test_op_files(max_files=3)
    if len(test_ops) < 2:
        pytest.skip("Need at least 2 test operators")
    
    async def insert_task(op_info):
        await db.insert(
            impl_code=op_info["impl_code"],
            framework_code=op_info["framework_code"],
            backend=TEST_BACKEND,
            arch=TEST_ARCH,
            dsl=TEST_DSL,
            framework=TEST_FRAMEWORK,
            mode="skip"
        )
        return op_info["op_name"]
    
    # 并发执行
    tasks = [insert_task(op) for op in test_ops]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 验证结果
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    
    db_dir = Path(test_path) / TEST_ARCH / TEST_DSL
    case_count = len(list(db_dir.iterdir())) if db_dir.exists() else 0
    
    print(f"✓ 并发操作完成: 成功 {len(successful)}, 失败 {len(failed)}, 数据库记录 {case_count}")
    
    assert len(successful) > 0, "At least one operation should succeed"
    assert case_count == len(successful), f"Database count mismatch: {case_count} vs {len(successful)}"


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.use_vector_store
@pytest.mark.asyncio
async def test_samples(temp_database_path, config):
    """测试：samples 采样功能"""
    # 使用独立的子目录
    test_path = str(Path(temp_database_path) / "samples_test")
    Path(test_path).mkdir(parents=True, exist_ok=True)
    
    db = CoderDatabase(
        database_path=test_path,
        config=config
    )
    
    # 先构建数据库
    await db.auto_update(
        dsl=TEST_DSL,
        framework=TEST_FRAMEWORK,
        backend=TEST_BACKEND,
        arch=TEST_ARCH,
        ref_type="impl",
        max_files=MAX_TEST_FILES,
        update_mode="skip"
    )
    
    # 获取一个测试算子用于查询
    test_ops = get_test_op_files(max_files=1)
    if not test_ops:
        pytest.skip("No test operators available")
    
    test_op = test_ops[0]
    
    # 执行 samples 查询
    try:
        results = await db.samples(
            output_content=["impl_code", "op_name"],
            sample_num=2,
            impl_code=test_op["impl_code"],
            framework_code=test_op["framework_code"],
            backend=TEST_BACKEND,
            arch=TEST_ARCH,
            dsl=TEST_DSL,
            framework=TEST_FRAMEWORK
        )
        
        assert isinstance(results, list), "samples should return a list"
        print(f"✓ samples 查询成功，返回 {len(results)} 个结果")
        
        for i, result in enumerate(results):
            op_name = result.get("op_name", "unknown")
            print(f"  - 结果 {i+1}: {op_name}")
            
    except ValueError as e:
        # 如果没有找到匹配的算子，这是可以接受的
        print(f"✓ samples 查询完成（无匹配结果）: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
