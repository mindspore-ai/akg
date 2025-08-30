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
import json
import tempfile
import time
from pathlib import Path

from ai_kernel_generator.utils.collector import get_collector


@pytest.mark.level0
@pytest.mark.asyncio
async def test_singleton_pattern():
    """测试单例模式"""
    c1 = await get_collector()
    c2 = await get_collector()
    assert c1 is c2


@pytest.mark.level0
@pytest.mark.asyncio
async def test_basic_collect_and_overwrite():
    """测试基础收集和覆盖功能"""
    collector = await get_collector()
    initial_count = collector._counter

    # 首次收集
    await collector.collect({"agent_name": "test_basic", "hash": "h1", "data": "v1"})
    assert collector._counter == initial_count + 1

    # 覆盖同一key
    await collector.collect({"agent_name": "test_basic", "hash": "h1", "data": "v2"})
    assert collector._counter == initial_count + 2  # 计数递增
    assert collector._store[("test_basic", "h1")]["data"]["data"] == "v2"


@pytest.mark.level0
@pytest.mark.asyncio
async def test_concurrent_collection():
    """测试并发收集能力"""
    collector = await get_collector()
    initial_store_size = len(collector._store)
    initial_counter = collector._counter

    async def batch(prefix: str, n: int):
        for i in range(n):
            await collector.collect({"agent_name": f"conc_{prefix}", "hash": f"h{i}", "v": i})

    await asyncio.gather(batch("a", 2), batch("b", 3))

    # 验证新增的数据
    assert len(collector._store) == initial_store_size + 5
    assert collector._counter == initial_counter + 5


@pytest.mark.level0
@pytest.mark.asyncio
async def test_task_data_preparation():
    """测试任务数据准备和文件操作"""
    collector = await get_collector()

    with tempfile.TemporaryDirectory() as temp_dir:
        collector.set_config({"log_dir": str(Path(temp_dir) / "test.log")})

        # 收集测试专用数据
        await collector.collect({"agent_name": "task_test_a", "hash": "h1", "task_id": "test_t1"})
        await collector.collect({"agent_name": "task_test_b", "hash": "h2", "task_id": "test_t2"})
        await collector.collect({"agent_name": "task_test_c", "hash": "h3"})  # 无task_id

        # 准备test_t1数据
        files = await collector.prepare_and_remove_data(task_id="test_t1")

        # 验证至少包含我们的测试数据
        assert len(files) >= 2  # test_t1 + 无task_id数据，可能还有其他数据

        # 验证文件有效性
        for f in files:
            assert Path(f).exists()
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                assert "agent_name" in data


@pytest.mark.level0
@pytest.mark.asyncio
async def test_database_data_preparation():
    """测试数据库数据准备"""
    collector = await get_collector()

    with tempfile.TemporaryDirectory() as temp_dir:
        collector.set_config({"log_dir": str(Path(temp_dir) / "test.log")})

        task_info = {
            "backend": "test_backend",
            "arch_name": "test_arch",
            "framework": "test_framework",
            "dsl": "test_dsl",
            "task_desc": "desc",
            "coder_code": "code",
            "profile_res": (1.0, 2.0)
        }

        db_file = collector.prepare_database_data(task_info)
        assert db_file != ""
        assert Path(db_file).exists()

        with open(db_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert data["hash"] == "database"
            assert data["backend"] == "test_backend"


@pytest.mark.level0
@pytest.mark.asyncio
async def test_config_management():
    """测试配置管理和错误处理"""
    collector = await get_collector()

    # 正常配置
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "test.log"
        collector.set_config({"log_dir": str(log_path)})
        assert collector._save_dir == log_path.parent / "save_data"

    # 默认配置
    collector.set_config(None)
    assert collector._save_dir == Path.cwd() / "save_data"

    # 错误处理
    collector._save_dir = None
    assert collector._save_json_file({"test": "data"}, "test.json") is False


@pytest.mark.level0
@pytest.mark.asyncio
async def test_filename_generation():
    """测试文件名生成"""
    collector = await get_collector()

    # 格式测试: {session_id}_{sequence_id}_{agent_name}_{hash}.json
    current_counter = collector._counter
    name = collector._generate_filename("a/b", "c\\d")
    assert "_a_b_c_d" in name  # 检查格式包含预期的agent_name和hash
    assert name.endswith(".json")
    assert "/" not in name and "\\" not in name
    assert name.startswith("session_")  # 检查以session_开头
    assert f"_{current_counter}_" in name  # 检查包含当前counter

    # 唯一性测试 - 每次调用counter会递增，所以文件名应该不同
    name1 = collector._generate_filename("agent", "hash")
    name2 = collector._generate_filename("agent", "hash")
    assert name1 != name2  # 因为counter递增，所以文件名不同


@pytest.mark.level0
@pytest.mark.asyncio
async def test_comprehensive_workflow():
    """测试完整工作流"""
    collector = await get_collector()
    initial_store_size = len(collector._store)

    with tempfile.TemporaryDirectory() as temp_dir:
        collector.set_config({"log_dir": str(Path(temp_dir) / "test.log")})

        # 使用唯一标识避免与其他测试数据冲突
        task_prefix = "workflow_test"

        # 并发模拟agent工作流
        async def agent_work(name: str, task: str):
            for stage in ["input", "process", "output"]:
                await collector.collect({
                    "agent_name": f"{task_prefix}_{name}",
                    "hash": f"{stage}_h",
                    "task_id": f"{task_prefix}_{task}",
                    "stage": stage
                })

        await asyncio.gather(
            agent_work("coder", "t1"),
            agent_work("designer", "t1"),
            agent_work("verifier", "t2"),
            collector.collect({"agent_name": f"{task_prefix}_monitor", "hash": "global"})
        )

        # 验证新增的数据（10个）
        assert len(collector._store) == initial_store_size + 10

        # 处理任务数据
        t1_files = await collector.prepare_and_remove_data(task_id=f"{task_prefix}_t1")
        # t1数据(6) + 全局数据(1)，可能还有其他无task_id数据
        assert len(t1_files) >= 7

        t2_files = await collector.prepare_and_remove_data(task_id=f"{task_prefix}_t2")
        # 至少包含t2数据(3)
        assert len(t2_files) >= 3

        # 验证文件完整性
        for file_path in t1_files + t2_files:
            assert Path(file_path).exists()
            # 验证是我们的测试数据
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert task_prefix in data.get("agent_name", "")


@pytest.mark.level0
@pytest.mark.asyncio
async def test_data_field_validation():
    """测试数据字段校验功能"""
    collector = await get_collector()

    # 测试1: 标准agent完整数据（不应有警告）
    complete_data = {
        'hash': 'test_hash',
        'agent_name': 'designer',
        'op_name': 'test_op',
        'dsl': 'triton',
        'backend': 'cuda',
        'arch': 'A100',
        'framework': 'torch',
        'task_desc': 'test task',
        'model_name': 'deepseek_r1_default',
        'content': 'generated code',
        'formatted_prompt': 'test prompt',
        'reasoning_content': 'reasoning',
        'response_metadata': 'metadata'
    }
    await collector.collect(complete_data)

    # 测试2: 标准agent缺少字段（应该有警告）
    incomplete_data = {
        'hash': 'test_hash2',
        'agent_name': 'coder',
        'op_name': 'test_op2'
        # 缺少多个必需字段
    }
    await collector.collect(incomplete_data)

    # 测试3: feature_extractor完整数据（不应有警告）
    feature_complete = {
        'hash': 'feature_hash',
        'agent_name': 'feature_extractor',
        'model_name': 'deepseek_r1_default',
        'content': 'feature content',
        'formatted_prompt': 'feature prompt',
        'reasoning_content': 'feature reasoning',
        'response_metadata': 'feature metadata'
    }
    await collector.collect(feature_complete)

    # 测试4: feature_extractor缺少字段（应该有警告）
    feature_incomplete = {
        'agent_name': 'feature_extractor',
        'hash': 'feature_hash2'
        # 缺少其他字段
    }
    await collector.collect(feature_incomplete)


@pytest.mark.level0
@pytest.mark.asyncio
async def test_empty_value_detection():
    """测试空值检测逻辑"""
    collector = await get_collector()

    # 测试各种空值和边界情况
    test_data = {
        'hash': 'test_hash',
        'agent_name': 'verifier',
        'op_name': '',     # 空字符串
        'dsl': None,       # None值
        'backend': False,  # False应该是合法值
        'arch': 0,         # 0应该是合法值
        'framework': 'mindspore',
        'task_desc': 'test task',
        'model_name': 'deepseek_r1_default',
        'content': 'generated code',
        'formatted_prompt': 'test prompt',
        'reasoning_content': 'reasoning',
        'response_metadata': 'metadata'
    }
    await collector.collect(test_data)


@pytest.mark.level0
@pytest.mark.asyncio
async def test_is_empty_value_helper():
    """测试_is_empty_value辅助方法的关键用例"""
    collector = await get_collector()

    # 测试关键用例
    assert collector._is_empty_value(None) == True
    assert collector._is_empty_value('') == True
    assert collector._is_empty_value('   ') == True
    assert collector._is_empty_value([]) == True
    assert collector._is_empty_value({}) == True
    assert collector._is_empty_value(False) == False
    assert collector._is_empty_value(0) == False
    assert collector._is_empty_value('hello') == False
