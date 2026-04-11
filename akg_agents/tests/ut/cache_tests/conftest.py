# Copyright 2026 Huawei Technologies Co., Ltd
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

import tempfile

import pytest

from akg_agents.core_v2.llm.cache import LLMCache


@pytest.fixture
def temp_cache_file():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        yield f.name


@pytest.fixture
def llm_cache(temp_cache_file):
    cache = LLMCache(
        max_memory_size=10,
        cache_file_path=temp_cache_file,
        expire_seconds=3600,
    )
    yield cache
    cache.clear_all_cache()


@pytest.fixture
def test_messages():
    return [{"role": "user", "content": "测试AKG算子生成"}]


@pytest.fixture
def test_result():
    return {
        "content": "生成的算子代码：xxx",
        "reasoning_content": "算子生成推理过程",
        "tool_calls": [],
        "usage": {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50}
    }


@pytest.fixture
def test_params():
    return {"temperature": 0.2, "max_tokens": 8192}
