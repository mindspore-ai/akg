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

import pytest
from akg_agents.core_v2.llm.cache import LLMCache, generate_cache_key


class TestLLMCacheCore:
    def test_generate_cache_key_consistent(self, test_messages, test_params):
        key1 = generate_cache_key(test_messages, **test_params)
        key2 = generate_cache_key(test_messages, **test_params)
        assert key1 == key2

    def test_generate_cache_key_different_messages(self, test_messages, test_params):
        key1 = generate_cache_key(test_messages, **test_params)
        key2 = generate_cache_key([{"role": "user", "content": "不同的内容"}], **test_params)
        assert key1 != key2

    def test_cache_set_and_get(self, llm_cache, test_messages, test_result, test_params):
        llm_cache.set(test_messages, test_result, **test_params)
        cached_result = llm_cache.get(test_messages, **test_params)
        assert cached_result == test_result

    def test_cache_miss(self, llm_cache, test_params):
        cached_result = llm_cache.get([{"role": "user", "content": "不存在的请求"}], **test_params)
        assert cached_result is None

    def test_cache_clear_all(self, llm_cache, test_messages, test_result, test_params):
        llm_cache.set(test_messages, test_result, **test_params)
        assert len(llm_cache.get_all_cache_keys()) == 1
        llm_cache.clear_all_cache()
        assert len(llm_cache.get_all_cache_keys()) == 0

    def test_cache_messages_complete_saved(self, llm_cache, test_messages, test_result, test_params):
        llm_cache.set(test_messages, test_result, **test_params)
        cache_key = generate_cache_key(test_messages, **test_params)
        cache_item = llm_cache.get_cache_by_key(cache_key)
        assert cache_item.get("messages") == test_messages

    def test_cache_expire(self, temp_cache_file, test_messages, test_result, test_params):
        expiring_cache = LLMCache(
            max_memory_size=10,
            cache_file_path=temp_cache_file,
            expire_seconds=1,
        )
        expiring_cache.set(test_messages, test_result, **test_params)

        cache_key = generate_cache_key(test_messages, **test_params)
        expiring_cache._memory_cache[cache_key]["create_time"] -= 2
        expiring_cache._local_cache[cache_key]["create_time"] -= 2

        assert expiring_cache.get(test_messages, **test_params) is None

    def test_cache_custom_key_set_and_get(self, llm_cache, test_messages, test_result, test_params):
        custom_key = "session_hash:agent_hash"
        llm_cache.set(test_messages, test_result, cache_key=custom_key, **test_params)

        cached_result = llm_cache.get(
            [{"role": "user", "content": "不同消息也可命中自定义键"}],
            cache_key=custom_key,
            **test_params,
        )
        assert cached_result == test_result

    def test_cache_legacy_enable_kwarg_is_ignored(self, temp_cache_file, test_messages, test_result, test_params):
        cache = LLMCache(enable=False, cache_file_path=temp_cache_file)
        cache.set(test_messages, test_result, **test_params)
        assert cache.get(test_messages, **test_params) == test_result
