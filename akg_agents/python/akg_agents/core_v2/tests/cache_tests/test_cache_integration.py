import pytest

from akg_agents.core_v2.llm.cache import LLMCache, attach_cache_to_client
from akg_agents.core_v2.llm.client import LLMClient


class FakeProvider:
    def __init__(self):
        self.model_name = "fake-model"
        self.calls = 0

    async def generate(self, messages, tools=None, **config):
        self.calls += 1
        return {
            "content": f"ok-{self.calls}",
            "reasoning_content": "",
            "tool_calls": [],
            "usage": {},
        }


@pytest.mark.asyncio
async def test_cache_attach_and_hit(temp_cache_file):
    cache = LLMCache(
        max_memory_size=10,
        cache_file_path=temp_cache_file,
        expire_seconds=3600,
        enable=True,
    )
    provider = FakeProvider()
    client = LLMClient(provider=provider)
    attach_cache_to_client(client, cache=cache)

    messages = [{"role": "user", "content": "hello"}]

    result1 = await client.generate(messages)
    result2 = await client.generate(messages)

    assert provider.calls == 1
    assert result1 == result2

    cache.clear_all_cache()


@pytest.mark.asyncio
async def test_cache_refresh_and_disable(temp_cache_file):
    cache = LLMCache(
        max_memory_size=10,
        cache_file_path=temp_cache_file,
        expire_seconds=3600,
        enable=True,
    )
    provider = FakeProvider()
    client = LLMClient(provider=provider)
    attach_cache_to_client(client, cache=cache)

    messages = [{"role": "user", "content": "hello"}]

    await client.generate(messages)
    await client.generate(messages, cache_refresh=True)
    await client.generate(messages, cache_enable=False)

    assert provider.calls == 3

    cache.clear_all_cache()


@pytest.mark.asyncio
async def test_cache_session_record_and_replay(temp_cache_file):
    cache = LLMCache(
        max_memory_size=10,
        cache_file_path=temp_cache_file,
        expire_seconds=3600,
        enable=True,
    )
    provider = FakeProvider()
    client = LLMClient(provider=provider)
    attach_cache_to_client(client, cache=cache)

    messages_a = [{"role": "user", "content": "step-a"}]
    messages_b = [{"role": "user", "content": "step-b"}]
    session_hash = "session-001"

    result_a_record = await client.generate(
        messages_a,
        cache_mode="record",
        cache_session_hash=session_hash,
        cache_agent_hash="0@1",
    )
    result_b_record = await client.generate(
        messages_b,
        cache_mode="record",
        cache_session_hash=session_hash,
        cache_agent_hash="0@2",
    )
    calls_after_record = provider.calls

    result_a_replay = await client.generate(
        [{"role": "user", "content": "different-a"}],
        cache_mode="replay",
        cache_session_hash=session_hash,
        cache_agent_hash="0@1",
    )
    result_b_replay = await client.generate(
        [{"role": "user", "content": "different-b"}],
        cache_mode="replay",
        cache_session_hash=session_hash,
        cache_agent_hash="0@2",
    )

    assert provider.calls == calls_after_record
    assert result_a_replay == result_a_record
    assert result_b_replay == result_b_record


@pytest.mark.asyncio
async def test_cache_session_replay_miss_fallback_and_fill(temp_cache_file):
    cache = LLMCache(
        max_memory_size=10,
        cache_file_path=temp_cache_file,
        expire_seconds=3600,
        enable=True,
    )
    provider = FakeProvider()
    client = LLMClient(provider=provider)
    attach_cache_to_client(client, cache=cache)

    session_hash = "session-miss"
    result_first = await client.generate(
        [{"role": "user", "content": "first-call"}],
        cache_mode="replay",
        cache_session_hash=session_hash,
        cache_agent_hash="0@3",
    )
    calls_after_first = provider.calls

    result_second = await client.generate(
        [{"role": "user", "content": "second-call"}],
        cache_mode="replay",
        cache_session_hash=session_hash,
        cache_agent_hash="0@3",
    )

    assert calls_after_first == 1
    assert provider.calls == 1
    assert result_second == result_first
