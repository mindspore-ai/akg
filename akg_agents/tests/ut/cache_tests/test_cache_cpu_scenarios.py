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

from pathlib import Path

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
            "content": f"fake-content-{self.calls}",
            "reasoning_content": f"fake-reasoning-{self.calls}",
            "tool_calls": [],
            "usage": {"total_tokens": 42, "prompt_tokens": 21, "completion_tokens": 21},
            "meta": {
                "messages": messages,
                "tools": tools or [],
                "config": config,
            },
        }


SCENARIOS = [
    {
        "name": "attention_small_baseline",
        "session_hash": "cpu_attn_small_record_v1",
        "agent_hash": "0@1",
        "shape": "B1_H8_S128_D64",
        "temperature": 0.2,
        "max_tokens": 2048,
    },
    {
        "name": "attention_medium_longseq",
        "session_hash": "cpu_attn_mid_record_v1",
        "agent_hash": "0@2",
        "shape": "B4_H16_S1024_D64",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    {
        "name": "attention_edge_expire_guard",
        "session_hash": "cpu_attn_edge_record_v1",
        "agent_hash": "0@3",
        "shape": "B1_H32_S2048_D128",
        "temperature": 0.0,
        "max_tokens": 8192,
    },
]


def _scenario_cache_file(cache_dir: Path, scenario_name: str) -> Path:
    return cache_dir / f"llm_cache_cpu_{scenario_name}.json"


def _build_messages(scenario: dict) -> list:
    return [
        {
            "role": "system",
            "content": "Generate x86_64 C++ torch attention kernel code.",
        },
        {
            "role": "user",
            "content": (
                f"Scenario={scenario['name']} Shape={scenario['shape']} "
                "Backend=cpu Arch=x86_64 DSL=cpp Framework=torch"
            ),
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
async def test_record_and_replay_cpu_cache_scenario(tmp_path, scenario):
    cache_file = _scenario_cache_file(tmp_path, scenario["name"])
    if cache_file.exists():
        cache_file.unlink()

    provider = FakeProvider()
    client = LLMClient(provider=provider)
    cache = LLMCache(
        max_memory_size=20,
        cache_file_path=str(cache_file),
        expire_seconds=3600,
    )
    attach_cache_to_client(client, cache=cache)

    messages = _build_messages(scenario)

    recorded = await client.generate(
        messages,
        cache_mode="record",
        cache_session_hash=scenario["session_hash"],
        cache_agent_hash=scenario["agent_hash"],
        temperature=scenario["temperature"],
        max_tokens=scenario["max_tokens"],
    )

    replayed = await client.generate(
        [{"role": "user", "content": "different message should still hit session cache"}],
        cache_mode="replay",
        cache_session_hash=scenario["session_hash"],
        cache_agent_hash=scenario["agent_hash"],
        temperature=scenario["temperature"],
        max_tokens=scenario["max_tokens"],
    )

    assert provider.calls == 1
    assert replayed == recorded
    assert cache_file.exists()
    assert cache_file.stat().st_size > 0


@pytest.mark.asyncio
async def test_replay_miss_strict_no_live_call(tmp_path):
    cache_file = tmp_path / "llm_cache_cpu_replay_strict_guard.json"
    if cache_file.exists():
        cache_file.unlink()

    provider = FakeProvider()
    client = LLMClient(provider=provider)
    cache = LLMCache(cache_file_path=str(cache_file), expire_seconds=3600)
    attach_cache_to_client(client, cache=cache)

    with pytest.raises(RuntimeError, match="Replay cache miss"):
        await client.generate(
            [{"role": "user", "content": "replay-miss"}],
            cache_mode="replay",
            cache_session_hash="strict-miss",
            cache_agent_hash="0@9",
        )

    assert provider.calls == 0


@pytest.mark.asyncio
async def test_off_mode_fast_fakeprovider_suggestion(tmp_path):
    cache_file = tmp_path / "llm_cache_cpu_off_mode_guard.json"
    if cache_file.exists():
        cache_file.unlink()

    provider = FakeProvider()
    client = LLMClient(provider=provider)
    cache = LLMCache(cache_file_path=str(cache_file), expire_seconds=3600)
    attach_cache_to_client(client, cache=cache)

    messages = [{"role": "user", "content": "off-mode-check"}]
    r1 = await client.generate(messages, cache_mode="off")
    r2 = await client.generate(messages, cache_mode="off")

    assert provider.calls == 2
    assert r1 != r2


@pytest.mark.asyncio
async def test_replay_session_prefix_fallback_when_agent_hash_drift(tmp_path):
    cache_file = tmp_path / "llm_cache_cpu_replay_prefix_fallback.json"
    if cache_file.exists():
        cache_file.unlink()

    provider = FakeProvider()
    client = LLMClient(provider=provider)
    cache = LLMCache(cache_file_path=str(cache_file), expire_seconds=3600)
    attach_cache_to_client(client, cache=cache)

    recorded = await client.generate(
        _build_messages(SCENARIOS[0]),
        cache_mode="record",
        cache_session_hash="prefix-fallback-session",
        cache_agent_hash="0@1",
    )

    replayed = await client.generate(
        [{"role": "user", "content": "agent-hash changed but should still replay"}],
        cache_mode="replay",
        cache_session_hash="prefix-fallback-session",
        cache_agent_hash="coder@1",
    )

    assert replayed == recorded
    assert provider.calls == 1
