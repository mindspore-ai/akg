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

import json
import os
import re
from copy import deepcopy
from pathlib import Path

import pytest

from akg_agents.op.config.config_validator import load_config
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.environment_check import check_env_for_task
from akg_agents import get_project_root


DEFAULT_CACHE_FILE = Path("~/.akg/llm_cache/llm_test_cache.json").expanduser()
OUTPUT_CACHE_DIR = Path(get_project_root()).resolve().parent.parent / ".cache"

SCENARIOS = [
    {
        "name": "attention_small_baseline",
        "file": "llm_cache_cpu_attention_small_baseline.json",
        "session_hash": "cpu_attn_real_small_record_v1",
        "shape": (1, 8, 128, 64),
        "sm_scale": 0.5,
    },
    {
        "name": "attention_medium_longseq",
        "file": "llm_cache_cpu_attention_medium_longseq.json",
        "session_hash": "cpu_attn_real_medium_record_v1",
        "shape": (4, 16, 1024, 64),
        "sm_scale": 0.125,
    },
    {
        "name": "attention_edge_large_head",
        "file": "llm_cache_cpu_attention_edge_expire_guard.json",
        "session_hash": "cpu_attn_real_edge_record_v1",
        "shape": (1, 32, 2048, 128),
        "sm_scale": 0.08838834764831845,
    },
]


def _build_attention_task_desc(shape, sm_scale: float) -> str:
    b, h, l, d = shape
    return f'''
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, sm_scale):
        super().__init__()
        self.sm_scale = sm_scale

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            scale=self.sm_scale
        )


def get_inputs():
    B, H, L, D = {b}, {h}, {l}, {d}
    dtype = torch.float32
    q = torch.empty((B, H, L, D), dtype=dtype).normal_(mean=0.0, std=0.5)
    k = torch.empty((B, H, L, D), dtype=dtype).normal_(mean=0.0, std=0.5)
    v = torch.empty((B, H, L, D), dtype=dtype).normal_(mean=0.0, std=0.5)
    return [q, k, v]


def get_init_inputs():
    sm_scale = {sm_scale}
    return [sm_scale]
'''


def _clear_default_cache_file() -> None:
    DEFAULT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CACHE_FILE.write_text("{}", encoding="utf-8")


def _read_default_cache_file() -> dict:
    if not DEFAULT_CACHE_FILE.exists():
        return {}
    content = DEFAULT_CACHE_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    return json.loads(content)


def _looks_like_coder_payload(content: str) -> bool:
    if not isinstance(content, str) or not content.strip():
        return False
    return "class ModelNew" in content


def _normalize_to_final_converged_result(recorded_cache: dict, session_hash: str) -> dict:
    """Keep replay export stable by promoting the final converged coder result.

    Record may contain iterative steps like `session:0@1`, `session:0@2`...
    Replay usually hits early step first, so we rewrite export to canonicalize
    `session:0@1` and coder payload entries to the final converged result.
    """
    normalized = deepcopy(recorded_cache)
    pattern = re.compile(rf"^{re.escape(session_hash)}:0@(\d+)$")

    coder_steps = []
    for key, item in normalized.items():
        match = pattern.match(key)
        if not match:
            continue
        result = item.get("result") if isinstance(item, dict) else None
        content = (result or {}).get("content") if isinstance(result, dict) else ""
        if _looks_like_coder_payload(content):
            coder_steps.append((int(match.group(1)), key))

    if not coder_steps:
        return normalized

    coder_steps.sort(key=lambda x: x[0])
    final_step, final_key = coder_steps[-1]
    final_item = normalized[final_key]
    final_result = deepcopy(final_item.get("result") or {})

    canonical_key = f"{session_hash}:0@1"
    normalized[canonical_key] = deepcopy(final_item)
    normalized[canonical_key]["result"] = deepcopy(final_result)

    for _, key in coder_steps:
        if key != canonical_key:
            normalized.pop(key, None)

    for item in normalized.values():
        if not isinstance(item, dict):
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        content = result.get("content", "")
        if _looks_like_coder_payload(content):
            item["result"] = deepcopy(final_result)

    # Keep an explicit converged step hint for easier manual inspection.
    normalized.setdefault("_metadata", {})["final_converged_coder_step"] = final_step
    return normalized


async def _run_single_real_workflow(cache_mode: str, session_hash: str, scenario: dict) -> None:
    os.environ["AKG_AGENTS_STREAM_OUTPUT"] = "off"
    await register_local_worker([0], backend="cpu", arch="x86_64")

    config = load_config("cpp")
    config["cache_mode"] = cache_mode
    config["cache_session_hash"] = session_hash

    check_env_for_task("torch", "cpu", "cpp", config)

    task = LangGraphTask(
        op_name=f"akg_cache_{scenario['name']}",
        task_desc=_build_attention_task_desc(scenario["shape"], scenario["sm_scale"]),
        task_id="0",
        dsl="cpp",
        backend="cpu",
        arch="x86_64",
        config=config,
        framework="torch",
        workflow="coder_only_workflow",
    )

    _, success, final_state = await task.run()
    if not success:
        raise AssertionError(f"workflow failed for {scenario['name']}: {final_state}")


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.cpp
@pytest.mark.cpu
@pytest.mark.x86_64
@pytest.mark.use_model
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
async def test_record_replay_and_export_real_cache_samples(scenario, monkeypatch):
    _clear_default_cache_file()
    output_path = OUTPUT_CACHE_DIR / scenario["file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Real record path: should call online model and write real cache entries.
    await _run_single_real_workflow("record", scenario["session_hash"], scenario)
    recorded_cache = _read_default_cache_file()

    assert recorded_cache, "record mode did not write cache file"
    assert any(k.startswith(f"{scenario['session_hash']}:") for k in recorded_cache)

    normalized_cache = _normalize_to_final_converged_result(
        recorded_cache,
        scenario["session_hash"],
    )

    output_path.write_text(
        json.dumps(normalized_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2) Replay strict guard: override API key to invalid value;
    # replay should still pass without live model call.
    monkeypatch.setenv("AKG_AGENTS_API_KEY", "invalid_for_replay_guard")
    await _run_single_real_workflow("replay", scenario["session_hash"], scenario)
