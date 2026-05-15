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

"""ST: cache replay -> kernel generation -> compile+accuracy verification."""

import ast
import re
from pathlib import Path

import pytest

from akg_agents import get_project_root
from akg_agents.core.worker.manager import get_worker_manager, register_local_worker
from akg_agents.core_v2.llm.cache import discover_cpu_attention_replay_scenarios
from akg_agents.op.config.config_validator import load_config
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.utils.environment_check import check_env_for_task
from ..utils import get_device_id


def _extract_model_init_arg_names(task_desc: str) -> list[str]:
    tree = ast.parse(task_desc)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    arg_names = [arg.arg for arg in fn.args.args]
                    if arg_names and arg_names[0] == "self":
                        return arg_names[1:]
                    return arg_names
    return []


def _normalize_replayed_modelnew_init(coder_code: str, init_arg_names: list[str]) -> str:
    if not init_arg_names:
        return coder_code

    expected_sig = ", ".join(["self", *init_arg_names])
    init_def_pattern = r"def\s+__init__\(self\s*\):"
    if not re.search(init_def_pattern, coder_code):
        return coder_code

    patched = re.sub(init_def_pattern, f"def __init__({expected_sig}):", coder_code, count=1)
    first_arg = init_arg_names[0]
    assign_pattern = rf"self\.{first_arg}\s*=\s*[^\n]+"
    if re.search(assign_pattern, patched):
        patched = re.sub(assign_pattern, f"self.{first_arg} = {first_arg}", patched, count=1)

    return patched


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


SCENARIOS = discover_cpu_attention_replay_scenarios()


@pytest.mark.level2
@pytest.mark.st
@pytest.mark.cache
@pytest.mark.replay
@pytest.mark.torch
@pytest.mark.cpp
@pytest.mark.cpu
@pytest.mark.x86_64
@pytest.mark.use_model
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
async def test_cache_replay_kernel_verification_cpu_cpp(scenario, monkeypatch):
    assert scenario.cache_file_path.exists(), f"cache sample not found: {scenario.cache_file_path}"

    device_id = get_device_id()
    await register_local_worker([device_id], backend="cpu", arch="x86_64")

    config_path = Path(get_project_root()) / "op" / "config" / "cpp_coderonly_config.yaml"
    config = load_config(config_path=str(config_path))
    config["cache_mode"] = "replay"
    config["cache_session_hash"] = scenario.session_hash

    monkeypatch.setenv("AKG_AGENTS_STREAM_OUTPUT", "off")
    monkeypatch.setenv("AKG_AGENTS_CACHE_FILE_PATH", str(scenario.cache_file_path))

    check_env_for_task("torch", "cpu", "cpp", config)

    framework_code = _build_attention_task_desc(scenario.shape, scenario.sm_scale)

    task = LangGraphTask(
        op_name=f"akg_cache_replay_{scenario.name}",
        task_desc=framework_code,
        task_id=f"st_cache_replay_{scenario.name}",
        dsl="cpp",
        backend="cpu",
        arch="x86_64",
        config=config,
        framework="torch",
        workflow="coder_only_workflow",
    )

    _, success, final_state = await task.run()

    coder_code = str(final_state.get("coder_code") or "").strip()
    assert coder_code, f"replay workflow did not produce coder_code, state={final_state}"

    # Historical replay samples may contain hardcoded __init__ values while
    # verification always instantiates ModelNew(*get_init_inputs()).
    expected_init_args = _extract_model_init_arg_names(framework_code)
    coder_code = _normalize_replayed_modelnew_init(coder_code, expected_init_args)

    worker = await get_worker_manager().select(backend="cpu", arch="x86_64")
    if not worker:
        raise RuntimeError("No available worker for backend=cpu, arch=x86_64")

    verifier = KernelVerifier(
        op_name=f"cache_replay_{scenario.name}",
        framework_code=_build_attention_task_desc(scenario.shape, scenario.sm_scale),
        task_id=f"st_cache_verify_{scenario.name}",
        framework="torch",
        dsl="cpp",
        backend="cpu",
        arch="x86_64",
        impl_func_name="ModelNew",
        config=config,
        worker=worker,
    )
    verify_ok, verify_error = await verifier.run({"coder_code": coder_code}, device_id=device_id)
    assert verify_ok, f"KernelVerifier failed: {verify_error}"

    if not success:
        replay_error = final_state.get("verifier_error") or final_state.get("error_message")
        assert replay_error, "workflow failed but did not expose replay verifier error"
