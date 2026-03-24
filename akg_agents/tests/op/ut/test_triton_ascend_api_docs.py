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

"""Triton Ascend API 文档加载逻辑单元测试。"""

from __future__ import annotations

import asyncio
import warnings
from types import SimpleNamespace

from akg_agents.op.agents.kernel_gen import KernelGen
from akg_agents.op.utils.triton_ascend_api_docs import (
    get_offline_triton_ascend_api_docs,
    resolve_triton_ascend_api_docs,
    update_offline_triton_ascend_api_docs,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from akg_agents.core.agent.coder import Coder


TRITON_ASCEND_DOCS_DIR = "op/resources/docs/triton_ascend_docs"


def test_offline_doc_contains_missing_api_section():
    offline_doc = get_offline_triton_ascend_api_docs()

    assert "# Triton API 参考手册" in offline_doc
    assert "## 当前版本不存在的 API" in offline_doc
    assert "tl.extract_slice" in offline_doc


def test_update_offline_doc_writes_current_snapshot(tmp_path, monkeypatch):
    target_path = tmp_path / "api.md"

    monkeypatch.setattr(
        "akg_agents.op.utils.triton_ascend_api_docs.get_aggregated_triton_ascend_api_docs",
        lambda: "# sdk snapshot\n",
    )

    updated_path = update_offline_triton_ascend_api_docs(target_path)

    assert updated_path == target_path
    assert target_path.read_text(encoding="utf-8") == "# sdk snapshot\n"


def test_resolve_triton_ascend_api_docs_falls_back_to_offline(monkeypatch):
    offline_doc = get_offline_triton_ascend_api_docs()

    async def fake_list_matching(*args, **kwargs):
        return []

    fake_manager = SimpleNamespace(list_matching=fake_list_matching)
    monkeypatch.setattr(
        "akg_agents.op.utils.triton_ascend_api_docs.get_aggregated_triton_ascend_api_docs",
        lambda: (_ for _ in ()).throw(RuntimeError("sdk unavailable")),
    )

    assert asyncio.run(
        resolve_triton_ascend_api_docs(backend="ascend", arch="ascend910b4", worker_manager=fake_manager)
    ) == offline_doc


def test_resolve_triton_ascend_api_docs_prefers_worker(monkeypatch):
    remote_doc = "# remote triton ascend api"

    class FakeRemoteWorker:
        async def get_doc(self, doc_name: str) -> str:
            assert doc_name == "triton_ascend_api"
            return remote_doc

    async def fake_list_matching(*args, **kwargs):
        return [FakeRemoteWorker()]

    fake_manager = SimpleNamespace(list_matching=fake_list_matching)
    monkeypatch.setattr(
        "akg_agents.op.utils.triton_ascend_api_docs.get_aggregated_triton_ascend_api_docs",
        lambda: (_ for _ in ()).throw(RuntimeError("sdk unavailable")),
    )

    assert asyncio.run(
        resolve_triton_ascend_api_docs(backend="ascend", arch="ascend910b4", worker_manager=fake_manager)
    ) == remote_doc


def test_kernel_gen_uses_resolve_triton_ascend_api_docs(monkeypatch):
    kernel_gen = KernelGen()
    remote_doc = "# remote triton ascend api"

    monkeypatch.setattr(
        "akg_agents.op.agents.kernel_gen.resolve_triton_ascend_api_docs",
        lambda backend="", arch="", worker_manager=None: asyncio.sleep(0, result=remote_doc),
    )

    assert asyncio.run(
        kernel_gen._load_aggregated_api_docs("triton_ascend", backend="ascend", arch="ascend910b4")
    ) == remote_doc
    assert asyncio.run(
        kernel_gen._load_aggregated_api_docs("triton_cuda", backend="cuda", arch="a100")
    ) == ""


def test_coder_uses_same_resolve_logic_for_triton_ascend(monkeypatch):
    expected = "# local remote offline docs"
    monkeypatch.setattr(
        "akg_agents.core.agent.coder.resolve_triton_ascend_api_docs",
        lambda backend="", arch="", worker_manager=None: asyncio.sleep(0, result=expected),
    )

    coder = Coder(
        op_name="test_op",
        task_desc="task",
        dsl="triton_ascend",
        framework="torch",
        backend="ascend",
        arch="ascend910b4",
        config={
            "agent_model_config": {},
            "docs_dir": {"coder": TRITON_ASCEND_DOCS_DIR},
        },
    )

    assert coder.base_doc["api_docs"] == ""
    assert asyncio.run(coder._ensure_api_docs_loaded()) == expected
    assert coder.base_doc["api_docs"] == expected
