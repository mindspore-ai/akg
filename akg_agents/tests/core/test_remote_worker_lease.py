# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""RemoteWorker exact-lease propagation tests (no HTTP server required)."""

import asyncio
from types import SimpleNamespace

import pytest

from akg_agents.core.worker import remote_worker as remote_worker_module
from akg_agents.core.worker.remote_worker import RemoteWorker


@pytest.mark.asyncio
async def test_remote_worker_attaches_and_clears_active_lease(monkeypatch):
    worker = RemoteWorker("http://worker.invalid")
    calls = []

    async def fake_post(url, files, data, read_timeout, task_id):
        calls.append((url, dict(data)))
        if url.endswith("/acquire_device"):
            return {"device_id": 3, "lease_id": 17}
        if url.endswith("/verify"):
            return {"success": True, "log": "", "artifacts": {}}
        return {"status": "ok"}

    monkeypatch.setattr(worker, "_post_with_reconnect", fake_post)

    device_id, lease_id = await worker.acquire_device("same-op")
    await worker.verify(b"tar", "same-op", "op", timeout=1)
    verify_data = calls[-1][1]
    assert verify_data["device_id"] == "3"
    assert verify_data["lease_id"] == "17"

    await worker.release_device(device_id, lease_id, "same-op")
    await worker.verify(b"tar", "same-op", "op", timeout=1)
    verify_after_release = calls[-1][1]
    assert "device_id" not in verify_after_release
    assert "lease_id" not in verify_after_release


@pytest.mark.asyncio
async def test_remote_worker_lease_context_is_per_coroutine(monkeypatch):
    worker = RemoteWorker("http://worker.invalid")
    seen = {}
    both_acquired = asyncio.Event()
    acquire_count = 0

    async def fake_post(url, files, data, read_timeout, task_id):
        nonlocal acquire_count
        task_name = asyncio.current_task().get_name()
        if url.endswith("/acquire_device"):
            acquire_count += 1
            if acquire_count == 2:
                both_acquired.set()
            return ({"device_id": 0, "lease_id": 101}
                    if task_name == "lease-a"
                    else {"device_id": 1, "lease_id": 202})
        if url.endswith("/verify"):
            seen[task_name] = (data.get("device_id"), data.get("lease_id"))
            return {"success": True, "log": "", "artifacts": {}}
        return {"status": "ok"}

    monkeypatch.setattr(worker, "_post_with_reconnect", fake_post)

    async def run_one():
        device_id, lease_id = await worker.acquire_device("same-op")
        await both_acquired.wait()
        await worker.verify(b"tar", "same-op", "op", timeout=1)
        await worker.release_device(device_id, lease_id, "same-op")

    await asyncio.gather(
        asyncio.create_task(run_one(), name="lease-a"),
        asyncio.create_task(run_one(), name="lease-b"),
    )
    assert seen == {"lease-a": ("0", "101"), "lease-b": ("1", "202")}


@pytest.mark.asyncio
async def test_remote_profile_network_failure_has_canonical_shape(monkeypatch):
    worker = RemoteWorker("http://worker.invalid")

    async def fail(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(worker, "_post_with_reconnect", fail)
    result = await worker.profile(
        b"tar", "task", "op", {"timeout": 1})

    assert result["gen_time"] is None
    assert result["base_time"] is None
    assert result["per_shape_gen_us"] == []
    assert result["artifacts"] == {}
    assert result["error"] == "network down"


@pytest.mark.asyncio
async def test_remote_profile_read_timeout_covers_base_and_generation(
        monkeypatch):
    worker = RemoteWorker("http://worker.invalid")
    seen = {}

    async def fake_post(url, files, data, read_timeout, task_id):
        seen["read_timeout"] = read_timeout
        return {"artifacts": {}}

    monkeypatch.setattr(worker, "_post_with_reconnect", fake_post)
    monkeypatch.setattr(
        remote_worker_module, "worker_timing",
        lambda: SimpleNamespace(http_read_margin=3.0),
    )

    await worker.profile(b"tar", "task", "op", {"timeout": 7})

    # LocalWorker may spend one complete budget on each serial section.
    assert seen["read_timeout"] == 17.0
