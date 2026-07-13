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

"""Pure-asyncio tests for the lease-based DevicePool (no NPU / torch needed)."""

import asyncio
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "python"))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from akg_agents.core.async_pool.device_pool import DevicePool  # noqa: E402
from akg_agents.core.worker.interface import WorkerInterface  # noqa: E402

_fails = []


class _FakeWorker(WorkerInterface):
    """Minimal WorkerInterface to exercise the device_lease CM without torch."""

    def __init__(self):
        self.released = []
        self._next = 0

    async def verify(self, *a, **k):
        return True, "", {}

    async def profile(self, *a, **k):
        return {}

    async def generate_reference(self, *a, **k):
        return True, "", b""

    async def profile_single_task(self, *a, **k):
        return {}

    async def get_doc(self, *a, **k):
        return ""

    async def acquire_device(self, task_id="unknown", timeout=None):
        d, self._next = self._next, self._next + 1
        return d, d  # (device_id, lease_id)

    async def release_device(self, device_id, lease_id, task_id="unknown"):
        self.released.append(device_id)


def check(cond, label):
    print(("[ok]   " if cond else "[FAIL] ") + label)
    if not cond:
        _fails.append(label)


async def test_cm_releases_on_normal_exit():
    pool = DevicePool([0])
    async with pool.lease("t") as dev:
        check(dev == 0, "cm: yields the device")
        check(pool.free_count() == 0, "cm: device removed from free set while held")
    check(pool.free_count() == 1, "cm: device returned after normal exit")


async def test_cm_releases_on_exception_and_cancel():
    pool = DevicePool([0])
    try:
        async with pool.lease("t"):
            raise ValueError("boom")
    except ValueError:
        pass
    check(pool.free_count() == 1, "cm: device returned after exception")

    async def hold():
        async with pool.lease("t"):
            await asyncio.sleep(100)
    task = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    check(pool.free_count() == 0, "cm: device held by running task")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    check(pool.free_count() == 1, "cm: device returned after task cancellation")


async def test_acquire_timeout():
    pool = DevicePool([0])
    dev, lid = await pool.acquire_device("a")
    raised = False
    try:
        await pool.acquire_device("b", timeout=0.1)
    except TimeoutError:
        raised = True
    check(raised, "acquire: raises TimeoutError when pool exhausted")
    await pool.release_device(dev, lid)
    # now acquire should succeed quickly
    dev, lid = await pool.acquire_device("c", timeout=1.0)
    check(dev == 0, "acquire: succeeds once a device frees")
    await pool.release_device(dev, lid)


async def test_idempotent_release():
    pool = DevicePool([0])
    dev, lid = await pool.acquire_device("t")
    first = await pool.release_device(dev, lid)
    second = await pool.release_device(dev, lid)  # double release
    check(first is True and second is False, "release: idempotent (double-release is a no-op)")
    check(pool.free_count() == 1, "release: no double-free into the queue")


async def test_lease_checked_release():
    pool = DevicePool([0])
    dev, lid = await pool.acquire_device("t")
    rejected = await pool.release_device(dev, lid + 999)  # wrong lease_id
    check(rejected is False, "release: wrong lease_id is rejected (not freed)")
    check(pool.free_count() == 0, "release: device still held after rejected release")
    ok = await pool.release_device(dev, lid)
    check(ok is True and pool.free_count() == 1, "release: correct lease_id frees")
    # force_release_device is the explicit admin path (no token); ordinary
    # release now requires lease_id (no silent downgrade).
    dev, _ = await pool.acquire_device("x")
    forced = await pool.force_release_device(dev)
    check(forced is True, "force_release_device: frees current lease without a token")


async def test_same_task_id_stale_release_rejected():
    # The High finding: the SAME task_id re-acquires the same device (e.g.
    # several rounds of "gelu" reuse one owner). A late release from the old
    # lease must NOT free the successor lease — owner alone can't tell them
    # apart, the lease_id can.
    pool = DevicePool([0])
    d1, lid1 = await pool.acquire_device("same-task")
    await pool.release_device(d1, lid1)                  # round 1 done
    d2, lid2 = await pool.acquire_device("same-task")    # round 2, same device + owner
    check(lid2 != lid1, "lease: re-acquire gets a fresh lease_id")
    leaked = await pool.release_device(d2, lid1)         # round 1's late release
    check(leaked is False, "release: same-owner stale lease can't free the successor")
    check(pool.free_count() == 0, "release: successor's device stays held")
    check(pool.leased().get(d2) == "same-task", "release: successor lease intact")


async def test_keepalive_protects_active_lease():
    pool = DevicePool([0], lease_ttl_s=0.3)
    pool.start_reaper(interval=0.05)
    try:
        dev, lid = await pool.acquire_device("worker", renewable=True)
        async with pool.keepalive("worker"):
            await asyncio.sleep(0.6)  # >> TTL; keepalive must keep it fresh
            check(dev in pool.leased(), "keepalive: active lease survives past TTL")
            check(pool.free_count() == 0, "keepalive: device not reaped while kept alive")
        await pool.release_device(dev, lid)
        check(pool.free_count() == 1, "keepalive: device released after block")
    finally:
        await pool.stop_reaper()


async def test_exact_keepalive_does_not_renew_same_owner_sibling():
    """One same-named live request must not preserve a dead sibling lease."""
    pool = DevicePool([0, 1], lease_ttl_s=0.3)
    pool.start_reaper(interval=0.05)
    try:
        dead_dev, _dead_lid = await pool.acquire_device(
            "same-op", renewable=True)
        live_dev, live_lid = await pool.acquire_device(
            "same-op", renewable=True)
        async with pool.keepalive(
                "same-op", device_id=live_dev, lease_id=live_lid):
            await asyncio.sleep(0.6)
            check(dead_dev not in pool.leased(),
                  "exact keepalive: same-owner dead sibling is reaped")
            check(live_dev in pool.leased(),
                  "exact keepalive: selected lease survives")
    finally:
        await pool.stop_reaper()


async def test_reaper_reclaims_and_renew_protects():
    # tiny TTL so the test runs fast
    pool = DevicePool([0, 1], lease_ttl_s=0.3)
    pool.start_reaper(interval=0.05)
    try:
        # dead holder: acquires renewable, never renews -> reaped
        d_dead, _ = await pool.acquire_device("dead", renewable=True)
        # live holder: acquires renewable, keeps renewing -> survives
        d_live, _ = await pool.acquire_device("live", renewable=True)
        check(pool.free_count() == 0, "reaper: both devices held initially")

        # Keep the live lease fresh CONTINUOUSLY through the check window,
        # so only the un-renewed dead lease crosses its TTL.
        stop = False

        async def renewer():
            while not stop:
                await pool.renew("live")
                await asyncio.sleep(0.05)

        renew_task = asyncio.create_task(renewer())
        await asyncio.sleep(0.6)     # >> TTL(0.3) + reaper interval, with live kept fresh
        check(pool.free_count() == 1, "reaper: reclaimed the un-renewed (dead) lease")
        check(d_dead not in pool.leased(), "reaper: dead lease dropped")
        check(d_live in pool.leased(), "reaper: renewed (live) lease survived")
        stop = True
        renew_task.cancel()
        try:
            await renew_task
        except asyncio.CancelledError:
            pass
    finally:
        await pool.stop_reaper()


async def test_worker_device_lease_cm():
    w = _FakeWorker()
    async with w.device_lease("t") as dev:
        check(dev == 0, "device_lease: yields the acquired device")
    check(w.released == [0], "device_lease: releases on normal exit")

    try:
        async with w.device_lease("t"):
            raise ValueError("boom")
    except ValueError:
        pass
    check(w.released == [0, 1], "device_lease: releases on exception")

    async def hold():
        async with w.device_lease("t"):
            await asyncio.sleep(100)
    task = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    check(w.released == [0, 1, 2], "device_lease: releases on cancellation")


async def test_no_default():
    raised = False
    try:
        DevicePool([])
    except ValueError:
        raised = True
    check(raised, "ctor: empty device_list raises (no implicit [0] default)")


async def main():
    for t in (
        test_cm_releases_on_normal_exit,
        test_cm_releases_on_exception_and_cancel,
        test_acquire_timeout,
        test_idempotent_release,
        test_lease_checked_release,
        test_same_task_id_stale_release_rejected,
        test_keepalive_protects_active_lease,
        test_exact_keepalive_does_not_renew_same_owner_sibling,
        test_reaper_reclaims_and_renew_protects,
        test_worker_device_lease_cm,
        test_no_default,
    ):
        # isolate AKG_AGENTS_DEVICES_LIST so a host env var can't skew the pool
        os.environ.pop("AKG_AGENTS_DEVICES_LIST", None)
        await t()
    if _fails:
        print(f"\n{len(_fails)} check(s) failed:")
        for f in _fails:
            print(f"  - {f}")
        return 1
    print("\nAll DevicePool lease checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
