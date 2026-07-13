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

"""Device leasing.

A device is held via a *lease* whose release is guaranteed by construction:

  - In-process, use ``async with pool.lease(owner) as dev:`` — the device is
    returned on normal exit, exception, OR task cancellation. Forgetting to
    release is structurally impossible.

  - Across a process boundary (the daemon's HTTP /acquire_device path), a
    lease carries a TTL and must be *renewed* by ongoing work (each verify /
    profile request renews it). A background reaper reclaims any lease not
    renewed within the TTL, so a client killed mid-task (e.g. the batch
    wall-clock SIGKILL) cannot permanently leak a device.

The free set is a single ``asyncio.Queue`` — it already blocks on empty and
wakes one waiter per ``put``, so no extra Condition is layered on top.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_NEVER = float("inf")  # non-expiring lease deadline (in-process CM leases)


@dataclass
class _Lease:
    device_id: int
    owner: str
    lease_id: int    # unique per acquire; distinguishes a re-acquire of the same device
    deadline: float  # time.monotonic() at which an un-renewed lease expires


class DevicePool:
    """Lease-based pool of Ascend/CUDA device ids.

    Args:
        device_list: the device ids this pool owns. Required and non-empty —
            there is no implicit ``[0]`` default (a silent default hides
            misconfiguration). ``AKG_AGENTS_DEVICES_LIST`` overrides it.
        lease_ttl_s: TTL for *renewable* leases. ``None`` (the default, used
            by short-lived in-process pools) means leases never expire and no
            reaper runs — release is then guaranteed by the ``lease()`` CM or
            the holder's own process lifetime. The daemon passes a finite TTL
            and calls ``start_reaper(interval=...)``.
    """

    def __init__(self, device_list: List[int], *, lease_ttl_s: Optional[float] = None):
        env_devices = os.environ.get("AKG_AGENTS_DEVICES_LIST")
        if env_devices:
            try:
                device_list = [int(x.strip()) for x in env_devices.split(",") if x.strip()]
                logger.info(f"使用环境变量 AKG_AGENTS_DEVICES_LIST: {device_list}")
            except ValueError:
                logger.warning(
                    f"环境变量 AKG_AGENTS_DEVICES_LIST 格式错误: {env_devices}, "
                    f"回退到传入的 device_list: {device_list}"
                )

        if not device_list:
            raise ValueError(
                "DevicePool requires a non-empty device_list — refusing to "
                "default to [0] (a silent default masks a misconfigured worker)."
            )

        self.device_list: List[int] = list(device_list)
        self.available_devices: "asyncio.Queue[int]" = asyncio.Queue()
        for device_id in self.device_list:
            self.available_devices.put_nowait(device_id)

        self._leases: Dict[int, _Lease] = {}
        self._leases_lock = asyncio.Lock()
        self._lease_seq = 0  # monotonic; stamps each acquire with a unique lease_id
        self._ttl = lease_ttl_s
        self._reaper: Optional[asyncio.Task] = None
        self._reap_interval: Optional[float] = None

    # ----- core acquire / release -------------------------------------------

    async def acquire_device(self, owner: str = "unknown", *,
                             timeout: Optional[float] = None,
                             renewable: bool = False) -> Tuple[int, int]:
        """Take a device, blocking until one is free (bounded by ``timeout``).
        Returns ``(device_id, lease_id)`` — ``lease_id`` is the token a later
        release must present, so a stale release of a superseded lease can't
        free the device that re-acquired it. ``renewable=True`` stamps a TTL
        deadline (daemon path) so the reaper can reclaim a dead holder.
        """
        if timeout is None:
            device_id = await self.available_devices.get()
        else:
            try:
                device_id = await asyncio.wait_for(self.available_devices.get(), timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"DevicePool.acquire_device: no device free within {timeout}s "
                    f"(pool={self.device_list}, all busy)"
                )
        # Device is dequeued but not yet leased; put it back on any interruption
        # so a cancellation here can't orphan it from both queue and _leases.
        try:
            deadline = (time.monotonic() + self._ttl) if (renewable and self._ttl) else _NEVER
            async with self._leases_lock:
                self._lease_seq += 1
                lease_id = self._lease_seq
                self._leases[device_id] = _Lease(device_id, owner, lease_id, deadline)
        except BaseException:
            self.available_devices.put_nowait(device_id)
            raise
        logger.debug(f"Acquired device {device_id} (owner={owner}, lease={lease_id})")
        return device_id, lease_id

    async def release_device(self, device_id: int, lease_id: int) -> bool:
        """Return a device acquired under ``lease_id``. Idempotent (releasing an
        already-freed/reaped device is a no-op) and lease-checked: a late
        release from a superseded lease — even same-owner — is rejected, so it
        can't free the device its successor now holds. ``lease_id`` is required;
        for an unconditional admin release use ``force_release_device``.
        """
        return await self._release(device_id, expect_lease=lease_id)

    async def force_release_device(self, device_id: int) -> bool:
        """Admin/debug escape hatch: release whatever lease currently holds
        ``device_id`` with no token check. Off the normal path — ordinary
        releases must present their lease_id (so a missed old-style call can't
        silently bypass the check)."""
        return await self._release(device_id, expect_lease=None)

    async def _release(self, device_id: int, *, expect_lease: Optional[int]) -> bool:
        # Pop and re-queue together under the lock (put_nowait never blocks the
        # unbounded queue) — no cancellation window between them.
        async with self._leases_lock:
            lease = self._leases.get(device_id)
            if lease is None:
                logger.debug(f"release_device({device_id}): no active lease (already freed/reaped) — no-op")
                return False
            if expect_lease is not None and lease.lease_id != expect_lease:
                logger.warning(
                    f"release_device({device_id}, lease={expect_lease}) ignored — "
                    f"device now held by lease {lease.lease_id} (owner={lease.owner!r}); "
                    f"stale release of a superseded lease, NOT freeing"
                )
                return False
            del self._leases[device_id]
            self.available_devices.put_nowait(device_id)
        logger.debug(f"Released device {device_id}")
        return True

    async def renew(self, owner: str) -> int:
        """Extend the TTL of every renewable lease held by ``owner``. Called by
        each work request so an active holder is never reaped. Returns the
        number of leases renewed."""
        if not self._ttl:
            return 0
        async with self._leases_lock:
            n = 0
            for lease in self._leases.values():
                if lease.owner == owner and lease.deadline != _NEVER:
                    lease.deadline = time.monotonic() + self._ttl
                    n += 1
        return n

    async def renew_lease(self, device_id: int, lease_id: int, *,
                          owner: Optional[str] = None) -> bool:
        """Renew one exact lease token instead of every lease by ``owner``.

        Concurrent jobs can legitimately share a human-readable task id.  A
        live request must not keep a dead sibling's lease alive, so new remote
        clients identify the exact device/lease pair they are using.
        """
        if not self._ttl:
            return False
        async with self._leases_lock:
            lease = self._leases.get(device_id)
            if (lease is None or lease.lease_id != lease_id
                    or (owner is not None and lease.owner != owner)
                    or lease.deadline == _NEVER):
                return False
            lease.deadline = time.monotonic() + self._ttl
            return True

    @asynccontextmanager
    async def lease(self, owner: str = "unknown", *, timeout: Optional[float] = None):
        """In-process lease: acquire on enter, release on exit — including on
        exception and task cancellation. The structural cure for forgotten
        releases."""
        device_id, lease_id = await self.acquire_device(owner, timeout=timeout, renewable=False)
        try:
            yield device_id
        finally:
            await self.release_device(device_id, lease_id)

    @asynccontextmanager
    async def keepalive(self, owner: str, *, device_id: Optional[int] = None,
                        lease_id: Optional[int] = None):
        """Renew ``owner``'s renewable leases now and periodically until the
        block exits, so a long request's device can't be reaped mid-flight.

        Passing both token fields renews exactly one lease.  Omitting both
        retains owner-wide renewal for older clients; passing only one is an
        error.  No-op on a TTL-less pool.
        """
        if (device_id is None) != (lease_id is None):
            raise ValueError("device_id and lease_id must be supplied together")

        async def _renew_once() -> int:
            if device_id is None:
                return await self.renew(owner)
            renewed = await self.renew_lease(
                device_id, lease_id, owner=owner)  # type: ignore[arg-type]
            if not renewed:
                raise LookupError(
                    f"inactive/stale device lease: device={device_id}, "
                    f"lease={lease_id}, owner={owner!r}")
            return 1

        await _renew_once()
        beat: Optional[asyncio.Task] = None
        if self._ttl:
            interval = self._ttl / 3  # renew well before the TTL expires

            async def _beat():
                while True:
                    await asyncio.sleep(interval)
                    await _renew_once()

            beat = asyncio.create_task(_beat())
        try:
            yield
        finally:
            if beat is not None:
                beat.cancel()
                try:
                    await beat
                except asyncio.CancelledError:
                    pass

    # ----- reaper (cross-process leak recovery) -----------------------------

    def start_reaper(self, interval: float) -> None:
        """Start the background reaper (idempotent). No-op when the pool has no
        TTL (in-process pools don't need reaping — their holder dies with the
        process or releases via the CM)."""
        if not self._ttl or self._reaper is not None:
            return
        self._reap_interval = interval
        self._reaper = asyncio.create_task(self._reap_loop())
        logger.info(f"DevicePool reaper started (ttl={self._ttl}s, interval={self._reap_interval}s)")

    async def stop_reaper(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            try:
                await self._reaper
            except asyncio.CancelledError:
                pass
            self._reaper = None

    async def _reap_loop(self) -> None:
        while True:
            assert self._reap_interval is not None
            await asyncio.sleep(self._reap_interval)
            now = time.monotonic()
            reclaimed: List[_Lease] = []
            # Drop the lease and re-queue the device together under the lock
            # (put_nowait can't block on the unbounded queue), so a reclaimed
            # device is never in limbo between _leases and the free queue.
            async with self._leases_lock:
                for device_id, lease in list(self._leases.items()):
                    if lease.deadline < now:
                        del self._leases[device_id]
                        self.available_devices.put_nowait(device_id)
                        reclaimed.append(lease)
            for lease in reclaimed:
                logger.warning(
                    f"Reaping expired device lease {lease.device_id} "
                    f"(owner={lease.owner!r} idle > {self._ttl}s) — client likely "
                    f"died mid-task; returning device to pool"
                )

    # ----- introspection -----------------------------------------------------

    def free_count(self) -> int:
        return self.available_devices.qsize()

    def leased(self) -> Dict[int, str]:
        """device_id -> owner snapshot of currently-held leases."""
        return {d: l.owner for d, l in self._leases.items()}
