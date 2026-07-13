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

"""Single-source-of-truth config loader for ``akg_cli worker``.

Replaces four sibling helpers (``load_default_port`` / ``load_default_dsl``
/ ``load_default_backend`` / ``_worker_setting``) that each re-resolved
and re-read the same yaml. One ``WorkerConfig.load(path)`` call returns a
frozen dataclass; everything downstream reads from it instead of poking
yaml again."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from akg_agents.op.utils.hw_detect import derive_arch
# config.yaml 解析/读取的唯一实现住在 eval_config（core 层）；cli 复用它，
# 只把 walk_parents/tag 调成 worker 侧语义。core←cli 是正确的依赖方向。
from akg_agents.core.worker.eval_config import _resolve, _load_yaml


# worker timing 的 11 个旋钮：字段名 -> env var 名。唯一一张表，as_env()/
# worker_timing() 都走它，load() 直接按字段名读 yaml（yaml key == 字段名）。
_TIMING_ENV = {
    "ready_timeout": "AKG_WORKER_READY_TIMEOUT",
    "ready_poll_interval": "AKG_WORKER_READY_POLL_INTERVAL",
    "ready_probe_timeout": "AKG_WORKER_READY_PROBE_TIMEOUT",
    "status_timeout": "AKG_WORKER_STATUS_TIMEOUT",
    "lease_ttl": "AKG_WORKER_LEASE_TTL_S",
    "lease_reap_interval": "AKG_WORKER_LEASE_REAP_INTERVAL_S",
    "acquire_timeout": "AKG_WORKER_ACQUIRE_TIMEOUT_S",
    "http_read_margin": "AKG_WORKER_HTTP_READ_MARGIN_S",
    "release_timeout": "AKG_WORKER_RELEASE_TIMEOUT_S",
    "doc_timeout": "AKG_WORKER_DOC_TIMEOUT_S",
    "health_timeout": "AKG_WORKER_HEALTH_TIMEOUT_S",
}


@dataclass(frozen=True)
class WorkerTiming:
    """``worker.*`` timing knobs from config.yaml. All in seconds (float)."""
    ready_timeout: float = 60.0          # 总等 daemon /status ready 多久
    ready_poll_interval: float = 5.0     # 心跳 tick 间隔
    ready_probe_timeout: float = 3.0     # 每次 /status probe 单次 timeout
    status_timeout: float = 3.0          # idle --status 探活的单次 timeout
    lease_ttl: float = 120.0             # 请求结束/客户端失联后多久回收 lease
    lease_reap_interval: float = 30.0    # daemon 扫描过期 lease 的间隔
    acquire_timeout: float = 600.0       # /acquire_device 等空闲设备多久
    http_read_margin: float = 10.0       # client read timeout 额外余量
    release_timeout: float = 10.0        # /release_device 读超时
    doc_timeout: float = 20.0            # /docs/<name> 读超时
    health_timeout: float = 5.0          # daemon /health 事件循环探活超时

    def as_env(self) -> Dict[str, str]:
        """转成 detached worker daemon 消费的环境变量。

        远端 worker 启动时 cwd 不一定有 config.yaml；akg_cli 先解析一次
        worker.*，再通过 env 透传给 daemon，避免 daemon 侧再长出一套默认值。
        """
        return {env: str(getattr(self, field)) for field, env in _TIMING_ENV.items()}


@dataclass(frozen=True)
class WorkerConfig:
    """Top-level worker config — read once, passed around as a value object.

    All non-DSL fields have concrete defaults baked in（``port=9001``、
    ``backend="cuda"`` 等），所以 ``WorkerConfig`` 就是 single source of
    truth：callers 直接读 ``cfg.port`` / ``cfg.backend`` 不再各处 ``or
    "cuda"`` / ``else 9001`` 兜底。覆盖只允许向上 (CLI > env > yaml)，
    fallback 永远只这一处。

    ``dsl`` 保留 Optional —— ``None`` 有语义（"未指定 DSL"，触发 classify
    的 warn 而不是 fatal），不能用 "" 顶替。"""
    port: int = 9001
    backend: str = "cuda"
    arch: str = "a100"
    devices: str = "0"
    dsl: Optional[str] = None
    hosts: Dict[str, dict] = field(default_factory=dict)
    timing: WorkerTiming = field(default_factory=WorkerTiming)
    source_path: Optional[str] = None    # 解析 yaml 的绝对路径，diag 用

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "WorkerConfig":
        """Load from explicit path or default ``cwd/config.yaml``. yaml 缺
        失或字段缺失 → 用 dataclass 默认。绝不返回 None，callers 不用
        null-check。"""
        resolved = _resolve(config_path, walk_parents=False)
        if resolved is None:
            return cls()
        data = _load_yaml(resolved, tag="akg_cli")
        if data is None:
            return cls(source_path=resolved)

        worker = data.get("worker") or {}
        defaults = data.get("defaults") or {}
        hosts = ((data.get("remote_worker") or {}).get("hosts") or {})

        port_v = _int_in_range(worker.get("port"), 1, 65535, cls.port)
        backend_v = _str_or(defaults.get("backend"), cls.backend).lower()
        arch_v = _str_or(defaults.get("arch"), cls.arch)
        devices_v = _str_or(defaults.get("devices"), cls.devices)
        dsl_raw = defaults.get("dsl")
        dsl_v: Optional[str] = str(dsl_raw) if isinstance(dsl_raw, str) else None

        # Timing 默认值只从 WorkerTiming 取，避免两个 dataclass 重复写数字。
        td = WorkerTiming()
        timing = WorkerTiming(**{
            field: _float(worker.get(field), getattr(td, field))
            for field in _TIMING_ENV
        })
        return cls(
            port=port_v, backend=backend_v, arch=arch_v, devices=devices_v,
            dsl=dsl_v, hosts=dict(hosts), timing=timing, source_path=resolved,
        )

    def host(self, alias: str) -> Optional[dict]:
        """Look up ``remote_worker.hosts.<alias>``. None if absent."""
        return self.hosts.get(alias)


def _float(val, default: float) -> float:
    if isinstance(val, (int, float)) and val > 0:
        return float(val)
    return default


def _int_in_range(val, lo: int, hi: int, default: int) -> int:
    if isinstance(val, int) and lo <= val <= hi:
        return val
    return default


def _str_or(val, default: str) -> str:
    return str(val).strip() if isinstance(val, str) and str(val).strip() else default


def _env_float(key: str, default: float) -> float:
    try:
        v = float(os.environ.get(key, ""))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def worker_timing(config_path: Optional[str] = None) -> WorkerTiming:
    """解析最终生效的 worker timing 配置。

    优先级：AKG_WORKER_* env（detached/remote daemon 路径）>
    config.yaml worker.* > WorkerTiming dataclass 默认值。
    """
    cfg = WorkerConfig.load(config_path).timing
    return WorkerTiming(**{
        field: _env_float(env, getattr(cfg, field))
        for field, env in _TIMING_ENV.items()
    })


def probe_local_arch(backend: str, device_id: int = 0) -> Optional[str]:
    """Best-effort local arch probe so ``akg_cli worker --start`` (no
    --remote-host) doesn't fall back to the baked ``a100`` default on hosts
    the operator forgot to flag. Delegates to the shared
    :func:`akg_agents.op.utils.hw_detect.derive_arch` — one probe
    implementation for both the CLI worker and the workspace scaffold.

    Returns None on any failure (binary not on PATH, non-zero exit,
    unparseable output, unknown backend); caller falls through to ``cfg.arch``.
    """
    return derive_arch(device_id, backend)
