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

"""``akg_cli worker --remote-host`` dispatch — thin orchestration layer.

Responsibilities: idempotent ``--start``, ``--stop``, ``--status``,
``--reconnect``. All heavy lifting delegated to siblings:

  - ``tunnel.py``           — local ssh -L lifecycle, port ownership
  - ``remote_probe.py``     — one-shot SSH probe → raw facts
  - ``diagnostics.py``      — facts → ``list[Finding]`` + rich.Table render

Module name was ``worker_remote`` previously — too easy to confuse with
``core/worker/remote_worker.py`` (the HTTP client class). Renamed so the
two layers can't be mistyped into each other."""

# pylint: disable=missing-function-docstring,broad-exception-caught,import-outside-toplevel
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from typing import Optional

from .diagnostics import classify, has_fatal, render_findings
from .remote_env import source_env_script_bash
from .remote_probe import probe_remote
from .tunnel import kill_pid_hint, tunnel_start, tunnel_stop_silent, who_holds_port
from .worker_config import WorkerConfig, WorkerTiming, worker_timing
from akg_agents.core.worker.eval_config import eval_defaults


# Back-compat thin wrappers — misc.py / akg_eval.py still import these
# names. New code should construct ``WorkerConfig.load(...)`` directly.

def load_remote_host_config(alias: str,
                            config_path: Optional[str]) -> Optional[dict]:
    return WorkerConfig.load(config_path).host(alias)


def load_default_port(config_path: Optional[str]) -> Optional[int]:
    return WorkerConfig.load(config_path).port


# ---------------------------------------------------------------------------
# HTTP probes (local-tunnel-side)
# ---------------------------------------------------------------------------


def _curl_status(host: str, port: int,
                 timeout: Optional[float] = None) -> Optional[dict]:
    """``/api/v1/status`` probe. ``timeout`` defaults to ``status_timeout``
    from config —— the ready loop should explicitly pass
    ``ready_probe_timeout`` instead since the two have different roles."""
    import urllib.request
    if timeout is None:
        timeout = worker_timing().status_timeout
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v1/status", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _is_ready(st: Optional[dict]) -> bool:
    """True iff ``/status`` 返回 dict 且 status 字段是 ready/ok。daemon
    刚 spawn 时 server.py 返回 ``initializing``（HTTP 已通但 worker 还
    没装好）——这个状态既不能跳过 spawn，也不能算 poll-loop 完成。"""
    if not isinstance(st, dict):
        return False
    return str(st.get("status", "")).lower() in ("ready", "ok")


def _curl_health(host: str, port: int,
                 timeout: Optional[float] = None) -> Optional[dict]:
    """``/health``：非阻塞 device queue 探活。

    ``timeout`` 默认比 daemon 侧 health_timeout 多留一段 client 余量。
    传输失败或老 daemon 缺少 endpoint 时返回 None。
    """
    import urllib.request
    if timeout is None:
        timing = worker_timing()
        timeout = (
            max(timing.status_timeout, timing.health_timeout)
            + timing.http_read_margin
        )
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v1/health", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Remote spawn helpers
# ---------------------------------------------------------------------------


def _build_remote_start_cmd(host_cfg: dict, backend: str, arch: str,
                            devices: str, port: int,
                            timing: WorkerTiming) -> str:
    """Compose the bash payload sent over SSH to spawn the daemon. The
    recursive ``akg_cli`` on remote goes through the local branch of
    ``worker_cmd`` → ``worker_service.start``, which Popen-detaches the
    daemon (``preexec_fn=os.setsid`` + ``stdin=DEVNULL``) so this SSH
    returns promptly.

    ``PYTHONPATH`` is pinned to ``<repo_path>/akg_agents/python`` so the
    daemon runs the checkout source, not whatever pip pinned.

    ``env_script`` may contain plain ``conda activate``; bootstrap the
    conda shell hook before sourcing it so non-interactive SSH behaves like
    the user's login shell.

    worker.* timing 通过 env 透传，所以远端 ``worker_service.start`` 不
    再硬编码固定启动等待值 —— config.yaml worker.* 一处改、本机和递归远端都
    生效。"""
    repo_path = host_cfg["repo_path"]
    env_script = host_cfg.get("env_script")

    parts: list = [source_env_script_bash(env_script)]
    parts.append(
        f"export PYTHONPATH={shlex.quote(repo_path)}/akg_agents/python:"
        f"${{PYTHONPATH:-}}"
    )
    # daemon 只绑 loopback（tunnel 转发 :<port> 到远端 127.0.0.1）。
    parts.append("export WORKER_HOST=127.0.0.1")
    # 递归远端 akg_cli 跳过启动表格和心跳噪声；本机命令负责用户可见输出。
    parts.append("export AKG_CLI_QUIET=1")
    for key, value in timing.as_env().items():
        parts.append(f"export {key}={shlex.quote(value)}")
    for key, value in eval_defaults().as_env().items():
        parts.append(f"export {key}={shlex.quote(value)}")
    parts.append(
        " ".join([
            "python", "-m", "akg_agents.cli.cli", "worker",
            "--start",
            "--backend", shlex.quote(backend),
            "--arch", shlex.quote(arch),
            "--devices", shlex.quote(devices),
            "--port", str(port),
        ])
    )
    return "\n".join(parts)


def _build_remote_stop_cmd(host_cfg: dict, port: int) -> str:
    """Compose exact daemon termination plus predecessor-tree cleanup.

    A single SIGTERM is not a completed stop: Uvicorn waits for an in-flight
    eval, while that eval may run for many minutes.  Escalate the one listener
    PID after the configured NPU teardown grace, then invoke the shared,
    PID-fingerprinted eval-group reaper from the same checkout/environment.
    """
    repo_path = host_cfg["repo_path"]
    env_script = host_cfg.get("env_script")
    defaults = eval_defaults()
    polls = max(1, int(defaults.kill_grace_s * 10) + 1)
    registry = f"/tmp/akg_worker_{port}_process_groups.json"
    state_lookup = (
        "from akg_agents.cli.utils.worker_state import live_worker_pid; "
        f"print(live_worker_pid({port}) or '')"
    )
    cleanup = (
        "import json; "
        "from akg_agents.utils.process_utils import "
        "reap_orphaned_process_groups; "
        "from akg_agents.cli.utils.worker_state import "
        "load_worker_state, remove_worker_entry, save_worker_state; "
        "reaped = reap_orphaned_process_groups(); "
        "state = load_worker_state(); "
        f"remove_worker_entry(state, {port}); "
        "save_worker_state(state); "
        "print(json.dumps({'reaped_process_groups': reaped}))"
    )
    parts: list[str] = [source_env_script_bash(env_script)]
    parts.append(
        f"export PYTHONPATH={shlex.quote(repo_path)}/akg_agents/python:"
        f"${{PYTHONPATH:-}}"
    )
    for key, value in defaults.as_env().items():
        parts.append(f"export {key}={shlex.quote(value)}")
    parts.append(
        f"export AKG_WORKER_PROCESS_REGISTRY={shlex.quote(registry)}"
    )
    parts.extend([
        f'listener_pid="$(lsof -tiTCP:{port} -sTCP:LISTEN | head -n 1)"',
        f"state_pid=\"$(python -c {shlex.quote(state_lookup)})\"",
        'pid="${listener_pid:-$state_pid}"',
        (
            'if [ -n "$pid" ]; then '
            'kill -TERM "$pid" 2>/dev/null || true; '
            f'for _ in $(seq 1 {polls}); do '
            'kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done; '
            'if kill -0 "$pid" 2>/dev/null; then '
            'kill -KILL "$pid" 2>/dev/null || true; fi; '
            'fi'
        ),
        f"python -c {shlex.quote(cleanup)}",
    ])
    return "\n".join(parts)


def _ssh_dispatch(ssh_alias: str, bash_cmd: str) -> int:
    """SSH-run bash_cmd on alias，stdout 透传给本机终端（让远端 akg_cli
    递归 print 流回来）。``-o LogLevel=ERROR`` 抑制 SSH banner 和无关
    RemoteForward warning，保留真实 ssh 错误。"""
    return subprocess.call([
        "ssh", "-o", "LogLevel=ERROR",
        ssh_alias, f"bash -lc {shlex.quote(bash_cmd)}",
    ])


def _step(msg: str) -> None:
    """Step log to stderr with ``flush=True`` — Windows terminals
    occasionally buffer stderr until the producer terminates, which makes
    a 30s probe look "frozen". Flushing per line eliminates that."""
    print(f"[akg_cli] {msg}", file=sys.stderr, flush=True)


def _device_ids_from_arg(devices: Optional[str]) -> Optional[list[int]]:
    if devices is None:
        return None
    ids = [int(p.strip()) for p in str(devices).split(",") if p.strip()]
    return ids or None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_start(alias: str, host_cfg: dict, backend: Optional[str],
                   arch: Optional[str], devices: Optional[str],
                   port: int, dsl: Optional[str] = None) -> int:
    """SSH-dispatch worker --start with idempotent recovery.

    Flow: probe /status → rebuild tunnel + reprobe → run diagnostic probe
    (also fills missing CLI defaults) → spawn daemon → poll /status
    with heartbeat. Any fatal finding aborts before spawn.

    ``backend / arch / devices`` may be None. Arch is filled from the
    backend-specific remote probe (Ascend: npu-smi, CUDA: nvidia-smi,
    CPU: platform.machine) unless the caller passes ``--arch``. ``dsl`` drives
    ``classify(require_triton=...)``: ``triton_*`` makes missing triton
    fatal, others (catlass / pypto / ...) keep it warn. Pass None to let
    dispatch read ``defaults.dsl`` from config.yaml."""
    if "repo_path" not in host_cfg:
        _step(f"remote_worker.hosts.{alias} 缺 repo_path")
        return 2
    ssh_alias = host_cfg.get("ssh_alias") or alias
    log_file = f"/tmp/akg_worker_{port}.log"
    env_script = host_cfg.get("env_script")
    repo_path = host_cfg.get("repo_path")

    # Resolve all defaults up front so every classify call sees a
    # consistent effective_backend / dsl / timing — without this,
    # tunnel-fail diagnostics used user-passed `backend` (often None)
    # and read different defaults than the probe-success path.
    cfg = WorkerConfig.load(None)
    effective_backend = (backend or cfg.backend)   # CLI > config.yaml
    effective_dsl = dsl or cfg.dsl
    timing = worker_timing()
    probe_device_ids = _device_ids_from_arg(devices)

    _step(f"[1/4] 探活 127.0.0.1:{port}/api/v1/status ...")
    st = _curl_status("127.0.0.1", port, timeout=timing.status_timeout)
    if _is_ready(st):
        _step(f"[1/4] daemon 已就绪 — nothing to do")
        print(json.dumps(st, indent=2, ensure_ascii=False))
        return 0
    if st is not None:
        _step(f"[1/4] /status 返回 {st.get('status')!r}，不是 ready —— 继续往下")
    else:
        _step(f"[1/4] 不通 → tunnel 或 daemon 至少一个不可达")

    _step(f"[2/4] 重建本机 ssh -L :{port} → {ssh_alias} ...")
    tunnel_stop_silent(port, ssh_alias)
    pid = tunnel_start(ssh_alias, port)
    if pid == 0:
        # tunnel ssh -f 静默吞了 stderr —— 跑一次 probe_remote 复用同一
        # 个 alias 反向探 SSH 透传：VPN 没开 / 网络不通 / 免密配置错 /
        # alias 不存在都会通过 _SSH_ERROR 路径出现在诊断表里第一行。
        _step(f"[2/4] tunnel 起失败 —— 反向诊断 SSH 透传：")
        facts = probe_remote(ssh_alias, env_script, port, log_file, repo_path,
                             probe_device_ids)
        render_findings(
            classify(facts, port, backend=effective_backend,
                     dsl=effective_dsl, for_start=True),
            facts.get("LOG_TAIL", ""),
        )
        return 1
    _step(f"[2/4] tunnel pid={pid}, 再探活 /status ...")
    st = _curl_status("127.0.0.1", port, timeout=timing.status_timeout)
    if _is_ready(st):
        _step(f"[2/4] 是 tunnel 那条线断的；daemon 还在 — 完成")
        print(json.dumps(st, indent=2, ensure_ascii=False))
        return 0
    _step(f"[2/4] tunnel 通了但 /status 未就绪 → daemon 未运行或还在 init")

    _step(f"[3/4] 远端诊断（env / backend deps / triton / disk / port / log）...")
    facts = probe_remote(ssh_alias, env_script, port, log_file, repo_path,
                         probe_device_ids)
    # Final fallback: 只有 CLI/env + config.yaml 都没给时才用 probe-based
    # 推断（torch_npu 能 import 就 ascend，否则 cuda）。
    if effective_backend is None:
        effective_backend = "ascend" if facts.get("TORCH_NPU") == "ok" else "cuda"
    findings = classify(facts, port, backend=effective_backend,
                        dsl=effective_dsl, for_start=True)
    if has_fatal(findings):
        _step(f"[3/4] fatal 项，不启动 daemon。诊断：")
        render_findings(findings, facts.get("LOG_TAIL", ""))
        return 1

    backend = effective_backend
    if arch is None:
        if backend == "ascend":
            raw_arch = (facts.get("ARCH") or "").strip().lower()
            if not raw_arch:
                _step(f"[3/4] 无法自动推断 ascend arch，--arch 必须显式传")
                render_findings(findings, facts.get("LOG_TAIL", ""))
                return 1
            arch = raw_arch
        elif backend == "cuda":
            raw_arch = (facts.get("CUDA_ARCH") or "").strip().lower()
            if not raw_arch:
                _step(f"[3/4] 无法自动推断 cuda arch，--arch 必须显式传")
                render_findings(findings, facts.get("LOG_TAIL", ""))
                return 1
            arch = raw_arch
        elif backend == "cpu":
            raw_arch = (facts.get("CPU_ARCH") or "").strip().lower()
            if not raw_arch:
                _step(f"[3/4] 无法自动推断 cpu arch，--arch 必须显式传")
                render_findings(findings, facts.get("LOG_TAIL", ""))
                return 1
            arch = raw_arch
        else:
            _step(f"[3/4] backend={backend!r} 没有远端 arch 自动推断，--arch 必须显式传")
            render_findings(findings, facts.get("LOG_TAIL", ""))
            return 1
    if devices is None:
        devices = "0"
    _step(f"[3/4] 探针 OK: backend={backend}, arch={arch}, "
          f"devices={devices}, dsl={effective_dsl or '(any)'}")

    _step(f"[4/4] SSH 起远端 daemon at {ssh_alias}:{port} ...")
    remote_cmd = _build_remote_start_cmd(
        host_cfg, backend=backend, arch=arch, devices=devices, port=port,
        timing=timing,
    )
    rc = _ssh_dispatch(ssh_alias, remote_cmd)
    if rc != 0:
        _step(f"[4/4] remote daemon launch rc={rc} —— 重新诊断：")
        facts2 = probe_remote(ssh_alias, env_script, port, log_file, repo_path,
                              probe_device_ids)
        render_findings(
            classify(facts2, port, backend=backend, dsl=effective_dsl,
                     for_start=True),
            facts2.get("LOG_TAIL", ""),
        )
        return rc

    _step(f"[4/4] daemon spawned，poll /status ready（最长 {timing.ready_timeout}s）...")
    deadline = time.time() + timing.ready_timeout
    last_beat = time.time()
    while time.time() < deadline:
        # ready 阶段用 ready_probe_timeout（轮询语义），区别于 idle
        # --status 的 status_timeout（一次性查询）。_is_ready 只接受
        # ready/ok；initializing 不算（daemon 启动期 HTTP 已通但 worker
        # 还没装好，继续等）。
        st = _curl_status("127.0.0.1", port,
                          timeout=timing.ready_probe_timeout)
        if _is_ready(st):
            _step(f"[4/4] /status ready — 完成")
            print(json.dumps(st, indent=2, ensure_ascii=False))
            return 0
        now = time.time()
        if now - last_beat >= timing.ready_poll_interval:
            _step(f"   /status 未就绪 "
                  f"({int(now - deadline + timing.ready_timeout)}s"
                  f"/{timing.ready_timeout}s)...")
            last_beat = now
        time.sleep(1)

    _step(f"[4/4] /status {timing.ready_timeout}s 未就绪 —— 重新诊断：")
    facts2 = probe_remote(ssh_alias, env_script, port, log_file, repo_path,
                          probe_device_ids)
    render_findings(
        classify(facts2, port, backend=backend, dsl=effective_dsl,
                 for_start=True),
        facts2.get("LOG_TAIL", ""),
    )
    return 1


def dispatch_stop(alias: str, host_cfg: dict, port: int) -> int:
    """Tear down the tunnel, stop the exact listener, reap its eval trees."""
    ssh_alias = host_cfg.get("ssh_alias") or alias
    tunnel_stop_silent(port, ssh_alias)
    print(f"[akg_cli] tore down local tunnel for :{port}")
    if "repo_path" not in host_cfg:
        print(f"[akg_cli] remote_worker.hosts.{alias} 缺 repo_path",
              file=sys.stderr)
        return 2
    rc = _ssh_dispatch(ssh_alias, _build_remote_stop_cmd(host_cfg, port))
    if rc != 0:
        print(f"[akg_cli] remote daemon stop rc={rc}", file=sys.stderr)
        return rc
    print(f"[akg_cli] stopped remote daemon and reaped owned eval trees on "
          f"{ssh_alias}:{port}")
    return 0


def dispatch_status(alias: str, host_cfg: dict, port: int, *,
                    backend: Optional[str] = None,
                    dsl: Optional[str] = None) -> int:
    """Curl tunneled ``/status`` + ``/health``.

    On /status failure, identify the local port holder and, for remote
    aliases, run the same SSH/env/NPU probe used by start preflight.
    /health output surfaces ``free`` + ``note`` so 'healthy busy' and
    'healthy idle' are distinguishable."""
    st = _curl_status("127.0.0.1", port)
    if st is None:
        holder = who_holds_port(port)
        if holder is None:
            print(f"Worker 127.0.0.1:{port} 不可达；本机 :port 空闲 → 跑 `--start`。")
        else:
            print(
                f"Worker 127.0.0.1:{port} 不可达；:port 被 PID={holder['pid']} 占着\n"
                f"  cmdline: {holder['cmdline'][:120]}\n"
                f"  → 残留 tunnel：`{kill_pid_hint(holder['pid'])}` 后 --start；"
                f"远端 daemon 已停：--stop + --start"
            )
        ssh_alias = host_cfg.get("ssh_alias") or alias
        if ssh_alias != "local":
            cfg = WorkerConfig.load(None)
            effective_backend = backend or cfg.backend
            effective_dsl = dsl or cfg.dsl
            log_file = f"/tmp/akg_worker_{port}.log"
            env_script = host_cfg.get("env_script")
            repo_path = host_cfg.get("repo_path")
            facts = probe_remote(ssh_alias, env_script, port, log_file, repo_path)
            render_findings(
                classify(
                    facts, port, backend=effective_backend,
                    dsl=effective_dsl, for_start=False,
                ),
                facts.get("LOG_TAIL", ""),
            )
        return 1

    health = _curl_health("127.0.0.1", port)
    out = dict(st)
    if health is not None:
        out["health"] = {
            "healthy": bool(health.get("healthy")),
            "probed_device": health.get("probed_device"),
            "free": health.get("free"),
            "note": health.get("note"),
            "error": health.get("error"),
        }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if health is not None and not health.get("healthy"):
        print(
            f"\n[akg_cli] /status OK 但 /health 报 degraded —— "
            f"daemon handler 可能阻塞。错误：{health.get('error')!r}",
            file=sys.stderr,
        )
        return 1
    return 0


def dispatch_reconnect_tunnel(alias: str, host_cfg: dict, port: int) -> int:
    """Rebuild only the local tunnel; leave remote daemon alone. Use when
    a long batch silently lost its tunnel (server-side SSH reset / network
    drop) but the daemon is still alive. Falls back to --stop+--start if
    the daemon is also gone."""
    ssh_alias = host_cfg.get("ssh_alias") or alias
    tunnel_stop_silent(port, ssh_alias)
    pid = tunnel_start(ssh_alias, port)
    if pid:
        print(f"[akg_cli] ssh -L 127.0.0.1:{port} → "
              f"{ssh_alias}:{port} reconnected (tunnel pid={pid})")
    st = _curl_status("127.0.0.1", port)
    if st is None:
        print(
            f"[akg_cli] /status 仍不通；daemon 可能也已停 — 用 --stop + --start。",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(st, indent=2, ensure_ascii=False))
    return 0
