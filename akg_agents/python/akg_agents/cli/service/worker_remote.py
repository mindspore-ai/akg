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

"""Remote worker dispatch + ssh -L tunnel management for `akg_cli worker`.

`akg_cli worker --remote-host <alias> --start/stop/status` SSH-dispatches the
worker lifecycle to a config-defined remote host. For --start, also opens a
local ssh -L tunnel so 127.0.0.1:<port> forwards to the remote daemon's
loopback port — the daemon never binds a public interface.

The remote process is `python -m akg_agents.worker.server` invoked with
`PYTHONPATH=<repo_path>/akg_agents/python` so the running worker matches the
checked-out source (NOT whatever akg_agents the remote pip install pinned).
That avoids the version-drift class of bug where the daemon exposes an older
endpoint surface than the local `RemoteWorker` client expects.

Config: `remote_worker.hosts.<alias>` under any yaml file passed via
`--remote-config` (default: `./config.yaml` if it exists). Fields:

    repo_path:  required. Abs path to the akg checkout on remote.
    env_script: optional. Sourced before running python, must yield a shell
                where `python -c "import torch_npu, triton"` succeeds.
    python:     optional. Default `python`.
    ssh_alias:  optional. Default = the key under `hosts.`.

Tunnel pid is stashed at `~/.akg_agents/tunnels/<port>.pid` so `--stop` knows
which ssh fork to SIGTERM.
"""

# pylint: disable=missing-function-docstring,broad-exception-caught,import-outside-toplevel
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional


STATE_DIR = Path.home() / ".akg_agents"


def _tunnel_pid_path(port: int) -> Path:
    return STATE_DIR / "tunnels" / f"{port}.pid"


def load_remote_host_config(alias: str,
                            config_path: Optional[str]) -> Optional[dict]:
    """Look up `remote_worker.hosts.<alias>` from the yaml at config_path.
    Returns None when the file is missing or the alias is absent — caller
    surfaces the error."""
    if config_path is None:
        default = Path.cwd() / "config.yaml"
        config_path = str(default) if default.is_file() else None
    if config_path is None or not Path(config_path).is_file():
        return None
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[akg_cli] failed to read {config_path}: {e}", file=sys.stderr)
        return None
    hosts = ((data.get("remote_worker") or {}).get("hosts") or {})
    return hosts.get(alias)


def _build_remote_start_cmd(host_cfg: dict, backend: str, arch: str,
                            devices: str, port: int) -> str:
    """Compose the bash command we send through ssh to launch the daemon.
    Recurses through akg_cli on the remote: `python -m akg_agents.cli.cli
    worker --start ...`. PYTHONPATH is pinned to `repo_path/akg_agents/
    python` so the running daemon matches the checkout (avoids the
    pip-install version drift class of bug).

    The remote akg_cli's `worker --start` path goes through
    `services.worker_service.start`, which `Popen`s the daemon with
    `preexec_fn=os.setsid` + `stdin=subprocess.DEVNULL` — that's what
    detaches it from the SSH channel so this `bash -lc '...'` returns
    promptly. Aligns with the recursive-dispatch pattern in
    claude-autoresearch `scripts/ar_cli.py`."""
    repo_path = host_cfg["repo_path"]  # required
    env_script = host_cfg.get("env_script")
    python = host_cfg.get("python") or "python"

    parts: list[str] = []
    if env_script:
        parts.append(f"source {shlex.quote(env_script)}")
    parts.append(
        f"export PYTHONPATH={shlex.quote(repo_path)}/akg_agents/python:"
        f"${{PYTHONPATH:-}}"
    )
    parts.append(
        " ".join([
            shlex.quote(python), "-m", "akg_agents.cli.cli", "worker",
            "--start",
            "--backend", shlex.quote(backend),
            "--arch", shlex.quote(arch),
            "--devices", shlex.quote(devices),
            "--host", "127.0.0.1",
            "--port", str(port),
        ])
    )
    return " && ".join(parts)


def _ssh_dispatch(ssh_alias: str, bash_cmd: str) -> int:
    """Run `ssh <alias> 'bash -lc "<bash_cmd>"'`. Returns subprocess rc."""
    return subprocess.call(["ssh", ssh_alias, f"bash -lc {shlex.quote(bash_cmd)}"])


def _curl_status(host: str, port: int, timeout: float = 3.0) -> Optional[dict]:
    """GET /api/v1/status. Returns parsed JSON on 200, None otherwise."""
    import urllib.request
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v1/status", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _tunnel_start(ssh_alias: str, port: int) -> int:
    """Open `ssh -L <port>:127.0.0.1:<port> <alias> -f -N -T`, stash the
    forked ssh pid at <STATE_DIR>/tunnels/<port>.pid. Returns the pid on
    success, 0 on soft failure (caller verifies via status probe).

    Deliberately does NOT pass ExitOnForwardFailure=yes — the user's
    ~/.ssh/config may declare unrelated RemoteForward entries (IDE relays,
    etc.) whose failure shouldn't take down the -L we need."""
    (STATE_DIR / "tunnels").mkdir(parents=True, exist_ok=True)

    _tunnel_stop_silent(port, ssh_alias)  # clear any stale tunnel first

    # Keepalive: ping every 30s, drop after 10 missed (~5 min idle
    # tolerance). Default OpenSSH (60s / 3x) drops idle long-running
    # tunnels on flaky multi-tenant networks; bumped here because
    # a long autoresearch batch can sit in PLAN for several minutes
    # without using the tunnel, and we don't want the next eval call
    # to find a dead tunnel.
    cmd = [
        "ssh", "-f", "-N", "-T",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
        "-L", f"{port}:127.0.0.1:{port}",
        ssh_alias,
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        print(
            f"[akg_cli] ssh exited rc={rc} (unrelated forward may have "
            f"failed; checking -L {port} via status probe).",
            file=sys.stderr,
        )
    pid = _find_tunnel_pid(port, ssh_alias)
    if pid:
        _tunnel_pid_path(port).write_text(str(pid))
    return pid


def _find_tunnel_pid(port: int, ssh_alias: str) -> int:
    """Locate the forked `ssh -L <port>:...` process by cmdline scan.
    pgrep on POSIX, Get-CimInstance on Windows. 0 = not found."""
    if os.name == "posix":
        try:
            out = subprocess.run(
                ["pgrep", "-f", f"ssh.*-L {port}:.*{ssh_alias}"],
                capture_output=True, text=True, timeout=5,
            )
            for ln in out.stdout.splitlines():
                ln = ln.strip()
                if ln.isdigit():
                    return int(ln)
        except Exception:
            pass
        return 0

    try:
        ps_cmd = (
            f"Get-CimInstance Win32_Process -Filter \"Name='ssh.exe'\" | "
            f"Where-Object {{ $_.CommandLine -like '*-L {port}:*{ssh_alias}*' }} | "
            f"Select-Object -ExpandProperty ProcessId"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=10,
        )
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if ln.isdigit():
                return int(ln)
    except Exception:
        pass
    return 0


def _tunnel_stop_silent(port: int, ssh_alias: str = "") -> None:
    """Best-effort tunnel teardown. Prefer the stashed pid, fall back to
    a fresh cmdline scan (handles --start that couldn't capture the pid)."""
    pid_path = _tunnel_pid_path(port)
    pid = 0
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text().strip())
        except (ValueError, FileNotFoundError):
            pid = 0
    if pid == 0 and ssh_alias:
        pid = _find_tunnel_pid(port, ssh_alias)
    if pid:
        try:
            if os.name == "posix":
                os.kill(pid, signal.SIGTERM)
            else:
                try:
                    os.kill(pid, signal.SIGTERM)
                except (PermissionError, OSError):
                    subprocess.call(["taskkill", "/PID", str(pid), "/F"])
        except (ProcessLookupError, ValueError, FileNotFoundError):
            pass
        except Exception as e:
            print(f"[akg_cli] tunnel stop failed: {e}", file=sys.stderr)
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass


def dispatch_start(alias: str, host_cfg: dict, backend: str, arch: str,
                   devices: str, port: int) -> int:
    """SSH-dispatch worker --start, then open the local ssh -L tunnel."""
    if "repo_path" not in host_cfg:
        print(f"[akg_cli] remote_worker.hosts.{alias} missing `repo_path`",
              file=sys.stderr)
        return 2
    ssh_alias = host_cfg.get("ssh_alias") or alias
    remote_cmd = _build_remote_start_cmd(
        host_cfg, backend=backend, arch=arch, devices=devices, port=port,
    )
    print(f"[akg_cli] remote ({ssh_alias}) start: {remote_cmd}", file=sys.stderr)
    rc = _ssh_dispatch(ssh_alias, remote_cmd)
    if rc != 0:
        print(f"[akg_cli] remote daemon launch exited rc={rc}", file=sys.stderr)
        return rc
    pid = _tunnel_start(ssh_alias, port)
    if pid:
        print(f"[akg_cli] ssh -L 127.0.0.1:{port} -> "
              f"{ssh_alias}:{port} (tunnel pid={pid})")
    st = _curl_status("127.0.0.1", port)
    if st is None:
        print(f"[akg_cli] tunneled status probe failed; remote daemon may "
              f"not be ready or tunnel didn't bind.", file=sys.stderr)
        return 1
    print(json.dumps(st, indent=2))
    return 0


def dispatch_stop(alias: str, host_cfg: dict, port: int) -> int:
    """Tear down local tunnel + kill the remote daemon listening on port."""
    ssh_alias = host_cfg.get("ssh_alias") or alias
    _tunnel_stop_silent(port, ssh_alias)
    print(f"[akg_cli] tore down local tunnel for :{port}")
    # `lsof -ti :<port>` returns only the listening PID — safer than pkill
    # which might match unrelated python processes.
    remote_cmd = f"lsof -ti :{port} | xargs -r kill"
    rc = _ssh_dispatch(ssh_alias, remote_cmd)
    if rc != 0:
        print(f"[akg_cli] remote daemon stop exited rc={rc}", file=sys.stderr)
        return rc
    print(f"[akg_cli] killed remote daemon on {ssh_alias}:{port}")
    return 0


def dispatch_status(alias: str, host_cfg: dict, port: int) -> int:
    """Curl the tunneled /status. Assumes --start already established the
    tunnel; surfaces 'unreachable' otherwise."""
    st = _curl_status("127.0.0.1", port)
    if st is None:
        print(
            f"Worker tunnel 127.0.0.1:{port} unreachable "
            f"(--start may not have been called, or tunnel died — "
            f"try --reconnect-tunnel)."
        )
        return 1
    print(json.dumps(st, indent=2))
    return 0


def dispatch_reconnect_tunnel(alias: str, host_cfg: dict, port: int) -> int:
    """Tear down the local ssh -L tunnel and reopen it, WITHOUT touching
    the remote daemon. Use when a long batch silently lost its tunnel
    (server-side SSH reset, network drop) but the daemon is still alive.

    If `--status` after this still fails, the daemon itself is gone —
    fall back to --stop + --start."""
    ssh_alias = host_cfg.get("ssh_alias") or alias
    _tunnel_stop_silent(port, ssh_alias)
    pid = _tunnel_start(ssh_alias, port)
    if pid:
        print(f"[akg_cli] ssh -L 127.0.0.1:{port} -> "
              f"{ssh_alias}:{port} reconnected (tunnel pid={pid})")
    st = _curl_status("127.0.0.1", port)
    if st is None:
        print(
            f"[akg_cli] tunneled status probe failed after reconnect; "
            f"remote daemon may have died too — use --stop + --start.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(st, indent=2))
    return 0
