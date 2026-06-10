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

"""Local ssh -L tunnel + port-owner detection for ``akg_cli worker``.

This layer only deals with local-side processes (tunnel ssh forks + local
TCP port ownership). No SSH RPC, no diagnostic classification, no
rendering — those live in ``remote_probe.py`` / ``diagnostics.py``.

All ``subprocess.run`` calls use ``stdout=PIPE, stderr=DEVNULL`` — Windows
PowerShell / OpenSSH deadlock when ``capture_output=True`` because both
PIPEs need concurrent draining and the timeout cannot reliably kill the
child (see commit history)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

STATE_DIR = Path.home() / ".akg_agents"


def tunnel_pid_path(port: int) -> Path:
    return STATE_DIR / "tunnels" / f"{port}.pid"


def kill_pid_hint(pid: int) -> str:
    return f"taskkill /F /PID {pid}" if os.name != "posix" else f"kill {pid}"


def who_holds_port(port: int) -> Optional[dict]:
    """Return ``{"pid", "cmdline"}`` for the LISTEN owner of local TCP
    ``port``, or None if free. Cross-platform: ``lsof`` on POSIX,
    ``Get-NetTCPConnection`` on Windows."""
    if os.name == "posix":
        try:
            out = subprocess.run(
                ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=5,
            )
            pids = [int(p) for p in out.stdout.split() if p.isdigit()]
            if not pids:
                return None
            pid = pids[0]
            ps = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=5,
            )
            return {"pid": pid, "cmdline": ps.stdout.strip()}
        except Exception:
            return None
    # Windows. ``$pid`` is reserved (current shell PID); use ``$owner_pid``.
    try:
        ps_cmd = (
            f"$conn = Get-NetTCPConnection -LocalPort {port} -State Listen "
            f"-ErrorAction SilentlyContinue | Select-Object -First 1; "
            f"if (-not $conn) {{ exit 0 }}; "
            f"$owner_pid = $conn.OwningProcess; "
            f"$cmd = (Get-CimInstance Win32_Process "
            f"-Filter \"ProcessId=$owner_pid\").CommandLine; "
            f"Write-Output \"$owner_pid|$cmd\""
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=10,
        )
        line = out.stdout.strip()
        if not line or "|" not in line:
            return None
        pid_s, cmd = line.split("|", 1)
        return {"pid": int(pid_s.strip()), "cmdline": cmd.strip()}
    except Exception:
        return None


def find_tunnel_pid(port: int, ssh_alias: str) -> int:
    """Locate the forked ``ssh -L <port>:...`` PID by cmdline scan. 0 if
    not found. Used to back-fill the pid file when ``ssh -f`` doesn't
    surface its forked child."""
    if os.name == "posix":
        try:
            out = subprocess.run(
                ["pgrep", "-f", f"ssh.*-L {port}:.*{ssh_alias}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=5,
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
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=10,
        )
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if ln.isdigit():
                return int(ln)
    except Exception:
        pass
    return 0


def tunnel_stop_silent(port: int, ssh_alias: str = "") -> None:
    """Best-effort tunnel teardown. Prefers the stashed pid; falls back to
    a cmdline scan (handles --start that lost its pid file)."""
    pid_path = tunnel_pid_path(port)
    pid = 0
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text().strip())
        except (ValueError, FileNotFoundError):
            pid = 0
    if pid == 0 and ssh_alias:
        pid = find_tunnel_pid(port, ssh_alias)
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


def tunnel_start(ssh_alias: str, port: int) -> int:
    """Open ``ssh -f -N -T -L <port>:127.0.0.1:<port> <alias>``, stash the
    forked pid. Returns the pid on success, 0 on soft failure.

    Pre-bind check: if local ``port`` is already taken (orphan tunnel /
    foreign squatter), refuses to spawn and prints the holder + remediation.
    Without this, ``ssh -f`` swallows the bind error in the forked child
    and the caller hangs on a silent /status timeout.

    Does NOT pass ExitOnForwardFailure=yes — user ``~/.ssh/config`` may
    declare unrelated RemoteForward entries whose failure shouldn't take
    down the -L we need."""
    (STATE_DIR / "tunnels").mkdir(parents=True, exist_ok=True)
    tunnel_stop_silent(port, ssh_alias)

    # 旧 ssh 拿到 SIGTERM 后到真正释放 socket 有一小段窗口（OS 走 TIME_WAIT
    # 也可能多几百 ms）。立刻 who_holds_port 会把刚被杀的 ssh 当外部占用，
    # --start 误报"端口被占"返回 0。Poll 最多 ~2s 等端口空出来；如果到点
    # 还有占用方，那才算真外部进程，给 operator 报错。
    holder = who_holds_port(port)
    for _ in range(20):
        if holder is None:
            break
        time.sleep(0.1)
        holder = who_holds_port(port)
    if holder:
        print(
            f"[akg_cli] :{port} 被 PID={holder['pid']} 占着\n"
            f"  cmdline: {holder['cmdline'][:160]}\n"
            f"  → `{kill_pid_hint(holder['pid'])}` 后再 --start",
            file=sys.stderr,
        )
        return 0

    # Keepalive bumped from OpenSSH default 60s/3x → 30s/10x (~5min idle
    # tolerance) so long PLAN phases don't lose the tunnel between evals.
    cmd = [
        "ssh", "-f", "-N", "-T",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
        "-L", f"{port}:127.0.0.1:{port}",
        ssh_alias,
    ]
    # Windows-specific: ssh -f doesn't reliably daemonize when invoked via
    # Python subprocess —— even with all 3 std streams to DEVNULL, the
    # parent ssh.exe stays attached and subprocess.call() blocks forever.
    # Sidestep entirely: Popen with DETACHED_PROCESS + NEW_PROCESS_GROUP
    # so ssh.exe is independent from akg_cli's console, don't wait, then
    # poll for the tunnel pid via cmdline scan. On POSIX, ssh -f detaches
    # correctly via setsid + we can keep subprocess.call.
    kwargs = dict(
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.name == "posix":
        rc = subprocess.call(cmd, **kwargs)
        if rc != 0:
            print(f"[akg_cli] ssh -L exit rc={rc}", file=sys.stderr)
    else:
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP
                 | getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
        try:
            subprocess.Popen(cmd, creationflags=flags, **kwargs)
        except Exception as e:
            print(f"[akg_cli] ssh -L spawn failed: {e}", file=sys.stderr)
            return 0
        # Poll up to ~5s for ssh to authenticate + bind the local port.
        for _ in range(10):
            time.sleep(0.5)
            if find_tunnel_pid(port, ssh_alias):
                break
    pid = find_tunnel_pid(port, ssh_alias)
    if pid:
        tunnel_pid_path(port).write_text(str(pid))
    return pid
