# Copyright 2025 Huawei Technologies Co., Ltd
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

from __future__ import annotations

import ipaddress
import os
import signal
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from typing import List, Optional, Tuple

import logging

from akg_agents.cli.constants import DisplayStyle, UISymbol

logger = logging.getLogger(__name__)
from akg_agents.cli.console import AKGConsole
from akg_agents.cli.service.worker_config import worker_timing
from akg_agents.core.worker.eval_config import eval_defaults
from akg_agents.cli.utils.paths import (
    get_akg_agents_pkg_dir,
    get_process_log_dir,
)
from akg_agents.cli.utils.worker_state import (
    get_worker_entry,
    load_worker_state,
    pid_alive,
    remove_worker_entry,
    save_worker_state,
    set_worker_entry,
    terminate_pid,
)


# Silence httpx/httpcore DEBUG loggers — their connect_tcp.started /
# connect_tcp.failed trace messages call logger.debug 上千次/s。当远端 /
# 本机磁盘满（ENOSPC）时，Python logging 试 flush 失败，触发 "Logging
# error" 元 traceback，再 cascade 报错把终端灌成洪水。这些 trace 对终
# 端用户没用，强制 WARNING 级。
import logging as _logging
_logging.getLogger("httpx").setLevel(_logging.WARNING)
_logging.getLogger("httpcore").setLevel(_logging.WARNING)


class WorkerService:
    """管理 worker 注册和本地 worker service 子进程。"""

    def __init__(self) -> None:
        self.workers: List[str] = []
        self._process: Optional[subprocess.Popen] = None
        self._log_file: Optional[str] = None
        self._url: Optional[str] = None
        self._port: Optional[int] = None

    def clear(self) -> None:
        """清空 worker 列表。"""
        self.workers = []

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    @property
    def log_file(self) -> Optional[str]:
        return self._log_file

    @property
    def url(self) -> Optional[str]:
        return self._url

    @staticmethod
    def parse_workers(workers: str) -> List[str]:
        """解析 worker URL 字符串为标准化 URL 列表。"""
        if workers is None:
            return []
        raw_items = [x.strip() for x in str(workers).split(",") if x.strip()]
        if not raw_items:
            return []

        normalized: List[str] = []
        for item in raw_items:
            url = item
            if "://" not in url:
                if url.count(":") > 1 and "[" not in url:
                    host_part, port_part = url.rsplit(":", 1)
                    if port_part.isdigit():
                        url = f"http://[{host_part}]:{port_part}"
                    else:
                        url = f"http://{url}"
                else:
                    url = f"http://{url}"

            parsed = urllib.parse.urlparse(url)
            if not parsed.scheme or not parsed.hostname or parsed.port is None:
                raise ValueError(
                    f"worker_url 格式非法: {item}（期望形如 https://host:port 或 host:port）"
                )

            hostname = parsed.hostname
            if hostname and ":" in hostname:
                hostname = f"[{hostname}]"
            normalized.append(f"{parsed.scheme}://{hostname}:{parsed.port}")

        # 去重但保持顺序
        dedup: List[str] = []
        seen = set()
        for u in normalized:
            if u in seen:
                continue
            dedup.append(u)
            seen.add(u)
        return dedup

    @staticmethod
    def _tail_log(path: str, lines: int = 30) -> str:
        """Best-effort read of last ``lines`` lines for in-CLI dump."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-lines:])
        except OSError as e:
            logger.debug("[WorkerService] tail log failed, path=%s", path, exc_info=e)
            return ""

    @staticmethod
    def _probe_local_worker(url: str, timeout: Optional[float] = None,
                            ready_only: bool = True) -> bool:
        """探测本地 worker。

        - ``ready_only=True``（默认，poll loop / state-reuse 用）：daemon
          的 /status 必须返回 ``ready`` 或 ``ok`` 才算可用。``initializing``
          意味着 HTTP 在线但 worker 还没装好，把它当 ready 会让 --start
          跳过 spawn / 给一个永远转 init 的 daemon 写 state。
        - ``ready_only=False``：仅查 HTTP 是否 200，状态 ``initializing``
          也算 alive；用于纯探活场景。

        ``timeout`` 不传时使用 ``worker_timing().ready_probe_timeout``。"""
        try:
            import httpx
            if timeout is None:
                timeout = worker_timing().ready_probe_timeout
            status_url = f"{url.rstrip('/')}/api/v1/status"
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                resp = client.get(status_url)
                if resp.status_code != 200:
                    return False
                data = resp.json()
            status = (
                str(data.get("status", "")).lower() if isinstance(data, dict) else ""
            )
            ready_set = ("ready", "ok")
            alive_set = ("ready", "ok", "initializing")
            return status in (ready_set if ready_only else alive_set)
        except Exception as e:
            logger.debug(f"[WorkerService] probe local worker failed, url={url}", exc_info=e)
            return False

    @staticmethod
    def _format_url_host(host: str) -> str:
        """格式化 URL 主机名。"""
        host_n = (host or "").strip()
        if host_n in ["0.0.0.0", ""]:
            host_n = "127.0.0.1"
        elif host_n in ["::", "[::]"]:
            host_n = "::1"
        elif host_n.startswith("[") and host_n.endswith("]"):
            host_n = host_n[1:-1]

        try:
            is_ipv6 = isinstance(ipaddress.ip_address(host_n), ipaddress.IPv6Address)
        except ValueError:
            is_ipv6 = ":" in host_n

        if is_ipv6:
            return f"[{host_n}]"
        return host_n

    def start(
        self,
        console: AKGConsole,
        *,
        backend: str,
        arch: str,
        devices: List[int],
        host: str,
        port: int,
    ) -> Tuple[Optional[subprocess.Popen], str, str]:
        """启动本地 worker service 子进程。host/port 由调用方解析。"""
        if self._process is not None and self._process.poll() is None:
            # 已在运行：直接返回已有信息
            return self._process, self._log_file or "", self._url or ""

        state = load_worker_state()
        entry = get_worker_entry(state, port)
        if entry:
            pid = entry.get("pid")
            if isinstance(pid, int) and pid > 0 and pid_alive(pid):
                url = str(entry.get("url") or "")
                if not url:
                    access_host = self._format_url_host(host)
                    url = f"http://{access_host}:{port}"
                if self._probe_local_worker(url):
                    console.print(
                        f"[{DisplayStyle.DIM}]   检测到 worker 已在端口 {port} 运行，跳过启动。[/{DisplayStyle.DIM}]"
                    )
                else:
                    console.print(
                        f"[{DisplayStyle.YELLOW}]{UISymbol.WARNING} 检测到端口 {port} 的 worker PID 仍在运行，但状态探测失败；已视为运行中（如需重启请先 --stop）。[/{DisplayStyle.YELLOW}]"
                    )
                self._process = None
                self._log_file = str(entry.get("log_file") or "")
                self._url = url
                self._port = int(port)
                return None, self._log_file or "", url
            # 记录里有残留 pid，先清理
            remove_worker_entry(state, port)
            save_worker_state(state)

        # log_file 在 POSIX 上落 /tmp/akg_worker_$port.log —— akg_cli 的远端
        # 探针 (cli/service/remote_probe.py) 默认 tail 这条路径；两边对齐到
        # 单一约定避免 probe "(no log)" 误诊断。Windows 仍走 process_log_dir
        # （本机起 daemon 通常只在 Linux NPU 上发生）。
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.name == "posix":
            log_file = f"/tmp/akg_worker_{port}.log"
        else:
            log_dir = get_process_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir / f"worker_server_{timestamp}.log")

        pkg_dir = get_akg_agents_pkg_dir()
        worker_module = pkg_dir / "worker" / "server.py"
        if not worker_module.exists():
            raise FileNotFoundError(f"Worker Service not found at: {worker_module}")

        # 调用方（misc.py worker_cmd）已经吃过 CLI → env → WorkerConfig
        # 的 precedence，传过来的 backend/arch 不会是空；这里只 normalize
        # 大小写不做兜底，避免 defensive double-fallback 跟 WorkerConfig
        # 的默认值分裂。
        backend_n = (backend or "").strip().lower()
        arch_n = (arch or "").strip()

        if not devices:
            devices = [0]

        # AKG_CLI_QUIET=1 表示被 SSH 递归调用（remote_dispatch 起远端 daemon），
        # 本机已经印过 [N/4] 进度，远端这边只回一两行关键信息就够。
        quiet = os.environ.get("AKG_CLI_QUIET") == "1"
        if not quiet:
            console.print(
                f"[{DisplayStyle.CYAN}]{UISymbol.ROCKET} 启动 worker service (端口: {port})...[/{DisplayStyle.CYAN}]"
            )
            console.print(
                f"[{DisplayStyle.DIM}]   日志文件: {log_file}[/{DisplayStyle.DIM}]"
            )

        log_f = open(log_file, "w", encoding="utf-8")
        env = os.environ.copy()
        env["WORKER_BACKEND"] = backend_n
        env["WORKER_ARCH"] = arch_n
        env["WORKER_DEVICES"] = ",".join(str(x) for x in devices)
        env["WORKER_HOST"] = str(host)
        env["WORKER_PORT"] = str(port)
        # daemon 通过 /api/v1/status 暴露这个路径，让 akg_cli probe 不再靠
        # 猜（process_log_dir/worker_server_*.log vs /tmp/akg_worker_*.log）。
        env["AKG_WORKER_LOG_FILE"] = log_file
        if os.name == "posix":
            # Eval subprocesses are separate sessions so timeout can kill their
            # whole tree.  Persist their PGIDs per worker port as well, allowing
            # a successor daemon to reap them after this daemon is SIGKILLed.
            env["AKG_WORKER_PROCESS_REGISTRY"] = (
                f"/tmp/akg_worker_{port}_process_groups.json"
            )
        timing = worker_timing()
        for key, value in timing.as_env().items():
            env.setdefault(key, value)
        for key, value in eval_defaults().as_env().items():
            env.setdefault(key, value)

        # stdin=DEVNULL is what lets this Popen detach cleanly when the
        # akg_cli that's calling us was itself spawned over SSH — without
        # it the child inherits the SSH channel 0 fd, setsid orphans the
        # group but ssh still sees an open channel and won't return.
        # See ar_cli.py:cmd_worker_start in claude-autoresearch for the
        # upstream reference (same pattern).
        process = subprocess.Popen(
            [sys.executable, str(worker_module)],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        access_host = self._format_url_host(host)
        url = f"http://{access_host}:{port}"
        # Timing：递归 SSH 路径下，remote_dispatch 把 worker.* timing
        # 通过 env 传过来；本机直跑则用 config.yaml / WorkerTiming 默认。
        # 这是 worker.* config 的单一事实源能传到远端 daemon 的关键，
        # 否则远端 worker_service 永远卡 60s 不能调。
        ready_timeout = timing.ready_timeout
        ready_tick = timing.ready_poll_interval
        if not quiet:
            console.print(
                f"[{DisplayStyle.DIM}]   等待 worker /status ready（最长 {ready_timeout}s）...[/{DisplayStyle.DIM}]"
            )
        deadline = time.time() + ready_timeout
        last_beat = time.time()
        while time.time() < deadline:
            if process.poll() is not None:
                log_f.close()
                console.print(
                    f"[{DisplayStyle.RED}]Worker 启动期退出 rc={process.returncode}，"
                    f"log tail:[/{DisplayStyle.RED}]\n{self._tail_log(log_file)}"
                )
                raise RuntimeError(
                    f"Worker 启动失败 rc={process.returncode}（log: {log_file}）"
                )
            if self._probe_local_worker(url):
                break
            now = time.time()
            if not quiet and now - last_beat >= ready_tick:
                console.print(
                    f"[{DisplayStyle.DIM}]   /status 未就绪 "
                    f"({int(now - deadline + ready_timeout)}s/{ready_timeout}s),"
                    f" PID={process.pid} 仍在运行[/{DisplayStyle.DIM}]"
                )
                last_beat = now
            time.sleep(1)
        else:
            # /status 在 ready_timeout 内都不通：daemon 进程仍在运行但不可用
            # （torch_npu 卡 init / 端口被抢 / FastAPI bind 失败等）。不写
            # worker_state，不打"已启动"，raise 让上层 typer 退非 0 —— 比
            # "伪装成功"诚实。
            raise RuntimeError(
                f"Worker PID={process.pid} 仍在运行但 /status "
                f"{ready_timeout}s 未就绪 —— 可能 daemon 卡在初始化"
                f"（torch_npu import / device init / FastAPI bind 失败）。"
                f"log tail:\n{self._tail_log(log_file)}\n"
                f"完整 log: {log_file}"
            )

        if quiet:
            # Recursive SSH 调用下，本机 dispatch_start 会自己打"daemon spawned"
            # + 后续 poll heartbeat —— 这里只回一行远端 ready 简讯。
            console.print(
                f"[{DisplayStyle.DIM}]   远端 worker ready (PID {process.pid}, log {log_file})[/{DisplayStyle.DIM}]"
            )
        else:
            console.print(
                f"[{DisplayStyle.GREEN}]{UISymbol.DONE} Worker 已启动 (PID: {process.pid})[/{DisplayStyle.GREEN}]"
            )
            console.print(
                f"[{DisplayStyle.DIM}]   健康检查: {url}/api/v1/status[/{DisplayStyle.DIM}]"
            )

        entry = {
            "pid": process.pid,
            "port": int(port),
            "host": str(host),
            "url": url,
            "backend": backend_n,
            "arch": arch_n,
            "devices": devices,
            "log_file": log_file,
            "started_at": datetime.now().isoformat(),
        }
        state = load_worker_state()
        set_worker_entry(state, port, entry)
        save_worker_state(state)

        self._process = process
        self._log_file = log_file
        self._url = url
        self._port = int(port)
        return process, log_file, url

    def stop(self, console: AKGConsole, port: Optional[int] = None) -> None:
        """停止本地 worker service 子进程。"""
        had_process = self._process is not None
        if self._process is not None:
            console.print(
                f"\n[{DisplayStyle.YELLOW}]{UISymbol.STOP} 停止 worker service...[/{DisplayStyle.YELLOW}]"
            )

            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                else:
                    self._process.terminate()

                try:
                    self._process.wait(timeout=5)
                    console.print(
                        f"[{DisplayStyle.GREEN}]{UISymbol.DONE} Worker 已停止[/{DisplayStyle.GREEN}]"
                    )
                except subprocess.TimeoutExpired:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    else:
                        self._process.kill()
                    console.print(
                        f"[{DisplayStyle.YELLOW}]{UISymbol.WARNING} Worker 被强制终止[/{DisplayStyle.YELLOW}]"
                    )
            except Exception as e:
                console.print(
                    f"[{DisplayStyle.YELLOW}]{UISymbol.WARNING} 停止 worker 时出错: {e}[/{DisplayStyle.YELLOW}]"
                )

            self._process = None

        if self._log_file:
            console.print(
                f"[{DisplayStyle.DIM}]   日志已保存到: {self._log_file}[/{DisplayStyle.DIM}]"
            )
            self._log_file = None
        if self._port is not None:
            state = load_worker_state()
            remove_worker_entry(state, self._port)
            save_worker_state(state)
            self._port = None
        self._url = None

        if not had_process and port is not None:
            state = load_worker_state()
            entry = get_worker_entry(state, port)
            if not entry:
                console.print(
                    f"[{DisplayStyle.YELLOW}]{UISymbol.WARNING} 未找到端口 {port} 的 worker 记录。[/{DisplayStyle.YELLOW}]"
                )
                return
            pid = entry.get("pid")
            if not isinstance(pid, int) or pid <= 0:
                console.print(
                    f"[{DisplayStyle.YELLOW}]{UISymbol.WARNING} worker 记录缺少有效 PID（端口 {port}）。[/{DisplayStyle.YELLOW}]"
                )
                remove_worker_entry(state, port)
                save_worker_state(state)
                return
            if not pid_alive(pid):
                console.print(
                    f"[{DisplayStyle.YELLOW}]{UISymbol.WARNING} worker 进程不存在（PID: {pid}），已清理记录。[/{DisplayStyle.YELLOW}]"
                )
                remove_worker_entry(state, port)
                save_worker_state(state)
                return
            console.print(
                f"\n[{DisplayStyle.YELLOW}]{UISymbol.STOP} 停止 worker service (端口: {port})...[/{DisplayStyle.YELLOW}]"
            )
            ok = terminate_pid(pid)
            if ok:
                console.print(
                    f"[{DisplayStyle.GREEN}]{UISymbol.DONE} Worker 已停止[/{DisplayStyle.GREEN}]"
                )
            else:
                console.print(
                    f"[{DisplayStyle.YELLOW}]{UISymbol.WARNING} Worker 停止失败（PID: {pid}）[/{DisplayStyle.YELLOW}]"
                )
            log_file = entry.get("log_file")
            if log_file:
                console.print(
                    f"[{DisplayStyle.DIM}]   日志已保存到: {log_file}[/{DisplayStyle.DIM}]"
                )
            remove_worker_entry(state, port)
            save_worker_state(state)
