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
import socket
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from typing import List, Optional, Tuple

import logging

from rich.console import Console

from ai_kernel_generator.cli.constants import DisplayStyle, UISymbol

logger = logging.getLogger(__name__)
from ai_kernel_generator.cli.console import AKGConsole
from ai_kernel_generator.cli.utils.paths import (
    get_ai_kernel_generator_pkg_dir,
    get_process_log_dir,
)
from ai_kernel_generator.cli.utils.worker_state import (
    get_worker_entry,
    load_worker_state,
    pid_alive,
    remove_worker_entry,
    save_worker_state,
    set_worker_entry,
    terminate_pid,
)


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
    def _probe_local_worker(url: str) -> bool:
        """探测本地 worker 是否可用。"""
        try:
            import httpx

            status_url = f"{url.rstrip('/')}/api/v1/status"
            with httpx.Client(timeout=1.0, trust_env=False) as client:
                resp = client.get(status_url)
                if resp.status_code != 200:
                    return False
                data = resp.json()
            status = (
                str(data.get("status", "")).lower() if isinstance(data, dict) else ""
            )
            return status in ["ready", "ok", "initializing"]
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
        host: str = "127.0.0.1",
        port: int = 9001,
    ) -> Tuple[Optional[subprocess.Popen], str, str]:
        """启动本地 worker service 子进程。"""
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = get_process_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / f"worker_server_{timestamp}.log")

        pkg_dir = get_ai_kernel_generator_pkg_dir()
        worker_module = pkg_dir / "worker" / "server.py"
        if not worker_module.exists():
            raise FileNotFoundError(f"Worker Service not found at: {worker_module}")

        backend_n = (backend or "").strip().lower()
        arch_n = (arch or "").strip()
        if not backend_n:
            backend_n = (os.environ.get("WORKER_BACKEND") or "").strip().lower()
        if not arch_n:
            arch_n = (os.environ.get("WORKER_ARCH") or "").strip()
        if not backend_n:
            backend_n = "cuda"
        if not arch_n:
            arch_n = "a100"

        if not devices:
            devices = [0]

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

        process = subprocess.Popen(
            [sys.executable, str(worker_module)],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        console.print(
            f"[{DisplayStyle.DIM}]   等待 worker 启动...[/{DisplayStyle.DIM}]"
        )
        time.sleep(2)

        if process.poll() is not None:
            log_f.close()
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    error_log = f.read()
            except OSError as e:
                logger.debug(
                    "[WorkerService] read worker log failed",
                    path=str(log_file),
                    exc_info=e,
                )
                error_log = ""
            raise RuntimeError(
                f"Worker 启动失败，退出码: {process.returncode}\n日志:\n{error_log}"
            )

        # url：用于 server 注册（本地默认用 localhost/127.0.0.1）
        access_host = self._format_url_host(host)
        url = f"http://{access_host}:{port}"

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
