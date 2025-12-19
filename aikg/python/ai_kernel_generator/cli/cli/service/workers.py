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

import time
import urllib.parse
from typing import Any, Dict, List, Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from textual import log

from ai_kernel_generator.cli.cli.constants import DisplayStyle


class WorkerRegistry:
    """负责解析/保存 workers，并在需要时向 server 注册。"""

    def __init__(self) -> None:
        self.workers: List[str] = []

    def clear(self) -> None:
        self.workers = []

    @staticmethod
    def parse_workers(workers: str) -> List[str]:
        if workers is None:
            return []
        raw_items = [x.strip() for x in str(workers).split(",") if x.strip()]
        if not raw_items:
            return []

        normalized: List[str] = []
        for item in raw_items:
            url = item
            if "://" not in url:
                url = f"http://{url}"

            parsed = urllib.parse.urlparse(url)
            if not parsed.scheme or not parsed.hostname or parsed.port is None:
                raise ValueError(
                    f"worker_url 格式非法: {item}（期望形如 https://host:port 或 host:port）"
                )

            normalized.append(f"{parsed.scheme}://{parsed.hostname}:{parsed.port}")

        # 去重但保持顺序
        dedup: List[str] = []
        seen = set()
        for u in normalized:
            if u in seen:
                continue
            dedup.append(u)
            seen.add(u)
        return dedup

    def configure(self, *, workers_cli: Optional[str], config: Dict[str, Any]) -> None:
        resolved_workers = workers_cli
        if not resolved_workers and isinstance(config, dict):
            cfg_workers = config.get("workers")
            if isinstance(cfg_workers, str) and cfg_workers.strip():
                resolved_workers = cfg_workers
            elif isinstance(cfg_workers, list) and cfg_workers:
                resolved_workers = ",".join(str(x) for x in cfg_workers)

        if resolved_workers:
            self.workers = self.parse_workers(resolved_workers)

    def register_if_any(self, console: Console, server_url: str) -> None:
        if not self.workers:
            return

        import httpx

        server_url = server_url.rstrip("/")
        workers = list(self.workers)

        console.print(
            f"[{DisplayStyle.CYAN}]正在注册 worker_url 到 server: {server_url}[/{DisplayStyle.CYAN}]"
        )
        table = Table(title="Worker 注册结果", box=box.ROUNDED, show_header=True)
        table.add_column("worker_url", style=DisplayStyle.CYAN)
        table.add_column("backend/arch", style=DisplayStyle.YELLOW)
        table.add_column("capacity", style=DisplayStyle.YELLOW, justify="right")
        table.add_column("结果", style=DisplayStyle.GREEN)

        with httpx.Client(timeout=10.0) as client:
            for w in workers:
                try:
                    status_url = f"{w}/api/v1/status"
                    # worker 启动后可能需要短暂 warmup：做几次重试，减少时序误伤
                    last_exc: Optional[Exception] = None
                    status = None
                    for _ in range(10):
                        try:
                            resp = client.get(status_url)
                            resp.raise_for_status()
                            status = resp.json()
                            # status=ready 才算 OK
                            if isinstance(status, dict) and str(
                                status.get("status", "")
                            ).lower() in ["ready", "ok"]:
                                break
                        except Exception as e:
                            last_exc = e
                            time.sleep(0.3)
                            continue
                    if status is None:
                        raise RuntimeError(f"worker status 获取失败: {last_exc}")
                    backend = status.get("backend", "")
                    arch = status.get("arch", "")
                    devices = status.get("devices", [])
                    capacity = (
                        len(devices)
                        if isinstance(devices, list) and len(devices) > 0
                        else 1
                    )

                    if not backend or not arch:
                        raise RuntimeError(f"worker status 缺少 backend/arch: {status}")

                    reg_url = f"{server_url}/api/v1/workers/register"
                    payload = {
                        "url": w,
                        "backend": backend,
                        "arch": arch,
                        "capacity": capacity,
                        "tags": [],
                    }
                    resp = client.post(reg_url, json=payload)
                    resp.raise_for_status()
                    table.add_row(
                        w,
                        f"{backend}/{arch}",
                        str(capacity),
                        f"[{DisplayStyle.BOLD_GREEN}]OK[/{DisplayStyle.BOLD_GREEN}]",
                    )
                except Exception as e:
                    log.warning(
                        "[Workers] register failed",
                        worker_url=str(w or ""),
                        server_url=str(server_url or ""),
                        exc_info=e,
                    )
                    table.add_row(
                        w,
                        "-",
                        "-",
                        f"[{DisplayStyle.BOLD_RED}]FAIL[/{DisplayStyle.BOLD_RED}] {e}",
                    )
                    console.print(table)
                    raise typer.Exit(code=2)

        console.print(table)

    def server_has_worker(self, server_url: str, backend: str, arch: str) -> bool:
        try:
            import httpx

            url = f"{server_url.rstrip('/')}/api/v1/workers/status"
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                workers = resp.json() or []
            for w in workers:
                if str(w.get("backend", "")).lower() != str(backend).lower():
                    continue
                if str(w.get("arch", "")).lower() != str(arch).lower():
                    continue
                return True
            return False
        except Exception as e:
            # 网络错误/接口不支持：保守返回 True（避免误伤）
            log.debug(
                "[Workers] server_has_worker failed; fallback True",
                server_url=str(server_url or ""),
                backend=str(backend or ""),
                arch=str(arch or ""),
                exc_info=e,
            )
            return True
