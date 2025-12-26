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

from rich import box
from rich.console import Console
from rich.table import Table

from ai_kernel_generator.cli.cli.constants import DisplayStyle
from textual import log


class JobInspector:
    """尽力而为的 job 状态摘要输出（用于 Ctrl+C/异常收尾）。"""

    def try_print_job_summary(
        self, console: Console, server_url: str, job_id: str
    ) -> None:
        if not job_id:
            return
        try:
            import httpx

            url = f"{server_url.rstrip('/')}/api/v1/jobs/{job_id}/status"
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                status = resp.json() or {}
            result = status.get("result") or {}
            meta = result.get("metadata") or {}

            table = Table(title="取消/结束状态摘要", box=box.ROUNDED, show_header=True)
            table.add_column("项目", style=DisplayStyle.CYAN, width=18)
            table.add_column("内容", style=DisplayStyle.YELLOW)

            table.add_row("job_id", job_id)
            table.add_row("status", str(status.get("status")))
            if isinstance(meta, dict):
                if meta.get("log_dir"):
                    table.add_row("log_dir", str(meta.get("log_dir")))
                if meta.get("task_desc_path"):
                    table.add_row("task_desc_path", str(meta.get("task_desc_path")))
                if meta.get("kernel_code_path"):
                    table.add_row("kernel_code_path", str(meta.get("kernel_code_path")))
            console.print(table)
        except Exception as e:
            log.debug("[Jobs] try_print_job_summary failed; ignore", exc_info=e)
            return
