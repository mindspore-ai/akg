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

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from ai_kernel_generator.utils.common_utils import load_yaml

from ai_kernel_generator.cli.cli.constants import DisplayStyle


class ResolvedValue(NamedTuple):
    key: str
    value: Any
    source: str  # cli/config/env/default/ask
    required: bool = False


class CLIConfigManager:
    """负责 --config 加载、读取与“CLI > config > env > default”的解析。"""

    def __init__(self) -> None:
        self.config_path: Optional[str] = None
        self.config: Dict[str, Any] = {}

    def reset(self) -> None:
        self.config_path = None
        self.config = {}

    def load(self, config_path: str) -> None:
        cfg_path = Path(config_path).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = Path(os.getcwd()) / cfg_path
        if not cfg_path.exists():
            raise FileNotFoundError(f"--config 指定的文件不存在: {cfg_path}")

        cfg = load_yaml(str(cfg_path))
        if not isinstance(cfg, dict):
            raise ValueError(
                f"--config 内容必须是 YAML 字典（mapping），实际类型: {type(cfg)}"
            )

        self.config_path = config_path
        self.config = cfg

    def get(self, *keys: str) -> Any:
        if not isinstance(self.config, dict):
            return None
        for k in keys:
            if k in self.config and self.config.get(k) is not None:
                return self.config.get(k)
        return None

    def print_report_if_any(self, console: Console) -> None:
        if not self.config_path:
            return

        cfg = self.config or {}
        known_keys = {
            "server_url",
            "framework",
            "backend",
            "arch",
            "dsl",
            "stream",
            "use_stream",
        }

        unknown = sorted([k for k in cfg.keys() if k not in known_keys])

        table = Table(
            title="配置检查（--config）", box=box.SIMPLE_HEAVY, show_header=True
        )
        table.add_column("字段", style=DisplayStyle.CYAN, width=20)
        table.add_column("值", style=DisplayStyle.YELLOW)
        table.add_column("类型", style=DisplayStyle.DIM)

        table.add_row("config_path", str(self.config_path), "path")
        for key in sorted([k for k in cfg.keys() if k in known_keys]):
            val = cfg.get(key)
            table.add_row(key, str(val), type(val).__name__)

        if unknown:
            table.add_row("unknown_keys", ", ".join(unknown), "list[str]")
        console.print(table)

    @staticmethod
    def print_resolution_table(
        console: Console, items: List[ResolvedValue], title: str = "配置解析与校验"
    ) -> None:
        table = Table(title=title, box=box.ROUNDED, show_header=True)
        table.add_column("项", style=DisplayStyle.CYAN, width=18)
        table.add_column("值", style=DisplayStyle.YELLOW, width=40)
        table.add_column("来源", style=DisplayStyle.DIM, width=10)
        table.add_column("必需", style=DisplayStyle.DIM, width=6)

        for it in items:
            val = it.value
            if isinstance(val, list):
                val_str = ",".join(str(x) for x in val)
            else:
                val_str = str(val)
            required_str = "Y" if it.required else "-"
            table.add_row(it.key, val_str, it.source, required_str)
        console.print(table)

    def resolve_from_sources(
        self,
        ctx: Optional[typer.Context],
        *,
        param_name: str,
        cli_value: Any,
        config_keys: List[str],
        env_keys: List[str],
        default_value: Any,
        cast: Optional[Callable[[Any], Any]] = None,
    ) -> ResolvedValue:
        """统一参数解析优先级：CLI（显式传参） > config > env > default"""

        # 1) CLI 显式传参
        if ctx is not None and hasattr(ctx, "get_parameter_source"):
            src = ctx.get_parameter_source(param_name)
            if getattr(src, "name", "") == "COMMANDLINE":
                v = cli_value
                if cast:
                    v = cast(v)
                return ResolvedValue(param_name, v, "cli", required=False)

        # 2) config
        for k in config_keys:
            v = self.get(k)
            if v is not None:
                if cast:
                    v = cast(v)
                return ResolvedValue(param_name, v, "config", required=False)

        # 3) env
        for ek in env_keys:
            ev = os.environ.get(ek)
            if ev is not None and str(ev).strip() != "":
                v2: Any = ev
                if cast:
                    v2 = cast(v2)
                return ResolvedValue(param_name, v2, "env", required=False)

        # 4) default
        v = default_value
        if cast:
            v = cast(v)
        return ResolvedValue(param_name, v, "default", required=False)
