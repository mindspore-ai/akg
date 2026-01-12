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

from ai_kernel_generator.cli.cli.constants import Defaults
from textual import log


def get_log_dir() -> Path:
    """获取统一日志目录（避免在源码目录/工作目录下产生大量运行产物）。"""

    env_dir = (os.environ.get("AIKG_LOG_DIR") or "").strip()
    base = env_dir or Defaults.LOG_DIR
    p = Path(base).expanduser()
    try:
        p = p.resolve()
    except (OSError, RuntimeError) as e:
        # resolve 失败也不影响使用
        log.debug(
            "[Paths] resolve failed; use non-resolved path", path=str(p), exc_info=e
        )
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_panel_log_base_dir() -> Path:
    return get_log_dir() / "panels"


def get_stream_save_dir() -> Path:
    return get_log_dir() / "streams"


def get_process_log_dir() -> Path:
    return get_log_dir() / "processes"


def get_ai_kernel_generator_pkg_dir() -> Path:
    """定位 ai_kernel_generator 包目录（用于拼接 server/worker 脚本路径）。"""

    import ai_kernel_generator  # 延迟导入，避免循环依赖

    return Path(ai_kernel_generator.__file__).resolve().parent
