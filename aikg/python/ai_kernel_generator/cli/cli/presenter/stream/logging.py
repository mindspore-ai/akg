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

import logging
import os
from pathlib import Path

from textual import log as textual_log

from ai_kernel_generator.cli.cli.utils.paths import get_log_dir


_LOG_LEVEL_MAP = {
    "0": logging.DEBUG,
    "1": logging.INFO,
    "2": logging.WARNING,
    "3": logging.ERROR,
}


def init_stream_renderer_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = _LOG_LEVEL_MAP.get(os.getenv("AIKG_LOG_LEVEL", "1"), logging.INFO)
    logger.setLevel(log_level)

    # 仅在需要时创建文件 Handler
    if log_level <= logging.INFO:
        try:
            log_dir = get_log_dir() / "internal"
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(
                log_dir / "stream_renderer.log", encoding="utf-8"
            )
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)
        except Exception as e:
            textual_log.warning(
                "[StreamLogger] init file handler failed; continue", exc_info=e
            )

    logger.propagate = False
    return logger
