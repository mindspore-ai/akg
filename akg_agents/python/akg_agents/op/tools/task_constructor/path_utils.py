# Copyright 2026 Huawei Technologies Co., Ltd
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

"""共享路径解析工具"""

import os
from pathlib import Path


def normalize_path_str(file_path: str) -> str:
    """标准化路径字符串，确保跨平台兼容"""
    return file_path.replace('\\', '/').replace('/', os.sep)


def resolve_path(file_path: str, workspace_dir: Path = None, output_dir: Path = None) -> Path:
    """解析文件路径，支持 workspace/ 和 output/ 短路径，跨平台兼容"""
    normalized = normalize_path_str(file_path)
    p = Path(normalized).expanduser()
    path_fwd = file_path.replace('\\', '/')

    if not p.is_absolute():
        if workspace_dir:
            if path_fwd.startswith("workspace/"):
                ws_path = workspace_dir / path_fwd[len("workspace/"):]
                if ws_path.exists():
                    return ws_path.resolve()
            ws_path = workspace_dir / p
            if ws_path.exists():
                return ws_path.resolve()
        if output_dir:
            if path_fwd.startswith("output/"):
                out_path = output_dir / path_fwd[len("output/"):]
                if out_path.exists():
                    return out_path.resolve()
            out_path = output_dir / p
            if out_path.exists():
                return out_path.resolve()
    return p.resolve()
