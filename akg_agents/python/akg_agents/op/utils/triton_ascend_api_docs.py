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

"""Triton Ascend API docs aggregation helpers."""

from __future__ import annotations

import importlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from akg_agents import get_project_root


MANIFEST_FILENAME = "api_manifest.json"
OFFLINE_API_FILENAME = "api.md"
DEFAULT_API_DIR_REL = "op/resources/docs/triton_ascend_docs/api"

logger = logging.getLogger(__name__)


def _default_api_dir() -> Path:
    return Path(get_project_root()) / DEFAULT_API_DIR_REL


def _default_offline_api_path() -> Path:
    return _default_api_dir() / OFFLINE_API_FILENAME


def _normalize_api_title(title: str) -> str:
    api_name = title.strip()
    if api_name.startswith("@"):
        api_name = api_name[1:]
    api_name = api_name.split("(", 1)[0].strip()

    if api_name.startswith("tl."):
        return f"triton.language.{api_name[3:]}"
    if api_name.startswith("bl."):
        return f"triton.extension.buffer.language.{api_name[3:]}"
    return api_name


def _api_exists(title: str) -> bool:
    full_name = _normalize_api_title(title)
    module_name, _, attr_name = full_name.rpartition(".")
    if not module_name or not attr_name:
        return False

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False
    return hasattr(module, attr_name)


def _load_manifest(api_dir: Path, manifest_name: str) -> dict:
    return json.loads((api_dir / manifest_name).read_text(encoding="utf-8"))


def _entry_title(api_dir: Path, entry: str) -> str:
    content = (api_dir / entry).read_text(encoding="utf-8")
    match = re.search(r"(?m)^### (.+)$", content)
    if not match:
        raise ValueError(f"cannot infer API title from {entry}")
    return match.group(1).strip()


def _aggregate(api_dir: Path, manifest_name: str) -> str:
    manifest = _load_manifest(api_dir, manifest_name)

    parts = [manifest["preamble"]]
    missing_entries: list[dict[str, str]] = []
    available_entry_count = 0

    for section in manifest["sections"]:
        section_parts: list[str] = []
        for entry in section["entries"]:
            title = _entry_title(api_dir, entry)
            if not _api_exists(title):
                missing_entries.append({"title": title})
                continue
            section_parts.append((api_dir / entry).read_text(encoding="utf-8"))
            available_entry_count += 1
        if section_parts:
            parts.append(section["prefix"])
            parts.extend(section_parts)

    if available_entry_count == 0:
        raise RuntimeError("no available Triton Ascend APIs found in current environment")

    if missing_entries:
        if parts and not parts[-1].endswith("\n"):
            parts.append("\n")
        parts.append("\n## 当前版本不存在的 API\n\n")
        for entry in missing_entries:
            parts.append(f"### {entry['title']}\n")
            parts.append("这个api 在当前版本不存在.\n\n")

    return "".join(parts)


@lru_cache(maxsize=1)
def get_aggregated_triton_ascend_api_docs() -> str:
    """获取当前 SDK 环境下过滤后的 Triton Ascend API 文档。"""
    return _aggregate(_default_api_dir(), MANIFEST_FILENAME)


@lru_cache(maxsize=1)
def get_offline_triton_ascend_api_docs() -> str:
    """读取仓库中固化的 Triton Ascend API 离线快照。"""
    return _default_offline_api_path().read_text(encoding="utf-8")


def load_triton_ascend_api_docs() -> str:
    """最佳努力加载 Triton Ascend API 文档。

    优先级：
    1. 直接尝试获取当前环境的真实聚合结果
    2. 若当前环境拿不到任何 Triton Ascend API，则回退到离线 `api/api.md`

    注意：
    - 离线 `api.md` 被视为一次真实 SDK 环境下跑出的快照产物
    - 运行时不会再根据 manifest 重新推导离线内容
    """
    try:
        return get_aggregated_triton_ascend_api_docs()
    except Exception as e:
        logger.warning("load local Triton Ascend API docs failed, fallback to offline: %s", e)

    return get_offline_triton_ascend_api_docs()


async def resolve_triton_ascend_api_docs(
    backend: str = "",
    arch: str = "",
    worker_manager: Any = None,
) -> str:
    """统一执行 Triton Ascend API 文档的 local -> remote -> offline 加载。"""
    try:
        return get_aggregated_triton_ascend_api_docs()
    except Exception as e:
        logger.warning("load local Triton Ascend API docs failed: %s", e)

    if backend:
        try:
            if worker_manager is None:
                from akg_agents.core.worker.manager import get_worker_manager
                worker_manager = get_worker_manager()

            from akg_agents.core.worker.remote_worker import RemoteWorker

            workers = await worker_manager.list_matching(backend=backend, arch=arch or None)
            workers = sorted(
                workers,
                key=lambda worker: 0 if isinstance(worker, RemoteWorker) else 1,
            )
            for worker in workers:
                try:
                    content = await worker.get_doc("triton_ascend_api")
                    if content:
                        logger.info(
                            "Loaded Triton Ascend API docs from worker %s",
                            type(worker).__name__,
                        )
                        return content
                except Exception as e:
                    logger.warning(
                        "Failed to fetch Triton Ascend API docs from worker %s: %s",
                        type(worker).__name__,
                        e,
                    )
        except Exception as e:
            logger.warning("load remote Triton Ascend API docs failed: %s", e)

    return get_offline_triton_ascend_api_docs()


def update_offline_triton_ascend_api_docs(output_path: Path | None = None) -> Path:
    """使用当前 SDK 环境更新仓库内置的离线 API 快照。"""
    target_path = output_path or _default_offline_api_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(get_aggregated_triton_ascend_api_docs(), encoding="utf-8")
    get_offline_triton_ascend_api_docs.cache_clear()
    logger.info("Updated offline Triton Ascend API docs: %s", target_path)
    return target_path


def main() -> int:
    updated_path = update_offline_triton_ascend_api_docs()
    print(f"Updated offline Triton Ascend API docs: {updated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
