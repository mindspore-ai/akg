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

"""manifest.json 的读写与数据模型。

manifest 是调优阶段唯一的部署产物（可选附带 model.pkl）。它包含：
    - schema_version
    - axis_names         : shape 维度名（与装饰器 key 一致）
    - all_candidates     : 用户原始全部 candidate config（包含被 compile_gate 剔除的）
    - selector_config_ids: selector 实际可选的 config_id 子集（顺序对齐 selector 内部 index）
    - selector           : { kind, payload, runtime_deps }
    - compile_gate       : { rejections: [{config_id, reason}, ...] }
    - tune_meta          : 调优时的元数据（warmup/repeat/path_used/notes）

不包含：baseline 时延、加速比、CSV 报表——按设计不引入 baseline。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from akg_agents.op.dynamic_tune.config import Config

MANIFEST_FILE_NAME = "manifest.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CandidateRecord:
    config: Config
    status: str  # "kept" | "rejected"
    reject_reason: str = ""

    def __post_init__(self) -> None:
        normalized_status = str(self.status).strip().lower()
        if normalized_status not in {"kept", "rejected"}:
            raise ValueError(f"非法 status={self.status!r}")
        object.__setattr__(self, "status", normalized_status)
        if normalized_status == "rejected" and not self.reject_reason.strip():
            raise ValueError("rejected 必须带 reject_reason")
        if normalized_status == "kept" and self.reject_reason:
            raise ValueError("kept 不应带 reject_reason")


@dataclass(frozen=True)
class SelectorPayload:
    kind: str
    payload: Mapping[str, Any]
    runtime_deps: tuple[str, ...] = ("numpy",)
    config_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TuneMeta:
    path_used: str
    warmup: int
    repeat: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Manifest:
    axis_names: tuple[str, ...]
    candidates: tuple[CandidateRecord, ...]
    selector: SelectorPayload
    tune_meta: TuneMeta
    schema_version: int = SCHEMA_VERSION
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.axis_names:
            raise ValueError("axis_names 不能为空")
        if not self.candidates:
            raise ValueError("candidates 不能为空")
        kept_ids = [
            cand.config.config_id for cand in self.candidates if cand.status == "kept"
        ]
        if not kept_ids:
            raise ValueError("manifest 至少需要一个 kept candidate")
        for selector_id in self.selector.config_ids:
            if selector_id not in kept_ids:
                raise ValueError(
                    f"selector.config_id={selector_id!r} 不在 kept candidates 内"
                )

    def kept_configs(self) -> tuple[Config, ...]:
        return tuple(
            cand.config for cand in self.candidates if cand.status == "kept"
        )

    def find_config_by_id(self, config_id: str) -> Config:
        target = str(config_id)
        for cand in self.candidates:
            if cand.config.config_id == target:
                return cand.config
        raise KeyError(target)


def _candidate_to_payload(record: CandidateRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "config_id": record.config.config_id,
        "params": [
            {"name": name, "value": int(value)}
            for name, value in record.config.params
        ],
        "runtime_meta": record.config.runtime_meta(),
        "status": record.status,
    }
    if record.reject_reason:
        payload["reject_reason"] = record.reject_reason
    return payload


def _candidate_from_payload(payload: Mapping[str, Any]) -> CandidateRecord:
    runtime_meta = payload.get("runtime_meta") or {}
    config = Config(
        [(str(item["name"]), int(item["value"])) for item in payload.get("params", [])],
        num_warps=int(runtime_meta.get("num_warps", 4)),
        num_stages=int(runtime_meta.get("num_stages", 2)),
        num_ctas=int(runtime_meta.get("num_ctas", 1)),
        maxnreg=(
            None
            if runtime_meta.get("maxnreg") in (None, "null")
            else int(runtime_meta["maxnreg"])
        ),
        config_id=str(payload["config_id"]),
    )
    return CandidateRecord(
        config=config,
        status=str(payload.get("status", "kept")),
        reject_reason=str(payload.get("reject_reason", "")),
    )


def manifest_to_dict(manifest: Manifest) -> dict[str, Any]:
    return {
        "schema_version": int(manifest.schema_version),
        "axis_names": list(manifest.axis_names),
        "candidates": [_candidate_to_payload(rec) for rec in manifest.candidates],
        "selector": {
            "kind": manifest.selector.kind,
            "payload": dict(manifest.selector.payload),
            "runtime_deps": list(manifest.selector.runtime_deps),
            "config_ids": list(manifest.selector.config_ids),
        },
        "tune_meta": {
            "path_used": manifest.tune_meta.path_used,
            "warmup": int(manifest.tune_meta.warmup),
            "repeat": int(manifest.tune_meta.repeat),
            "notes": list(manifest.tune_meta.notes),
        },
        "extras": dict(manifest.extras),
    }


def manifest_from_dict(payload: Mapping[str, Any]) -> Manifest:
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version={schema_version} 与当前实现={SCHEMA_VERSION} 不一致"
        )
    selector_payload = payload.get("selector") or {}
    tune_meta_payload = payload.get("tune_meta") or {}
    return Manifest(
        axis_names=tuple(str(name) for name in payload.get("axis_names", [])),
        candidates=tuple(
            _candidate_from_payload(item) for item in payload.get("candidates", [])
        ),
        selector=SelectorPayload(
            kind=str(selector_payload.get("kind", "")),
            payload=dict(selector_payload.get("payload") or {}),
            runtime_deps=tuple(
                str(dep) for dep in selector_payload.get("runtime_deps") or ("numpy",)
            ),
            config_ids=tuple(
                str(item) for item in selector_payload.get("config_ids") or ()
            ),
        ),
        tune_meta=TuneMeta(
            path_used=str(tune_meta_payload.get("path_used", "")),
            warmup=int(tune_meta_payload.get("warmup", 0)),
            repeat=int(tune_meta_payload.get("repeat", 0)),
            notes=tuple(str(note) for note in tune_meta_payload.get("notes") or ()),
        ),
        extras=dict(payload.get("extras") or {}),
    )


def dump_manifest(manifest: Manifest, cache_dir: str | Path) -> Path:
    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path / MANIFEST_FILE_NAME
    payload = manifest_to_dict(manifest)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def load_manifest(cache_dir: str | Path) -> Manifest:
    cache_path = Path(cache_dir).expanduser()
    manifest_path = cache_path / MANIFEST_FILE_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_from_dict(payload)


def manifest_exists(cache_dir: str | Path) -> bool:
    return (Path(cache_dir).expanduser() / MANIFEST_FILE_NAME).is_file()


def build_candidates(
    *,
    all_configs: Sequence[Config],
    rejected: Mapping[str, str],
) -> tuple[CandidateRecord, ...]:
    """根据 compile_gate 输出组装 candidates 列表。

    rejected: {config_id: reject_reason}
    """

    out: list[CandidateRecord] = []
    for config in all_configs:
        reason = rejected.get(config.config_id, "")
        if reason:
            out.append(
                CandidateRecord(config=config, status="rejected", reject_reason=reason)
            )
        else:
            out.append(CandidateRecord(config=config, status="kept"))
    return tuple(out)


__all__ = [
    "CandidateRecord",
    "MANIFEST_FILE_NAME",
    "Manifest",
    "SCHEMA_VERSION",
    "SelectorPayload",
    "TuneMeta",
    "build_candidates",
    "dump_manifest",
    "load_manifest",
    "manifest_exists",
    "manifest_from_dict",
    "manifest_to_dict",
]
