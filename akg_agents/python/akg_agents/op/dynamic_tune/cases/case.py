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

from __future__ import annotations

import importlib.util
import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _CaseSpec:
    name: str
    case_dir: Path
    base_path: Path
    impl_path: Path
    sample_path: Path
    payload: dict[str, Any]

    @classmethod
    def from_case_dir(cls, case_dir: str | Path) -> _CaseSpec:
        resolved_case_dir = Path(case_dir).expanduser().resolve()
        if not resolved_case_dir.is_dir():
            raise FileNotFoundError(f"case_dir 不存在: {resolved_case_dir}")
        base_path = cls._require_case_file(resolved_case_dir, "base.py")
        impl_path = cls._require_case_file(resolved_case_dir, "impl.py")
        sample_path = cls._require_case_file(resolved_case_dir, "sample.json")
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        return cls(
            name=resolved_case_dir.name,
            case_dir=resolved_case_dir,
            base_path=base_path,
            impl_path=impl_path,
            sample_path=sample_path,
            payload=payload,
        )

    @staticmethod
    def _require_case_file(case_dir: Path, filename: str) -> Path:
        path = case_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"case_dir 缺少必需文件 `{filename}`: {case_dir}")
        return path

    @property
    def autotune_spec(self) -> dict[str, Any]:
        return dict(self.payload["autotune"])

    @property
    def op_name(self) -> str:
        return str(self.payload["op_name"])

    @property
    def shape_expr(self) -> str:
        return str(self.payload["shape_expr"])


@dataclass(frozen=True)
class _RuntimeBundle:
    case_spec: _CaseSpec
    base_module: types.ModuleType
    runtime_module: types.ModuleType

    @classmethod
    def load(cls, case_spec: _CaseSpec, impl_path: Path) -> _RuntimeBundle:
        return cls(
            case_spec=case_spec,
            base_module=cls._load_module(f"txa_base_{case_spec.name}", case_spec.base_path),
            runtime_module=cls._load_module(f"txa_impl_{case_spec.name}", impl_path),
        )

    @staticmethod
    def _load_module(name: str, path: Path) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载模块: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
