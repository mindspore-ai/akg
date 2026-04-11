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

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ReplayCacheScenario:
    name: str
    cache_filename: str
    session_hash: str
    shape: Tuple[int, int, int, int]
    sm_scale: float
    cache_file_path: Path


_CPU_ATTENTION_REPLAY_BASE = (
    {
        "name": "attention_small_baseline",
        "cache_filename": "llm_cache_cpu_attention_small_baseline.json",
        "session_hash": "cpu_attn_real_small_record_v1",
        "shape": (1, 8, 128, 64),
        "sm_scale": 0.5,
    },
    {
        "name": "attention_medium_longseq",
        "cache_filename": "llm_cache_cpu_attention_medium_longseq.json",
        "session_hash": "cpu_attn_real_medium_record_v1",
        "shape": (4, 16, 1024, 64),
        "sm_scale": 0.125,
    },
    {
        "name": "attention_edge_expire_guard",
        "cache_filename": "llm_cache_cpu_attention_edge_expire_guard.json",
        "session_hash": "cpu_attn_real_edge_record_v1",
        "shape": (1, 32, 2048, 128),
        "sm_scale": 0.08838834764831845,
    },
)


def get_akg_agents_repo_root() -> Path:
    """Resolve akg_agents repo root by searching parent dirs."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "python" / "akg_agents").is_dir():
            return parent
    raise RuntimeError(f"Cannot resolve akg_agents repo root from {current}")


def get_project_cache_dir() -> Path:
    return get_akg_agents_repo_root() / ".cache"


def discover_cpu_attention_replay_scenarios(
    cache_dir: Optional[Path] = None,
    require_all: bool = True,
) -> List[ReplayCacheScenario]:
    """Discover the three canonical CPU attention replay cache samples."""
    resolved_dir = (cache_dir or get_project_cache_dir()).expanduser().resolve()
    scenarios: List[ReplayCacheScenario] = []
    missing_files = []

    for item in _CPU_ATTENTION_REPLAY_BASE:
        file_path = resolved_dir / item["cache_filename"]
        if not file_path.exists():
            missing_files.append(str(file_path))
            continue
        scenarios.append(
            ReplayCacheScenario(
                name=item["name"],
                cache_filename=item["cache_filename"],
                session_hash=item["session_hash"],
                shape=item["shape"],
                sm_scale=item["sm_scale"],
                cache_file_path=file_path,
            )
        )

    if require_all and missing_files:
        raise FileNotFoundError(
            "Missing replay cache sample files: " + ", ".join(missing_files)
        )

    return scenarios
