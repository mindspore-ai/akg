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

from pathlib import Path

from akg_agents.core_v2.llm.cache import (
    discover_cpu_attention_replay_scenarios,
    get_project_cache_dir,
)
from akg_agents.core_v2.llm.cache.cache_config import load_cache_config


def test_discover_cpu_attention_replay_scenarios():
    scenarios = discover_cpu_attention_replay_scenarios(cache_dir=get_project_cache_dir())
    assert len(scenarios) == 3

    names = [s.name for s in scenarios]
    assert names == [
        "attention_small_baseline",
        "attention_medium_longseq",
        "attention_edge_expire_guard",
    ]
    assert all(s.cache_file_path.exists() for s in scenarios)


def test_load_cache_config_supports_env_cache_file_path(monkeypatch, tmp_path):
    custom_cache_file = tmp_path / "replay_sample.json"
    monkeypatch.setenv("AKG_AGENTS_CACHE_FILE_PATH", str(custom_cache_file))

    cache_config = load_cache_config()
    assert Path(cache_config["cache_file_path"]).as_posix() == custom_cache_file.as_posix()
