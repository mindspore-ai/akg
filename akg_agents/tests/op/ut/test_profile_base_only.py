# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

import asyncio
import sys
import textwrap

import pytest

from akg_agents.op.verifier.profiler_utils import (
    run_profile_scripts_and_collect_results,
)


@pytest.mark.asyncio
async def test_profile_utils_allows_base_only_package(tmp_path):
    op_name = "toy"
    base_script = tmp_path / f"profile_{op_name}_base.py"
    base_script.write_text(
        textwrap.dedent(
            """
            import json

            with open("base_profile_result.json", "w", encoding="utf-8") as f:
                json.dump({
                    "avg_time_us": 12.5,
                    "per_case_us": [10.0, 15.0],
                    "method": "unit_timer",
                }, f)
            """
        ),
        encoding="utf-8",
    )

    async def run_script(name, _label):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, name, cwd=str(tmp_path))
        await proc.communicate()
        return proc.returncode == 0

    sections = await run_profile_scripts_and_collect_results(
        str(tmp_path), op_name, run_script)

    assert sections["base"] == {
        "avg_us": 12.5,
        "per_case_us": [10.0, 15.0],
        "method": "unit_timer",
    }
    assert sections["gen"] is None


@pytest.mark.asyncio
async def test_profile_utils_runs_base_then_generation(tmp_path):
    op_name = "toy"
    (tmp_path / f"profile_{op_name}_base.py").write_text("", encoding="utf-8")
    (tmp_path / f"profile_{op_name}_generation.py").write_text("", encoding="utf-8")
    calls = []

    async def fake_run(name, label):
        calls.append(label)
        output = (
            "base_profile_result.json"
            if label == "base_profile"
            else "generation_profile_result.json"
        )
        (tmp_path / output).write_text(
            '{"avg_time_us": 1.0, "per_case_us": [1.0], "method": "unit"}',
            encoding="utf-8",
        )
        return True

    sections = await run_profile_scripts_and_collect_results(
        str(tmp_path), op_name, fake_run)

    assert calls == ["base_profile", "generation_profile"]
    assert sections["base"]["avg_us"] == 1.0
    assert sections["gen"]["avg_us"] == 1.0
