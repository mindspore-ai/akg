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

"""Regression for `akg_cli worker` CLI surface.

Pins the contract of the typer subcommand: all expected flags parse,
the start/stop/status actions are mutually exclusive, and the local
--status invocation still validates (never reaches the spawn).

Run: pytest -x tests/op/ut/test_worker_cli.py
"""
import subprocess
import sys

import pytest


def _akg_cli_worker(*args, timeout=15):
    """Invoke `python -m akg_agents.cli.cli worker <args>` subprocess.
    Returns (rc, stdout, stderr)."""
    cmd = [sys.executable, "-m", "akg_agents.cli.cli", "worker", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


@pytest.mark.level0
def test_worker_help_has_all_flags():
    """--help renders and includes every documented flag (regression
    against accidental flag removal in misc.py)."""
    rc, out, _ = _akg_cli_worker("--help")
    assert rc == 0
    combined = out
    for flag in ("--start", "--stop", "--status", "--backend", "--arch",
                 "--devices", "--port", "--remote-host"):
        assert flag in combined, f"--help missing flag: {flag}"


@pytest.mark.level0
def test_worker_no_action_rejected():
    """Calling `worker` with zero of start/stop/status must error out
    at the mutex check, not silently spawn or hang."""
    rc, _, _ = _akg_cli_worker()
    assert rc != 0


@pytest.mark.level0
def test_worker_two_actions_rejected():
    """Combining --start with --stop violates the mutex; reject."""
    rc, _, _ = _akg_cli_worker("--start", "--stop")
    assert rc != 0


@pytest.mark.level0
def test_status_local_passes_arg_validation():
    """`worker --status --port N` (no remote-host) must validate cleanly
    and reach the status probe — the rc is whatever the local probe
    returns (likely 1 because no daemon is running on the test port),
    but it must NOT be a parse error."""
    rc, _, _ = _akg_cli_worker("--status", "--port", "65530")
    # rc=1 = unreachable (expected on a free port). rc=0 would mean some
    # daemon is on the port. rc=2 would mean arg validation failed
    # (regression). Accept 0 or 1, reject 2.
    assert rc in (0, 1), f"unexpected rc={rc} (regression on local --status path)"


@pytest.mark.level0
def test_remote_host_missing_config_rejected():
    """--remote-host alias with no corresponding entry in ./config.yaml
    must reject (typer.Exit code=2) before any SSH attempt."""
    rc, _, _ = _akg_cli_worker(
        "--status", "--remote-host", "nonexistent_alias_xyz_for_regression_test",
        "--port", "9101",
    )
    assert rc != 0
