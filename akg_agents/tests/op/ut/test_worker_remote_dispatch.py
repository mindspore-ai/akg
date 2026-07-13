# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Remote worker stop is exact, bounded, and owns orphan-tree cleanup."""

from akg_agents.cli.service import remote_dispatch as rd


def test_remote_stop_command_is_exact_and_escalating():
    cmd = rd._build_remote_stop_cmd({
        "repo_path": "/checkout/akg",
        "env_script": "/env.sh",
    }, 9111)
    assert "lsof -tiTCP:9111 -sTCP:LISTEN" in cmd
    assert "live_worker_pid" in cmd
    assert 'pid="${listener_pid:-$state_pid}"' in cmd
    assert 'kill -TERM "$pid"' in cmd
    assert 'kill -KILL "$pid"' in cmd
    assert "AKG_WORKER_PROCESS_REGISTRY=/tmp/akg_worker_9111_process_groups.json" in cmd
    assert "/checkout/akg/akg_agents/python" in cmd
    assert "reap_orphaned_process_groups" in cmd
    assert "remove_worker_entry" in cmd
    assert "pkill" not in cmd


def test_dispatch_stop_tears_tunnel_then_runs_owned_cleanup(monkeypatch):
    events = []
    monkeypatch.setattr(
        rd, "tunnel_stop_silent",
        lambda port, alias: events.append(("tunnel", port, alias)))
    monkeypatch.setattr(
        rd, "_ssh_dispatch",
        lambda alias, cmd: events.append(("ssh", alias, cmd)) or 0)
    rc = rd.dispatch_stop("npu", {
        "ssh_alias": "npu-alias",
        "repo_path": "/checkout/akg",
        "env_script": "/env.sh",
    }, 9111)
    assert rc == 0
    assert events[0] == ("tunnel", 9111, "npu-alias")
    assert events[1][0:2] == ("ssh", "npu-alias")
    assert "reap_orphaned_process_groups" in events[1][2]
