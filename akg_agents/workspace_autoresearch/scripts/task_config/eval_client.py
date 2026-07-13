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

"""Eval entry point — WA-side shim into ``utils.akg_eval``.

`run_eval(task_dir, config, device_id=None, worker_urls=None) -> EvalResult`
is the contract baseline / pipeline call. CA's standalone implementation
ships a self-contained `eval_request` / `eval_assemble` / `package_builder`
stack that drives `utils.eval_runner.local_eval` locally or POSTs a tarball
to a CA worker. WA reuses ``akg_agents.op.verifier.KernelVerifier`` +
``akg_agents.core.worker.manager`` via ``utils.akg_eval.eval_kernel`` instead,
so this module is a one-call adapter that maps the bridge's dict result onto
the ``EvalResult`` shape downstream consumers (baseline / pipeline /
keep_or_discard / dashboard) already understand.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from .loader import TaskConfig
from .metric_policy import EvalOutcome, EvalResult


_OUTCOME_VALUES = {e.value for e in EvalOutcome}


def _resolve_worker_url(worker_urls: Optional[list],
                        config: TaskConfig) -> Optional[str]:
    """Pick the first non-empty URL. CA's eval_client probes /api/v1/status
    on every URL and ranks by free device slots — that's an HTTP worker
    protocol the AKG worker doesn't expose. Multi-URL fallback is left
    out on purpose; multi-worker scheduling lives in akg's
    `core.worker.manager` once the URL is registered there."""
    candidates = worker_urls or getattr(config, "worker_urls", None) or []
    for u in candidates:
        if u and str(u).strip():
            return str(u).strip()
    return None


def _resolve_device_arg(device_id: Optional[int], config: TaskConfig,
                        worker_url: Optional[str]):
    if device_id is not None:
        return int(device_id)
    devices = getattr(config, "devices", None)
    if devices:
        parsed = [int(d) for d in devices]
        return parsed if worker_url else parsed[0]
    if worker_url:
        return None
    print(
        "[akg_eval] WARNING: no device specified (no device_id arg, "
        "no `devices` field in task.yaml). Defaulting to local device 0.",
        file=sys.stderr,
    )
    return 0


def run_eval(task_dir: str, config: TaskConfig,
             device_id: Optional[int] = None,
             worker_urls: Optional[list] = None,
             current_step: int = 0) -> EvalResult:
    """Route the eval through ``utils.akg_eval.eval_kernel`` → ``EvalResult``.
    ``worker_urls`` empty → local worker on ``device_id``; else first URL is a
    RemoteWorker. ``current_step`` (caller-owned: pipeline=round_num, seed=0)
    numbers the verify dir so each round's artifacts are kept, not overwritten."""
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from utils.akg_eval import eval_kernel  # noqa: E402

    worker_url = _resolve_worker_url(worker_urls, config)
    dev_id = _resolve_device_arg(device_id, config, worker_url)

    try:
        raw = eval_kernel(task_dir, config,
                          device_id=dev_id,
                          worker_url=worker_url,
                          current_step=current_step)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return EvalResult(
            outcome=EvalOutcome.INFRA_FAIL,
            error=f"akg_eval.eval_kernel raised {type(e).__name__}: {e}",
            error_source="infra",
        )

    outcome_str = raw.get("outcome") or "infra_fail"
    if outcome_str not in _OUTCOME_VALUES:
        outcome_str = "infra_fail"
    return EvalResult(
        outcome=EvalOutcome(outcome_str),
        metrics=raw.get("metrics") or {},
        error=raw.get("error"),
        raw_output=str(raw.get("raw_output_tail") or ""),
        error_source=raw.get("error_source"),
        fail_report=raw.get("fail_report"),
        failure_signals=raw.get("failure_signals") or {},
    )
