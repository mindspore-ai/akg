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

"""Bridge: sync workspace eval entry → async ``akg_agents`` verifier + worker.

``eval_kernel(task_dir, config, device_id, worker_url)`` reads the current
``kernel.py`` + reference, registers a LocalWorker (or RemoteWorker), runs
``KernelVerifier.run()`` then ``run_profile()``, and returns a dict in the
schema ``phase_machine`` / ``workflow`` already consume (``outcome`` /
``correctness`` / ``metrics{...}`` / ``error`` / ``error_source``).
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring
from __future__ import annotations

import asyncio
import logging
import math
import os
import traceback
from typing import Any, Dict, Optional, Tuple

from akg_agents.op.verifier.adapters.factory import get_dsl_adapter

from .settings import (
    eval_warmup, eval_repeats,
    target_backend, target_framework, target_dsl,
)

logger = logging.getLogger(__name__)


def eval_kernel(task_dir: str, config, device_id: Any = 0,
                worker_url: Optional[str] = None,
                current_step: int = 0,
                verify_only: bool = False) -> Dict[str, Any]:
    """Run akg verify + profile against current kernel.py.

    `worker_url` non-None → RemoteWorker; otherwise LocalWorker on `device_id`.
    `current_step` distinguishes akg log subdirs across rounds.
    """
    return asyncio.run(_eval_async(task_dir, config, device_id, worker_url,
                                   current_step, verify_only))


def _normalize_device_ids(device_id: Any) -> list[int]:
    if device_id is None:
        return []
    if isinstance(device_id, str):
        text = device_id.strip()
        if not text:
            return []
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    if isinstance(device_id, (list, tuple, set)):
        return [int(x) for x in device_id]
    return [int(device_id)]


def _load_seed_files(task_dir: str, ref_file: str, kernel_file: str):
    """Return (kernel_code, ref_code) or (None, infra_fail_dict).

    ``kernel_file`` is the primary editable filename (TaskConfig.
    editable_files[0]); DSL-driven so e.g. ascendc / catlass / triton all
    work the same way without ``kernel.py`` literal hardcoding here."""
    kernel_path = os.path.join(task_dir, kernel_file)
    ref_path = os.path.join(task_dir, ref_file)
    if not os.path.exists(kernel_path):
        return None, _infra_fail(
            f"primary editable {kernel_file} not found in {task_dir}")
    if not os.path.exists(ref_path):
        return None, _infra_fail(
            f"reference file {ref_file} not found in {task_dir}")
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_code = f.read()
    return (kernel_code, ref_code), None


def _remote_host_for_url(worker_url: str) -> Optional[Tuple[str, str]]:
    """Find the ``remote_worker.hosts.<alias>`` entry whose tunneled port
    matches ``worker_url``. Used to wire an auto-reconnect callback into
    ``register_remote_worker`` — only valid when worker_url points at a
    tunneled local port that akg_cli set up.

    Returns ``(alias, cfg_path)`` so callers pass the same yaml back to
    ``load_remote_host_config`` — eval might be running outside the
    workspace cwd, can't rely on the cwd-search fallback in akg_cli."""
    import urllib.parse
    try:
        port = urllib.parse.urlparse(
            worker_url if "://" in worker_url else f"http://{worker_url}"
        ).port
    except Exception:
        return None
    if port is None:
        return None
    # workspace_autoresearch/config.yaml is the canonical place; akg_cli
    # uses the same yaml via --remote-config (default cwd/config.yaml).
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "..", "config.yaml")
    cfg_path = os.path.abspath(cfg_path)
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None
    # Match by yaml `worker.port` (single configured tunnel port).
    if int((data.get("worker") or {}).get("port", -1)) != port:
        return None
    hosts = (data.get("remote_worker") or {}).get("hosts") or {}
    # Pick the first host; multi-host setups would need a per-host port
    # mapping, but the current config model is single-tunnel-per-yaml.
    alias = next(iter(hosts.keys()), None)
    if alias is None:
        return None
    return (alias, cfg_path)


def _make_reconnect_callback(worker_url: str):
    """Build a hook that reconnects the ssh -L tunnel matching this
    worker_url. Returns None when the url isn't a tunneled local port
    (e.g. direct remote host, or url not declared in config.yaml)."""
    resolved = _remote_host_for_url(worker_url)
    if not resolved:
        return None
    alias, cfg_path = resolved
    import urllib.parse
    def _reconnect():
        from akg_agents.cli.service import remote_dispatch
        # Pass the absolute cfg_path resolved above —  if eval runs outside
        # workspace cwd, the load_remote_host_config(alias, None) fallback
        # to cwd/config.yaml silently misses, and reconnect goes no-op.
        host_cfg = remote_dispatch.load_remote_host_config(alias, cfg_path)
        if host_cfg is None:
            return
        port = int(urllib.parse.urlparse(
            worker_url if "://" in worker_url else f"http://{worker_url}"
        ).port)
        remote_dispatch.dispatch_reconnect_tunnel(alias, host_cfg, port)
    return _reconnect


async def _acquire_worker(backend: str, arch: str, device_ids: list[int],
                          worker_url: Optional[str]):
    """Register + select a worker. Returns (worker, manager) on success,
    or (None, infra_fail_dict) on failure."""
    from akg_agents.core.worker.manager import (
        register_local_worker, register_remote_worker, get_worker_manager,
    )
    try:
        if worker_url:
            # AKG's register_remote_worker uses urllib's URL parser, which
            # needs an explicit scheme. CA / WA CLIs accept bare host:port
            # (e.g. `--worker-url 127.0.0.1:9111`); normalise here so the
            # call site stays terse.
            url = worker_url if worker_url.startswith(("http://", "https://")) \
                else f"http://{worker_url}"
            await register_remote_worker(
                backend=backend, arch=arch, worker_url=url,
                expected_device_ids=device_ids or None,
                on_transient_failure=_make_reconnect_callback(url),
            )
        else:
            await register_local_worker(device_ids=device_ids or [0],
                                        backend=backend, arch=arch)
    except Exception as e:
        return None, _infra_fail(f"worker registration failed: {e}")
    wm = get_worker_manager()
    worker = await wm.select(backend=backend, arch=arch)
    if worker is None:
        return None, _infra_fail(
            f"no worker available for backend={backend} arch={arch}")
    return (worker, wm), None


def _resolve_eval_timeout_s(task_dir: str, config) -> int:
    """Compute the wall-clock subprocess timeout for one verify / profile
    invocation as ``eval_timeout * num_cases``.

    ``TaskConfig.eval_timeout`` is a PER-SHAPE budget (see TaskConfig
    docstring). For a multi-shape op the verify subprocess iterates
    every input group, so the wall-clock cap is N × per-shape budget —
    otherwise a 30-case op with a 600s per-shape budget hits the worker
    protocol default (300s) long before it can finish.

    ``num_cases`` resolution order: explicit ``config.num_cases``
    (scaffold-probed at task creation, written into task.yaml) > runtime
    probe via ``utils.input_groups.num_cases`` (covers the case where
    scaffold's probe failed because the dev host lacks torch) > 1
    (single-shape fallback)."""
    num_cases = int(getattr(config, "num_cases", 0) or 0)
    if num_cases <= 0:
        try:
            from .input_groups import num_cases as probe_num_cases
            ref_path = os.path.join(task_dir, config.ref_file)
            if os.path.isfile(ref_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "reference", ref_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    num_cases = int(probe_num_cases(mod))
        except Exception:
            num_cases = 0
    if num_cases <= 0:
        num_cases = 1
    return int(config.eval_timeout) * num_cases


def _build_verifier(task_dir: str, config, ref_code: str, backend: str,
                    arch: str, framework: str, dsl: str, worker):
    from akg_agents.op.verifier.kernel_verifier import KernelVerifier
    log_dir = os.path.join(task_dir, ".ar_state", "akg_verify")
    os.makedirs(log_dir, exist_ok=True)
    # warmup/repeats are global eval knobs in config.yaml — not on
    # TaskConfig — mirroring CA's `eval_runner` -> `settings.
    # eval_warmup/eval_repeats` pattern. Both runs (baseline ref +
    # per-round kernel) read the same values so timing stays comparable.
    config_dict: Dict[str, Any] = {
        "log_dir": log_dir,
        "verify_timeout": _resolve_eval_timeout_s(task_dir, config),
        "warmup_times": eval_warmup(),
        "run_times": eval_repeats(),
        "task_dir": task_dir,
        "framework_filename": os.path.basename(str(config.ref_file)),
        "framework_module_name": os.path.splitext(
            os.path.basename(str(config.ref_file))
        )[0],
    }
    # Forward all per-DSL knobs from TaskConfig.dsl_config. adapter's
    # prepare_config() reads them at run/run_profile time. New DSL
    # adding a new key is zero code change here.
    config_dict.update(getattr(config, "dsl_config", None) or {})
    # Sidecar files are materialized exactly as declared in task.yaml.
    # AR tasks normally use reference.py/reference.json; the verifier gets
    # the same framework filename/module above, so no sidecar rename is
    # needed.
    data_files = getattr(config, "data_files", None) or []
    framework_aux_files: Dict[str, Any] = {}
    for rel in data_files:
        if not isinstance(rel, str) or not rel:
            continue
        src = os.path.join(task_dir, rel)
        if not os.path.isfile(src):
            continue
        with open(src, "rb") as f:
            framework_aux_files[rel] = f.read()
    if framework_aux_files:
        config_dict["framework_aux_files"] = framework_aux_files
    return KernelVerifier(
        op_name=config.name,
        task_id=config.name,
        framework_code=ref_code,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config_dict,
        worker=worker,
    )


def _verify_fail_payload(verify_log: Optional[str],
                         sidecar: Optional[dict] = None) -> Dict[str, Any]:
    """Build the dict downstream eval_client consumes. When the verify
    script wrote a per-case sidecar (kernel_verify_template_refactored.j2
    -> verify_result.json -> KernelVerifier.last_verify_sidecar), use its
    `error_source` so a ref-side crash flips to "ref" instead of being
    blamed on the kernel. The per-case list is forwarded under `per_case`
    for failure_extractor / DIAGNOSE consumption."""
    payload: Dict[str, Any] = {
        "outcome": "kernel_fail",
        "correctness": False,
        "metrics": {},
        "error": (verify_log or "verify failed")[-2000:],
        "error_source": "kernel",
        "raw_output_tail": (verify_log or "")[-4000:],
    }
    if isinstance(sidecar, dict):
        es = sidecar.get("error_source")
        if es in ("ref", "kernel"):
            payload["error_source"] = es
        if sidecar.get("per_case"):
            payload["per_case"] = sidecar["per_case"]
        if sidecar.get("failed_indices") is not None:
            payload["failed_indices"] = sidecar["failed_indices"]
    try:
        from .failure_extractor import extract_failure_signals
        diag = extract_failure_signals(verify_log or "")
        if not diag.is_empty:
            payload["failure_signals"] = diag.to_dict()
    except Exception:
        pass
    return payload


def _profile_fail_payload(exc: Exception) -> Dict[str, Any]:
    tb_str = traceback.format_exc()
    payload = {
        "outcome": "kernel_fail",
        "correctness": True,
        "metrics": {},
        "error": f"profile raised: {exc}",
        "error_source": "kernel",
        "raw_output_tail": tb_str[-4000:],
    }
    try:
        from .failure_extractor import extract_failure_signals
        diag = extract_failure_signals(tb_str)
        if not diag.is_empty:
            payload["failure_signals"] = diag.to_dict()
    except Exception:
        pass
    return payload


def _iteration_verify_dir(task_dir: str, op_name: str, current_step: int) -> str:
    """Mirror KernelVerifier._create_verify_dir naming; needed because the
    verifier doesn't expose ``last_verify_dir`` and we want the adapter's
    post_iteration_cleanup hook to run after each round."""
    return os.path.join(
        task_dir, ".ar_state", "akg_verify", op_name,
        f"Iteration{op_name}_Step{current_step:02d}_verify",
    )


def _make_verify_ok_payload(sidecar: Optional[dict]) -> Dict[str, Any]:
    """Return a success payload for verify-only callers."""
    sidecar = sidecar if isinstance(sidecar, dict) else {}
    metrics = {
        "num_cases": int(sidecar.get("num_cases") or 1),
        "max_abs_diff": sidecar.get("worst_max_abs_diff"),
    }
    payload: Dict[str, Any] = {
        "outcome": "ok",
        "correctness": True,
        "metrics": metrics,
        "error": None,
        "error_source": None,
    }
    if sidecar.get("per_case"):
        payload["per_case"] = sidecar["per_case"]
    return payload


def _make_ok_payload(profile_result: dict, op_name: str) -> Dict[str, Any]:
    """Pack profile_result into the workspace's eval-result schema. Returns
    an ``infra_fail`` payload when gen_time is missing / non-positive, or
    when the canonical per-shape arrays are absent (chain regression).

    ``profile_result`` is the canonical per-shape dict surfaced by
    ``KernelVerifier.run_profile``:

        {
          "gen_time": float,                    # aggregate (mean of per_shape)
          "base_time": float | None,            # cross-backend → None
          "speedup": float | None,
          "per_shape_gen_us": list[float],      # always populated; len == num_cases
          "per_shape_base_us": list[float],     # [] when base skipped
          "case_descs": list[str],              # from verify sidecar
          "gen_method": str | None,
          "base_method": str | None,
          ...roofline fields...
        }

    No fallback for missing ``per_shape_gen_us`` — the profile template +
    profiler_utils + local_worker + KernelVerifier chain guarantees it.
    Empty here means a chain regression and we surface that explicitly.
    Missing ``case_descs`` is softer (verify sidecar may genuinely be
    skipped on some paths) — we synthesize generic ``caseN`` labels."""
    gen_us = _float(profile_result.get("gen_time"))
    if gen_us is None or gen_us <= 0:
        return _infra_fail(
            "profile returned invalid "
            f"gen_time={profile_result.get('gen_time')!r}"
        )

    per_shape_gen = list(profile_result.get("per_shape_gen_us") or [])
    if not per_shape_gen:
        return _infra_fail(
            "profile_result missing per_shape_gen_us "
            f"(gen_time={gen_us:.2f}us is set, but per-case breakdown is "
            "empty — chain regression in profiler_utils / LocalWorker / "
            "KernelVerifier)"
        )
    num_cases = len(per_shape_gen)

    base_us = _float(profile_result.get("base_time"))
    per_shape_base = list(profile_result.get("per_shape_base_us") or [])
    if base_us is not None and len(per_shape_base) != num_cases:
        # Schema invariant: when base is present, the per-shape arrays
        # must align across gen / base / descs. A mismatch means base ran
        # over a different set of cases than gen — caller can't compute
        # per-shape speedup. Soften to no-base rather than emit garbage.
        logger.warning(
            f"[{op_name}] dropping per_shape_base_us "
            f"(len {len(per_shape_base)} != per_shape_gen_us len {num_cases})"
        )
        per_shape_base = []
        base_us = None

    speedup = _float(profile_result.get("speedup"))
    if speedup is None and base_us and base_us > 0:
        speedup = base_us / gen_us

    descs = list(profile_result.get("case_descs") or [])
    if len(descs) != num_cases:
        if descs:
            logger.warning(
                f"[{op_name}] case_descs len {len(descs)} != per_shape_gen_us "
                f"len {num_cases} — regenerating with generic caseN labels"
            )
        descs = [f"case{i}" for i in range(num_cases)]

    per_shape_speedup: list = []
    if per_shape_base:
        for g, b in zip(per_shape_gen, per_shape_base):
            per_shape_speedup.append(b / g if g and g > 0 else 0.0)

    gen_method = profile_result.get("gen_method") or "akg_kernel_verifier"
    base_method = profile_result.get("base_method") or (
        "akg_kernel_verifier" if per_shape_base else None)

    return {
        "outcome": "ok",
        "correctness": True,
        "metrics": {
            "latency_us": gen_us,
            "ref_latency_us": base_us,
            "speedup_vs_ref": speedup,
            "num_cases": num_cases,
            "per_shape_gen_us": per_shape_gen,
            "per_shape_gen_method": [gen_method] * num_cases,
            "timing_method_gen": gen_method,
            "per_shape_base_us": per_shape_base,
            "per_shape_base_method":
                [base_method] * num_cases if per_shape_base else [],
            "timing_method_base": base_method,
            "per_shape_speedup": per_shape_speedup,
            "speedup_aggregation": "geomean",
            "per_shape_descs": descs,
        },
        "error": None,
        "error_source": None,
    }


async def _eval_async(task_dir: str, config, device_id: Any,
                      worker_url: Optional[str],
                      current_step: int,
                      verify_only: bool) -> Dict[str, Any]:
    # Entry file name = adapter's structural convention (kernel.py for
    # most DSLs; pure C++ DSLs override). Routed through the adapter
    # rather than indexing config.editable_files so the lookup stays
    # consistent with what scaffold wrote.
    kernel_file = get_dsl_adapter(target_dsl()).entry_filename_template.format(
        op_name=config.name)
    seed, err = _load_seed_files(task_dir, config.ref_file, kernel_file)
    if err is not None:
        return err
    kernel_code, ref_code = seed

    # Workspace target triple is pinned per repo in config.yaml's
    # `defaults.{backend,framework,dsl}` (see scripts/task_config/loader.py
    # for why these don't live on TaskConfig). Only `arch` is carried
    # per-task — it varies per machine, scaffold autodetects via the
    # backend-appropriate probe (npu-smi / nvidia-smi / platform.machine).
    backend = target_backend()
    arch = config.arch
    framework = target_framework()
    dsl = target_dsl()
    device_ids = _normalize_device_ids(device_id)
    if not arch and worker_url:
        # Remote eval: scaffold deliberately leaves arch=None because the
        # orchestrator may not have a local device to probe. Fetch from
        # the worker daemon's /api/v1/status (already up when we reach
        # here, by construction of the worker --start → scaffold ordering).
        arch = _arch_from_worker(worker_url)
    if not arch:
        return _infra_fail(
            "task.yaml missing arch and could not derive it from worker "
            f"/status (worker_url={worker_url!r})")

    acq, err = await _acquire_worker(backend, arch, device_ids, worker_url)
    if err is not None:
        return err
    worker, wm = acq

    verifier = _build_verifier(task_dir, config, ref_code, backend, arch,
                               framework, dsl, worker)
    # task_info carries the catlass-specific paths so the adapter's
    # prepare_config (called inside verifier.run / run_profile) can pick
    # them up without going through config; falls back to config for
    # DSLs that don't use task_info.
    task_info: Dict[str, Any] = {
        "coder_code": kernel_code,
        "task_dir": task_dir,
        **(getattr(config, "dsl_config", None) or {}),
    }
    try:
        verify_ok, verify_log = await verifier.run(
            task_info, current_step=current_step)
        if not verify_ok:
            return _verify_fail_payload(
                verify_log, getattr(verifier, "last_verify_sidecar", None))
        if verify_only:
            return _make_verify_ok_payload(
                getattr(verifier, "last_verify_sidecar", None))
        try:
            # Profile HTTP read timeout = ``eval_timeout × num_cases``
            # (single owner: :func:`_resolve_eval_timeout_s`). Mirrors
            # verify_timeout — both layers share the same per-shape budget
            # so multi-shape ops don't trip the worker's protocol default.
            profile_result = await verifier.run_profile(
                task_info, current_step=current_step,
                profile_settings={
                    "timeout": _resolve_eval_timeout_s(task_dir, config),
                },
            )
        except Exception as e:
            return _profile_fail_payload(e)
        return _make_ok_payload(profile_result, config.name)
    finally:
        try:
            verifier.dsl_adapter.post_iteration_cleanup(
                _iteration_verify_dir(task_dir, config.name, current_step))
        except Exception:
            pass
        try:
            await wm.release(worker)
        except Exception:
            pass


def _arch_from_worker(worker_url: str) -> Optional[str]:
    """Curl http://<worker_url>/api/v1/status and return the arch field.
    Returns None on any failure; the caller surfaces the infra_fail."""
    import json as _json
    import urllib.request
    url = worker_url if worker_url.startswith(("http://", "https://")) \
        else f"http://{worker_url}"
    try:
        with urllib.request.urlopen(f"{url}/api/v1/status", timeout=5) as r:
            return _json.loads(r.read().decode("utf-8")).get("arch")
    except Exception:
        return None


def _float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _infra_fail(msg: str) -> Dict[str, Any]:
    return {
        "outcome": "infra_fail",
        "correctness": False,
        "metrics": {},
        "error": msg,
        "error_source": None,
    }
