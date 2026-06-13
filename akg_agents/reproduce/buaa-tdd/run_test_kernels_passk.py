import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


default_workflow = "mathir_coder_workflow"
# default_workflow = "mathir_multi_kernel_gen_workflow"


triton_cuda_path = f"{PYTHON_ROOT}/akg_agents/op/config/triton_cuda_mathir_config.yaml"
triton_ascend_path = f"{PYTHON_ROOT}/akg_agents/op/config/triton_ascend_mathir_config.yaml"

BACKEND_ALIASES = {
    "nvidia": "cuda",
    "npu": "ascend",
}
CONFIG_BY_BACKEND = {
    "cuda": triton_cuda_path,
    "ascend": triton_ascend_path,
}
DSL_BY_BACKEND = {
    "cuda": "triton_cuda",
    "ascend": "triton_ascend",
}
ARCH_BY_BACKEND = {
    "cuda": "a100",
    "ascend": "ascend910b3",
}


def _detect_default_backend() -> str:
    if os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME") or shutil.which("npu-smi"):
        return "ascend"
    return "cuda"


default_backend = _detect_default_backend()
default_config_path = CONFIG_BY_BACKEND[default_backend]

os.environ.setdefault("AKG_AGENTS_STREAM_OUTPUT", "on")


def _kernelbench_root() -> Path:
    return REPO_ROOT / "thirdparty" / "KernelBench" / "KernelBench"


def _normalize_level(level: str) -> str:
    level = str(level).strip().lower()
    if level.startswith("level"):
        return level
    return f"level{level}"


def _parse_task_indices(raw: str) -> Optional[List[int]]:
    if not raw:
        return None

    indices = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid task range: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))

    return sorted(indices)


def _extract_problem_id(path: Path) -> int:
    match = re.match(r"(\d+)_", path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def _select_kernelbench_files(level: str, task_indices: Optional[Sequence[int]]) -> List[Path]:
    level_dir = _kernelbench_root() / _normalize_level(level)
    if not level_dir.exists():
        raise FileNotFoundError(
            f"KernelBench level directory not found: {level_dir}. "
            "Please initialize akg_agents/thirdparty/KernelBench first."
        )

    files = sorted(level_dir.glob("*.py"), key=lambda p: (_extract_problem_id(p), p.name))
    if task_indices is None:
        return files

    wanted = set(task_indices)
    selected = [path for path in files if _extract_problem_id(path) in wanted]
    missing = sorted(wanted - {_extract_problem_id(path) for path in selected})
    if missing:
        print(f"Warning: missing KernelBench task ids in {level_dir}: {missing}")
    return selected


def _op_name_from_file(path: Path) -> str:
    return f"aikg_kernelbench_{path.stem}"


def _read_task_desc(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_path(path: str) -> str:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = REPO_ROOT / path_obj
    return str(path_obj)


async def _run_task_safe(task: Any, *, op_name: str) -> Tuple[str, bool, Dict[str, Any]]:
    try:
        return await task.run()
    except Exception as exc:
        return op_name, False, {"error": f"{type(exc).__name__}: {exc}"}


def _build_task(
    *,
    op_name: str,
    task_desc: str,
    task_id: str,
    attempt_index: int,
    base_config: Dict[str, Any],
    args: argparse.Namespace,
) -> Any:
    from akg_agents.op.langgraph_op.task import LangGraphTask  # type: ignore[reportMissingImports]
    from akg_agents.utils.task_label import resolve_task_label  # type: ignore[reportMissingImports]

    task_config = copy.deepcopy(base_config)
    task_config["task_label"] = resolve_task_label(
        op_name=op_name,
        parallel_index=attempt_index,
    )
    task_config["bench_type"] = "kernelbench"

    if args.workflow_timeout is not None:
        task_config["workflow_timeout"] = args.workflow_timeout
    if args.max_step is not None:
        task_config["max_step"] = args.max_step

    return LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id=task_id,
        backend=args.backend,
        arch=args.arch,
        dsl=args.dsl,
        config=task_config,
        framework=args.framework,
        task_type=args.task_type,
        workflow=args.workflow,
        bench_type="kernelbench",
    )


def _summarize_results(results: Sequence[Tuple[str, bool, Dict[str, Any]]]) -> Dict[str, Any]:
    per_op: Dict[str, Dict[str, Any]] = {}
    attempts = []

    for op_name, success, task_info in results:
        stats = per_op.setdefault(op_name, {"passed": 0, "total": 0, "errors": []})
        stats["total"] += 1
        if success:
            stats["passed"] += 1
        else:
            error = (
                task_info.get("verifier_error")
                or task_info.get("error")
                or task_info.get("error_message")
                or ""
            )
            if error:
                stats["errors"].append(str(error)[-1000:])

        attempts.append(
            {
                "op_name": op_name,
                "success": bool(success),
                "task_id": task_info.get("task_id"),
                "task_label": task_info.get("task_label"),
                "profile": task_info.get("profile_res") or {},
                "error": (
                    task_info.get("verifier_error")
                    or task_info.get("error")
                    or task_info.get("error_message")
                    or ""
                ),
            }
        )

    passed_ops = [op for op, stats in per_op.items() if stats["passed"] > 0]
    failed_ops = [op for op, stats in per_op.items() if stats["passed"] == 0]
    total_ops = len(per_op)

    return {
        "total_ops": total_ops,
        "passed_ops": len(passed_ops),
        "failed_ops": len(failed_ops),
        "pass_rate": (len(passed_ops) / total_ops) if total_ops else 0.0,
        "per_op": per_op,
        "failed_op_names": failed_ops,
        "attempts": attempts,
    }


def _write_report(
    *,
    summary: Dict[str, Any],
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = Path(config.get("log_dir", "~/akg_agents_logs")).expanduser()
        output_dir = output_dir / "buaa_tdd_kernelbench_passk"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"passk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    payload = {
        "created_at": datetime.now().isoformat(),
        "settings": {
            "level": _normalize_level(args.level),
            "tasks": args.tasks,
            "pass_k": args.pass_k,
            "parallel_num": args.parallel_num,
            "devices": args.devices,
            "framework": args.framework,
            "dsl": args.dsl,
            "backend": args.backend,
            "arch": args.arch,
            "workflow": args.workflow,
            "task_type": args.task_type,
            "config_path": args.config_path,
        },
        "summary": summary,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report_path


def _print_summary(summary: Dict[str, Any], report_path: Path) -> None:
    total = summary["total_ops"]
    passed = summary["passed_ops"]
    failed = summary["failed_ops"]
    pass_rate = summary["pass_rate"]

    print("\n" + "=" * 80)
    print("KernelBench pass@k summary")
    print("=" * 80)
    print(f"Passed ops: {passed}/{total}")
    print(f"Failed ops: {failed}/{total}")
    print(f"Pass rate : {pass_rate:.2%}")

    for op_name, stats in sorted(summary["per_op"].items()):
        mark = "PASS" if stats["passed"] > 0 else "FAIL"
        print(f"[{mark}] {op_name}: {stats['passed']}/{stats['total']}")

    if summary["failed_op_names"]:
        print("\nFailed op names:")
        for op_name in summary["failed_op_names"]:
            print(f"  - {op_name}")

    print(f"\nReport saved to: {report_path}")
    print("=" * 80)


async def run_kernelbench_passk(args: argparse.Namespace) -> Dict[str, Any]:
    from akg_agents.core.async_pool.task_pool import TaskPool  # type: ignore[reportMissingImports]
    from akg_agents.core.worker.manager import register_worker  # type: ignore[reportMissingImports]
    from akg_agents.op.config.config_validator import load_config  # type: ignore[reportMissingImports]
    from akg_agents.utils.environment_check import check_env_for_task  # type: ignore[reportMissingImports]

    args.workflow = args.workflow or default_workflow
    args.backend = BACKEND_ALIASES.get(str(args.backend).strip().lower(), str(args.backend).strip().lower())
    if args.backend not in CONFIG_BY_BACKEND:
        raise ValueError(f"Unsupported backend for KernelBench pass@k: {args.backend}")
    args.dsl = args.dsl or DSL_BY_BACKEND[args.backend]
    args.arch = args.arch or ARCH_BY_BACKEND[args.backend]
    args.config_path = args.config_path or CONFIG_BY_BACKEND[args.backend]
    task_indices = _parse_task_indices(args.tasks)
    task_files = _select_kernelbench_files(args.level, task_indices)
    if not task_files:
        raise RuntimeError("No KernelBench tasks selected.")

    device_ids = [int(item) for item in args.devices.split(",") if item.strip()]
    await register_worker(backend=args.backend, arch=args.arch, device_ids=device_ids)

    if args.config_path:
        config = load_config(config_path=_resolve_path(args.config_path))
    else:
        config = load_config(dsl=args.dsl, backend=args.backend, workflow=args.workflow)

    check_env_for_task(args.framework, args.backend, args.dsl, config)

    print(f"Selected {len(task_files)} KernelBench tasks from {_normalize_level(args.level)}")
    print(f"Running pass@{args.pass_k}, parallel_num={args.parallel_num}, devices={device_ids}")

    task_pool = TaskPool(max_concurrency=args.parallel_num)
    init_failures: List[Tuple[str, bool, Dict[str, Any]]] = []

    for problem_idx, task_file in enumerate(task_files, start=1):
        task_desc = _read_task_desc(task_file)
        op_name = _op_name_from_file(task_file)

        for attempt in range(1, args.pass_k + 1):
            task_id = f"{_extract_problem_id(task_file)}_attempt{attempt}"
            try:
                task = _build_task(
                    op_name=op_name,
                    task_desc=task_desc,
                    task_id=task_id,
                    attempt_index=attempt,
                    base_config=config,
                    args=args,
                )
            except Exception as exc:
                init_failures.append(
                    (
                        op_name,
                        False,
                        {
                            "task_id": task_id,
                            "error": f"TaskInitError: {type(exc).__name__}: {exc}",
                        },
                    )
                )
                continue

            task_pool.create_task(
                lambda t=task, name=op_name: _run_task_safe(t, op_name=name),
                task_name=f"{problem_idx}:{task_file.stem}:attempt{attempt}",
            )

    results = init_failures + await task_pool.wait_all()
    summary = _summarize_results(results)
    report_path = _write_report(summary=summary, args=args, config=config)
    _print_summary(summary, report_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KernelBench pass@k with AKG Agents LangGraphTask."
    )
    parser.add_argument("--level", default="level1", help="KernelBench level, e.g. level1 or 1")
    parser.add_argument(
        "--tasks",
        default="",
        help="Task ids to run, e.g. '1,2,5-8'. Empty means all tasks in the level.",
    )
    parser.add_argument("--pass-k", type=int, default=1, help="Attempts per KernelBench task")
    parser.add_argument("--parallel-num", type=int, default=1, help="Max concurrent attempts")
    parser.add_argument("--devices", default="0", help="Comma-separated local device ids")
    parser.add_argument("--framework", default="torch")
    parser.add_argument(
        "--dsl",
        default="",
        help=f"DSL type. Default follows --backend ({DSL_BY_BACKEND[default_backend]}).",
    )
    parser.add_argument(
        "--backend",
        default=default_backend,
        help=f"Backend type. Auto-detected default: {default_backend}",
    )
    parser.add_argument(
        "--arch",
        default="",
        help=f"Hardware arch. Default follows --backend ({ARCH_BY_BACKEND[default_backend]}).",
    )
    parser.add_argument("--task-type", default="precision_only", choices=["precision_only", "profile"])
    parser.add_argument("--workflow", default="", help=f"Workflow name. Default: {default_workflow}")
    parser.add_argument("--config-path", default="", help=f"Config YAML path. Default: {default_config_path}")
    parser.add_argument("--output-dir", default="", help="Optional report output directory")
    parser.add_argument("--max-step", type=int, default=None, help="Override workflow max_step")
    parser.add_argument("--workflow-timeout", type=int, default=None, help="Override workflow timeout seconds")

    args = parser.parse_args()
    if args.pass_k <= 0:
        parser.error("--pass-k must be positive")
    if args.parallel_num <= 0:
        parser.error("--parallel-num must be positive")
    if not args.devices.strip():
        parser.error("--devices must not be empty")
    return args


if __name__ == "__main__":
    asyncio.run(run_kernelbench_passk(parse_args()))
