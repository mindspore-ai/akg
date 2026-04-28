"""
评测执行器 — 以子进程方式运行评测脚本, 解析结果

评测脚本协议:
  - 退出码 0 = 正确性通过
  - 退出码 1 = 正确性失败或 crash
  - stdout 最后一行必须是 JSON, 格式:
    {"correctness": true/false, "latency_ms": 0.466, "tflops": 27.66, ...}

评测脚本确定优先级:
  1. config.eval_script — 自定义脚本 (任务自带)
  2. config.dsl + framework + backend — adapter 生成的评测脚本
"""

import asyncio
import json
import operator as _op
import os
import subprocess
import sys
import time
from typing import Optional

from .config import EvalResult, TaskConfig


# 缓存: (task_dir, dsl, framework, backend) → 生成的脚本路径
_generated_eval_cache: dict[tuple[str, str, str, str], str] = {}


def _resolve_eval_command(task_dir: str, config: TaskConfig) -> list[str]:
    """确定评测命令.

    优先级:
      1. config.eval_script — 自定义脚本 (任务自带)
      2. adapter 生成 — dsl/framework/backend 三元组驱动
    """
    # 1. 自定义脚本
    if config.eval_script:
        eval_script = os.path.join(task_dir, config.eval_script)
        if not os.path.exists(eval_script):
            return None  # caller 会处理 error
        return [sys.executable, eval_script]

    # 2. Adapter 生成
    if not (config.dsl and config.framework and config.backend):
        return None  # 缺少适配器声明

    cache_key = (os.path.abspath(task_dir), config.dsl, config.framework, config.backend)
    if cache_key not in _generated_eval_cache:
        try:
            from .eval_generator import generate_eval_script_file
        except ImportError:
            generate_eval_script_file = None  # not available in AKG mode
        if generate_eval_script_file is None:
            return None
        path = generate_eval_script_file(
            dsl=config.dsl,
            framework=config.framework,
            backend=config.backend,
            output_dir=os.path.join(task_dir, ".eval_cache"),
        )
        _generated_eval_cache[cache_key] = path
    eval_path = _generated_eval_cache[cache_key]
    return [sys.executable, eval_path, "--task-dir", task_dir]


def _resolve_env(config: TaskConfig, device_id: int = None) -> dict:
    """确定环境变量 (通过 BackendAdapter)."""
    env = os.environ.copy()
    if device_id is None:
        return env

    if config.backend:
        try:
            from .adapters.registry import get_backend_adapter
        except ImportError:
            get_backend_adapter = None  # not available in AKG mode
        if get_backend_adapter is not None:
            be = get_backend_adapter(config.backend)
            env.update(be.env_vars(device_id))

    return env


def run_ref_benchmark(task_dir: str, config: TaskConfig, device_id: int = None) -> Optional[float]:
    """单独测一次 ref model latency, 返回 ref_latency_us. 失败返回 None."""
    cmd = _resolve_eval_command(task_dir, config)
    if cmd is None:
        return None
    cmd = cmd + ["--ref-only"]
    env = _resolve_env(config, device_id)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=config.eval_timeout, cwd=task_dir, env=env,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    stdout_lines = result.stdout.strip().split("\n") if result.stdout else []
    for line in reversed(stdout_lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                return data.get("ref_latency_us")
            except json.JSONDecodeError:
                continue
    return None


def run_eval(task_dir: str, config: TaskConfig, device_id: int = None) -> EvalResult:
    """
    执行评测脚本, 返回 EvalResult

    评测脚本在 task_dir 下运行, 超时后自动 kill.
    device_id 不为 None 时注入设备可见性环境变量.
    """
    cmd = _resolve_eval_command(task_dir, config)
    if cmd is None:
        if config.eval_script:
            msg = f"eval script not found: {config.eval_script}"
        else:
            msg = "cannot resolve eval command: set dsl/framework/backend or eval_script in task.yaml"
        return EvalResult(correctness=False, error=msg)

    env = _resolve_env(config, device_id)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.eval_timeout,
            cwd=task_dir,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return EvalResult(
            correctness=False,
            error=f"eval timed out after {config.eval_timeout}s",
        )
    except Exception as e:
        return EvalResult(
            correctness=False,
            error=f"eval failed to launch: {e}",
        )

    raw_output = result.stdout + result.stderr

    # 解析 stdout 最后一行的 JSON
    metrics = {}
    correctness = False
    parsed_json = False

    # 尝试从 stdout 中提取 JSON 结果
    stdout_lines = result.stdout.strip().split("\n") if result.stdout else []
    for line in reversed(stdout_lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                correctness = data.get("correctness", False)
                metrics = {k: v for k, v in data.items() if k != "correctness"}
                parsed_json = True
                break
            except json.JSONDecodeError:
                continue

    # 如果没有找到 JSON, 尝试解析 key: value 格式 (兼容 karpathy 风格)
    if not parsed_json:
        for line in stdout_lines:
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, _, val = line.partition(":")
                key = key.strip().replace(" ", "_")
                val = val.strip()
                try:
                    metrics[key] = float(val)
                except ValueError:
                    if val.lower() in ("true", "pass", "yes"):
                        if key.lower() in ("correctness", "correct", "pass"):
                            correctness = True
                    elif val.lower() in ("false", "fail", "no"):
                        if key.lower() in ("correctness", "correct", "pass"):
                            correctness = False

    # Determine error: only set when we have a non-zero exit AND could not
    # parse a structured result.  When JSON was parsed with correctness=False,
    # that is a normal correctness failure — NOT a crash.
    error = None
    if result.returncode != 0 and not parsed_json:
        stderr_tail = result.stderr[-500:] if result.stderr else ""
        stdout_tail = result.stdout[-500:] if result.stdout else ""
        error = f"exit code {result.returncode}\n{stderr_tail}\n{stdout_tail}".strip()

    return EvalResult(
        correctness=correctness,
        metrics=metrics,
        error=error,
        raw_output=raw_output,
    )


async def run_eval_robust(task_dir: str, config: TaskConfig, device_id: int = None,
                          eval_fn=None, round_num: int = 0) -> EvalResult:
    """
    执行一次评测, 返回 EvalResult.

    eval_fn: Optional async callable(task_dir, config, round_num=0) -> EvalResult.
             When provided, called once per round (AKG mode).
             When None, uses subprocess-based run_eval (standalone mode).

    Both paths respect config.eval_timeout.
    """
    if eval_fn is not None:
        try:
            return await asyncio.wait_for(
                eval_fn(task_dir, config, round_num=round_num),
                timeout=config.eval_timeout,
            )
        except asyncio.TimeoutError:
            return EvalResult(
                correctness=False,
                error=f"eval_fn timed out after {config.eval_timeout}s",
            )
    return run_eval(task_dir, config, device_id=device_id)


_CONSTRAINT_OPS = {
    "<=": _op.le,
    ">=": _op.ge,
    "<": _op.lt,
    ">": _op.gt,
    "==": _op.eq,
}

VALID_CONSTRAINT_OPS = set(_CONSTRAINT_OPS.keys())


def validate_constraints(constraints: dict) -> None:
    """
    Validate constraints config at load time. Raises ValueError with a
    readable message if the format is wrong.

    Expected format: {metric_name: (operator_str, threshold)}
    """
    if not isinstance(constraints, dict):
        raise ValueError(
            f"constraints must be a dict, got {type(constraints).__name__}: {constraints!r}"
        )
    for metric_name, spec in constraints.items():
        if not isinstance(metric_name, str):
            raise ValueError(
                f"constraints key must be a string, got {type(metric_name).__name__}: {metric_name!r}"
            )
        if not isinstance(spec, (tuple, list)) or len(spec) != 2:
            raise ValueError(
                f"constraints[{metric_name!r}] must be a (operator, threshold) pair, "
                f"got {spec!r}. Example: {{\"latency\": (\"<=\", 500.0)}}"
            )
        op_str, threshold = spec
        if op_str not in VALID_CONSTRAINT_OPS:
            raise ValueError(
                f"constraints[{metric_name!r}] has unknown operator {op_str!r}. "
                f"Valid operators: {', '.join(sorted(VALID_CONSTRAINT_OPS))}"
            )
        if not isinstance(threshold, (int, float)):
            raise ValueError(
                f"constraints[{metric_name!r}] threshold must be numeric, "
                f"got {type(threshold).__name__}: {threshold!r}"
            )


def check_constraints(
    result: EvalResult,
    constraints: dict,
) -> list[str]:
    """
    Check hard constraints on evaluation metrics.

    Args:
        result: evaluation result to check
        constraints: {metric_name: (operator_str, threshold)} from TaskConfig

    Returns:
        List of human-readable violation strings. Empty list = all constraints satisfied.
    """
    violations = []
    for metric_name, (op_str, threshold) in constraints.items():
        func = _CONSTRAINT_OPS.get(op_str)
        if func is None:
            violations.append(f"{metric_name}: unknown operator '{op_str}'")
            continue

        value = result.metrics.get(metric_name)
        if value is None:
            violations.append(f"{metric_name}: metric missing (required {op_str} {threshold})")
            continue

        if not isinstance(value, (int, float)):
            violations.append(f"{metric_name}: non-numeric value {value!r} (required {op_str} {threshold})")
            continue

        if not func(value, threshold):
            violations.append(f"{metric_name}: {value} violates {op_str} {threshold}")

    return violations


def is_improvement(
    current: EvalResult,
    best: EvalResult,
    metric: str = "latency_ms",
    lower_is_better: bool = True,
    threshold: float = 0.0,
) -> bool:
    """
    判断当前结果是否优于历史最优

    两个结果都必须通过正确性检查.
    threshold 为相对提升百分比阈值 (e.g. 2.0 表示需要 >2% 的提升).
    """
    if not current.correctness:
        return False

    cur_val = current.metrics.get(metric)
    best_val = best.metrics.get(metric)

    if cur_val is None:
        return False
    if best_val is None:
        return True  # 首次有效结果

    if best_val == 0:
        return cur_val < 0 if lower_is_better else cur_val > 0

    # 计算相对提升百分比 (正值 = 改进)
    if lower_is_better:
        relative_pct = (best_val - cur_val) / abs(best_val) * 100
    else:
        relative_pct = (cur_val - best_val) / abs(best_val) * 100

    return relative_pct > threshold


def format_result_summary(result: EvalResult) -> str:
    """格式化结果摘要, 用于 agent prompt 上下文"""
    if not result.correctness:
        if result.error:
            return f"FAILED: {result.error}"
        return f"CORRECTNESS FAILED (metrics: {result.metrics})"

    parts = [f"correctness: PASS"]
    for key, val in result.metrics.items():
        if isinstance(val, float):
            parts.append(f"{key}: {val:.4f}")
        else:
            parts.append(f"{key}: {val}")
    return "  |  ".join(parts)
