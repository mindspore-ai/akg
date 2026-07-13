from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class EvalDefaults:
    """verify/profile/generate-reference 的集中默认配置。"""

    eval_timeout: int = 600              # verify/profile 单次预算（秒）
    reference_timeout: int = 120         # generate_reference 预算（秒）
    warmup_times: int = 5                # profile 预热次数
    run_times: int = 50                  # profile 正式测量次数

    # SIGTERM 后等 graceful 退出的秒数；NPU 上需要 >= 2s 给 PyTorch+CANN
    # atexit 释放 ACL context，否则 SIGKILL 把 TS 残留状态留在 device 上
    # （6/17 device 5 wedge 同 path）。SIGKILL 后排干管道的秒数。
    kill_grace_s: float = 5.0
    kill_drain_s: float = 2.0

    def as_env(self) -> Dict[str, str]:
        """转成 detached / remote worker daemon 消费的环境变量。"""
        return {
            "AKG_EVAL_TIMEOUT_S": str(self.eval_timeout),
            "AKG_EVAL_REFERENCE_TIMEOUT_S": str(self.reference_timeout),
            "AKG_EVAL_WARMUP_TIMES": str(self.warmup_times),
            "AKG_EVAL_RUN_TIMES": str(self.run_times),
            "AKG_EVAL_KILL_GRACE_S": str(self.kill_grace_s),
            "AKG_EVAL_KILL_DRAIN_S": str(self.kill_drain_s),
        }


def eval_defaults(config_path: Optional[str] = None) -> EvalDefaults:
    """解析最终生效的 eval/profile 默认值。

    优先级：AKG_EVAL_* env > config.yaml > EvalDefaults dataclass 默认值。
    config.yaml 读取 ``defaults.eval_timeout``、
    ``defaults.reference_data_timeout``、``eval.warmup``、``eval.repeats``、
    ``defaults.kill_grace_s``、``defaults.kill_drain_s``。
    """
    base = _from_config(config_path)
    return EvalDefaults(
        eval_timeout=_env_int("AKG_EVAL_TIMEOUT_S", base.eval_timeout),
        reference_timeout=_env_int(
            "AKG_EVAL_REFERENCE_TIMEOUT_S", base.reference_timeout),
        warmup_times=_env_int("AKG_EVAL_WARMUP_TIMES", base.warmup_times),
        run_times=_env_int("AKG_EVAL_RUN_TIMES", base.run_times),
        kill_grace_s=_env_float("AKG_EVAL_KILL_GRACE_S", base.kill_grace_s),
        kill_drain_s=_env_float("AKG_EVAL_KILL_DRAIN_S", base.kill_drain_s),
    )


def resolve_eval_timeout(value: Optional[int] = None) -> int:
    return _positive_int(value, eval_defaults().eval_timeout)


def resolve_reference_timeout(value: Optional[int] = None) -> int:
    return _positive_int(value, eval_defaults().reference_timeout)


def resolve_warmup_times(value: Optional[int] = None) -> int:
    return _positive_int(value, eval_defaults().warmup_times)


def resolve_run_times(value: Optional[int] = None) -> int:
    return _positive_int(value, eval_defaults().run_times)


def resolve_kill_grace_s(value: Optional[float] = None) -> float:
    """SIGTERM 后等 graceful 的秒数。设 0 等价于直接 SIGKILL（测试用）。"""
    return _positive_float(value, eval_defaults().kill_grace_s)


def resolve_kill_drain_s(value: Optional[float] = None) -> float:
    """SIGKILL 后排干 stdout/stderr 管道的等待秒数。"""
    return _positive_float(value, eval_defaults().kill_drain_s)


def _from_config(config_path: Optional[str]) -> EvalDefaults:
    defaults = EvalDefaults()
    resolved = _resolve(config_path)
    if resolved is None:
        return defaults
    data = _load_yaml(resolved)
    if not isinstance(data, dict):
        return defaults
    defaults_block = data.get("defaults") or {}
    eval_block = data.get("eval") or {}
    return EvalDefaults(
        eval_timeout=_positive_int(
            defaults_block.get("eval_timeout"), defaults.eval_timeout),
        reference_timeout=_positive_int(
            defaults_block.get("reference_data_timeout"),
            defaults.reference_timeout,
        ),
        warmup_times=_positive_int(eval_block.get("warmup"),
                                   defaults.warmup_times),
        run_times=_positive_int(eval_block.get("repeats"),
                                defaults.run_times),
        kill_grace_s=_positive_float(
            defaults_block.get("kill_grace_s"), defaults.kill_grace_s),
        kill_drain_s=_positive_float(
            defaults_block.get("kill_drain_s"), defaults.kill_drain_s),
    )


# config.yaml 解析 + 读取的唯一实现。worker_config.py 复用这两个 helper，
# 各自只通过参数表达差异：eval 侧向上 walk parents 找 config.yaml，worker
# 侧只看 cwd（walk_parents=False）；读失败的 stderr tag 也各用各的。
def _resolve(config_path: Optional[str], *, walk_parents: bool = True) -> Optional[str]:
    if config_path is not None:
        p = Path(config_path)
        return str(p) if p.is_file() else None
    cur = Path.cwd()
    candidates = (cur, *cur.parents) if walk_parents else (cur,)
    for candidate in candidates:
        p = candidate / "config.yaml"
        if p.is_file():
            return str(p)
    return None


def _load_yaml(config_path: str, tag: str = "akg_eval") -> Optional[dict]:
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[{tag}] failed to read {config_path}: {e}", file=sys.stderr)
        return None


def _env_int(key: str, default: int) -> int:
    return _positive_int(os.environ.get(key), default)


def _env_float(key: str, default: float) -> float:
    return _positive_float(os.environ.get(key), default)


def _positive_int(value, default: int) -> int:
    try:
        v = int(value)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _positive_float(value, default: float) -> float:
    """允许 0（kill_grace_s=0 → 跳过 graceful），负数和不可解析 fallback。"""
    try:
        v = float(value)
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default
