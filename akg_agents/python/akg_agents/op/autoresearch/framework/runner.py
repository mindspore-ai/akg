"""
实验循环执行器 — 驱动 agent 进行自主迭代优化

两种使用方式:
  1. Agent 内部调用: agent 调用 run_one_round() 评测
  2. 外部驱动: python -m framework.runner --task <task_dir>

在 Agent 模式下 (推荐), agent 自己是循环的一部分:
  - agent 读取历史、提出假设、编辑可编辑文件
  - agent 调用 run_one_round() 评测
  - agent 根据结果决定 keep/discard
  - agent 继续下一轮

本 runner 提供的是评测 + 记录的胶水代码. 所有 git 操作都委托给
``framework.git_repo.GitRepo``，文件回滚委托给
``framework.file_state.FileStateManager``，两者都是 ExperimentRunner 的
公开属性 (``self.git`` / ``self.file_state``)，外部组件 (AgentLoop /
TurnExecutor / SessionStore) 也通过这两个属性访问。
"""

import os
import time
from typing import Optional

from .config import EvalResult, RoundRecord, TaskConfig
from .evaluator import run_eval, run_eval_robust, is_improvement, check_constraints, validate_constraints, format_result_summary
from .file_state import FileStateManager
from .git_repo import GitRepo
from .logger import RoundLogger


def load_task_config(task_dir: str) -> TaskConfig:
    """从 task_dir/task.yaml 加载 TaskConfig."""
    from .config_loader import load_yaml_config

    cfg = load_yaml_config(task_dir)
    if cfg is None:
        raise FileNotFoundError(
            f"task.yaml not found in {task_dir}"
        )
    if cfg.constraints != {}:
        validate_constraints(cfg.constraints)
    return cfg


class ExperimentRunner:
    """
    实验循环管理器 — 评测 + keep/discard 判定 + git 操作 + 日志.

    典型用法:
        runner = ExperimentRunner("<task_dir>")
        result = runner.run_one_round("baseline — no changes")
    """

    def __init__(self, task_dir: str, skip_branch_switch: bool = False,
                 device_id: int = None, eval_fn=None):
        self.task_dir = os.path.abspath(task_dir)
        self.device_id = device_id
        self._eval_fn = eval_fn  # async callable(task_dir, config, round_num) -> EvalResult
        self.config = load_task_config(self.task_dir)
        self._best_result: Optional[EvalResult] = None
        self.extra_commit_files: list[str] = []

        # Public sub-components: GitRepo for all git ops, FileStateManager
        # for editable-file rollback (snapshot/restore + git rollback). Both
        # are accessible from external callers (AgentLoop, TurnExecutor,
        # SessionStore) via self.git / self.file_state.
        self.git = GitRepo(self.task_dir)
        self.file_state = FileStateManager(
            self.task_dir, self.config.editable_files, self.git,
        )

        self.original_branch: Optional[str] = None
        if not skip_branch_switch:
            self.original_branch = self.git.current_branch()
            branch = self.config.git_branch or f"exp/{self.config.name}"
            self.branch_name = self.git.ensure_branch(branch)
        else:
            self.branch_name = None

        self.logger = RoundLogger(self.task_dir, self.config)

        # Single ref baseline measurement (set once at round 0)
        self._ref_latency: Optional[float] = None

        best = self.logger.get_best()
        if best:
            self._best_result = EvalResult(
                correctness=True,
                metrics=best.get("metrics", {}),
            )

        # Restore ref from history (for --resume).
        # Reject inf — it indicates a failed measurement, not a real latency.
        for rec in self.logger.load_history():
            metrics = rec.get("metrics", {})
            ref_val = metrics.get("_ref_latency_raw") or metrics.get("ref_latency_us")
            if isinstance(ref_val, (int, float)) and 0 < ref_val < float('inf'):
                self._ref_latency = float(ref_val)
                break  # only need the first one

    @property
    def ref_latency(self) -> Optional[float]:
        """Ref model latency measured at baseline. None if not yet measured."""
        return self._ref_latency

    async def run_one_round(self, description: str) -> RoundRecord:
        """
        执行一轮实验: eval → keep/discard → git → log → return record.

        Contract: always returns a RoundRecord, never raises.
        Any internal failure (eval, git, logging) is captured as a
        FAIL record with rollback.
        """
        round_num = self.logger.next_round_num()
        t0 = time.time()

        print(f"\n[Runner] {'='*56}")
        print(f"[Runner] Round {round_num}: {description}")
        print(f"[Runner] {'='*56}")

        try:
            return await self._run_one_round_inner(
                round_num, description, t0)
        except Exception as e:
            # Last-resort catch: something after eval crashed (git, log, etc).
            # Rollback to ensure no dirty state leaks into the next round.
            duration = time.time() - t0
            print(f"[Runner] ROUND CRASHED: {e}", flush=True)
            self.file_state.rollback_to_head()
            record = RoundRecord(
                round_num=round_num,
                description=description,
                result=EvalResult(correctness=False,
                                  error=f"round crashed: {e}"),
                accepted=False,
                commit_hash=None,
                duration_sec=duration,
            )
            try:
                self.logger.log_round(record)
            except Exception:
                pass  # logging failure must not prevent return
            return record

    async def _run_one_round_inner(
        self, round_num: int, description: str, t0: float,
    ) -> RoundRecord:
        """Inner implementation — may raise, caller guarantees catch."""

        # 1. Eval
        if self._eval_fn is not None:
            eval_desc = "eval_fn (KernelVerifier)"
        elif self.config.eval_script:
            eval_desc = self.config.eval_script
        else:
            eval_desc = f"generated ({self.config.dsl}/{self.config.framework}/{self.config.backend})"
        print(f"[Runner] Running eval: {eval_desc} ...", flush=True)
        try:
            result = await run_eval_robust(self.task_dir, self.config,
                                           device_id=self.device_id,
                                           eval_fn=self._eval_fn,
                                           round_num=round_num)
        except Exception as e:
            result = EvalResult(correctness=False, error=f"eval crashed: {e}")
        duration = time.time() - t0

        # Ref baseline: read from eval result metrics (set by eval_fn).
        if self._ref_latency is None:
            ref_us = result.metrics.get("ref_latency_us")
            if ref_us is not None and 0 < ref_us < float('inf'):
                self._ref_latency = ref_us
                print(f"[Runner] Ref baseline: {ref_us:.2f} us", flush=True)

        print(f"\n[Runner] Result: {format_result_summary(result)}", flush=True)

        # 2. Keep/discard decision
        accepted = False
        prev_best = self._best_result

        constraint_violations = []
        if self.config.constraints and result.correctness:
            constraint_violations = check_constraints(result, self.config.constraints)

        if not result.correctness:
            if result.error:
                print(f"[Runner] FAILED: {result.error}", flush=True)
            else:
                print("[Runner] CORRECTNESS FAILED — discarding", flush=True)
        elif constraint_violations:
            print("[Runner] CONSTRAINT VIOLATED — discarding:", flush=True)
            for v in constraint_violations:
                print(f"[Runner]   - {v}")
        elif self._best_result is None:
            accepted = True
            self._best_result = result
            print("[Runner] First valid result — KEEP (baseline)", flush=True)
        elif is_improvement(result, self._best_result,
                            self.config.primary_metric,
                            self.config.lower_is_better,
                            threshold=self.config.improvement_threshold):
            accepted = True
            old_val = self._best_result.metrics.get(self.config.primary_metric, "N/A")
            new_val = result.metrics.get(self.config.primary_metric, "N/A")
            print(f"[Runner] IMPROVEMENT: {old_val} → {new_val} — KEEP", flush=True)
            self._best_result = result
        else:
            cur_val = result.metrics.get(self.config.primary_metric, "N/A")
            best_val = self._best_result.metrics.get(self.config.primary_metric, "N/A")
            print(f"[Runner] NO IMPROVEMENT: {cur_val} vs best {best_val} — DISCARD", flush=True)

        # 3. Build record
        record = RoundRecord(
            round_num=round_num,
            description=description,
            result=result,
            accepted=accepted,
            commit_hash=None,
            duration_sec=duration,
            constraint_violations=constraint_violations,
        )

        # 4. Git: commit (KEEP) or rollback (DISCARD)
        if accepted:
            primary_val = result.metrics.get(self.config.primary_metric, "")
            if isinstance(primary_val, float):
                primary_val = f"{primary_val:.4f}"
            commit_msg = f"R{round_num}: {description} | {self.config.primary_metric}={primary_val}"
            commit_files = list(self.config.editable_files) + self.extra_commit_files
            cr = self.git.commit(
                commit_msg,
                files=commit_files,
                push=self.config.git_push,
                task_name=self.config.name,
                expected_branch=self.branch_name,
            )
            if cr.committed:
                record.commit_hash = cr.hash
                print(f"[Runner] Committed: {cr.hash}", flush=True)
            elif cr.nothing_to_commit:
                print("[Runner] No code changes to commit (baseline or no-op)")
            else:
                print(f"[Runner] WARNING: git commit failed ({cr.error}) — demoting to DISCARD", flush=True)
                accepted = False
                record.accepted = False
                self._best_result = prev_best
                self.file_state.rollback_to_head()
        else:
            self.file_state.rollback_to_head()
            print("[Runner] Rolled back editable files", flush=True)

        # 5. Log (AFTER git so commit_hash is populated)
        # Logging failure must not override an already-committed result.
        try:
            self.logger.log_round(record)
        except Exception as e:
            print(f"[Runner] WARNING: log_round failed: {e}")

        return record

    def status(self) -> str:
        """返回当前状态摘要"""
        lines = [
            f"Task: {self.config.name}",
            f"Description: {self.config.description}",
        ]
        if self.config.metadata:
            for k, v in self.config.metadata.items():
                lines.append(f"{k}: {v}")
        lines.append("")
        lines.append(self.logger.get_history_summary())
        return "\n".join(lines)

    def get_editable_contents(self) -> dict[str, str]:
        """读取所有可编辑文件的当前内容"""
        contents = {}
        for f in self.config.editable_files:
            fpath = os.path.join(self.task_dir, f)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as fp:
                    contents[f] = fp.read()
        return contents

    def generate_report(self, output_dir: Optional[str] = None) -> str:
        """生成实验报告, 返回报告文件路径"""
        from .report import generate_report
        return generate_report(self.task_dir, self.config, output_dir=output_dir)

    @property
    def best_result(self) -> Optional[EvalResult]:
        return self._best_result


# ---- CLI 入口 ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one experiment round")
    parser.add_argument("--task", required=True, help="Path to task directory")
    parser.add_argument("--desc", default="manual run", help="Round description")
    parser.add_argument("--eval-only", action="store_true", help="Only run eval, no git ops")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task)

    import asyncio

    if args.eval_only:
        # Read-only: no ExperimentRunner, no branch switch.
        cfg = load_task_config(task_dir)
        result = run_eval(task_dir, cfg)
        print(format_result_summary(result))
    else:
        runner = ExperimentRunner(task_dir)
        record = asyncio.run(runner.run_one_round(args.desc))
        print(f"\nStatus:\n{runner.status()}")