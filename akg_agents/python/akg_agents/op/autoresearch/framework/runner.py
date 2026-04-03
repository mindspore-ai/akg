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

本 runner 提供的是评测 + 记录 + git 管理的胶水代码.
"""

import os
import subprocess
import sys
import time
from typing import Optional

from .config import CommitResult, EvalResult, RoundRecord, TaskConfig
from .evaluator import run_eval, run_eval_robust, is_improvement, check_constraints, validate_constraints, format_result_summary
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


def _git_repo_root(task_dir: str) -> str:
    """获取 git 仓库根目录, 失败时抛异常"""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=task_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Not a git repository: {task_dir}\n{result.stderr}")
    return result.stdout.strip()


def _git_add(repo_root: str, rel_path: str) -> bool:
    """git add 单个文件, 返回是否成功"""
    # Refresh git index for this path — on WSL2 with /mnt/c/ the stat cache
    # can be stale, causing git add to skip genuinely modified files.
    subprocess.run(
        ["git", "diff", "--", rel_path],
        cwd=repo_root, capture_output=True,
    )
    result = subprocess.run(
        ["git", "add", rel_path],
        cwd=repo_root, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[git] WARNING: git add failed for {rel_path}: {result.stderr.strip()}")
        return False
    return True


def git_commit(task_dir: str, message: str, files: Optional[list[str]] = None,
               push: bool = False, task_name: Optional[str] = None,
               expected_branch: Optional[str] = None) -> CommitResult:
    """
    提交当前更改, 返回 CommitResult.

    三种结果:
      - committed: hash 非空, 提交成功
      - nothing_to_commit: 没有需要提交的内容 (正常情况, 如 baseline)
      - error: 提交命令失败

    If expected_branch is set, refuse to commit when on a different branch.
    """
    try:
        repo_root = _git_repo_root(task_dir)

        # Branch guard: abort if on wrong branch
        if expected_branch:
            current = git_current_branch(task_dir)
            if current and current != expected_branch:
                raise RuntimeError(
                    f"Branch mismatch: on '{current}' but expected "
                    f"'{expected_branch}'. Aborting to prevent commits "
                    f"on the wrong branch."
                )

        add_failures = []
        if files:
            for f in files:
                fpath = os.path.join(task_dir, f)
                rel_path = os.path.relpath(fpath, repo_root)
                if not _git_add(repo_root, rel_path):
                    add_failures.append(rel_path)
        else:
            rel_dir = os.path.relpath(task_dir, repo_root)
            if not _git_add(repo_root, rel_dir):
                add_failures.append(rel_dir)

        # 检查是否有 staged 内容
        diff_result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_root, capture_output=True,
        )
        if diff_result.returncode == 0:
            if add_failures:
                # git add failed AND nothing staged → real error, not "nothing to commit".
                err = f"git add failed for: {', '.join(add_failures)}"
                print(f"[git] ERROR: {err}")
                return CommitResult(error=err)
            return CommitResult(nothing_to_commit=True)

        author_name = task_name or "agent"
        git_cmd = [
            "git",
            "-c", f"user.name={author_name}",
            "-c", "user.email=agent@autoresearch",
            "commit", "-m", message,
        ]
        commit_result = subprocess.run(
            git_cmd,
            cwd=repo_root, capture_output=True, text=True,
        )
        if commit_result.returncode != 0:
            err = commit_result.stderr.strip()
            print(f"[git] ERROR: commit failed: {err}")
            return CommitResult(error=err)

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo_root,
        )
        commit_hash = result.stdout.strip()

        if push:
            push_result = subprocess.run(
                ["git", "push", "-u", "origin", "HEAD"],
                cwd=repo_root, capture_output=True, text=True,
            )
            if push_result.returncode != 0:
                print(f"[git] WARNING: push failed: {push_result.stderr.strip()}")
            else:
                print(f"[git] Pushed {commit_hash} to remote")

        return CommitResult(hash=commit_hash)
    except Exception as e:
        print(f"[git] ERROR: git_commit exception: {e}")
        return CommitResult(error=str(e))


def git_current_branch(task_dir: str) -> Optional[str]:
    """Return the current branch name, or None on error."""
    try:
        repo_root = _git_repo_root(task_dir)
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=repo_root,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def git_cleanup_branch(task_dir: str, exp_branch: str, original_branch: str,
                       session_dir: str = "agent_session",
                       heartbeat_file: str = "RUNNING"):
    """Switch back to original branch, keep exp branch for result inspection."""
    import shutil
    repo_root = _git_repo_root(task_dir)
    rel_dir = os.path.relpath(os.path.abspath(task_dir), repo_root)

    # 1. Remove experiment artifacts FIRST (before checkout, to avoid dirty-tree conflicts)
    experiment_artifacts = [
        "agent.log", "log.jsonl", "perf_log.md",
        "report.png", "report.md", "plan.md", heartbeat_file,
    ]
    for fname in experiment_artifacts:
        fpath = os.path.join(task_dir, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass
    for dname in [session_dir, "__pycache__"]:
        dpath = os.path.join(task_dir, dname)
        if os.path.isdir(dpath):
            try:
                shutil.rmtree(dpath)
            except Exception:
                pass

    # 2. Discard any remaining uncommitted changes in task dir so checkout won't fail
    subprocess.run(
        ["git", "checkout", "--", rel_dir],
        capture_output=True, text=True, cwd=repo_root,
    )

    # 3. Switch back to original branch (exp branch preserved)
    current = git_current_branch(task_dir)
    if current == exp_branch:
        result = subprocess.run(
            ["git", "checkout", original_branch],
            capture_output=True, text=True, cwd=repo_root,
        )
        if result.returncode != 0:
            print(f"[git] WARNING: checkout {original_branch} failed: {result.stderr.strip()}")
            return
        print(f"[git] Switched back to '{original_branch}' (exp branch '{exp_branch}' preserved)")
    else:
        print(f"[git] WARNING: not on exp branch '{exp_branch}' (on '{current}'), skipping checkout")


def git_ensure_branch(task_dir: str, branch_name: str) -> str:
    """确保当前在指定的实验分支上.

    如果同名分支已存在 (上次实验残留), 先删除再从当前 HEAD 重建,
    保证每次实验从干净状态开始.
    """
    repo_root = _git_repo_root(task_dir)

    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, cwd=repo_root,
    )
    current_branch = result.stdout.strip()

    # Delete stale exp branch from previous run
    result = subprocess.run(
        ["git", "rev-parse", "--verify", branch_name],
        capture_output=True, text=True, cwd=repo_root,
    )
    branch_exists = result.returncode == 0

    if branch_exists:
        if current_branch == branch_name:
            # On the stale branch — need to switch away first
            # Find a branch to switch to (prefer main/master, else any other)
            for candidate in ["main", "master"]:
                check = subprocess.run(
                    ["git", "rev-parse", "--verify", candidate],
                    capture_output=True, cwd=repo_root,
                )
                if check.returncode == 0:
                    subprocess.run(
                        ["git", "checkout", candidate],
                        capture_output=True, cwd=repo_root,
                    )
                    break
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            capture_output=True, text=True, cwd=repo_root,
        )
        print(f"[git] Deleted stale branch '{branch_name}'")

    # Create fresh branch from current HEAD
    result = subprocess.run(
        ["git", "checkout", "-b", branch_name],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create branch '{branch_name}': {result.stderr.strip()}"
        )

    print(f"[git] Created and switched to branch '{branch_name}'")
    return branch_name


def git_rollback_files(task_dir: str, files: list[str]):
    """回滚指定文件到上一个 commit 的状态, 包括删除本轮新建的未跟踪文件"""
    try:
        repo_root = _git_repo_root(task_dir)

        for f in files:
            fpath = os.path.join(task_dir, f)
            rel_path = os.path.relpath(fpath, repo_root)

            ls_result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel_path],
                cwd=repo_root, capture_output=True,
            )
            if ls_result.returncode != 0:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    print(f"[git] Removed untracked file: {rel_path}")
                continue

            result = subprocess.run(
                ["git", "checkout", "HEAD", "--", rel_path],
                cwd=repo_root, capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"[git] WARNING: rollback failed for {rel_path}: {result.stderr.strip()}")
    except Exception as e:
        print(f"[git] ERROR: git_rollback_files exception: {e}")


def git_current_commit(task_dir: str) -> Optional[str]:
    """返回 HEAD 的 short commit hash，失败返回 None"""
    try:
        repo_root = _git_repo_root(task_dir)
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo_root,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def git_diff(task_dir: str, base_commit: str, head: str = "HEAD",
             paths: Optional[list[str]] = None) -> Optional[str]:
    """返回 git diff base_commit..head 的输出"""
    try:
        repo_root = _git_repo_root(task_dir)
        cmd = ["git", "diff", f"{base_commit}..{head}"]
        if paths:
            cmd.append("--")
            for p in paths:
                cmd.append(os.path.relpath(os.path.join(task_dir, p), repo_root))
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=repo_root,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def git_dirty_files(task_dir: str, files: list[str]) -> Optional[list[str]]:
    """Return subset of files that have uncommitted changes or are untracked."""
    try:
        repo_root = _git_repo_root(task_dir)
        dirty = set()
        for f in files:
            fpath = os.path.join(task_dir, f)
            rel_path = os.path.relpath(fpath, repo_root)

            result = subprocess.run(
                ["git", "diff", "HEAD", "--", rel_path],
                capture_output=True, text=True, cwd=repo_root,
            )
            if result.returncode == 0 and result.stdout.strip():
                dirty.add(f)
                continue

            if os.path.exists(fpath):
                result = subprocess.run(
                    ["git", "ls-files", "--others", "--exclude-standard", "--", rel_path],
                    capture_output=True, text=True, cwd=repo_root,
                )
                if result.returncode == 0 and result.stdout.strip():
                    dirty.add(f)

        return list(dirty)
    except Exception:
        return None


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

        self.original_branch: Optional[str] = None
        if not skip_branch_switch:
            self.original_branch = git_current_branch(self.task_dir)
            branch = self.config.git_branch or f"exp/{self.config.name}"
            self.branch_name = git_ensure_branch(self.task_dir, branch)
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
            git_rollback_files(self.task_dir, self.config.editable_files)
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
            cr = git_commit(
                self.task_dir, commit_msg,
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
                git_rollback_files(self.task_dir, self.config.editable_files)
        else:
            git_rollback_files(self.task_dir, self.config.editable_files)
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