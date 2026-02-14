#!/usr/bin/env python3
"""
Run a single SWE-bench Lite instance using akg_cli common as the agent.

Flow:
  1) Load one instance from the SWE-bench Lite dataset
  2) Clone the repo and checkout the base commit
  3) Run akg_cli common (programmatically) to generate a fix
  4) Write predictions.jsonl
  5) (Optional) run swebench.harness.run_evaluation for that instance

Notes:
  - Requires `datasets` and `swebench` installed.
  - Requires akg_agents deps + LLM API key configured in env.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"cwd={cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _ensure_repo(repo: str, base_commit: str, repo_dir: Path, force_reclone: bool) -> None:
    repo_url = f"https://github.com/{repo}.git"
    if repo_dir.exists():
        if force_reclone:
            # Explicit opt-in deletion only.
            _run(["rm", "-rf", str(repo_dir)])
        else:
            status = _run(["git", "status", "--porcelain"], cwd=repo_dir)
            if status:
                raise RuntimeError(
                    f"Repo has local changes: {repo_dir}\n"
                    "Use --force-reclone to delete and re-clone."
                )
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repo_url, str(repo_dir)])
    _run(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
    _run(["git", "checkout", base_commit], cwd=repo_dir)


def _build_prompt(instance: Dict[str, Any]) -> str:
    repo = instance.get("repo", "")
    base_commit = instance.get("base_commit", "")
    problem_statement = instance.get("problem_statement", "")
    fail_to_pass = instance.get("FAIL_TO_PASS") or instance.get("fail_to_pass") or []
    pass_to_pass = instance.get("PASS_TO_PASS") or instance.get("pass_to_pass") or []
    tests = []
    if isinstance(fail_to_pass, list):
        tests.extend(fail_to_pass)
    if isinstance(pass_to_pass, list):
        tests.extend(pass_to_pass[:2])  # keep prompt concise

    tests_text = "\n".join(f"- {t}" for t in tests) if tests else "- (not provided)"

    return textwrap.dedent(
        f"""
        You are an AI software engineer fixing a SWE-bench issue.

        Repo: {repo}
        Base commit: {base_commit}

        Problem statement:
        {problem_statement}

        Suggested tests (if available):
        {tests_text}

        Requirements:
        - Make minimal, correct changes.
        - Use repo-local tools (read/grep/edit/apply_patch/bash).
        - Run the suggested failing tests if possible.
        - When done, call finish() with a brief summary.
        """
    ).strip()


def _run_common_cli(prompt: str, repo_dir: Path, *, stream: bool) -> None:
    akg_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{akg_root / 'python'}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        "-m",
        "akg_agents.cli.cli",
        "common",
        "--intent",
        prompt,
        "--yolo",
        "--once",
    ]
    if stream:
        cmd.append("--stream")
    else:
        cmd.append("--no-stream")

    result = subprocess.run(cmd, cwd=str(repo_dir), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"akg_cli common failed with exit code {result.returncode}")


def _git_diff(repo_dir: Path) -> str:
    env = os.environ.copy()
    env["GIT_EXTERNAL_DIFF"] = ""
    env["GIT_PAGER"] = "cat"
    result = subprocess.run(
        [
            "git",
            "-c",
            "diff.external=",
            "-c",
            "pager.diff=false",
            "diff",
            "--no-color",
            "--no-ext-diff",
        ],
        cwd=str(repo_dir),
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git diff failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _write_predictions(path: Path, instance_id: str, model_name: str, patch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SWE-bench Lite with akg_cli common on one instance.")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    # Use the shortest problem statement in SWE-bench Lite test by default.
    parser.add_argument("--instance-id", default="pallets__flask-4045")
    parser.add_argument("--workdir", default="workspace/swebench_runs")
    parser.add_argument("--model-name", default="akg_cli_common")
    parser.add_argument("--force-reclone", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with:\n"
            "  pip install datasets"
        ) from exc

    ds = load_dataset(args.dataset, split=args.split)
    instance = None
    for item in ds:
        if item.get("instance_id") == args.instance_id:
            instance = item
            break
    if instance is None:
        raise SystemExit(f"Instance not found: {args.instance_id} in {args.dataset}:{args.split}")

    base_dir = Path(args.workdir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = base_dir / args.instance_id / "repo"
    _ensure_repo(instance["repo"], instance["base_commit"], repo_dir, args.force_reclone)

    prompt = _build_prompt(instance)
    _run_common_cli(prompt, repo_dir, stream=args.stream)

    patch = _git_diff(repo_dir)
    if not patch.strip():
        raise SystemExit("Agent produced an empty patch. Aborting.")

    pred_path = base_dir / args.instance_id / "predictions.jsonl"
    _write_predictions(pred_path, args.instance_id, args.model_name, patch)

    print(f"Predictions written to: {pred_path}")

    if args.run_eval:
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            args.dataset,
            "--predictions_path",
            str(pred_path),
            "--instance_ids",
            args.instance_id,
            "--max_workers",
            str(args.max_workers),
            "--run_id",
            f"akg_common_{args.instance_id}",
            "--split",
            args.split,
        ]
        print("Running SWE-bench evaluation:")
        print(" ".join(cmd))
        _run(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
