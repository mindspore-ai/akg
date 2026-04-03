"""
Autoresearch 手动评测工具 (辅助脚本, 不涉及 LLM)

主入口是 agent 模式: python -m agent --task <dir>

用法:
    # 运行单轮评测
    python manual_eval.py --task <task_dir> --desc "baseline"

    # 仅评测, 不做 git 操作
    python manual_eval.py --task <task_dir> --eval-only

    # 查看当前状态
    python manual_eval.py --task <task_dir> --status

    # 生成报告 (写到临时目录, 不污染当前分支)
    python manual_eval.py --task <task_dir> --report
"""

import argparse
import os
import sys

from .framework.runner import ExperimentRunner, load_task_config
from .framework.evaluator import run_eval, format_result_summary
from .framework.logger import RoundLogger


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch — 自主迭代优化框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python manual_eval.py --task <task_dir> --desc "baseline"
  python manual_eval.py --task <task_dir> --status
        """,
    )
    parser.add_argument("--task", required=True, help="任务目录路径")
    parser.add_argument("--desc", default=None, help="本轮改动描述")
    parser.add_argument("--eval-only", action="store_true", help="仅评测, 不做 git 操作")
    parser.add_argument("--status", action="store_true", help="显示当前状态")
    parser.add_argument("--report", action="store_true", help="生成实验报告 (带图表)")

    args = parser.parse_args()

    task_dir = os.path.abspath(args.task)
    if not os.path.isdir(task_dir):
        print(f"Error: task directory not found: {task_dir}")
        sys.exit(1)

    # ---- Read-only commands: no ExperimentRunner, no branch switch, no log writes ----

    if args.status:
        config = load_task_config(task_dir)
        logger = RoundLogger(task_dir, config)
        print(f"Task: {config.name}")
        print(f"Description: {config.description}")
        if config.metadata:
            for k, v in config.metadata.items():
                print(f"{k}: {v}")
        print()
        print(logger.get_history_summary())
        return

    if args.eval_only:
        config = load_task_config(task_dir)
        result = run_eval(task_dir, config)
        print(format_result_summary(result))
        return

    if args.report:
        from .framework.report import generate_report
        config = load_task_config(task_dir)
        report_dir = os.path.join(task_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        generate_report(task_dir, config, output_dir=report_dir)
        return

    # ---- Active mode: run one round (switches to exp branch) ----

    if args.desc is None:
        print("Error: --desc is required for running an experiment round")
        print("Usage: python manual_eval.py --task <path> --desc 'description'")
        sys.exit(1)

    import asyncio
    runner = ExperimentRunner(task_dir)
    record = asyncio.run(runner.run_one_round(args.desc))

    print(f"\n{'='*60}")
    print("Status:")
    print(f"{'='*60}")
    print(runner.status())


if __name__ == "__main__":
    main()