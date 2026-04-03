"""
CLI entry point for the programmatic agent.

Usage:
    # Single task (reads api_config.yaml for model/key/endpoint)
    python -m agent --task tasks/op_migrate/relu_rmsnorm
    python -m agent --task tasks/op_migrate/relu_rmsnorm --max-rounds 20 --device-id 0

    # CLI flags override api_config.yaml values
    python -m agent --task ... --model claude-sonnet-4-6 --provider anthropic

Config file:
    agent/api_config.yaml — stores model, api_key, base_url, provider, reasoning_effort.
    Gitignored to avoid leaking keys. CLI flags take precedence.
"""

import argparse
import asyncio
import os
import sys

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_api_config() -> dict:
    """Load agent/api_config.yaml if it exists. Returns dict of config values."""
    config_path = os.path.join(AGENT_DIR, "api_config.yaml")
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
    except ImportError:
        # Fallback: minimal YAML parser for simple key: value files
        cfg = {}
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, _, val = line.partition(":")
                    val = val.split("#")[0].strip().strip("'\"")
                    if val:
                        cfg[key.strip()] = val
        return cfg
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    # Load config file defaults
    file_cfg = _load_api_config()

    parser = argparse.ArgumentParser(
        description="Autoresearch — Programmatic Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--task", metavar="DIR", required=True,
        help="Task directory path",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Model ID (config: {file_cfg.get('model', 'claude-sonnet-4-6')})",
    )
    parser.add_argument(
        "--device-id", type=int, default=0,
        help="Device ID for CUDA GPU / NPU (default: 0)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=None,
        help="Max eval rounds (default: from task config)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (default: from api_config.yaml or env var)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="API base URL (default: from api_config.yaml or auto per provider)",
    )
    parser.add_argument(
        "--provider", default=None, choices=["anthropic", "openai"],
        help="LLM provider (default: from api_config.yaml or auto-detect)",
    )
    parser.add_argument(
        "--reasoning-effort", default=None, choices=["low", "medium", "high"],
        help="Reasoning effort for OpenAI models (default: from api_config.yaml)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose tool-call logging",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previous session from the task directory",
    )
    parser.add_argument(
        "--context-limit", type=int, default=None,
        help=(
            "Model context window size in tokens. Required to enable auto-compression "
            "(e.g. 200000 for Claude, 128000 for GPT-4). "
            f"config: {file_cfg.get('context_limit', 'not set')}"
        ),
    )
    parser.add_argument(
        "--compression-threshold", type=float, default=None,
        help=(
            "Fraction of context_limit at which to trigger compression (default: 0.75). "
            f"config: {file_cfg.get('compression_threshold', 'not set')}"
        ),
    )

    args = parser.parse_args()

    # Merge: CLI flags > api_config.yaml > hardcoded defaults
    model = args.model or file_cfg.get("model", "claude-sonnet-4-6")
    api_key = args.api_key or file_cfg.get("api_key")
    base_url = args.base_url or file_cfg.get("base_url")
    provider = args.provider or file_cfg.get("provider")
    reasoning_effort = args.reasoning_effort or file_cfg.get("reasoning_effort")

    # Validate task directory
    task_dir = os.path.abspath(args.task)
    if not os.path.isdir(task_dir):
        print(f"Error: task directory not found: {task_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(os.path.join(task_dir, "task.yaml")):
        print(f"Error: no task.yaml in {task_dir}", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet

    from .loop import AgentLoop

    loop = AgentLoop(
        task_dir=task_dir,
        model=model,
        device_id=args.device_id,
        max_rounds=args.max_rounds,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        reasoning_effort=reasoning_effort,
        verbose=verbose,
        resume=args.resume,
    )
    # CLI / yaml 覆盖压缩配置到 config.agent（优先级: CLI > yaml > AgentConfig 默认值）
    if args.context_limit is not None:
        loop.config.agent.context_limit = args.context_limit
    elif file_cfg.get("context_limit") is not None:
        loop.config.agent.context_limit = int(file_cfg["context_limit"])
    if args.compression_threshold is not None:
        loop.config.agent.compression_threshold = args.compression_threshold
    elif file_cfg.get("compression_threshold") is not None:
        loop.config.agent.compression_threshold = float(file_cfg["compression_threshold"])

    if verbose:
        ctx = loop.config.agent.context_limit
        ctx_str = f"{ctx:,}" if ctx else "not set (compression disabled)"
        print(f"[Config] model={model}, provider={provider or 'auto'}, "
              f"base_url={base_url or 'default'}, context_limit={ctx_str}", flush=True)
    result = asyncio.run(loop.run())
    print(f"\nDone: eval_rounds={result['eval_rounds']}, best={result['best_metrics']}")


if __name__ == "__main__":
    main()
