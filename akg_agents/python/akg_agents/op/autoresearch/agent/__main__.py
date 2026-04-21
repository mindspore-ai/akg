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

"""
CLI entry point for the programmatic agent (debug path).

Usage:
    # Uses settings.json (project ``.akg/settings.json`` or user
    # ``~/.akg/settings.json``) for model / provider / key / endpoint.
    python -m akg_agents.op.autoresearch.agent --task tasks/op_migrate/relu_rmsnorm

    # Override the model level used for the main agent loop.
    python -m akg_agents.op.autoresearch.agent --task ... --model-level complex

    # One-shot override without touching settings.json — translated to
    # the corresponding AKG_AGENTS_* env vars for this process.
    python -m akg_agents.op.autoresearch.agent \
        --task ... --model deepseek-chat \
        --base-url https://... --api-key sk-...

The production path goes through ``autoresearch_workflow.py`` which
builds the same ``AkgLLMAdapter`` on top of ``LLMClient``; this entry
exists only for standalone debugging.
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


def _apply_env_overrides(model: str, api_key: str, base_url: str) -> None:
    """Translate CLI / api_config.yaml values into AKG_AGENTS_* env vars.

    ``create_llm_client`` reads from settings.json, which in turn falls
    back to AKG_AGENTS_* env vars — so setting them here lets the debug
    entry override the resolved config without touching settings.json.
    Only non-empty values overwrite.
    """
    if model:
        os.environ["AKG_AGENTS_MODEL_NAME"] = model
    if api_key:
        os.environ["AKG_AGENTS_API_KEY"] = api_key
    if base_url:
        os.environ["AKG_AGENTS_BASE_URL"] = base_url


def main():
    file_cfg = _load_api_config()

    parser = argparse.ArgumentParser(
        description="Autoresearch — Programmatic Agent (debug)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--task", metavar="DIR", required=True,
                        help="Task directory path")
    parser.add_argument("--device-id", type=int, default=0,
                        help="Device ID for CUDA GPU / NPU (default: 0)")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Max eval rounds (default: from task config)")
    parser.add_argument("--model-level", default="coder",
                        choices=["complex", "standard", "fast", "coder"],
                        help="settings.json model level for the main agent "
                             "loop (default: coder)")
    parser.add_argument("--fast-model-level", default="fast",
                        help="settings.json model level for compact / "
                             "keyword-generation calls (default: fast)")
    parser.add_argument("--model", default=None,
                        help="Override model name; sets AKG_AGENTS_MODEL_NAME "
                             f"(config: {file_cfg.get('model', 'not set')})")
    parser.add_argument("--api-key", default=None,
                        help="Override API key; sets AKG_AGENTS_API_KEY "
                             "(default: settings.json / env)")
    parser.add_argument("--base-url", default=None,
                        help="Override API base URL; sets AKG_AGENTS_BASE_URL "
                             f"(config: {file_cfg.get('base_url', 'not set')})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose tool-call logging")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous session from the task directory")
    parser.add_argument("--context-limit", type=int, default=None,
                        help="Model context window size in tokens "
                             "(e.g. 200000 for Claude, 128000 for GPT-4). "
                             f"config: {file_cfg.get('context_limit', 'not set')}")
    parser.add_argument("--compression-threshold", type=float, default=None,
                        help="Fraction of context_limit that triggers "
                             "compression (default: 0.75). "
                             f"config: {file_cfg.get('compression_threshold', 'not set')}")

    args = parser.parse_args()

    # Merge: CLI > api_config.yaml > settings.json (env vars)
    model = args.model or file_cfg.get("model") or ""
    api_key = args.api_key or file_cfg.get("api_key") or ""
    base_url = args.base_url or file_cfg.get("base_url") or ""
    _apply_env_overrides(model, api_key, base_url)

    task_dir = os.path.abspath(args.task)
    if not os.path.isdir(task_dir):
        print(f"Error: task directory not found: {task_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(os.path.join(task_dir, "task.yaml")):
        print(f"Error: no task.yaml in {task_dir}", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet

    from akg_agents.core_v2.llm.factory import create_llm_client
    from ..adapters.llm_adapter import AkgLLMAdapter
    from .loop import AgentLoop

    llm_client = create_llm_client(model_level=args.model_level)
    try:
        fast_llm_client = create_llm_client(model_level=args.fast_model_level)
    except Exception as exc:
        if verbose:
            print(
                f"[Config] fast model level '{args.fast_model_level}' "
                f"unavailable ({exc}); compact calls will reuse the main "
                f"client.",
                flush=True,
            )
        fast_llm_client = None
    llm_adapter = AkgLLMAdapter(llm_client, fast_client=fast_llm_client,
                                verbose=verbose)

    loop = AgentLoop(
        task_dir=task_dir,
        llm_adapter=llm_adapter,
        device_id=args.device_id,
        max_rounds=args.max_rounds,
        verbose=verbose,
        resume=args.resume,
    )

    # CLI / yaml overrides for compression config (CLI > yaml > AgentConfig default)
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
        print(f"[Config] model={llm_adapter.model}, "
              f"level={args.model_level}/{args.fast_model_level}, "
              f"context_limit={ctx_str}", flush=True)

    result = asyncio.run(loop.run())
    print(f"\nDone: eval_rounds={result['eval_rounds']}, "
          f"best={result['best_metrics']}")


if __name__ == "__main__":
    main()
