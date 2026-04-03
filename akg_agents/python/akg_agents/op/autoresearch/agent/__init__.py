"""
Autoresearch Programmatic Agent

使用方式:
    python -m agent --task tasks/op_migrate/relu_rmsnorm --max-rounds 20

依赖:
    pip install anthropic   (Anthropic provider)
    pip install openai      (OpenAI/Codex provider)

配置:
    agent/api_config.yaml   API 密钥和端点 (gitignored)
"""

from .loop import AgentLoop

__all__ = ["AgentLoop"]
