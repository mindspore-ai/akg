import functools
import logging
from typing import Any, Dict, Optional

from .llm_cache import LLMCache
from .cache_config import load_cache_config

logger = logging.getLogger(__name__)


def _resolve_cache_params(client: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        base = getattr(client, "default_config", {}) or {}
        return {**base, **kwargs}
    except Exception:
        return dict(kwargs)


def attach_cache_to_client(
    client: Any,
    cache: Optional[LLMCache] = None,
    cache_config_path: Optional[str] = None,
    enable_cache: Optional[bool] = None,
) -> Any:
    """Attach cache to a client instance without changing its interface."""
    if cache is None:
        cache_config = load_cache_config(cache_config_path)
        if enable_cache is not None:
            cache_config["enable"] = enable_cache
        cache = LLMCache(**cache_config)
    elif enable_cache is not None:
        cache.enable = enable_cache

    if not cache.enable:
        logger.info("LLM cache disabled; skipping attach.")
        return client

    original_generate = client.generate

    @functools.wraps(original_generate)
    async def cached_generate(
        messages,
        stream: bool = False,
        agent_name: str = "",
        tools=None,
        **kwargs
    ):
        cache_enable = kwargs.pop("cache_enable", None)
        cache_refresh = kwargs.pop("cache_refresh", False)
        cache_enable = cache.enable if cache_enable is None else cache_enable

        if not cache_enable or stream:
            return await original_generate(
                messages, stream=stream, agent_name=agent_name, tools=tools, **kwargs
            )

        cache_params = _resolve_cache_params(client, kwargs)

        try:
            if not cache_refresh:
                cached = cache.get(messages, tools=tools, **cache_params)
                if cached is not None:
                    return cached
        except Exception as exc:
            logger.warning(f"LLM cache lookup failed: {exc}")

        result = await original_generate(
            messages, stream=stream, agent_name=agent_name, tools=tools, **kwargs
        )

        try:
            cache.set(messages, result, tools=tools, **cache_params)
        except Exception as exc:
            logger.warning(f"LLM cache write failed: {exc}")

        return result

    client.generate = cached_generate
    return client
