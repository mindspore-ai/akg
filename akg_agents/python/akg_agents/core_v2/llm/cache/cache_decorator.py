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

# 装饰器2次修改
import functools
import logging
from typing import Any, Dict, Optional

from .llm_cache import LLMCache
from .cache_config import load_cache_config
from .cache_utils import generate_cache_key

logger = logging.getLogger(__name__)


def _build_session_cache_key(cache_mode: str, session_hash: str, agent_hash: str) -> Optional[str]:
    mode = (cache_mode or "off").strip().lower()
    if mode not in {"record", "replay"}:
        return None
    if not session_hash or not agent_hash:
        return None
    return f"{session_hash}:{agent_hash}"


def _resolve_cache_params(client: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        base = getattr(client, "default_config", {}) or {}
        return {**base, **kwargs}
    except Exception:
        return dict(kwargs)


def _replay_cached_stream(client: Any, agent_name: str, cached_result: Dict[str, Any]) -> None:
    """Replay cached result to stream channel so stream callers keep UI behavior."""
    reasoning = cached_result.get("reasoning_content", "") if isinstance(cached_result, dict) else ""
    content = cached_result.get("content", "") if isinstance(cached_result, dict) else ""

    try:
        if reasoning and getattr(client, "display_reasoning", True):
            client._safe_send_stream(agent_name, reasoning, is_reasoning=True)
        if content and getattr(client, "display_content", True):
            client._safe_send_stream(agent_name, content, is_reasoning=False)
        if hasattr(client, "_safe_send_display"):
            client._safe_send_display("")
    except Exception as exc:
        logger.warning(f"Failed to replay cached stream: {exc}")


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
        cache_mode = str(kwargs.pop("cache_mode", "off") or "off").strip().lower()
        cache_session_hash = str(kwargs.pop("cache_session_hash", "") or "").strip()
        cache_agent_hash = str(kwargs.pop("cache_agent_hash", "") or "").strip()
        cache_enable = cache.enable if cache_enable is None else cache_enable

        if not cache_enable:
            return await original_generate(
                messages, stream=stream, agent_name=agent_name, tools=tools, **kwargs
            )

        if cache_mode not in {"off", "record", "replay"}:
            logger.warning(f"Unknown cache_mode={cache_mode}, fallback to off")
            cache_mode = "off"

        if cache_mode == "off":
            return await original_generate(
                messages, stream=stream, agent_name=agent_name, tools=tools, **kwargs
            )

        cache_params = _resolve_cache_params(client, kwargs)
        session_cache_key = _build_session_cache_key(cache_mode, cache_session_hash, cache_agent_hash)
        content_cache_key = generate_cache_key(messages, tools, **cache_params)

        try:
            if not cache_refresh:
                cached = None
                if session_cache_key:
                    cached = cache.get(messages, tools=tools, cache_key=session_cache_key, **cache_params)
                if cached is None:
                    cached = cache.get(messages, tools=tools, cache_key=content_cache_key, **cache_params)
                if cached is not None:
                    if stream:
                        _replay_cached_stream(client, agent_name, cached)
                    return cached
                if cache_mode == "replay":
                    raise RuntimeError(
                        "Replay cache miss for session/content keys; aborting to avoid live LLM call"
                    )
        except Exception as exc:
            logger.warning(f"LLM cache lookup failed: {exc}")
            if cache_mode == "replay":
                raise

        result = await original_generate(
            messages, stream=stream, agent_name=agent_name, tools=tools, **kwargs
        )

        if cache_mode == "record":
            try:
                if session_cache_key:
                    cache.set(messages, result, tools=tools, cache_key=session_cache_key, **cache_params)
                cache.set(messages, result, tools=tools, cache_key=content_cache_key, **cache_params)
            except Exception as exc:
                logger.warning(f"LLM cache write failed: {exc}")

        return result

    client.generate = cached_generate
    return client
