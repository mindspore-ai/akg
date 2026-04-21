"""
AkgLLMAdapter — Wraps AKG's LLMClient so autoresearch's agent loop
(AgentLoop, TurnExecutor, compress, subagents) can talk to it with a
stable interface.

AKG LLMClient uses OpenAI-compatible format internally; autoresearch
uses Anthropic-style messages (tool_use / tool_result blocks). This
adapter bridges the two formats and adds the retry / response-parsing
helpers the agent loop needs on top of ``LLMClient``.
"""

import asyncio
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class AkgLLMAdapter:
    """The single LLM access point for the autoresearch agent loop.

    Wraps AKG's LLMClient (OpenAI-compatible) and exposes the surface
    AgentLoop / TurnExecutor / compress / subagents consume:
      - ``call(system_prompt, messages, tools, **kwargs)`` with retry
      - ``extract_tool_calls`` / ``append_assistant`` / ``get_stop_reason``
        / ``get_response_text`` — response-parsing helpers
      - ``check_connection`` — reachability probe
      - ``client`` / ``model`` / ``provider`` properties (read by
        ``compress.py`` and logging)
    """

    def __init__(self, llm_client, *, fast_client=None,
                 retry_initial_backoff: float = 5.0,
                 retry_max_backoff_rate_limit: float = 120.0,
                 retry_max_backoff_other: float = 60.0,
                 max_retries: int = 5,
                 connection_check_timeout: float = 15.0,
                 verbose: bool = True):
        """
        Args:
            llm_client: Main AKG LLMClient instance (from create_llm_client()).
                        Used for the ReAct agent loop.
            fast_client: Optional LLMClient bound to the "fast" model level.
                        ``call(compact=True, ...)`` routes here — non-thinking,
                        low-latency tasks (auto_compact summaries, keyword bag
                        generation). If None, falls back to ``llm_client``.
            retry_initial_backoff: Initial backoff delay (seconds) on failure.
            retry_max_backoff_rate_limit: Backoff cap for RateLimitError.
            retry_max_backoff_other: Backoff cap for other transient errors.
            max_retries: Max attempts per ``call`` (callers can override
                         per-call via ``max_retries`` kwarg).
            connection_check_timeout: check_connection default timeout (seconds).
            verbose: Print progress messages.
        """
        self._llm = llm_client
        self._fast_llm = fast_client or llm_client
        self._retry_initial_backoff = retry_initial_backoff
        self._retry_max_backoff_rate_limit = retry_max_backoff_rate_limit
        self._retry_max_backoff_other = retry_max_backoff_other
        self._max_retries = max_retries
        self._connection_check_timeout = connection_check_timeout
        self.verbose = verbose

    # -- Properties (read by compress / logging) -------------------------------

    @property
    def client(self):
        """AsyncOpenAI instance — used by compress.py auto_compact."""
        return self._llm.provider.client

    @property
    def model(self) -> str:
        return self._llm.provider.model_name

    @property
    def provider(self) -> str:
        return "openai"

    # -- LLM call -------------------------------------------------------------

    async def call(self, system_prompt: str, messages: list, tools=None,
                   **kwargs):
        """Call the LLM. Returns a response dict (AKG LLMClient.generate format).

        Converts autoresearch internal messages (Anthropic-style) to
        OpenAI format, then calls LLMClient.generate().

        Special kwargs consumed by adapter (not forwarded to provider):
          compact=True: routes the call to the fast LLM client (bound to
                        the "fast" model level in settings.json, typically
                        configured without ``extra_body.thinking``) and
                        caps max_tokens=4000. Used by auto_compact
                        summaries and keyword bag generation.

        ``tools`` is required (pass ``[]`` to disable tool-calling for a
        given call). The adapter deliberately does NOT reach back into
        ``..agent.tools`` for a default — tools are a caller concern,
        and the adapter must stay a pure format bridge.
        """
        # compact mode: adapter picks the fast client and caps output.
        # max_tokens uses setdefault so callers can override via config.
        compact_mode = kwargs.pop("compact", False)
        max_retries = kwargs.pop("max_retries", None)
        if compact_mode:
            kwargs.setdefault("max_tokens", 4000)

        llm = self._fast_llm if compact_mode else self._llm
        oai_messages = self._convert_messages(system_prompt, messages)
        oai_tools = self._convert_tools(tools) if tools else []

        async def _do_call():
            return await llm.generate(
                messages=oai_messages,
                stream=False,
                tools=oai_tools if oai_tools else None,
                **kwargs,
            )

        return await self._retry_with_backoff(_do_call, max_retries=max_retries)

    # -- Retry ---------------------------------------------------------------

    async def _retry_with_backoff(self, fn, max_retries: Optional[int] = None):
        """Exponential-backoff retry around ``fn``.

        ``fn`` is an async zero-arg callable (the concrete provider call).
        Rate-limit errors get a longer backoff cap than transient errors.
        ``max_retries=0`` still invokes ``fn`` once — it caps the number
        of additional retries, matching long-running agent-loop
        expectations.
        """
        if max_retries is None:
            max_retries = self._max_retries
        max_attempts = max(max_retries, 1)

        try:
            from openai import RateLimitError
        except ImportError:
            RateLimitError = None  # fallback: all errors treated as "other"

        backoff = self._retry_initial_backoff
        model_label = getattr(self._llm.provider, "model_name", "?")
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            t0 = time.monotonic()
            if self.verbose:
                print(
                    f"  [LLM] calling {model_label} "
                    f"(attempt {attempt}/{max_retries}) …",
                    flush=True,
                )
            try:
                response = await fn()
            except Exception as e:
                last_exc = e
                etype = type(e).__name__
                logger.warning(f"{etype} on attempt {attempt}: {e}")
                if attempt == max_attempts:
                    raise
                cap = (
                    self._retry_max_backoff_rate_limit
                    if RateLimitError is not None and isinstance(e, RateLimitError)
                    else self._retry_max_backoff_other
                )
                if self.verbose:
                    print(
                        f"  [LLM] {etype} — waiting {backoff:.0f}s …",
                        flush=True,
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, cap)
                continue
            elapsed = time.monotonic() - t0
            if self.verbose:
                print(f"  [LLM] response in {elapsed:.1f}s", flush=True)
            return response
        # Defensive: unreachable — the loop always raises or returns.
        raise last_exc if last_exc is not None else RuntimeError("retry exhausted")

    # -- Response parsing -----------------------------------------------------

    def extract_tool_calls(self, response: dict) -> list[dict]:
        """Extract tool calls from LLMClient.generate() response dict."""
        raw_calls = response.get("tool_calls") or []
        calls = []
        for tc in raw_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                calls.append({
                    "tool_use_id": tc.get("id", ""),
                    "tool_name": func.get("name", ""),
                    "arguments": args,
                })
            else:
                # Handle OpenAI ChatCompletionMessageToolCall objects
                func = tc.function
                args = func.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                calls.append({
                    "tool_use_id": tc.id,
                    "tool_name": func.name,
                    "arguments": args,
                })
        return calls

    def get_stop_reason(self, response: dict) -> str:
        """Map finish_reason to autoresearch convention."""
        reason = response.get("finish_reason", "stop")
        if reason == "tool_calls":
            return "tool_use"
        return "end_turn"

    def append_assistant(self, messages: list, response: dict):
        """Append assistant turn to messages in autoresearch internal format.

        Converts OpenAI-style response to Anthropic-style content blocks
        so that subsequent _convert_messages() round-trips correctly.
        """
        content_blocks = []

        text = response.get("content") or ""
        if text:
            content_blocks.append(_TextBlock(text))

        raw_calls = response.get("tool_calls") or []
        for tc in raw_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                content_blocks.append(_ToolUseBlock(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    input=args,
                ))
            else:
                func = tc.function
                args = func.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                content_blocks.append(_ToolUseBlock(
                    id=tc.id, name=func.name, input=args,
                ))

        messages.append({
            "role": "assistant",
            "content": content_blocks if content_blocks else [_TextBlock("")],
        })

    def get_response_text(self, response: dict) -> str:
        return response.get("content") or ""

    # -- Connection check -----------------------------------------------------

    async def check_connection(self, timeout: float | None = None, verbose: bool = True):
        """Verify LLM endpoint is reachable via a minimal probe."""
        if timeout is None:
            timeout = self._connection_check_timeout
        if verbose:
            print("[LLM] Checking AKG LLM endpoint …", flush=True)
        try:
            await self._fast_llm.generate(
                messages=[{"role": "user", "content": "ping"}],
                stream=False,
                max_tokens=1,
                timeout=timeout,
            )
            if verbose:
                print("[LLM] AKG LLM endpoint OK", flush=True)
        except Exception as e:
            raise RuntimeError(f"Cannot reach AKG LLM endpoint: {e}") from e

    # -- Internal conversion --------------------------------------------------

    def _convert_messages(self, system_prompt: str, messages: list) -> list[dict]:
        """Convert autoresearch internal messages (Anthropic-style) to OpenAI format.

        Preserves Anthropic content-block semantics through the
        conversion so the agent loop can keep pairing tool_use and
        tool_result blocks correctly across turns.
        """
        oai_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if isinstance(content, str):
                    oai_messages.append(msg)
                elif isinstance(content, list):
                    # Tool results: convert from Anthropic to OpenAI format
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "tool_result":
                            oai_messages.append({
                                "role": "tool",
                                "tool_call_id": part["tool_use_id"],
                                "content": part.get("content", ""),
                            })

            elif role == "assistant":
                if isinstance(content, list):
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if hasattr(block, "type"):
                            if block.type == "text":
                                text_parts.append(block.text)
                            elif block.type == "tool_use":
                                tool_calls.append({
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": json.dumps(block.input),
                                    },
                                })
                    oai_msg = {"role": "assistant"}
                    if text_parts:
                        oai_msg["content"] = " ".join(text_parts)
                    if tool_calls:
                        oai_msg["tool_calls"] = tool_calls
                    oai_messages.append(oai_msg)
                else:
                    oai_messages.append(msg)

        return oai_messages

    @staticmethod
    def _convert_tools(tools: list) -> list[dict]:
        """Convert autoresearch tool schemas (Anthropic input_schema) to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]


# -- Lightweight block objects for Anthropic-style content --------------------
# These mimic Anthropic SDK's ContentBlock objects so that autoresearch's
# message conversion round-trips correctly (hasattr(block, "type") checks).

class _TextBlock:
    __slots__ = ("type", "text")

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    __slots__ = ("type", "id", "name", "input")

    def __init__(self, id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input
