"""
ConversationAdapter — Provider-agnostic LLM communication layer.

Encapsulates:
  - API client creation (Anthropic / OpenAI)
  - Message format conversion between providers
  - Retry with exponential backoff
  - Tool call extraction from responses
  - Connection health check
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

from .tools import TOOLS

logger = logging.getLogger(__name__)


class ConversationAdapter:
    """Provider-agnostic wrapper around Anthropic / OpenAI async clients."""

    def __init__(
        self,
        model: str,
        provider: str,
        call_timeout: float,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        thinking_budget: int = 0,
        llm_max_tokens: int = 16_384,
        retry_initial_backoff: float = 5.0,
        retry_max_backoff_rate_limit: float = 120.0,
        retry_max_backoff_other: float = 30.0,
        verbose: bool = True,
    ):
        self.model = model
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.thinking_budget = thinking_budget
        self.verbose = verbose
        self._llm_max_tokens = llm_max_tokens
        self._retry_initial_backoff = retry_initial_backoff
        self._retry_max_backoff_rate_limit = retry_max_backoff_rate_limit
        self._retry_max_backoff_other = retry_max_backoff_other

        self.client = self._create_client(api_key, base_url, call_timeout)

    # -- Client creation ----------------------------------------------------

    def _create_client(self, api_key: Optional[str], base_url: Optional[str],
                       call_timeout: float):
        if self.provider == "openai":
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required. pip install openai")
            resolved_key = api_key or os.getenv("OPENAI_API_KEY")
            if not resolved_key:
                raise ValueError("OpenAI API key required.")
            resolved_base = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            return AsyncOpenAI(api_key=resolved_key,
                               base_url=resolved_base.rstrip("/"),
                               timeout=call_timeout)
        else:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("anthropic package required. pip install anthropic")
            resolved_key = (api_key or os.getenv("ANTHROPIC_API_KEY")
                            or os.getenv("ANTHROPIC_AUTH_TOKEN"))
            if not resolved_key:
                raise ValueError("API key required. Set ANTHROPIC_API_KEY or pass --api-key.")
            resolved_base = base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1"
            return AsyncAnthropic(api_key=resolved_key,
                                  base_url=resolved_base.rstrip("/"),
                                  timeout=call_timeout)

    # -- Retry logic --------------------------------------------------------

    async def _retry_with_backoff(self, fn, max_retries: int = 5):
        if self.provider == "openai":
            from openai import RateLimitError
        else:
            from anthropic import RateLimitError

        backoff = self._retry_initial_backoff
        for attempt in range(1, max_retries + 1):
            t0 = time.monotonic()
            if self.verbose:
                print(f"  [LLM] calling {self.provider}:{self.model} "
                      f"(attempt {attempt}/{max_retries}) …", flush=True)
            try:
                response = await fn()
            except Exception as e:
                etype = type(e).__name__
                logger.warning(f"{etype} on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise
                cap = (self._retry_max_backoff_rate_limit
                       if isinstance(e, RateLimitError)
                       else self._retry_max_backoff_other)
                if self.verbose:
                    print(f"  [LLM] {etype} — waiting {backoff:.0f}s …", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, cap)
                continue
            elapsed = time.monotonic() - t0
            if self.verbose:
                print(f"  [LLM] response in {elapsed:.1f}s", flush=True)
            return response

    # -- LLM call ----------------------------------------------------------

    async def call(self, system_prompt: str, messages: list, tools=None):
        """Call LLM with the current messages list. Returns the raw API response.

        Args:
            tools: Tool schemas to send. Defaults to the main agent TOOLS.
                   Pass a custom list for subagents with different tool sets.
        """
        if tools is None:
            tools = TOOLS
        if self.provider == "openai":
            return await self._call_openai(system_prompt, messages, tools)
        else:
            return await self._call_anthropic(system_prompt, messages, tools)

    async def _call_anthropic(self, system_prompt: str, messages: list, tools):
        max_tok = self._llm_max_tokens
        thinking_budget = self.thinking_budget

        async def _do_call():
            kwargs = dict(
                model=self.model,
                system=system_prompt,
                messages=messages,
                tools=tools,
                max_tokens=max_tok,
            )
            if thinking_budget and thinking_budget > 0:
                kwargs["max_tokens"] = thinking_budget + max_tok
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            return await self.client.messages.create(**kwargs)
        return await self._retry_with_backoff(_do_call)

    async def _call_openai(self, system_prompt: str, messages: list, tools):
        oai_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    oai_messages.append(msg)
                elif isinstance(msg["content"], list):
                    for part in msg["content"]:
                        if isinstance(part, dict) and part.get("type") == "tool_result":
                            oai_messages.append({
                                "role": "tool",
                                "tool_call_id": part["tool_use_id"],
                                "content": part.get("content", ""),
                            })
            elif msg["role"] == "assistant":
                if isinstance(msg["content"], list):
                    text_parts = []
                    tool_calls = []
                    for block in msg["content"]:
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

        oai_tools = [
            {"type": "function", "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }}
            for t in tools
        ]

        max_tok = self._llm_max_tokens
        thinking_budget = self.thinking_budget
        async def _do_call():
            kwargs = {
                "model": self.model,
                "messages": oai_messages,
                "tools": oai_tools,
                "max_tokens": max_tok,
                "store": False,
            }
            if self.reasoning_effort:
                kwargs.setdefault("extra_body", {})
                kwargs["extra_body"]["reasoning"] = {"effort": self.reasoning_effort}
            if thinking_budget and thinking_budget > 0:
                kwargs["max_tokens"] = thinking_budget + max_tok
                kwargs.setdefault("extra_body", {})
                kwargs["extra_body"]["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            return await self.client.chat.completions.create(**kwargs)

        return await self._retry_with_backoff(_do_call)

    # -- Response parsing (provider-agnostic) ------------------------------

    def extract_tool_calls(self, response) -> list[dict]:
        """Extract tool calls from LLM response.
        Returns list of {tool_use_id, tool_name, arguments}.
        """
        if self.provider == "openai":
            choice = response.choices[0]
            if choice.message.tool_calls:
                return [
                    {
                        "tool_use_id": tc.id,
                        "tool_name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
                    }
                    for tc in choice.message.tool_calls
                ]
            return []
        else:
            calls = []
            for block in response.content:
                if block.type == "tool_use":
                    calls.append({
                        "tool_use_id": block.id,
                        "tool_name": block.name,
                        "arguments": block.input or {},
                    })
            return calls

    def get_stop_reason(self, response) -> str:
        if self.provider == "openai":
            return "tool_use" if response.choices[0].finish_reason == "tool_calls" else "end_turn"
        else:
            return response.stop_reason

    def append_assistant(self, messages: list, response):
        """Append assistant turn to messages (provider-agnostic)."""
        if self.provider == "openai":
            choice = response.choices[0]
            msg = {
                "role": "assistant",
                "content": choice.message.content or "",
            }
            if choice.message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]
            messages.append(msg)
        else:
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

    def get_response_text(self, response) -> str:
        """Extract text content from response for verbose logging."""
        if self.provider == "openai":
            return response.choices[0].message.content or ""
        else:
            parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return " ".join(parts)

    # -- Connection check --------------------------------------------------

    async def check_connection(self, timeout: float = 15.0, verbose: bool = True):
        import httpx
        if verbose:
            print("[LLM] Checking endpoint …", flush=True)
        base_url = str(self.client.base_url).rstrip("/")
        try:
            headers = {}
            if self.provider == "openai":
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            else:
                headers["x-api-key"] = self.client.api_key
                headers["anthropic-version"] = "2023-06-01"
            async with httpx.AsyncClient(timeout=timeout) as http:
                resp = await http.get(f"{base_url}/models", headers=headers)
            if resp.status_code < 500:
                if verbose:
                    print(f"[LLM] Endpoint OK (HTTP {resp.status_code})", flush=True)
                return
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise RuntimeError(f"Cannot reach endpoint {base_url}: {e}") from e
        except Exception:
            pass

        # Fallback: minimal LLM probe
        if self.provider == "openai":
            from openai import APITimeoutError, APIConnectionError
            try:
                await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
            except (APITimeoutError, APIConnectionError) as e:
                raise RuntimeError(f"Cannot reach endpoint: {e}") from e
            except Exception:
                pass
        else:
            from anthropic import APITimeoutError, APIConnectionError
            try:
                await self.client.messages.create(
                    model=self.model, system="ping",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
            except (APITimeoutError, APIConnectionError) as e:
                raise RuntimeError(f"Cannot reach endpoint: {e}") from e
            except Exception:
                pass
        if verbose:
            print("[LLM] Endpoint OK", flush=True)
