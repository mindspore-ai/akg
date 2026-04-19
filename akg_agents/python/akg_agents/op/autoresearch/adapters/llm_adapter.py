"""
AkgLLMAdapter — Wraps AKG's LLMClient to be duck-type compatible with
autoresearch's ConversationAdapter interface.

AKG LLMClient uses OpenAI-compatible format internally.
Autoresearch uses Anthropic-style messages internally (tool_use/tool_result blocks).
This adapter bridges the two formats.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AkgLLMAdapter:
    """Duck-type compatible with ConversationAdapter.

    Wraps AKG's LLMClient (OpenAI-compatible) so that autoresearch's
    AgentLoop, TurnExecutor, and compress module can use it transparently.
    """

    def __init__(self, llm_client, *, connection_check_timeout: float = 15.0,
                 verbose: bool = True):
        """
        Args:
            llm_client: AKG LLMClient instance (from create_llm_client()).
            connection_check_timeout: check_connection default timeout (seconds).
            verbose: Print progress messages.
        """
        self._llm = llm_client
        self._connection_check_timeout = connection_check_timeout
        self.verbose = verbose

    # -- Properties (duck-type ConversationAdapter) ---------------------------

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
          compact=True: caps max_tokens=4000, suppresses default extra_body
                        (disables thinking/reasoning for compact summarizer)

        ``tools`` is required (pass ``[]`` to disable tool-calling for a
        given call). The adapter deliberately does NOT reach back into
        ``..agent.tools`` for a default — tools are a caller concern,
        and the adapter must stay a pure format bridge.
        """
        # compact mode: adapter consumes, sets provider-level overrides
        # max_tokens uses setdefault so callers can override via config
        compact_mode = kwargs.pop("compact", False)
        # Autoresearch's ConversationAdapter supports per-call retry override.
        # AKG's LLMClient currently does not, so consume the kwarg here to
        # keep the duck-typed interface compatible.
        kwargs.pop("max_retries", None)
        if compact_mode:
            kwargs.setdefault("max_tokens", 4000)
            kwargs["suppress_extra_body"] = True

        oai_messages = self._convert_messages(system_prompt, messages)
        oai_tools = self._convert_tools(tools) if tools else []

        result = await self._llm.generate(
            messages=oai_messages,
            stream=False,
            tools=oai_tools if oai_tools else None,
            **kwargs,
        )
        return result

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
            await self._llm.generate(
                messages=[{"role": "user", "content": "ping"}],
                stream=False,
                max_tokens=1,
                timeout=timeout,
                suppress_extra_body=True,
            )
            if verbose:
                print("[LLM] AKG LLM endpoint OK", flush=True)
        except Exception as e:
            raise RuntimeError(f"Cannot reach AKG LLM endpoint: {e}") from e

    # -- Internal conversion --------------------------------------------------

    def _convert_messages(self, system_prompt: str, messages: list) -> list[dict]:
        """Convert autoresearch internal messages (Anthropic-style) to OpenAI format.

        Mirrors ConversationAdapter._call_openai message conversion logic.
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
