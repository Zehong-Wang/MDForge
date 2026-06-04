"""Claude Agent SDK backend for the LLMClient interface.

Uses the user's Claude Code subscription (via `claude_agent_sdk`) instead of
the raw Anthropic API. No ANTHROPIC_API_KEY needed; auth is whatever the
locally installed `claude` CLI uses.

The SDK is async-native; we wrap it in `asyncio.run` so callers can use the
sync `LLMClient.complete` interface unchanged.

We pin `allowed_tools=[]` so the SDK only generates text — it must not
spawn its own tool calls (file ops, bash, etc.) for our Writer/expert
prompts. Caller code, not the SDK agent, is what runs MD and edits files.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

_log = logging.getLogger(__name__)

from src.llm.base import LLMMessage, LLMResponse, LLMRole
from src.llm.usage_logger import get_global_usage_logger

DEFAULT_MODEL = os.environ.get("MD_AGENT_DEFAULT_MODEL", "claude-opus-4-7")


class ClaudeSDKClient:
    """LLMClient backed by Claude Agent SDK (uses local `claude` subscription)."""

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        cwd: str | None = None,
        allowed_tools: list[str] | None = None,
    ):
        self._default_model = default_model
        self._cwd = cwd
        # Default = text-only (no tool use). Pass e.g.
        # ["WebSearch", "WebFetch"] to enable literature recon.
        self._allowed_tools: list[str] = list(allowed_tools or [])

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        system: str | None = None,
        caller_tag: str | None = None,
    ) -> LLMResponse:
        msgs = list(messages)
        # Promote a SYSTEM-roled first message into the system kwarg.
        if msgs and msgs[0].role is LLMRole.SYSTEM:
            if system is None:
                system = msgs[0].content
            msgs = msgs[1:]

        # Concatenate USER and ASSISTANT history into a single prompt string.
        # The SDK's one-shot query() takes a single prompt; we encode roles
        # explicitly so the model can see prior turns.
        prompt_parts: list[str] = []
        for m in msgs:
            tag = m.role.value.upper()
            prompt_parts.append(f"[{tag}]\n{m.content}")
        prompt = "\n\n".join(prompt_parts) if prompt_parts else ""

        options = ClaudeAgentOptions(
            model=model or self._default_model,
            system_prompt=system if system else None,
            # `max_turns` caps total assistant turns (each Claude response
            # counts as one). With `allowed_tools=[]` the model shouldn't
            # take tool-use turns, but long generations can spill into a
            # continuation turn; for Writer-with-WebSearch we need many
            # more turns because each tool call is a turn.
            max_turns=40 if self._allowed_tools else 20,
            allowed_tools=self._allowed_tools,
            cwd=self._cwd,
            permission_mode="bypassPermissions",
        )

        # Bundled CLI/SDK can throw transient "Claude Code returned an
        # error result: success" or stream-interruption errors. Retry a
        # few times so a single transient blip doesn't kill a long run.
        # We use 5 attempts: these errors have highly variable latency
        # (intermittent infrastructure, not a hard limit), so extra attempts
        # have a real chance of getting through.
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                t0 = time.time()
                resp = asyncio.run(_run_query(prompt, options))
                latency = time.time() - t0
                # Log per-call usage if a global logger is configured.
                logger = get_global_usage_logger()
                if logger is not None:
                    logger.log(
                        caller=caller_tag or "unknown",
                        model=resp.model,
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                        cache_read_input_tokens=resp.cache_read_input_tokens,
                        cache_creation_input_tokens=resp.cache_creation_input_tokens,
                        latency_s=latency,
                    )
                return resp
            except Exception as exc:
                last_exc = exc
                _log.warning(
                    "ClaudeSDKClient.complete attempt %d/5 failed: %r",
                    attempt, exc,
                )
                if attempt < 5:
                    time.sleep(5 * attempt)
        # Give up after 5 attempts.
        assert last_exc is not None
        raise last_exc


async def _run_query(prompt: str, options: ClaudeAgentOptions) -> LLMResponse:
    """Drive the async query() iterator to completion and return a normalised LLMResponse."""
    text_chunks: list[str] = []
    model_used: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    stop_reason: str | None = None

    def _u(usage, key):
        # SDK's usage object can be a dataclass-ish thing or a dict.
        return (
            getattr(usage, key, None)
            or (usage.get(key) if isinstance(usage, dict) else None)
        )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_chunks.append(block.text)
            if model_used is None:
                model_used = getattr(message, "model", None)
        elif isinstance(message, ResultMessage):
            usage = getattr(message, "usage", None)
            if usage:
                input_tokens = _u(usage, "input_tokens")
                output_tokens = _u(usage, "output_tokens")
                cache_creation_input_tokens = _u(usage, "cache_creation_input_tokens")
                cache_read_input_tokens = _u(usage, "cache_read_input_tokens")
            stop_reason = getattr(message, "subtype", None) or getattr(
                message, "stop_reason", None
            )

    if not text_chunks:
        raise RuntimeError(
            "ClaudeSDKClient: no AssistantMessage TextBlock in response. "
            "The SDK may have errored silently; try ClaudeAgentOptions(debug_stderr=True)."
        )

    return LLMResponse(
        content="".join(text_chunks),
        model=model_used or "claude-agent-sdk",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        stop_reason=stop_reason,
    )
