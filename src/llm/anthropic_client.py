"""Anthropic SDK backend for the LLMClient interface.

Defaults to `claude-sonnet-4-6` for development and dry runs. Override
via the `model` argument or by setting the `MD_AGENT_DEFAULT_MODEL`
environment variable.

Authentication uses `ANTHROPIC_API_KEY` from the environment, the
standard SDK convention. Claude Code's subscription auth is *not*
shared with the API SDK; if shared auth is needed, swap in a
claude-agent-sdk-backed client.
"""

from __future__ import annotations

import os
import time

from src.llm.base import LLMMessage, LLMResponse, LLMRole
from src.llm.usage_logger import get_global_usage_logger

DEFAULT_MODEL = os.environ.get("MD_AGENT_DEFAULT_MODEL", "claude-sonnet-4-6")


class AnthropicClient:
    """Thin wrapper around `anthropic.Anthropic` that implements LLMClient."""

    def __init__(self, api_key: str | None = None, default_model: str = DEFAULT_MODEL):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicClient requires the `anthropic` package. "
                "Install with `pip install -e '.[llm]'`."
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._default_model = default_model

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system: str | None = None,
        caller_tag: str | None = None,
    ) -> LLMResponse:
        # The Anthropic API takes `system` as a top-level kwarg; the rest
        # of the messages go in the messages list. We accept either form
        # from callers (system arg, or a SYSTEM-roled message at index 0).
        msgs = list(messages)
        if msgs and msgs[0].role is LLMRole.SYSTEM:
            if system is None:
                system = msgs[0].content
            msgs = msgs[1:]

        api_messages = [{"role": m.role.value, "content": m.content} for m in msgs]

        kwargs = {
            "model": model or self._default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system

        t0 = time.time()
        response = self._client.messages.create(**kwargs)
        latency = time.time() - t0

        # SDK content is a list of blocks; concatenate the text blocks.
        text_chunks = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_chunks.append(block.text)
        text = "".join(text_chunks)

        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "input_tokens", None) if usage else None
        out_tokens = getattr(usage, "output_tokens", None) if usage else None

        # Log per-call usage if a global logger is configured (mirrors
        # ClaudeSDKClient so cost aggregation works across backends).
        logger = get_global_usage_logger()
        if logger is not None:
            logger.log(
                caller=caller_tag or "unknown",
                model=response.model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_s=latency,
            )

        return LLMResponse(
            content=text,
            model=response.model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            stop_reason=getattr(response, "stop_reason", None),
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )
