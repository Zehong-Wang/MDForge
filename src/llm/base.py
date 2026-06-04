"""Backend-agnostic LLM client interface."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class LLMRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    role: LLMRole
    content: str


class LLMResponse(BaseModel):
    """Normalised response shape across backends."""

    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    """Tokens written to the prompt cache (billed at ~1.25× input rate)."""
    cache_read_input_tokens: int | None = None
    """Tokens read from the prompt cache (billed at ~0.1× input rate). High
    values here mean caching is saving you money."""
    stop_reason: str | None = None
    raw: dict | None = Field(default=None, repr=False)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every backend must satisfy."""

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system: str | None = None,
        caller_tag: str | None = None,
    ) -> LLMResponse: ...


class NoLLMClient:
    """A client that refuses to call out, useful for tests and dry runs.

    Tests that should never hit a real model can inject this to make
    accidental network calls fail loudly.
    """

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
        raise RuntimeError(
            "NoLLMClient: no LLM is configured. Provide an LLM backend "
            "(ClaudeSDKClient or AnthropicClient from src.llm) before "
            "invoking model-driven components."
        )
