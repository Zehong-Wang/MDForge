"""LLM client abstraction for MDForge.

A pluggable interface so the Code agent and the expert verifiers can target
different model backends (Anthropic, Claude Code subscription, local)
without the rest of the codebase caring.

Available backends:
  - `ClaudeSDKClient` — wraps `claude_agent_sdk`, uses the local Claude Code
     subscription. No API key needed. Preferred for development.
  - `AnthropicClient` — direct Anthropic API. Requires ANTHROPIC_API_KEY.
  - `NoLLMClient` — refuses all calls; injected into tests that should
     never reach a real model.
"""

from src.llm.base import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMRole,
    NoLLMClient,
)

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "LLMRole",
    "NoLLMClient",
    "ClaudeSDKClient",
    "AnthropicClient",
]


def __getattr__(name):
    # Lazy import to avoid pulling claude_agent_sdk / anthropic on every
    # import of this module. Useful for environments where one backend is
    # installed but not the other.
    if name == "ClaudeSDKClient":
        from src.llm.claude_sdk_client import ClaudeSDKClient
        return ClaudeSDKClient
    if name == "AnthropicClient":
        from src.llm.anthropic_client import AnthropicClient
        return AnthropicClient
    raise AttributeError(name)
