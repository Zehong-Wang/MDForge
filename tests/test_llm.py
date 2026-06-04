"""Tests for LLM client interface (no real API calls)."""

from __future__ import annotations

import pytest

from src.llm import LLMClient, LLMMessage, LLMResponse, LLMRole, NoLLMClient


def test_message_round_trip():
    m = LLMMessage(role=LLMRole.USER, content="hello")
    blob = m.model_dump_json()
    m2 = LLMMessage.model_validate_json(blob)
    assert m == m2


def test_response_optional_fields():
    r = LLMResponse(content="hi", model="claude-sonnet-4-6")
    assert r.input_tokens is None
    assert r.output_tokens is None


def test_no_llm_client_raises_on_complete():
    client = NoLLMClient()
    with pytest.raises(RuntimeError):
        client.complete([LLMMessage(role=LLMRole.USER, content="hi")])


def test_no_llm_client_satisfies_protocol():
    # `runtime_checkable` protocol; NoLLMClient should pass isinstance.
    assert isinstance(NoLLMClient(), LLMClient)
