"""Tests for the Layer-2 expert agents (using a fake LLM client)."""

from __future__ import annotations

from src.llm.base import LLMResponse
from src.models import HostGuestSpec, Layer3Output, Pipeline
from src.verifiers.layer2.experts import (
    AnalysisExpert,
    DEFAULT_EXPERT_CLASSES,
    ForceFieldExpert,
    RestraintExpert,
    SamplingExpert,
)


_RESPONSE_PASS = """\
```json
{
  "label": "pass",
  "confidence": 0.8,
  "concerns": [],
  "reasoning": "Looks fine."
}
```
"""

_RESPONSE_FAIL = """\
```json
{
  "label": "fail",
  "confidence": 0.9,
  "concerns": [
    {"severity": 0.9, "description": "missing parameterisation", "suggested_focus": "force_field"}
  ],
  "reasoning": "No force-field assignment in the code."
}
```
"""


class _FakeLLM:
    def __init__(self, response_content: str):
        self._response = response_content
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, messages, *, model=None, max_tokens=4096, temperature=0.0, system=None, caller_tag=None):
        self.last_system = system
        self.last_user = messages[0].content if messages else None
        return LLMResponse(content=self._response, model=model or "fake")


def _pipeline() -> Pipeline:
    return Pipeline(
        target=HostGuestSpec(host_id="cb7", guest_id="adz"),
        code_files={"run.py": "import openmm\nprint('ok')\n"},
        entry_point="run.py",
        rationale="alchemical FEP with GAFF2 + TIP3P.",
    )


def test_default_expert_set_has_three_distinct_names():
    names = {cls.NAME for cls in DEFAULT_EXPERT_CLASSES}
    assert names == {"force_field", "sampling", "analysis"}


def test_force_field_expert_pre_eval_pass():
    expert = ForceFieldExpert(llm_client=_FakeLLM(_RESPONSE_PASS))
    vote = expert.pre_eval(_pipeline())
    assert vote.expert == "force_field"
    assert vote.label.value == "pass"
    assert vote.confidence == 0.8


def test_sampling_expert_pre_eval_fail_with_concern():
    expert = SamplingExpert(llm_client=_FakeLLM(_RESPONSE_FAIL))
    vote = expert.pre_eval(_pipeline())
    assert vote.label.value == "fail"
    assert len(vote.concerns) == 1
    assert vote.concerns[0].expert == "sampling"


def test_post_eval_passes_layer3_output_into_prompt():
    fake = _FakeLLM(_RESPONSE_PASS)
    expert = AnalysisExpert(llm_client=fake)
    layer3 = Layer3Output(
        delta_g_kcal_per_mol=-14.0,
        delta_g_uncertainty=0.4,
        exited_normally=True,
        wall_time_seconds=120.0,
        diagnostics={"window_overlap": "ok"},
    )
    expert.post_eval(_pipeline(), layer3_output=layer3)
    assert "delta_g_kcal_per_mol: -14.0" in fake.last_user
    assert "wall_time_seconds: 120.0" in fake.last_user
    assert "window_overlap" in fake.last_user


def test_restraint_expert_sees_peer_votes_in_user_prompt():
    fake = _FakeLLM(_RESPONSE_PASS)
    expert = RestraintExpert(llm_client=fake)
    # Build a peer vote with one concern.
    other_vote_response = _RESPONSE_FAIL
    peer_expert = ForceFieldExpert(llm_client=_FakeLLM(other_vote_response))
    peer_vote = peer_expert.pre_eval(_pipeline())

    expert.pre_eval(_pipeline(), peer_votes=[peer_vote])
    assert "Other experts'" in fake.last_user
    assert "force_field" in fake.last_user
    assert "missing parameterisation" in fake.last_user


def test_expert_records_debug_info():
    fake = _FakeLLM(_RESPONSE_PASS)
    expert = ForceFieldExpert(llm_client=fake)
    vote = expert.pre_eval(_pipeline())
    debug = expert.last_debug
    assert debug is not None
    assert debug.expert == "force_field"
    assert debug.parsed.label == vote.label
    assert "force-field expert" in debug.system_prompt.lower()


def test_system_prompt_includes_shared_output_contract():
    expert = ForceFieldExpert(llm_client=_FakeLLM(_RESPONSE_PASS))
    expert.pre_eval(_pipeline())
    # The {shared_output} placeholder should have been substituted with the
    # real output contract.
    assert "{shared_output}" not in expert.last_debug.system_prompt
    assert "Output contract" in expert.last_debug.system_prompt
    assert "label" in expert.last_debug.system_prompt
