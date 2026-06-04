"""Tests for Layer 2 multi-expert debate orchestrator."""

from __future__ import annotations

import pytest

from src.llm.base import LLMResponse
from src.models import (
    Concern,
    ExpertVote,
    HostGuestSpec,
    LayerName,
    Layer3Output,
    Pipeline,
    VerdictLabel,
)
from src.verifiers.layer2.debate import (
    Layer2Debate,
    update_reputations_from_pre_post,
)
from src.verifiers.layer2.experts import (
    AnalysisExpert,
    ForceFieldExpert,
    RestraintExpert,
    SamplingExpert,
)
from src.verifiers.layer2.reputation import ExpertReputationTracker


def _pipeline() -> Pipeline:
    return Pipeline(
        target=HostGuestSpec(host_id="cb7", guest_id="adz"),
        code_files={"run.py": "import openmm\n"},
        entry_point="run.py",
    )


class _ScriptedLLM:
    """Returns canned responses round-by-round.

    Each call() pops the next response. Each round has 4 calls (one per
    expert) so we record per-round responses by interleaving.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, model=None, max_tokens=4096, temperature=0.0, system=None, caller_tag=None):
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of canned responses")
        content = self._responses.pop(0)
        self.calls += 1
        return LLMResponse(content=content, model=model or "fake")


def _vote_json(label: str, confidence: float = 0.7, concerns: list[dict] | None = None, reasoning: str = "") -> str:
    import json
    payload = {
        "label": label,
        "confidence": confidence,
        "concerns": concerns or [],
        "reasoning": reasoning,
    }
    return f"```json\n{json.dumps(payload)}\n```"


def _all_pass(n: int = 8) -> list[str]:
    return [_vote_json("pass", 0.8) for _ in range(n)]


def _all_fail(n: int = 8) -> list[str]:
    return [_vote_json("fail", 0.9, [{"severity": 0.9, "description": "bad"}]) for _ in range(n)]


def _experts(llm) -> list:
    return [
        ForceFieldExpert(llm_client=llm),
        SamplingExpert(llm_client=llm),
        RestraintExpert(llm_client=llm),
        AnalysisExpert(llm_client=llm),
    ]


def test_unanimous_pass_aggregates_to_pass():
    llm = _ScriptedLLM(_all_pass(8))
    rep = ExpertReputationTracker(["force_field", "sampling", "restraint", "analysis"])
    debate = Layer2Debate(experts=_experts(llm), reputation=rep)
    crit = debate.pre_eval(_pipeline())
    assert crit.layer is LayerName.LAYER_2_PRE
    assert crit.label is VerdictLabel.PASS
    assert llm.calls == 8  # 4 experts x 2 rounds


def test_unanimous_fail_aggregates_to_fail():
    llm = _ScriptedLLM(_all_fail(8))
    rep = ExpertReputationTracker(["force_field", "sampling", "restraint", "analysis"])
    debate = Layer2Debate(experts=_experts(llm), reputation=rep)
    crit = debate.pre_eval(_pipeline())
    assert crit.label is VerdictLabel.FAIL
    assert len(crit.concerns) > 0


def test_split_verdict_aggregates_to_uncertain():
    # 2 pass, 2 fail in round 1, same in round 2 -> score 0.5 -> uncertain
    responses = [
        _vote_json("pass", 0.7),
        _vote_json("pass", 0.7),
        _vote_json("fail", 0.7, [{"severity": 0.7, "description": "x"}]),
        _vote_json("fail", 0.7, [{"severity": 0.7, "description": "y"}]),
    ] * 2
    llm = _ScriptedLLM(responses)
    rep = ExpertReputationTracker(["force_field", "sampling", "restraint", "analysis"])
    debate = Layer2Debate(experts=_experts(llm), reputation=rep)
    crit = debate.pre_eval(_pipeline())
    assert crit.label is VerdictLabel.UNCERTAIN


def test_higher_reputation_dominates_aggregate():
    # Round-1 and Round-2: force_field says pass, others say fail.
    # If reputation is uniform, the aggregate is FAIL (3 vs 1).
    # If we crank force_field's reputation way up, the aggregate flips toward PASS.
    responses = [
        _vote_json("pass", 0.9),  # force_field round 1
        _vote_json("fail", 0.6, [{"severity": 0.6, "description": "x"}]),  # sampling
        _vote_json("fail", 0.6, [{"severity": 0.6, "description": "y"}]),  # restraint
        _vote_json("fail", 0.6, [{"severity": 0.6, "description": "z"}]),  # analysis
        _vote_json("pass", 0.9),  # force_field round 2
        _vote_json("fail", 0.6, [{"severity": 0.6, "description": "x"}]),
        _vote_json("fail", 0.6, [{"severity": 0.6, "description": "y"}]),
        _vote_json("fail", 0.6, [{"severity": 0.6, "description": "z"}]),
    ]

    # Uniform reputation -> FAIL.
    llm1 = _ScriptedLLM(list(responses))
    rep_uniform = ExpertReputationTracker(["force_field", "sampling", "restraint", "analysis"])
    crit_uniform = Layer2Debate(experts=_experts(llm1), reputation=rep_uniform).pre_eval(_pipeline())
    assert crit_uniform.label is VerdictLabel.FAIL

    # Highly trusted force_field -> aggregate moves toward PASS.
    llm2 = _ScriptedLLM(list(responses))
    rep_skewed = ExpertReputationTracker(["force_field", "sampling", "restraint", "analysis"])
    for _ in range(40):
        rep_skewed.update("force_field", consistent=True)
    for name in ("sampling", "restraint", "analysis"):
        for _ in range(40):
            rep_skewed.update(name, consistent=False)
    crit_skewed = Layer2Debate(experts=_experts(llm2), reputation=rep_skewed).pre_eval(_pipeline())
    # Should at minimum no longer be a FAIL; depending on weights, it's PASS or UNCERTAIN.
    assert crit_skewed.label is not VerdictLabel.FAIL


def test_post_eval_uses_layer3_output():
    llm = _ScriptedLLM(_all_pass(8))
    rep = ExpertReputationTracker(["force_field", "sampling", "restraint", "analysis"])
    debate = Layer2Debate(experts=_experts(llm), reputation=rep)
    out = Layer3Output(
        delta_g_kcal_per_mol=-13.5,
        delta_g_uncertainty=0.5,
        exited_normally=True,
    )
    crit = debate.post_eval(_pipeline(), layer3_output=out)
    assert crit.layer is LayerName.LAYER_2_POST


def test_debate_records_full_transcript():
    llm = _ScriptedLLM(_all_pass(8))
    rep = ExpertReputationTracker(["force_field", "sampling", "restraint", "analysis"])
    debate = Layer2Debate(experts=_experts(llm), reputation=rep)
    debate.pre_eval(_pipeline())
    transcript = debate.last_result
    assert transcript is not None
    assert len(transcript.round1_votes) == 4
    assert len(transcript.round2_votes) == 4


def test_update_reputations_from_pre_post_records_consistency():
    rep = ExpertReputationTracker(["force_field", "sampling", "restraint", "analysis"])
    pre_votes = [
        ExpertVote(expert="force_field", label=VerdictLabel.PASS, confidence=0.8),
        ExpertVote(expert="sampling", label=VerdictLabel.PASS, confidence=0.7),
    ]
    post_votes = [
        ExpertVote(expert="force_field", label=VerdictLabel.PASS, confidence=0.8),  # consistent
        ExpertVote(expert="sampling", label=VerdictLabel.FAIL, confidence=0.9,
                   concerns=[Concern(layer=LayerName.LAYER_2_POST, severity=0.9, description="x")]),  # mismatch
    ]
    update_reputations_from_pre_post(rep, pre_votes, post_votes, evidence={"task_id": "t1"})

    ff_state = rep.state("force_field")
    sa_state = rep.state("sampling")
    assert ff_state.alpha == pytest.approx(2.0)  # 1 prior + 1 consistent
    assert ff_state.beta == pytest.approx(1.0)
    assert sa_state.alpha == pytest.approx(1.0)
    assert sa_state.beta == pytest.approx(2.0)  # 1 prior + 1 mismatch
    assert sa_state.history[-1]["evidence"]["pre_label"] == "pass"
    assert sa_state.history[-1]["evidence"]["post_label"] == "fail"


def test_debate_requires_at_least_one_expert():
    rep = ExpertReputationTracker(["x"])
    with pytest.raises(ValueError):
        Layer2Debate(experts=[], reputation=rep)
