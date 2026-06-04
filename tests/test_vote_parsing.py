"""Tests for the Layer-2 expert vote parser."""

from __future__ import annotations

import pytest

from src.models import LayerName, VerdictLabel
from src.verifiers.layer2.vote_parsing import VoteParseError, parse_expert_vote


def test_parses_fenced_json():
    text = """\
Some preamble.

```json
{
  "label": "pass",
  "confidence": 0.85,
  "concerns": [],
  "reasoning": "looks good"
}
```

Trailing prose.
"""
    vote = parse_expert_vote(text, expert="force_field", layer=LayerName.LAYER_2_PRE)
    assert vote.label is VerdictLabel.PASS
    assert vote.confidence == pytest.approx(0.85)
    assert vote.expert == "force_field"
    assert vote.reasoning == "looks good"
    assert vote.concerns == []


def test_parses_bare_json():
    text = """\
{"label": "fail", "confidence": 0.9, "concerns": [{"severity": 0.9, "description": "X", "suggested_focus": "force field"}], "reasoning": "bad"}
"""
    vote = parse_expert_vote(text, expert="sampling", layer=LayerName.LAYER_2_PRE)
    assert vote.label is VerdictLabel.FAIL
    assert vote.confidence == pytest.approx(0.9)
    assert len(vote.concerns) == 1
    assert vote.concerns[0].severity == pytest.approx(0.9)
    assert vote.concerns[0].suggested_focus == "force field"
    assert vote.concerns[0].layer is LayerName.LAYER_2_PRE
    assert vote.concerns[0].expert == "sampling"


def test_uncertain_label():
    text = '```json\n{"label": "uncertain", "confidence": 0.4, "concerns": [], "reasoning": "ambiguous"}\n```'
    vote = parse_expert_vote(text, expert="restraint", layer=LayerName.LAYER_2_POST)
    assert vote.label is VerdictLabel.UNCERTAIN
    assert vote.concerns[0:0] == []  # no concerns


def test_clamps_confidence_to_unit_interval():
    text = '{"label": "pass", "confidence": 1.5, "concerns": [], "reasoning": ""}'
    vote = parse_expert_vote(text, expert="x", layer=LayerName.LAYER_2_PRE)
    assert vote.confidence == 1.0


def test_clamps_severity():
    text = '{"label": "fail", "confidence": 0.5, "concerns": [{"severity": -0.2, "description": "y"}], "reasoning": ""}'
    vote = parse_expert_vote(text, expert="x", layer=LayerName.LAYER_2_PRE)
    assert vote.concerns[0].severity == 0.0


def test_skips_empty_descriptions():
    text = '{"label": "pass", "confidence": 0.7, "concerns": [{"severity": 0.5, "description": ""}, {"severity": 0.6, "description": "Real"}], "reasoning": ""}'
    vote = parse_expert_vote(text, expert="x", layer=LayerName.LAYER_2_PRE)
    assert len(vote.concerns) == 1
    assert vote.concerns[0].description == "Real"


def test_unknown_label_raises():
    text = '{"label": "maybe", "confidence": 0.5, "concerns": [], "reasoning": ""}'
    with pytest.raises(VoteParseError, match="not pass/fail/uncertain"):
        parse_expert_vote(text, expert="x", layer=LayerName.LAYER_2_PRE)


def test_no_json_object_raises():
    with pytest.raises(VoteParseError, match="No JSON object"):
        parse_expert_vote("just prose, no json here", expert="x", layer=LayerName.LAYER_2_PRE)


def test_concerns_must_be_list():
    text = '{"label": "fail", "confidence": 0.5, "concerns": "not a list", "reasoning": ""}'
    with pytest.raises(VoteParseError, match="must be a list"):
        parse_expert_vote(text, expert="x", layer=LayerName.LAYER_2_PRE)


def test_missing_label_raises():
    text = '{"confidence": 0.5, "concerns": [], "reasoning": ""}'
    with pytest.raises(VoteParseError, match="missing 'label'"):
        parse_expert_vote(text, expert="x", layer=LayerName.LAYER_2_PRE)
