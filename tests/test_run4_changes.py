"""Tests for the run-#4 changes:

- ExpertVote schema includes `strategic_insight` (co-designer field)
- vote_parsing handles both old (no strategic_insight) and new schemas
- TrialSummary model
- Orchestrator's `should_invoke_m2` routing (skip dock + equil; invoke prep / production / analysis)
- Orchestrator's `is_production_stage` budget-detection
- _build_trial_history collapses TrialRecord into TrialSummary correctly
- Writer.revise renders the trial_history into its prompt
"""

from __future__ import annotations

import textwrap

import pytest

from src.models import (
    Concern,
    Critique,
    ExpertVote,
    HostGuestSpec,
    Layer3Output,
    LayerName,
    Pipeline,
    TrialSummary,
    VerdictLabel,
)
from src.orchestrator import (
    TrialRecord,
    _build_trial_history,
    is_production_stage,
    should_invoke_m2,
)
from src.verifiers.layer2.vote_parsing import parse_expert_vote
from src.writer import _revise_prompt_template


# ---- Schema --------------------------------------------------------------


def test_expert_vote_has_strategic_insight():
    v = ExpertVote(
        expert="force_field",
        label=VerdictLabel.PASS,
        confidence=0.8,
        strategic_insight="For CB7 the SAMPL standard is APR with HG corrections.",
    )
    assert v.strategic_insight.startswith("For CB7")


def test_expert_vote_strategic_insight_defaults_to_empty():
    v = ExpertVote(expert="sampling", label=VerdictLabel.UNCERTAIN, confidence=0.5)
    assert v.strategic_insight == ""


# ---- Vote parsing --------------------------------------------------------


def test_vote_parsing_reads_strategic_insight():
    payload = """
    ```json
    {
        "label": "uncertain",
        "confidence": 0.65,
        "strategic_insight": "For CB7-amantadine the SAMPL-validated approach is APR.",
        "concerns": [
            {"severity": 0.4, "description": "11 windows is below recommended.", "suggested_focus": "lambda_schedule"}
        ],
        "reasoning": "method is sound, sampling is tight."
    }
    ```
    """
    v = parse_expert_vote(payload, expert="sampling", layer=LayerName.LAYER_2_PRE)
    assert v.strategic_insight.startswith("For CB7-amantadine")
    assert v.label == VerdictLabel.UNCERTAIN
    assert len(v.concerns) == 1
    assert v.concerns[0].severity == 0.4


def test_vote_parsing_back_compat_no_strategic_insight():
    """Older verifier-mode prompts may not include strategic_insight; parser must tolerate."""
    payload = """
    {
        "label": "fail",
        "confidence": 0.9,
        "concerns": [{"severity": 0.9, "description": "Mismatched force field.", "suggested_focus": null}],
        "reasoning": "FF wrong."
    }
    """
    v = parse_expert_vote(payload, expert="force_field", layer=LayerName.LAYER_2_PRE)
    assert v.strategic_insight == ""
    assert v.label == VerdictLabel.FAIL
    assert v.concerns[0].suggested_focus is None


# ---- M2 routing ----------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("00_dock.py", False),       # docking — mechanical, skip M2
        ("01_prep.py", True),        # parameterize + solvate + min — M2
        ("01_setup.py", True),       # legacy K=6 name — still M2
        ("02_solvate_minimize.py", False),  # legacy K=6 — skip (merged into prep)
        ("02_equilibrate.py", False),       # equilibration — mechanical, skip M2
        ("03_production.py", True),  # production — always M2
        ("04_analysis.py", True),    # analysis — always M2
        ("05_analysis.py", True),    # legacy K=6 numbering — still M2 (matches "analysis")
    ],
)
def test_should_invoke_m2_routing(filename: str, expected: bool):
    assert should_invoke_m2(filename) is expected


# ---- Production stage detection -----------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("03_production.py", True),
        ("04_production.py", True),  # K=6 layout
        ("03_prod.py", False),       # different name; production_check requires "production"
        ("02_equilibrate.py", False),
        ("00_dock.py", False),
    ],
)
def test_is_production_stage(filename: str, expected: bool):
    assert is_production_stage(filename) is expected


# ---- TrialSummary --------------------------------------------------------


def test_trial_summary_defaults():
    s = TrialSummary(revision=0, rationale="initial attempt")
    assert s.revision == 0
    assert s.rationale == "initial attempt"
    assert s.pre_eval_strategic_insights == []
    assert s.post_eval_strategic_insights_by_stage == {}
    assert s.failed_at_stage is None


# ---- _build_trial_history ------------------------------------------------


def _make_trial_record(
    revision: int,
    rationale: str,
    pre_vote_insight: str = "",
    stage_outputs: list[Layer3Output] | None = None,
    post_critiques: list[Critique] | None = None,
    failed_at: int | None = None,
) -> TrialRecord:
    pipeline = Pipeline(
        target=HostGuestSpec(host_id="cb7", guest_id="adz"),
        code_files={"00_dock.py": "pass"},
        entry_point="00_dock.py",
        rationale=rationale,
        revision=revision,
    )
    c_pre = None
    if pre_vote_insight:
        c_pre = Critique(
            layer=LayerName.LAYER_2_PRE,
            label=VerdictLabel.PASS,
            aggregate_confidence=0.7,
            expert_votes=[
                ExpertVote(
                    expert="force_field",
                    label=VerdictLabel.PASS,
                    confidence=0.7,
                    strategic_insight=pre_vote_insight,
                ),
            ],
        )
    tr = TrialRecord(trial_index=revision, pipeline=pipeline)
    tr.c_pre = c_pre
    tr.stage_outputs = stage_outputs or []
    tr.post_critiques = post_critiques or []
    tr.failed_at_stage = failed_at
    return tr


def test_build_trial_history_basic():
    so = Layer3Output(stage_id="00_dock", status="success", exited_normally=True, wall_time_seconds=0.4)
    tr = _make_trial_record(
        revision=3,
        rationale="Try APR with HG corrections.",
        pre_vote_insight="For CB7 with cationic guest, APR is the standard.",
        stage_outputs=[so],
        failed_at=None,
    )
    out = _build_trial_history([tr])
    assert len(out) == 1
    s = out[0]
    assert s.revision == 3
    assert s.rationale.startswith("Try APR")
    assert s.pre_eval_strategic_insights == [
        "force_field: For CB7 with cationic guest, APR is the standard."
    ]
    assert s.stage_outputs_summary == ["00_dock: success (0.4s)"]
    assert s.deepest_stage_reached == 0
    assert s.failed_at_stage is None


def test_build_trial_history_extracts_deepest_stage():
    outs = [
        Layer3Output(stage_id="00_dock", status="success", exited_normally=True, wall_time_seconds=0.5),
        Layer3Output(stage_id="01_prep", status="success", exited_normally=True, wall_time_seconds=2.0),
        Layer3Output(stage_id="02_equilibrate", status="diverged", exited_normally=True, wall_time_seconds=10.0),
    ]
    tr = _make_trial_record(revision=2, rationale="", stage_outputs=outs, failed_at=2)
    out = _build_trial_history([tr])
    assert out[0].deepest_stage_reached == 2
    assert out[0].failed_at_stage == 2


def test_build_trial_history_empty():
    assert _build_trial_history([]) == []


# ---- Writer prompt rendering with trial_history -------------------------


def _make_simple_task():
    from pathlib import Path

    from src.tasks import ExperimentalReference, Task

    return Task(
        task_id="test-task",
        target=HostGuestSpec(host_id="cb7", guest_id="adz"),
        description="Test pipeline design.",
        host_structure_files={},
        guest_structure_files={},
        bound_complex_pdb=Path("/tmp/__nonexistent_test_complex.pdb"),
        extra_context={"temperature_kelvin": 298.15},
        experimental_reference=ExperimentalReference(
            delta_g_kcal_per_mol=-14.1,
            delta_g_uncertainty_kcal_per_mol=0.1,
            technique="ITC",
            pH=7.0,
            temperature_kelvin=298.15,
        ),
    )


def test_revise_prompt_renders_trial_history():
    task = _make_simple_task()
    previous = Pipeline(
        target=task.target,
        code_files={"00_dock.py": "pass"},
        entry_point="00_dock.py",
        rationale="Latest attempt.",
        revision=2,
    )
    critique = Critique(
        layer=LayerName.LAYER_2_PRE,
        label=VerdictLabel.FAIL,
        aggregate_confidence=0.9,
        summary="failed pre-eval",
    )
    history = [
        TrialSummary(
            revision=0,
            rationale="First attempt.",
            pre_eval_strategic_insights=["force_field: Use GAFF2."],
            failed_at_stage=1,
            stage_outputs_summary=["00_dock: success (0.4s)", "01_prep: diverged (3.0s)"],
            deepest_stage_reached=1,
        ),
        TrialSummary(
            revision=1,
            rationale="Second attempt with fix to FF.",
            pre_eval_strategic_insights=[],
            failed_at_stage=None,
            stage_outputs_summary=["00_dock: success (0.3s)"],
            deepest_stage_reached=0,
        ),
    ]
    text = _revise_prompt_template(
        previous=previous,
        critiques=[critique],
        task=task,
        time_budget_minutes=120,
        trial_history=history,
    )
    # Should mention both earlier attempts
    assert "Rev 0" in text
    assert "Rev 1" in text
    assert "First attempt." in text
    assert "Second attempt" in text
    assert "GAFF2" in text
    # Should still show the immediate-previous full pipeline
    assert "Most recent attempt (revision 2)" in text


def test_revise_prompt_omits_history_section_when_empty():
    task = _make_simple_task()
    previous = Pipeline(
        target=task.target,
        code_files={"00_dock.py": "pass"},
        entry_point="00_dock.py",
        rationale="initial revise",
        revision=0,
    )
    critique = Critique(
        layer=LayerName.LAYER_1,
        label=VerdictLabel.FAIL,
        aggregate_confidence=1.0,
        summary="syntax error",
    )
    text = _revise_prompt_template(
        previous=previous,
        critiques=[critique],
        task=task,
        time_budget_minutes=120,
        trial_history=[],
    )
    assert "Earlier attempts" not in text
    assert "Most recent attempt (revision 0)" in text
