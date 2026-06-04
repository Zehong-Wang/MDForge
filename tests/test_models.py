"""Tests for data models."""

from __future__ import annotations

from src.models import (
    Concern,
    Critique,
    ExpertVote,
    HostGuestSpec,
    LayerName,
    Layer3Output,
    Pipeline,
    TaskState,
    VerdictLabel,
)


def _make_pipeline() -> Pipeline:
    return Pipeline(
        target=HostGuestSpec(host_id="CB7", guest_id="g1"),
        code_files={
            "run.py": "import openmm\n\nif __name__ == '__main__':\n    print('hi')\n",
            "helpers.py": "def add(a, b):\n    return a + b\n",
        },
        entry_point="run.py",
        rationale="Boresch restraints + GAFF2 + TIP3P; minimal demo.",
    )


def test_pipeline_round_trip():
    p = _make_pipeline()
    blob = p.model_dump_json()
    p2 = Pipeline.model_validate_json(blob)
    assert p == p2


def test_pipeline_defaults():
    p = _make_pipeline()
    assert p.revision == 0
    assert p.parent_revision is None


def test_critique_with_concerns():
    c = Critique(
        layer=LayerName.LAYER_1,
        label=VerdictLabel.UNCERTAIN,
        aggregate_confidence=0.6,
        concerns=[Concern(layer=LayerName.LAYER_1, severity=0.6, description="x")],
        summary="one concern",
    )
    assert c.label is VerdictLabel.UNCERTAIN
    assert len(c.concerns) == 1


def test_expert_vote_default_no_concerns():
    v = ExpertVote(expert="force_field", label=VerdictLabel.PASS, confidence=0.8)
    assert v.concerns == []
    assert v.reasoning == ""


def test_layer3_output_optional_fields():
    out = Layer3Output()
    assert out.delta_g_kcal_per_mol is None
    assert out.energy_components == {}
    assert out.exited_normally is False
    assert out.timed_out is False
    assert out.parse_failed is False


def test_task_state_history_grows():
    state = TaskState(target=HostGuestSpec(host_id="CB7", guest_id="g"))
    state.pipeline_history.append(_make_pipeline())
    assert len(state.pipeline_history) == 1
    assert not state.converged
