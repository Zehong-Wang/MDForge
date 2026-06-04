"""Tests for the bandit reputation tracker."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.verifiers.layer2.reputation import (
    ExpertReputationState,
    ExpertReputationTracker,
)


EXPERTS = ["force_field", "sampling", "restraint", "analysis"]


def test_initial_neutral_priors():
    rep = ExpertReputationTracker(EXPERTS)
    for w in rep.weights().values():
        assert w == pytest.approx(0.5)


def test_consistent_updates_increase_weight():
    rep = ExpertReputationTracker(EXPERTS)
    for _ in range(10):
        rep.update("force_field", consistent=True)
    assert rep.weight("force_field") > 0.85
    # other experts unchanged
    assert rep.weight("sampling") == pytest.approx(0.5)


def test_mismatch_updates_decrease_weight():
    rep = ExpertReputationTracker(EXPERTS)
    for _ in range(10):
        rep.update("sampling", consistent=False)
    assert rep.weight("sampling") < 0.15


def test_history_records_each_update():
    rep = ExpertReputationTracker(EXPERTS)
    rep.update("restraint", consistent=True, evidence={"task_id": "t1"})
    rep.update("restraint", consistent=False, evidence={"task_id": "t2"})
    state = rep.state("restraint")
    assert state.n_observations == 2
    assert len(state.history) == 2
    assert state.history[0]["consistent"] is True
    assert state.history[1]["consistent"] is False
    assert state.history[1]["evidence"]["task_id"] == "t2"


def test_thompson_sampling_returns_in_unit_interval():
    rep = ExpertReputationTracker(EXPERTS)
    rng = np.random.default_rng(seed=42)
    samples = rep.thompson_weights(rng=rng)
    for w in samples.values():
        assert 0.0 <= w <= 1.0


def test_unknown_expert_auto_registers():
    # Unknown experts are auto-registered with a neutral Beta(1,1) prior
    # rather than raising, so single-agent mode (which swaps the multi-expert
    # panel for one generalist) does not crash the tracker.
    rep = ExpertReputationTracker(EXPERTS)
    state = rep.update("nonexistent", consistent=True)
    assert state.expert == "nonexistent"
    assert rep.state("nonexistent").alpha == 2.0  # 1 prior + 1 consistent


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "rep.json"
    rep = ExpertReputationTracker(EXPERTS, persist_path=p)
    rep.update("analysis", consistent=True)
    rep.update("analysis", consistent=True)
    rep.update("analysis", consistent=False)

    rep2 = ExpertReputationTracker(EXPERTS, persist_path=p)
    s = rep2.state("analysis")
    assert s.alpha == 3.0  # 1 prior + 2 consistent
    assert s.beta == 2.0   # 1 prior + 1 mismatch
    assert s.n_observations == 3


def test_state_dataclass_posterior_mean():
    s = ExpertReputationState(expert="x", alpha=4.0, beta=1.0)
    assert s.posterior_mean() == pytest.approx(0.8)
