"""Bandit-style reputation tracker for Layer-2 expert verifiers.

Each expert has a Beta posterior over its "reliability" --- the probability
that the expert's pre-execution verdict is consistent with the post-execution
ground-truth assessment. Updates are driven by pre/post-eval comparison, and
reputation feeds the Layer-2 aggregator's voting weights.

Update rule:
  - On a CONSISTENT outcome: alpha += 1
  - On a MISMATCH: beta += 1
  - "Consistent" = the expert's pre-eval label agrees with whether the
    pipeline ultimately passed/failed Layer 3 (interpreted via post-eval).
  - Posterior mean alpha / (alpha + beta) is used as the deterministic
    weight; sampling from Beta(alpha, beta) is supported for Thompson-style
    exploration when desired.

Persistence is JSON-on-disk, one file per project, so reputations survive
restart and can be inspected.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ExpertReputationState:
    """Beta posterior over an expert's reliability.

    Priors are Beta(1, 1) = uniform by default --- experts start neutral.
    """

    expert: str
    alpha: float = 1.0
    beta: float = 1.0
    n_observations: int = 0
    history: list[dict] = field(default_factory=list)

    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        return float(rng.beta(self.alpha, self.beta))


class ExpertReputationTracker:
    """Manages reputation state for all Layer-2 experts.

    Thread-safe through a single lock; expected workloads are low rate
    (one update per task completion) so a coarse lock is fine.
    """

    def __init__(self, expert_names: list[str], persist_path: str | Path | None = None):
        self._lock = threading.Lock()
        self._persist_path: Path | None = Path(persist_path) if persist_path else None
        self._states: dict[str, ExpertReputationState] = {
            name: ExpertReputationState(expert=name) for name in expert_names
        }
        if self._persist_path and self._persist_path.exists():
            self._load()

    def update(
        self,
        expert: str,
        consistent: bool,
        evidence: dict | None = None,
    ) -> ExpertReputationState:
        """Apply one observation to the named expert's posterior.

        Args:
            expert: name of the expert; unknown names are auto-registered
                with a neutral Beta(1,1) prior.
            consistent: True if pre-eval matched post-eval ground truth.
            evidence: optional metadata about the update (which task,
                which pipeline revision, which concerns) for inspectability.
        """
        with self._lock:
            if expert not in self._states:
                # In single_agent_mode the multi-expert panel is replaced by
                # one "generalist" expert that the tracker may not have been
                # constructed with. Auto-register unknown experts with the
                # neutral Beta(1,1) prior rather than raising.
                self._states[expert] = ExpertReputationState(expert=expert)
            state = self._states[expert]
            if consistent:
                state.alpha += 1.0
            else:
                state.beta += 1.0
            state.n_observations += 1
            state.history.append({
                "consistent": bool(consistent),
                "alpha_after": state.alpha,
                "beta_after": state.beta,
                "evidence": evidence or {},
            })
            if self._persist_path:
                self._save()
            return state

    def weight(self, expert: str) -> float:
        """Return the posterior-mean weight in [0, 1] for the expert."""
        with self._lock:
            return self._states[expert].posterior_mean()

    def weights(self) -> dict[str, float]:
        """All experts' current posterior-mean weights."""
        with self._lock:
            return {n: s.posterior_mean() for n, s in self._states.items()}

    def thompson_weights(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        """All experts' Thompson-sampled weights (one Beta draw per expert)."""
        rng = rng or np.random.default_rng()
        with self._lock:
            return {n: s.sample(rng) for n, s in self._states.items()}

    def state(self, expert: str) -> ExpertReputationState:
        with self._lock:
            return self._states[expert]

    def all_states(self) -> dict[str, ExpertReputationState]:
        with self._lock:
            return dict(self._states)

    def _save(self) -> None:
        assert self._persist_path is not None
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {n: asdict(s) for n, s in self._states.items()}
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        assert self._persist_path is not None
        raw = json.loads(self._persist_path.read_text())
        for name, payload in raw.items():
            self._states[name] = ExpertReputationState(**payload)
