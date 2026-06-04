"""Layer-2 multi-expert debate orchestrator.

Debate format:

  Round 1 - independent **parallel** review (each expert produces a vote
            without seeing the others). All experts run concurrently via
            a ThreadPoolExecutor — each thread carries its own
            claude_agent_sdk subprocess, so concurrency is real.
  Round 2 - one cross-visibility revision (each expert sees the others'
            round-1 votes and may revise its own). Also runs in
            parallel across experts.

Aggregation: weighted by `ExpertReputationTracker.weights()`. The
aggregate verdict and confidence become the Layer-2 Critique.

The same orchestrator is used in two modes:
  - pre_eval(pipeline, task)               -> Critique
  - post_eval(pipeline, task, layer3_output) -> Critique

Reputation updates happen *outside* this module (in the Orchestrator).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

import logging

from src.models import (
    Critique,
    ExpertVote,
    Layer3Output,
    LayerName,
    Pipeline,
    VerdictLabel,
)
from src.verifiers.layer2.reputation import ExpertReputationTracker
from src.verifiers.layer2.vote_parsing import VoteParseError

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.tasks import Task
    from src.verifiers.layer2.base_expert import BaseExpert

# Aggregation thresholds. Score is a weighted combination where
#   PASS=1.0, UNCERTAIN=0.5, FAIL=0.0.
PASS_THRESHOLD = 0.7
FAIL_THRESHOLD = 0.3


@dataclass
class DebateResult:
    """Full transcript of a debate, kept for inspection / logging."""

    layer: LayerName
    round1_votes: list[ExpertVote]
    round2_votes: list[ExpertVote]
    aggregate: Critique


class Layer2Debate:
    """Multi-expert debate over a candidate pipeline."""

    def __init__(
        self,
        experts: list["BaseExpert"],
        reputation: ExpertReputationTracker,
        n_rounds: int = 2,
        single_agent_mode: bool = False,
    ):
        if not experts:
            raise ValueError("Layer2Debate needs at least one expert.")
        if n_rounds < 1:
            raise ValueError("Layer2Debate needs at least one round.")
        self._experts = list(experts)
        self._reputation = reputation
        self._n_rounds = n_rounds
        self._single_agent_mode = single_agent_mode
        # Single-agent ablation: replace the 3-expert × n-round debate with
        # a single GeneralistExpert × 1 round. The generalist's system prompt
        # covers the merged FF + sampling + analysis expertise, so the only
        # thing being ablated is *debate*, not *expertise breadth*. Borrow the
        # llm_client from one of the original experts so it uses the same
        # backend as the full panel.
        if self._single_agent_mode:
            # Local import to avoid circular import at module load.
            from src.verifiers.layer2.experts.generalist import (
                GeneralistExpert,
            )
            template_llm = getattr(self._experts[0], "_llm", None)
            self._experts = [GeneralistExpert(llm_client=template_llm)]
            self._n_rounds = 1
        self._last_result: DebateResult | None = None

    @property
    def last_result(self) -> DebateResult | None:
        return self._last_result

    def pre_eval(self, pipeline: Pipeline, task: "Task | None" = None) -> Critique:
        """Mode B: pre-pipeline-execution critique of an existing design."""
        return self._run(
            pipeline=pipeline,
            task=task,
            layer3_output=None,
            layer=LayerName.LAYER_2_PRE,
        )

    def post_eval_benchmark(
        self,
        pipeline: Pipeline,
        benchmark_result,
        task: "Task | None" = None,
    ) -> Critique:
        """Mode D: post-MD multi-molecule benchmark debate.

        After the pipeline has been APPLIED to N benchmark molecules, run
        the 3 experts × N rounds debate on the cross-molecule results.
        Returns a Critique whose ``expert_votes`` carry domain-specific
        strategic_insights, ready to be appended to ``trial.post_critiques``
        and consumed by the next ``Writer.revise``.
        """
        return self._run_post_eval_benchmark(
            pipeline=pipeline,
            benchmark_result=benchmark_result,
            task=task,
        )

    def design_discussion(self, task: "Task") -> Critique:
        """Phase 0: pre-pipeline method design discussion.

        Runs the same Round-1 (independent) + Round-2 (cross-visibility,
        if ``n_rounds >= 2``) debate format as ``pre_eval``, but each
        expert is asked to *recommend a method* rather than critique an
        existing pipeline. Experts may use ``WebSearch`` / ``WebFetch``
        if their LLM client has those tools enabled.

        The returned Critique aggregates the experts' final-round
        recommendations; ``expert_votes`` contains the per-expert
        strategic_insights. The orchestrator forwards this Critique to
        ``PipelineWriter.propose`` so the Writer's initial design is
        informed by the expert panel's pre-pipeline recommendations.
        """
        return self._run_design(task=task)

    def post_eval(
        self,
        pipeline: Pipeline,
        layer3_output: Layer3Output,
        task: "Task | None" = None,
    ) -> Critique:
        return self._run(
            pipeline=pipeline,
            task=task,
            layer3_output=layer3_output,
            layer=LayerName.LAYER_2_POST,
        )

    def _run(
        self,
        pipeline: Pipeline,
        task: "Task | None",
        layer3_output: Layer3Output | None,
        layer: LayerName,
    ) -> Critique:
        round_votes: list[list[ExpertVote]] = []
        prev_votes: list[ExpertVote] = []
        for round_idx in range(self._n_rounds):
            # PARALLEL: all experts in this round run concurrently. Each
            # thread has its own claude_agent_sdk subprocess, so calls
            # don't share state. Per-expert peer_votes are built BEFORE
            # the dispatch so each thread receives an immutable list.
            args_per_expert: list[tuple] = []
            for i, expert in enumerate(self._experts):
                if round_idx == 0:
                    peers: list[ExpertVote] = []
                else:
                    peers = [v for j, v in enumerate(prev_votes) if j != i]
                args_per_expert.append((i, expert, peers))

            this_round: list[ExpertVote | None] = [None] * len(self._experts)
            with ThreadPoolExecutor(max_workers=len(self._experts)) as pool:
                future_to_idx = {
                    pool.submit(
                        self._invoke_expert,
                        expert=expert,
                        pipeline=pipeline,
                        layer3_output=layer3_output,
                        task=task,
                        peer_votes=peers,
                    ): i
                    for (i, expert, peers) in args_per_expert
                }
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    this_round[idx] = fut.result()

            # All slots filled by now; cast to non-None
            round_votes.append([v for v in this_round if v is not None])
            prev_votes = round_votes[-1]

        final_votes = round_votes[-1]
        aggregate = self._aggregate(final_votes, layer)
        self._last_result = DebateResult(
            layer=layer,
            round1_votes=round_votes[0],
            round2_votes=final_votes if self._n_rounds > 1 else round_votes[0],
            aggregate=aggregate,
        )
        return aggregate

    def _run_post_eval_benchmark(
        self,
        pipeline: Pipeline,
        benchmark_result,
        task: "Task | None",
    ) -> Critique:
        """Parallel multi-round expert debate on multi-molecule benchmark results."""
        round_votes: list[list[ExpertVote]] = []
        prev_votes: list[ExpertVote] = []
        for round_idx in range(self._n_rounds):
            args_per_expert: list[tuple] = []
            for i, expert in enumerate(self._experts):
                peers = (
                    [v for j, v in enumerate(prev_votes) if j != i]
                    if round_idx > 0 else []
                )
                args_per_expert.append((i, expert, peers))

            this_round: list[ExpertVote | None] = [None] * len(self._experts)
            with ThreadPoolExecutor(max_workers=len(self._experts)) as pool:
                future_to_idx = {
                    pool.submit(
                        self._invoke_post_eval_benchmark_expert,
                        expert=expert,
                        pipeline=pipeline,
                        benchmark_result=benchmark_result,
                        task=task,
                        peer_votes=peers,
                        round_idx=round_idx,
                    ): i
                    for (i, expert, peers) in args_per_expert
                }
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    this_round[idx] = fut.result()
            round_votes.append([v for v in this_round if v is not None])
            prev_votes = round_votes[-1]

        final_votes = round_votes[-1]
        aggregate = self._aggregate(final_votes, LayerName.LAYER_2_POST)
        self._last_result = DebateResult(
            layer=LayerName.LAYER_2_POST,
            round1_votes=round_votes[0],
            round2_votes=final_votes if self._n_rounds > 1 else round_votes[0],
            aggregate=aggregate,
        )
        return aggregate

    def _invoke_post_eval_benchmark_expert(
        self,
        expert: "BaseExpert",
        pipeline: Pipeline,
        benchmark_result,
        task: "Task | None",
        peer_votes: list[ExpertVote],
        round_idx: int = 0,
    ) -> ExpertVote:
        try:
            return expert.post_eval_benchmark(
                pipeline=pipeline,
                benchmark_result=benchmark_result,
                task=task,
                peer_votes=peer_votes,
                round_idx=round_idx,
            )
        except VoteParseError as exc:
            _log.warning("Expert %s post_eval_benchmark JSON parse failed: %s", expert.NAME, exc)
            return ExpertVote(
                expert=expert.NAME,
                label=VerdictLabel.UNCERTAIN,
                confidence=0.0,
                strategic_insight="(abstain: post_eval_benchmark JSON parse failed)",
                reasoning=f"VoteParseError: {exc}",
            )
        except Exception as exc:
            _log.warning("Expert %s post_eval_benchmark call failed: %r", expert.NAME, exc)
            return ExpertVote(
                expert=expert.NAME,
                label=VerdictLabel.UNCERTAIN,
                confidence=0.0,
                strategic_insight="(abstain: post_eval_benchmark call failed)",
                reasoning=f"{type(exc).__name__}: {exc}",
            )

    def _run_design(self, task: "Task") -> Critique:
        """Parallel multi-round design discussion (mode A)."""
        round_votes: list[list[ExpertVote]] = []
        prev_votes: list[ExpertVote] = []
        for round_idx in range(self._n_rounds):
            args_per_expert: list[tuple] = []
            for i, expert in enumerate(self._experts):
                peers = (
                    [v for j, v in enumerate(prev_votes) if j != i]
                    if round_idx > 0 else []
                )
                args_per_expert.append((i, expert, peers))

            this_round: list[ExpertVote | None] = [None] * len(self._experts)
            with ThreadPoolExecutor(max_workers=len(self._experts)) as pool:
                future_to_idx = {
                    pool.submit(
                        self._invoke_design_expert,
                        expert=expert,
                        task=task,
                        peer_votes=peers,
                        round_idx=round_idx,
                    ): i
                    for (i, expert, peers) in args_per_expert
                }
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    this_round[idx] = fut.result()
            round_votes.append([v for v in this_round if v is not None])
            prev_votes = round_votes[-1]

        final_votes = round_votes[-1]
        aggregate = self._aggregate(final_votes, LayerName.LAYER_2_PRE)
        self._last_result = DebateResult(
            layer=LayerName.LAYER_2_PRE,
            round1_votes=round_votes[0],
            round2_votes=final_votes if self._n_rounds > 1 else round_votes[0],
            aggregate=aggregate,
        )
        return aggregate

    def _invoke_design_expert(
        self,
        expert: "BaseExpert",
        task: "Task",
        peer_votes: list[ExpertVote],
        round_idx: int = 0,
    ) -> ExpertVote:
        """Phase-0 dispatcher — same advisory-on-error guard as `_invoke_expert`."""
        try:
            return expert.design_recommendation(
                task=task, peer_votes=peer_votes, round_idx=round_idx,
            )
        except VoteParseError as exc:
            _log.warning(
                "Expert %s design vote parse failed (advisory abstain): %s",
                expert.NAME, exc,
            )
            return ExpertVote(
                expert=expert.NAME,
                label=VerdictLabel.UNCERTAIN,
                confidence=0.0,
                strategic_insight="(abstain: design recommendation JSON parse failed)",
                reasoning=f"VoteParseError: {exc}",
            )
        except Exception as exc:
            _log.warning(
                "Expert %s design call failed (advisory abstain): %r",
                expert.NAME, exc,
            )
            return ExpertVote(
                expert=expert.NAME,
                label=VerdictLabel.UNCERTAIN,
                confidence=0.0,
                strategic_insight="(abstain: design-mode expert call failed)",
                reasoning=f"{type(exc).__name__}: {exc}",
            )

    def _invoke_expert(
        self,
        expert: "BaseExpert",
        pipeline: Pipeline,
        layer3_output: Layer3Output | None,
        task: "Task | None",
        peer_votes: list[ExpertVote],
    ) -> ExpertVote:
        # Multi-agent is advisory under P4 — a single expert's malformed
        # JSON or transient transport hiccup must NEVER tank the whole
        # pipeline. Convert any per-expert failure into an UNCERTAIN
        # abstention so the orchestrator can keep moving.
        try:
            if layer3_output is None:
                return expert.pre_eval(pipeline, task=task, peer_votes=peer_votes)
            return expert.post_eval(
                pipeline,
                layer3_output=layer3_output,
                task=task,
                peer_votes=peer_votes,
            )
        except VoteParseError as exc:
            _log.warning(
                "Expert %s vote parse failed (advisory abstain): %s",
                expert.NAME, exc,
            )
            return ExpertVote(
                expert=expert.NAME,
                label=VerdictLabel.UNCERTAIN,
                confidence=0.0,
                strategic_insight="(abstain: vote JSON parse failed)",
                reasoning=f"VoteParseError: {exc}",
            )
        except Exception as exc:
            _log.warning(
                "Expert %s call failed (advisory abstain): %r",
                expert.NAME, exc,
            )
            return ExpertVote(
                expert=expert.NAME,
                label=VerdictLabel.UNCERTAIN,
                confidence=0.0,
                strategic_insight="(abstain: expert call failed)",
                reasoning=f"{type(exc).__name__}: {exc}",
            )

    def _aggregate(
        self,
        votes: list[ExpertVote],
        layer: LayerName,
    ) -> Critique:
        weights = self._reputation.weights()
        total_weight = 0.0
        score = 0.0
        confidence = 0.0
        for v in votes:
            w = weights.get(v.expert, 0.5)
            total_weight += w
            score += w * _label_score(v.label)
            confidence += w * v.confidence
        if total_weight == 0:
            score = 0.5
            confidence = 0.5
        else:
            score /= total_weight
            confidence /= total_weight

        if score >= PASS_THRESHOLD:
            label = VerdictLabel.PASS
        elif score <= FAIL_THRESHOLD:
            label = VerdictLabel.FAIL
        else:
            label = VerdictLabel.UNCERTAIN

        concerns = [c for v in votes for c in v.concerns]
        summary = (
            f"Aggregated over {len(votes)} expert(s); weighted score {score:.2f} "
            f"-> {label.value}; aggregate confidence {confidence:.2f}."
        )
        return Critique(
            layer=layer,
            label=label,
            aggregate_confidence=confidence,
            concerns=concerns,
            expert_votes=list(votes),
            summary=summary,
        )


def _label_score(label: VerdictLabel) -> float:
    return {
        VerdictLabel.PASS: 1.0,
        VerdictLabel.UNCERTAIN: 0.5,
        VerdictLabel.FAIL: 0.0,
    }[label]


def update_reputations_from_pre_post(
    reputation: ExpertReputationTracker,
    pre_votes: list[ExpertVote],
    post_votes: list[ExpertVote],
    evidence: dict | None = None,
) -> None:
    """Compare each expert's pre-eval and post-eval verdict; update reputation.

    'Consistent' = the labels match. (A finer scheme can credit "pre said
    pass and Layer 3 succeeded"-style outcomes, but that requires the
    Layer-3 ground-truth signal; pre-vs-post agreement is the cheap proxy
    we have available locally.)
    """
    by_pre = {v.expert: v for v in pre_votes}
    for post in post_votes:
        pre = by_pre.get(post.expert)
        if pre is None:
            continue
        consistent = pre.label == post.label
        ev = dict(evidence or {})
        ev["pre_label"] = pre.label.value
        ev["post_label"] = post.label.value
        reputation.update(post.expert, consistent=consistent, evidence=ev)
