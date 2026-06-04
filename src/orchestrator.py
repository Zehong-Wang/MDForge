"""Orchestrator implementing the MDForge verbal-RL design loop (PRISM).

Drives one task end-to-end:

  Phase 1 — Pre-trial verbal screening
    Layer1.verify  → on FAIL: Writer.revise → restart
    Layer2.pre_eval → on FAIL: Writer.revise → restart

  Phase 2 — Verbal reward shaping (per-stage)
    For each stage k = 0..K-1:
      Layer3.run_stage → stage_output (real MD)
      if status == "error" (Python crash):
          → bypass M2 (engineering bug, no physical insight needed)
          → break
      Layer2.post_eval → c_post_k (M2 deliberation; verbal shaping reward)
      if c_post_k.label == FAIL or status in ("timeout", "diverged"):
          → break (M2-mediated abort)

  Phase 3 — Sparse outcome reward
    If all K stages completed and final critique == PASS with high confidence,
    converged. (Outcome reward vs experimental_reference is computed by the
    caller and recorded; the orchestrator does not consume it.)

  Phase 4 — Verbal posterior update
    Writer.revise(prev_pipeline, [c_l1, c_pre, *c_post_*])

  Phase 5 — Cross-task slow-loop reputation update
    update_reputations(c_pre.expert_votes, last c_post.expert_votes)

Invariants:
  - Multi-agent (Layer 2) is invoked ONLY for physical-insight tasks.
    Layer 1 fails (engineering) and stage status="error" (Python crashes)
    bypass M2.
  - Budget metric is N = number of trials that reached production-stage
    completion. Early-aborted trials don't count.
  - All artifacts (Writer prompts, expert critiques, stage result.json,
    reputation snapshots) are persisted under `state.artifacts_dir` for
    later inspection.

This module is the runtime; the algorithm it implements (PRISM densification
of the sparse terminal reward) is described in the paper.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.models import (
    Critique,
    HostGuestSpec,
    Layer3Output,
    LayerName,
    Pipeline,
    TrialSummary,
    VerdictLabel,
)
from src.tasks import Task
from src.verifiers.layer1 import verify_layer1
from src.verifiers.layer2.debate import (
    Layer2Debate,
    update_reputations_from_pre_post,
)
from src.verifiers.layer2.reputation import ExpertReputationTracker
from src.verifiers.layer3 import Layer3Oracle, stage_index
from src.verifiers.multi_molecule_benchmark import MultiMoleculeBenchmark
from src.writer import PipelineWriter
from src.writer_engineer import EngineerResult, PipelineEngineer

log = logging.getLogger("mdforge.orchestrator")


# Convergence threshold for the outcome-reward phase.
DEFAULT_CONFIDENCE_THRESHOLD = 0.7


# Stages where M2 (multi-expert deliberation) is INVOKED by the routing rule.
# Other stages (e.g. cheap docking-pose load, mechanical equilibration) bypass
# M2: multi-agent deliberation is reserved for physical-insight stages.
# Match by substring on stage filename basename (e.g. "01_prep.py" matches
# "prep"). Production and analysis always get M2; setup/prep gets M2;
# dock and equilibrate skip M2.
M2_INVOKE_NAME_PATTERNS: tuple[str, ...] = (
    "prep",       # 01 prep stage (parameterize + tleap + solvate + min)
    "setup",      # alternate name for the prep stage (still supported)
    "production", # production sampling — the expensive stage
    "analysis",   # MBAR / final ΔG
)
M2_SKIP_NAME_PATTERNS: tuple[str, ...] = (
    "dock",       # cheap pose load
    "equil",      # mechanical equilibration; covers "equilibrate", "equilibration"
    "solvate",    # alternate stage name now folded into prep
    "minimize",   # alternate stage name now folded into prep
)


def should_invoke_m2(stage_filename: str) -> bool:
    """Return True if this stage should be sent to the multi-expert
    Layer-2 post-eval (generalised P1 principle: multi-agent only on
    stages with substantive physical signal).
    """
    name = stage_filename.lower()
    # Explicit skip list wins
    if any(pat in name for pat in M2_SKIP_NAME_PATTERNS):
        return False
    # Explicit invoke list
    if any(pat in name for pat in M2_INVOKE_NAME_PATTERNS):
        return True
    # Default: invoke M2 if uncertain (conservative)
    return True


def is_production_stage(stage_filename: str) -> bool:
    """Whether this stage is the production-sampling stage. Budget is
    counted by trials whose production stage exited_normally + status=success.
    """
    return "production" in stage_filename.lower()


@dataclass
class TrialRecord:
    """All artifacts produced by a single trial (one full attempt)."""

    trial_index: int
    pipeline: Pipeline
    c_l1: Critique | None = None
    c_pre: Critique | None = None
    stage_outputs: list[Layer3Output] = field(default_factory=list)
    post_critiques: list[Critique] = field(default_factory=list)
    failed_at_stage: int | None = None
    counted_toward_budget: bool = False
    converged: bool = False
    work_dir: str | None = None
    wall_time_seconds: float = 0.0


@dataclass
class RunReport:
    """Top-level report returned by Orchestrator.run."""

    task_id: str
    target: HostGuestSpec
    trials: list[TrialRecord] = field(default_factory=list)
    converged: bool = False
    budget_used: int = 0
    budget_max: int = 0
    artifacts_dir: str | None = None
    final_pipeline: Pipeline | None = None


class Orchestrator:
    """Drive one task end-to-end via PRISM."""

    def __init__(
        self,
        writer: PipelineWriter,
        layer2: Layer2Debate,
        layer3: Layer3Oracle,
        reputation: ExpertReputationTracker,
        budget_n: int = 5,
        max_trials: int = 12,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        artifacts_root: str | Path | None = None,
        engineer: PipelineEngineer | None = None,
        multi_molecule_benchmark: MultiMoleculeBenchmark | None = None,
        skip_intermediate: bool = False,
    ):
        self._writer = writer
        self._layer2 = layer2
        self._layer3 = layer3
        self._reputation = reputation
        self._budget_n = budget_n
        # Hard cap on TOTAL trials. budget_n only counts production-
        # succeeding trials; without this cap a cell whose production MD
        # keeps diverging (or whose Writer keeps failing Layer1) could loop
        # indefinitely. Healthy cells converge in 6-8 trials; 12 leaves
        # margin while terminating runaway cells.
        self._max_trials = max_trials
        self._confidence_threshold = confidence_threshold
        self._artifacts_root = Path(artifacts_root) if artifacts_root else None
        # When True, suppress all *intermediate* multi-agent reward
        # signals (Phase 0 design_discussion, Layer2.pre_eval, per-stage
        # Layer2.post_eval). The final-reward path (MultiMoleculeBenchmark
        # numeric critique + Layer2.post_eval_benchmark text critique)
        # ALWAYS runs regardless: that is the verbal-RL reward floor that
        # all 4 ablation cells share.
        self._skip_intermediate = skip_intermediate
        # Optional tool-using engineer agent. When set, the orchestrator's
        # per-stage execution loop is replaced by one engineer.run_to_completion()
        # call: the engineer drives the pipeline end-to-end (stages 00-04) in
        # the sandbox, edits code in-place to fix engineering bugs, and writes
        # stage_NN_*_result.json files. The orchestrator then reads those
        # results and runs L2 post-eval on each.
        self._engineer = engineer
        # Optional multi-molecule benchmark verifier. When set, after the
        # single-molecule trial's stage post-evals complete, the engineer-
        # debugged pipeline is APPLIED to N additional benchmark molecules
        # in parallel on N GPUs. The resulting aggregate ΔG-vs-ITC errors
        # are packaged as a Layer-2 post-eval Critique and APPENDED to the
        # trial's post-eval critique list, so the next Writer.revise sees
        # cross-molecule signal — not just single-molecule trial signal.
        # This is the structural fix for the verbal-RL "single-molecule
        # over-fit" failure mode observed in earlier experiments.
        self._multi_molecule_benchmark = multi_molecule_benchmark

    def run(
        self,
        task: Task,
        resume_pipeline: Pipeline | None = None,
        resume_critiques: list[Critique] | None = None,
        resume_history: list[TrialSummary] | None = None,
    ) -> RunReport:
        """Execute PRISM on a single task; return a full RunReport.

        If ``resume_pipeline`` is provided, the FIRST iteration goes
        through Writer.revise (not Writer.propose), seeded with the
        provided pipeline + critiques + trial_history. Use this to
        continue iterating on top of a prior run's last successful
        trial without losing the multi-trial context.
        """
        artifacts_dir = self._setup_artifacts_dir(task)
        log.info("orchestrator.run task=%s artifacts=%s", task.task_id, artifacts_dir)
        if resume_pipeline is not None:
            log.info(
                "orchestrator.run RESUMING from prior pipeline rev=%d "
                "(critiques=%d, prior_trial_history=%d)",
                resume_pipeline.revision,
                len(resume_critiques or []),
                len(resume_history or []),
            )

        report = RunReport(
            task_id=task.task_id,
            target=task.target,
            budget_max=self._budget_n,
            artifacts_dir=str(artifacts_dir) if artifacts_dir else None,
        )

        trial_idx = 0
        pipeline: Pipeline | None = resume_pipeline
        prior_critiques: list[Critique] = list(resume_critiques or [])
        # If resuming, the seed history is what TrialSummary list to give
        # the FIRST Writer.revise call. We do this by injecting fake
        # earlier-trial sentinels into report.trials so that the regular
        # _build_trial_history(report.trials[:-1]) machinery works.
        # Simplest: stash them in a separate slot used only at trial 1.
        self._seed_history: list[TrialSummary] = list(resume_history or [])

        # ---- Phase 0: Pre-pipeline multi-expert design discussion ----
        # Run once at the very top of the task (NOT per trial). The aggregate
        # recommendations are forwarded as starting context to Writer.propose
        # below. Still advisory — Writer retains final method authority.
        # skip_intermediate=True disables this entirely (intermediate-reward ablations).
        design_discussion: Critique | None = None
        if pipeline is None and not self._skip_intermediate:
            log.info("Phase 0: design_discussion (3 expert × %d round, advisory; experts have WebSearch)", self._layer2._n_rounds)
            try:
                design_discussion = self._layer2.design_discussion(task=task)
                log.info(
                    "Phase 0 done: panel verdict=%s, confidence=%.2f, %d expert votes",
                    design_discussion.label.value,
                    design_discussion.aggregate_confidence,
                    len(design_discussion.expert_votes),
                )
                self._save_critique(artifacts_dir, 0, "phase0_design_discussion", design_discussion)
            except Exception as exc:
                log.warning("Phase 0 design discussion failed (advisory; proceeding without): %r", exc)
                design_discussion = None
        elif self._skip_intermediate:
            log.info("Phase 0 SKIPPED (skip_intermediate=True; intermediate-reward ablation)")

        while (report.budget_used < self._budget_n and not report.converged
               and trial_idx < self._max_trials):
            trial_idx += 1
            t0 = time.monotonic()

            # ---- Writer: propose or revise ----
            if pipeline is None:
                log.info("trial %d: Writer.propose (with Phase-0 design recs)", trial_idx)
                pipeline = self._writer.propose(
                    task,
                    time_budget_minutes=720,
                    design_discussion=design_discussion,
                )
            else:
                # Build a TrialSummary list of all PRIOR trials (excluding the
                # immediately-previous one, which is passed in full as `previous`).
                # The Writer needs this so revisions consider the entire arc,
                # not just one trial back.
                full_history = _build_trial_history(report.trials[:-1])
                # If resuming, prepend the seeded history (from the prior run's
                # earlier trials) so the Writer sees the full lineage.
                if trial_idx == 1 and self._seed_history:
                    full_history = list(self._seed_history) + full_history
                elif self._seed_history:
                    full_history = list(self._seed_history) + full_history
                log.info(
                    "trial %d: Writer.revise (with %d earlier trial summaries)",
                    trial_idx, len(full_history),
                )
                pipeline = self._writer.revise(
                    previous=pipeline,
                    critiques=prior_critiques,
                    task=task,
                    time_budget_minutes=720,
                    trial_history=full_history,
                )

            # Save the pipeline and Writer debug for this trial.
            self._save_pipeline(artifacts_dir, trial_idx, pipeline)

            trial = TrialRecord(trial_index=trial_idx, pipeline=pipeline)
            report.trials.append(trial)

            # ---- Phase 1a: Layer 1 (engineering / static) ----
            log.info("trial %d: Layer1.verify", trial_idx)
            c_l1 = verify_layer1(pipeline)
            trial.c_l1 = c_l1
            self._save_critique(artifacts_dir, trial_idx, "l1", c_l1)
            if c_l1.label == VerdictLabel.FAIL:
                log.warning("trial %d: Layer1 FAIL — bypass M2, revise & restart", trial_idx)
                prior_critiques = [c_l1]
                trial.wall_time_seconds = time.monotonic() - t0
                continue

            # ---- Phase 1b: Layer 2 pre-eval (multi-expert debate, ADVISORY) ----
            # P4: Multi-agent never gates execution based on code-prediction.
            # Pre-eval strategic_insight is collected and forwarded to the
            # Writer as context for the *next* revision (and recorded in the
            # cross-trial history), but its label NEVER aborts the current
            # trial. The only "committed" signals that can abort are
            # Layer 1 (objective engineering bugs) and Layer 3 hard physical
            # signals (status=diverged / timeout / exit_code≠0).
            # skip_intermediate=True suppresses pre_eval entirely;
            # c_pre stays None and is NOT forwarded to the next Writer.revise.
            c_pre: Critique | None = None
            if not self._skip_intermediate:
                log.info("trial %d: Layer2.pre_eval (4 expert × 2 round, advisory)", trial_idx)
                c_pre = self._layer2.pre_eval(pipeline=pipeline, task=task)
                trial.c_pre = c_pre
                self._save_critique(artifacts_dir, trial_idx, "l2pre", c_pre)
                if c_pre.label == VerdictLabel.FAIL:
                    log.info(
                        "trial %d: L2 pre-eval FAIL (advisory only — proceeding to Layer 3 anyway per P4)",
                        trial_idx,
                    )
            else:
                log.info("trial %d: Layer2.pre_eval SKIPPED (skip_intermediate=True)", trial_idx)

            # ---- Phase 2: Staged execution + per-stage post-eval ----
            stage_filenames = pipeline.stage_files()
            if not stage_filenames:
                # Writer didn't produce stage files; treat as L1 failure.
                log.error("trial %d: Writer produced no stage files", trial_idx)
                fake = _make_no_stages_critique()
                prior_critiques = [c for c in (c_l1, c_pre, fake) if c is not None]
                trial.wall_time_seconds = time.monotonic() - t0
                continue

            work_dir = self._layer3.prepare_workdir(
                pipeline=pipeline,
                auxiliary_files=_collect_auxiliary_files(task),
            )
            trial.work_dir = str(work_dir)
            log.info("trial %d: work_dir=%s, K=%d stages", trial_idx, work_dir, len(stage_filenames))

            stage_outputs: list[Layer3Output] = []
            post_critiques: list[Critique] = []
            failed_at: int | None = None
            production_succeeded = False

            if self._engineer is not None:
                # Engineer-led mode: tool-using agent drives all stages end-to-end
                # in the sandbox, editing code in-place to fix engineering bugs.
                # The orchestrator then reads each stage's result file off disk
                # and runs multi-agent post-eval on the routed stages.
                log.info("trial %d: PipelineEngineer.run_to_completion (tool-using, ~60min budget)", trial_idx)
                # Defense-in-depth: the bundled claude CLI can be
                # SIGKILLed (exit code -9) if a long Bash stage trips its
                # process-group timeout. Retry the SAME engineer call (re-uses
                # work_dir; the engineer continues from current stage files)
                # up to 3 attempts total before giving up.
                _ENGINEER_MAX_ATTEMPTS = 3
                for _attempt in range(1, _ENGINEER_MAX_ATTEMPTS + 1):
                    try:
                        eng_result = self._engineer.run_to_completion(
                            pipeline=pipeline,
                            work_dir=work_dir,
                            task=task,
                            trial_idx=trial_idx,
                        )
                        break
                    except Exception as exc:
                        if "exit code -9" in str(exc) and _attempt < _ENGINEER_MAX_ATTEMPTS:
                            log.warning(
                                "trial %d: engineer crashed with 'exit code -9' "
                                "(attempt %d/%d) — retrying run_to_completion",
                                trial_idx, _attempt, _ENGINEER_MAX_ATTEMPTS,
                            )
                            continue
                        raise
                log.info(
                    "trial %d: engineer done — wall=%.1fs, tool_calls=%d "
                    "(Bash=%d, Edit=%d, Write=%d), stage_statuses=%s",
                    trial_idx, eng_result.wall_time_seconds,
                    eng_result.n_tool_calls, eng_result.n_bash_calls,
                    eng_result.n_edit_calls, eng_result.n_write_calls,
                    eng_result.final_stage_statuses,
                )
                # Engineer may have edited the code in-place; update pipeline.
                pipeline = eng_result.final_pipeline
                trial.pipeline = pipeline
                # Save engineer-edited code + transcript snippet for inspection.
                self._save_pipeline(artifacts_dir, trial_idx, pipeline)
                self._save_engineer_transcript(artifacts_dir, trial_idx, eng_result)
                # Re-discover stage files from the (possibly edited) pipeline.
                stage_filenames = pipeline.stage_files()

                # For each stage filename, read whatever the engineer's session
                # produced (or didn't) and build a Layer3Output for it.
                for stage_filename in stage_filenames:
                    k = stage_index(stage_filename)
                    out_k = self._layer3.read_stage_result_from_workdir(
                        work_dir=work_dir,
                        stage_filename=stage_filename,
                    )
                    stage_outputs.append(out_k)
                    self._save_stage_output(artifacts_dir, trial_idx, stage_filename, out_k)
                    log.info(
                        "trial %d: stage %02d (engineer ran) status=%s "
                        "wall_time=%.1fs delta_g=%s",
                        trial_idx, k, out_k.status,
                        out_k.wall_time_seconds or 0,
                        out_k.delta_g_kcal_per_mol,
                    )

                    if is_production_stage(stage_filename) and out_k.status == "success":
                        production_succeeded = True

                    # Note: per-stage L2 post-eval is SKIPPED when a
                    # MultiMoleculeBenchmark is configured. Engineer is
                    # only code-validation; the proper post-eval signal
                    # comes from the benchmark's cross-molecule errors
                    # (run below). When the benchmark is NOT configured
                    # we fall back to per-stage post-eval on engineer
                    # single-molecule output (legacy behaviour).
                    if self._multi_molecule_benchmark is not None:
                        # tracking only — no critique generation here
                        if out_k.parse_failed and out_k.status is None:
                            log.info(
                                "trial %d: stage %02d (%s) — engineer didn't reach this stage",
                                trial_idx, k, stage_filename,
                            )
                            failed_at = k
                            break
                        continue
                    # Legacy single-molecule post-eval path (when no benchmark)
                    if not should_invoke_m2(stage_filename):
                        continue
                    if out_k.parse_failed and out_k.status is None:
                        log.info(
                            "trial %d: stage %02d (%s) — no result file; skip M2",
                            trial_idx, k, stage_filename,
                        )
                        failed_at = k
                        break
                    # Per-stage post_eval is intermediate reward;
                    # suppressed entirely in intermediate-reward ablations.
                    if self._skip_intermediate:
                        log.info(
                            "trial %d: stage %02d — Layer2.post_eval SKIPPED (skip_intermediate=True)",
                            trial_idx, k,
                        )
                        continue
                    log.info("trial %d: stage %02d — Layer2.post_eval", trial_idx, k)
                    c_post_k = self._layer2.post_eval(
                        pipeline=pipeline, layer3_output=out_k, task=task
                    )
                    post_critiques.append(c_post_k)
                    self._save_critique(
                        artifacts_dir, trial_idx, f"l2post_stage{k:02d}", c_post_k
                    )
            else:
                # Legacy (engineer-less) mode: orchestrator runs each stage
                # in-process, post-evaluates, aborts on hard signals.
                for stage_filename in stage_filenames:
                    k = stage_index(stage_filename)
                    log.info("trial %d: stage %02d (%s) — running", trial_idx, k, stage_filename)
                    out_k = self._layer3.run_stage(
                        work_dir=work_dir,
                        pipeline=pipeline,
                        stage_filename=stage_filename,
                    )
                    stage_outputs.append(out_k)
                    self._save_stage_output(artifacts_dir, trial_idx, stage_filename, out_k)
                    log.info(
                        "trial %d: stage %02d done — exited_normally=%s status=%s "
                        "wall_time=%.1fs",
                        trial_idx, k, out_k.exited_normally, out_k.status,
                        out_k.wall_time_seconds or 0,
                    )
                    if is_production_stage(stage_filename) and out_k.exited_normally and out_k.status == "success":
                        production_succeeded = True
                    if not out_k.exited_normally and not out_k.timed_out and (out_k.status not in ("diverged",)):
                        log.warning("trial %d: stage %02d engineering crash — bypass M2", trial_idx, k)
                        failed_at = k
                        break
                    if not should_invoke_m2(stage_filename):
                        log.info(
                            "trial %d: stage %02d (%s) — skipping M2 post-eval per routing rule",
                            trial_idx, k, stage_filename,
                        )
                        if out_k.timed_out or out_k.status == "diverged":
                            log.warning(
                                "trial %d: stage %02d aborted on physical signal (status=%s timed_out=%s) — no M2 because not in invoke list",
                                trial_idx, k, out_k.status, out_k.timed_out,
                            )
                            failed_at = k
                            break
                        continue
                    # Per-stage post_eval is intermediate reward;
                    # suppressed entirely in intermediate-reward ablations.
                    if self._skip_intermediate:
                        log.info(
                            "trial %d: stage %02d — Layer2.post_eval SKIPPED (skip_intermediate=True)",
                            trial_idx, k,
                        )
                        if out_k.timed_out or out_k.status == "diverged":
                            log.warning(
                                "trial %d: stage %02d HARD ABORT — Layer 3 reports status=%s timed_out=%s",
                                trial_idx, k, out_k.status, out_k.timed_out,
                            )
                            failed_at = k
                            break
                        continue
                    log.info("trial %d: stage %02d — Layer2.post_eval", trial_idx, k)
                    c_post_k = self._layer2.post_eval(
                        pipeline=pipeline, layer3_output=out_k, task=task
                    )
                    post_critiques.append(c_post_k)
                    self._save_critique(
                        artifacts_dir, trial_idx, f"l2post_stage{k:02d}", c_post_k
                    )
                    if out_k.timed_out or out_k.status == "diverged":
                        log.warning(
                            "trial %d: stage %02d HARD ABORT — Layer 3 itself reports status=%s timed_out=%s (M2 label=%s is advisory)",
                            trial_idx, k, out_k.status, out_k.timed_out, c_post_k.label.value,
                        )
                        failed_at = k
                        break
                    elif c_post_k.label == VerdictLabel.FAIL:
                        log.info(
                            "trial %d: stage %02d — M2 post-eval label=fail (advisory only, continuing to next stage per P4)",
                            trial_idx, k,
                        )

            trial.stage_outputs = stage_outputs
            trial.post_critiques = post_critiques
            trial.failed_at_stage = failed_at

            # ---- Phase 2b: Multi-molecule benchmark + experts post-eval ----
            # Correct evaluation flow (per user spec):
            #   Engineer (single-mol code validation)
            #     → MultiMoleculeBenchmark (production MD on N guests in parallel)
            #     → 3 expert × 2 round post-eval on the cross-molecule results
            #     → expert critiques become the trial's post_critiques
            #     → next Writer.revise sees the multi-molecule signal
            # The benchmark itself also returns a rule-based Critique (with
            # MAE/RMSE/R²); both the rule-based Critique AND the experts'
            # Critique are appended for the Writer's revise prompt.
            if (
                self._multi_molecule_benchmark is not None
                and production_succeeded
                and not failed_at
            ):
                log.info(
                    "trial %d: MultiMoleculeBenchmark — applying pipeline "
                    "to %d benchmark guests on %d GPUs (production MD)",
                    trial_idx,
                    len(self._multi_molecule_benchmark.guest_indices),
                    len(self._multi_molecule_benchmark.gpu_ids),
                )
                try:
                    bench_out_dir = (
                        Path(trial.work_dir).parent
                        / f"benchmark_trial_{trial_idx:02d}"
                    ) if trial.work_dir else None
                    if bench_out_dir is None:
                        bench_out_dir = artifacts_dir / f"trial_{trial_idx:02d}" / "benchmark"
                    bench_result, bench_critique = self._multi_molecule_benchmark.evaluate(
                        pipeline=pipeline,
                        out_dir=bench_out_dir,
                    )
                    log.info(
                        "trial %d: MultiMoleculeBenchmark done — %d/%d valid; "
                        "MAE=%s; rule-based label=%s",
                        trial_idx,
                        bench_result.n_valid, len(bench_result.rows),
                        f"{bench_result.mae:.2f}" if bench_result.mae is not None else "N/A",
                        bench_critique.label.value,
                    )
                    # Save the rule-based critique
                    post_critiques.append(bench_critique)
                    self._save_critique(
                        artifacts_dir, trial_idx,
                        "l2post_benchmark_rule_based",
                        bench_critique,
                    )

                    # NOW run 3-expert × N-round post-eval on the benchmark
                    # result — this is the proper multi-agent verbal-RL
                    # signal that feeds the next Writer.revise.
                    log.info(
                        "trial %d: Layer2.post_eval_benchmark (3 expert × %d round on multi-molecule results)",
                        trial_idx, self._layer2._n_rounds,
                    )
                    expert_bench_critique = self._layer2.post_eval_benchmark(
                        pipeline=pipeline,
                        benchmark_result=bench_result,
                        task=task,
                    )
                    log.info(
                        "trial %d: Layer2.post_eval_benchmark done — verdict=%s conf=%.2f",
                        trial_idx,
                        expert_bench_critique.label.value,
                        expert_bench_critique.aggregate_confidence,
                    )
                    post_critiques.append(expert_bench_critique)
                    self._save_critique(
                        artifacts_dir, trial_idx,
                        "l2post_benchmark_experts",
                        expert_bench_critique,
                    )
                    trial.post_critiques = post_critiques
                except Exception as exc:
                    log.warning(
                        "trial %d: MultiMoleculeBenchmark or its post-eval FAILED "
                        "(advisory only — proceeding): %r",
                        trial_idx, exc,
                    )

            # Budget accounting: only count trials whose stage-04 production
            # actually ran to completion and self-reported success. Crashes,
            # divergences, and timeouts at production do NOT count: they
            # produced no real MD feedback.
            if production_succeeded:
                trial.counted_toward_budget = True
                report.budget_used += 1
                log.info("trial %d: production stage SUCCEEDED — budget %d/%d",
                         trial_idx, report.budget_used, self._budget_n)

            # ---- Phase 3: Convergence check ----
            converged = self._is_converged(failed_at, post_critiques, stage_filenames)
            trial.converged = converged
            if converged:
                report.converged = True
                report.final_pipeline = pipeline
                log.info("trial %d: CONVERGED", trial_idx)

            # ---- Phase 5: Per-trial cross-task reputation update ----
            if c_pre is not None and post_critiques:
                final_post = post_critiques[-1]
                update_reputations_from_pre_post(
                    self._reputation,
                    pre_votes=c_pre.expert_votes,
                    post_votes=final_post.expert_votes,
                    evidence={
                        "task_id": task.task_id,
                        "trial_index": trial_idx,
                        "failed_at_stage": failed_at,
                        "converged": converged,
                    },
                )
                self._save_reputation_snapshot(artifacts_dir, trial_idx)

            trial.wall_time_seconds = time.monotonic() - t0

            if converged:
                break

            # ---- Phase 4 prep: accumulate critiques for next Writer.revise ----
            # When skip_intermediate=True, c_pre is None and post_critiques
            # only contains the two benchmark-level entries (bench_critique +
            # expert_bench_critique); the per-stage post-eval entries never
            # get appended. The filter below drops the None c_pre cleanly so
            # the intermediate-reward ablations forward
            # [c_l1, bench_critique, expert_bench_critique] only.
            prior_critiques = [c for c in (c_l1, c_pre, *post_critiques) if c is not None]

        if (report.budget_used < self._budget_n and not report.converged
                and trial_idx >= self._max_trials):
            log.warning(
                "orchestrator.run: hit max_trials cap (%d) with budget %d/%d "
                "— ending cell (doom-loop guard)",
                self._max_trials, report.budget_used, self._budget_n,
            )
        log.info(
            "orchestrator.run done: converged=%s trials=%d budget_used=%d/%d",
            report.converged, len(report.trials), report.budget_used, self._budget_n,
        )
        return report

    # ---- Helpers --------------------------------------------------------

    def _is_converged(
        self,
        failed_at_stage: int | None,
        post_critiques: list[Critique],
        stage_filenames: list[str],
    ) -> bool:
        if failed_at_stage is not None:
            return False
        if not post_critiques:
            return False
        if len(post_critiques) != len(stage_filenames):
            return False
        last = post_critiques[-1]
        if last.label != VerdictLabel.PASS:
            return False
        if last.aggregate_confidence < self._confidence_threshold:
            return False
        return True

    def _setup_artifacts_dir(self, task: Task) -> Path | None:
        if self._artifacts_root is None:
            return None
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = self._artifacts_root / f"{task.task_id}-{ts}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_pipeline(self, artifacts_dir: Path | None, trial_idx: int, pipeline: Pipeline) -> None:
        if artifacts_dir is None:
            return
        d = artifacts_dir / f"trial_{trial_idx:02d}" / "pipeline"
        d.mkdir(parents=True, exist_ok=True)
        (d / "rationale.txt").write_text(pipeline.rationale or "")
        (d / "metadata.json").write_text(json.dumps({
            "revision": pipeline.revision,
            "parent_revision": pipeline.parent_revision,
            "entry_point": pipeline.entry_point,
            "stage_files": pipeline.stage_files(),
            "all_files": sorted(pipeline.code_files.keys()),
        }, indent=2))
        for filename, source in pipeline.code_files.items():
            target = d / "code" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        # Also save the Writer's full debug (system prompt, user prompt, raw response).
        if self._writer.last_debug is not None:
            dbg = self._writer.last_debug
            (d / "writer_debug.txt").write_text(
                f"=== SYSTEM PROMPT ===\n{dbg.system_prompt}\n\n"
                f"=== USER PROMPT ===\n{dbg.user_prompt}\n\n"
                f"=== RAW RESPONSE ===\n{dbg.raw_response}\n"
            )

    def _save_critique(self, artifacts_dir: Path | None, trial_idx: int, tag: str, critique: Critique) -> None:
        if artifacts_dir is None:
            return
        d = artifacts_dir / f"trial_{trial_idx:02d}" / "critiques"
        d.mkdir(parents=True, exist_ok=True)
        payload = critique.model_dump(mode="json")
        (d / f"{tag}.json").write_text(json.dumps(payload, indent=2, default=str))

    def _save_engineer_transcript(
        self,
        artifacts_dir: Path | None,
        trial_idx: int,
        result: EngineerResult,
    ) -> None:
        if artifacts_dir is None:
            return
        d = artifacts_dir / f"trial_{trial_idx:02d}" / "engineer"
        d.mkdir(parents=True, exist_ok=True)
        summary = {
            "summary_text": result.summary_text,
            "n_tool_calls": result.n_tool_calls,
            "n_bash_calls": result.n_bash_calls,
            "n_edit_calls": result.n_edit_calls,
            "n_write_calls": result.n_write_calls,
            "wall_time_seconds": result.wall_time_seconds,
            "final_stage_statuses": result.final_stage_statuses,
        }
        (d / "summary.json").write_text(json.dumps(summary, indent=2))
        (d / "transcript.txt").write_text("\n".join(result.raw_transcript))

    def _save_stage_output(
        self, artifacts_dir: Path | None, trial_idx: int, stage_filename: str, out: Layer3Output,
    ) -> None:
        if artifacts_dir is None:
            return
        d = artifacts_dir / f"trial_{trial_idx:02d}" / "stages"
        d.mkdir(parents=True, exist_ok=True)
        payload = out.model_dump(mode="json")
        (d / f"{stage_filename}.json").write_text(json.dumps(payload, indent=2, default=str))

    def _save_reputation_snapshot(self, artifacts_dir: Path | None, trial_idx: int) -> None:
        if artifacts_dir is None:
            return
        d = artifacts_dir / f"trial_{trial_idx:02d}"
        d.mkdir(parents=True, exist_ok=True)
        snapshot = {
            name: {
                "alpha": s.alpha,
                "beta": s.beta,
                "n_observations": s.n_observations,
                "posterior_mean": s.posterior_mean(),
            }
            for name, s in self._reputation.all_states().items()
        }
        (d / "reputation_after.json").write_text(json.dumps(snapshot, indent=2))


# ---- Module-level helpers -----------------------------------------------


def _collect_auxiliary_files(task: Task) -> dict[str, Path]:
    """Flatten task host/guest/complex files into the {basename: path} mapping
    that Layer3Oracle expects.

    Files are staged under BOTH their original basename AND a set of
    CANONICAL names, so Writer-emitted code can always reference the
    same input filenames regardless of which (host, guest) the task
    represents:

      complex.pdb              ← bound complex (REQUIRED)
      host.mol2 / host.pdb /
        host.sdf               ← host topology files
      guest.mol2 / guest.sdf   ← guest topology files
      task_metadata.json       ← guest_id, host_id, charges, n_sym, T, pH

    Pipelines MUST read from these canonical names so a single pipeline
    works on any task. Original-basename copies are kept for any
    backward-compat reference.
    """
    import json as _json
    import tempfile

    out: dict[str, Path] = {}
    # Host: alias every host_structure_file to host.<ext>
    for path in task.host_structure_files.values():
        out[path.name] = path
        ext = path.suffix.lstrip(".")
        if ext in ("mol2", "pdb", "sdf"):
            out[f"host.{ext}"] = path
    # Guest: alias every guest_structure_file to guest.<ext>
    for path in task.guest_structure_files.values():
        out[path.name] = path
        ext = path.suffix.lstrip(".")
        if ext in ("mol2", "sdf", "pdb"):
            out[f"guest.{ext}"] = path
    # Complex: original + canonical
    if task.bound_complex_pdb is not None:
        out[task.bound_complex_pdb.name] = task.bound_complex_pdb
        out["complex.pdb"] = task.bound_complex_pdb
    # Per-task metadata blob (charges, symmetry, T, pH)
    meta = {
        "task_id": task.task_id,
        "host_id": task.target.host_id,
        "guest_id": task.target.guest_id,
        "host_net_charge": task.extra_context.get("host_net_charge"),
        "guest_net_charge": task.extra_context.get("guest_net_charge"),
        "n_sym": task.extra_context.get("n_sym"),
        "temperature_kelvin": task.extra_context.get("temperature_kelvin"),
        "pH": task.extra_context.get("pH"),
        "atom_id": task.extra_context.get("atom_id"),
    }
    tf = tempfile.NamedTemporaryFile(
        mode="w", delete=False, prefix="task_metadata_", suffix=".json"
    )
    _json.dump(meta, tf, indent=2)
    tf.close()
    out["task_metadata.json"] = Path(tf.name)
    return out


def _build_trial_history(prior_trials: list[TrialRecord]) -> list[TrialSummary]:
    """Convert a list of TrialRecord into compact TrialSummary objects
    suitable for feeding the Writer's `revise` prompt as context.
    """
    out: list[TrialSummary] = []
    for tr in prior_trials:
        stage_summaries: list[str] = []
        deepest = -1
        for so in tr.stage_outputs:
            stage_id_str = so.stage_id or "?"
            wall = so.wall_time_seconds or 0.0
            status = so.status or ("ok" if so.exited_normally else "crash")
            stage_summaries.append(f"{stage_id_str}: {status} ({wall:.1f}s)")
            # Extract numeric stage prefix if present
            if so.stage_id and so.stage_id[:2].isdigit():
                try:
                    deepest = max(deepest, int(so.stage_id[:2]))
                except ValueError:
                    pass

        pre_insights: list[str] = []
        if tr.c_pre is not None:
            for v in tr.c_pre.expert_votes:
                if v.strategic_insight:
                    pre_insights.append(f"{v.expert}: {v.strategic_insight}")

        post_by_stage: dict[str, list[str]] = {}
        for i, c in enumerate(tr.post_critiques):
            # Stage_id from the matching stage_outputs entry if available
            stage_id = (
                tr.stage_outputs[i].stage_id
                if i < len(tr.stage_outputs) and tr.stage_outputs[i].stage_id
                else f"stage_{i}"
            )
            ins_list = [v.strategic_insight for v in c.expert_votes if v.strategic_insight]
            if ins_list:
                post_by_stage[stage_id] = [
                    f"{v.expert}: {v.strategic_insight}"
                    for v in c.expert_votes
                    if v.strategic_insight
                ]

        final_label: str | None = None
        final_summary: str = ""
        if tr.post_critiques:
            final_label = tr.post_critiques[-1].label.value
            final_summary = tr.post_critiques[-1].summary
        elif tr.c_pre is not None:
            final_label = tr.c_pre.label.value
            final_summary = tr.c_pre.summary
        elif tr.c_l1 is not None:
            final_label = tr.c_l1.label.value
            final_summary = tr.c_l1.summary

        out.append(TrialSummary(
            revision=tr.pipeline.revision,
            rationale=tr.pipeline.rationale or "",
            pre_eval_strategic_insights=pre_insights,
            post_eval_strategic_insights_by_stage=post_by_stage,
            final_critique_label=final_label,
            final_critique_summary=final_summary,
            failed_at_stage=tr.failed_at_stage,
            stage_outputs_summary=stage_summaries,
            deepest_stage_reached=deepest if deepest >= 0 else None,
        ))
    return out


def load_resume_state_from_orch_dir(
    orch_dir: Path,
    *,
    target_trial: int | None = None,
) -> tuple[Pipeline, list[Critique], list[TrialSummary]]:
    """Reconstruct (resume_pipeline, resume_critiques, resume_history)
    from a prior orchestrator artifacts directory.

    Args:
        orch_dir: path like ``/tmp/prism_run/<run-ts>/sampl4-cb7-adz-<ts>``.
        target_trial: if provided, resume from this specific trial (e.g. 1
            to seed the new run with that trial's pipeline as the
            "immediately previous"). If None, picks the latest trial that
            has a saved pipeline.

    Returns:
        (pipeline, critiques, history) where:
            - pipeline = the Pipeline object reconstructed from the chosen trial
            - critiques = the L1 + L2pre + L2post[stage_NN] critiques of that trial
            - history = TrialSummary list of trials BEFORE the chosen trial
    """
    import json as _json
    trial_dirs = sorted([p for p in orch_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")])
    if not trial_dirs:
        raise FileNotFoundError(f"No trial_* dirs under {orch_dir}")

    if target_trial is not None:
        chosen = orch_dir / f"trial_{target_trial:02d}"
    else:
        chosen = trial_dirs[-1]
    if not chosen.exists():
        raise FileNotFoundError(f"trial dir not found: {chosen}")

    # Reconstruct the pipeline
    pipe_dir = chosen / "pipeline"
    if not pipe_dir.exists():
        raise FileNotFoundError(f"No pipeline/ in {chosen}")
    meta = _json.loads((pipe_dir / "metadata.json").read_text())
    rationale = (pipe_dir / "rationale.txt").read_text() if (pipe_dir / "rationale.txt").exists() else ""
    code_files: dict[str, str] = {}
    code_dir = pipe_dir / "code"
    for f in code_dir.iterdir():
        if f.is_file():
            # Drop any 00_*.py — docking was removed in the K=4 redesign;
            # resumed pipelines from K=5 era keep stages 01-04 only.
            if f.name.startswith("00_") and f.name.endswith(".py"):
                continue
            code_files[f.name] = f.read_text()

    # Pull the target identification — use metadata to construct HostGuestSpec.
    # If metadata doesn't have target details we leave host/guest fields blank
    # (the orchestrator only uses pipeline.target.host_id/guest_id for naming).
    pipeline = Pipeline(
        target=HostGuestSpec(
            host_id=meta.get("target_host_id", "cb7"),
            guest_id=meta.get("target_guest_id", "adz"),
        ),
        code_files=code_files,
        entry_point=meta.get("entry_point", "01_prep.py"),
        revision=meta.get("revision", 0),
        parent_revision=meta.get("parent_revision"),
        rationale=rationale,
    )

    # Reconstruct critiques (L1, L2pre, L2post_stageNN, L2post_benchmark_*)
    # The benchmark critiques are saved by the orchestrator as
    # l2post_benchmark_rule_based.json and l2post_benchmark_experts.json
    # — they MUST be loaded on resume so the next Writer.revise sees the
    # cross-molecule signal (the whole point of the multi-molecule loop).
    critiques: list[Critique] = []
    crit_dir = chosen / "critiques"
    if crit_dir.exists():
        order = ["l1.json", "l2pre.json"] + sorted(
            f.name for f in crit_dir.iterdir() if f.name.startswith("l2post_stage")
        ) + sorted(
            f.name for f in crit_dir.iterdir() if f.name.startswith("l2post_benchmark")
        )
        for fname in order:
            f = crit_dir / fname
            if not f.exists():
                continue
            data = _json.loads(f.read_text())
            critiques.append(Critique(**data))

    # Build TrialSummary history from EARLIER trials only
    earlier_trial_dirs = [t for t in trial_dirs if t.name < chosen.name]
    history: list[TrialSummary] = []
    for tdir in earlier_trial_dirs:
        try:
            history.append(_trial_dir_to_summary(tdir))
        except Exception as exc:
            log.warning("Failed to load trial summary from %s: %s", tdir, exc)
            continue

    return pipeline, critiques, history


def _trial_dir_to_summary(trial_dir: Path) -> TrialSummary:
    """Convert one saved trial_NN dir into a compact TrialSummary."""
    import json as _json
    pipe_dir = trial_dir / "pipeline"
    meta = _json.loads((pipe_dir / "metadata.json").read_text()) if (pipe_dir / "metadata.json").exists() else {}
    rationale = (pipe_dir / "rationale.txt").read_text() if (pipe_dir / "rationale.txt").exists() else ""

    stage_outputs_summary: list[str] = []
    deepest = None
    stages_dir = trial_dir / "stages"
    if stages_dir.exists():
        for f in sorted(stages_dir.iterdir()):
            if f.suffix != ".json":
                continue
            data = _json.loads(f.read_text())
            sid = data.get("stage_id", f.stem)
            wt = data.get("wall_time_seconds") or 0
            status = data.get("status") or ("ok" if data.get("exited_normally") else "crash")
            stage_outputs_summary.append(f"{sid}: {status} ({wt:.1f}s)")
            if sid and sid[:2].isdigit():
                try:
                    n = int(sid[:2])
                    deepest = n if deepest is None else max(deepest, n)
                except ValueError:
                    pass

    pre_insights: list[str] = []
    crit_dir = trial_dir / "critiques"
    if (crit_dir / "l2pre.json").exists():
        pre = _json.loads((crit_dir / "l2pre.json").read_text())
        for v in pre.get("expert_votes", []):
            ins = v.get("strategic_insight", "")
            if ins:
                pre_insights.append(f"{v.get('expert')}: {ins}")

    post_by_stage: dict[str, list[str]] = {}
    if crit_dir.exists():
        for f in sorted(crit_dir.iterdir()):
            if not f.name.startswith("l2post_stage"):
                continue
            stage_id = f.stem.replace("l2post_", "")
            data = _json.loads(f.read_text())
            ins_list = [
                f"{v.get('expert')}: {v.get('strategic_insight','')}"
                for v in data.get("expert_votes", [])
                if v.get("strategic_insight")
            ]
            if ins_list:
                post_by_stage[stage_id] = ins_list

    final_label = None
    final_summary = ""
    if crit_dir.exists():
        last_post = sorted(f for f in crit_dir.iterdir() if f.name.startswith("l2post_stage"))
        if last_post:
            data = _json.loads(last_post[-1].read_text())
            final_label = data.get("label")
            final_summary = data.get("summary", "")

    return TrialSummary(
        revision=meta.get("revision", 0),
        rationale=rationale,
        pre_eval_strategic_insights=pre_insights,
        post_eval_strategic_insights_by_stage=post_by_stage,
        final_critique_label=final_label,
        final_critique_summary=final_summary,
        failed_at_stage=None,  # we don't reconstruct this; mark None
        stage_outputs_summary=stage_outputs_summary,
        deepest_stage_reached=deepest,
    )


def _make_no_stages_critique() -> Critique:
    """Synthesise a FAIL critique when the Writer didn't emit any \\d{2}_*.py files."""
    from src.models import Concern

    return Critique(
        layer=LayerName.LAYER_2_PRE,
        label=VerdictLabel.FAIL,
        aggregate_confidence=1.0,
        concerns=[Concern(
            layer=LayerName.LAYER_2_PRE,
            severity=1.0,
            description=(
                "Writer produced no stage entry files matching the convention "
                "`^\\d{2}_.*\\.py$`. The pipeline must have at least one stage "
                "entry (typically 00_dock.py through 05_analysis.py)."
            ),
            suggested_focus="output_contract",
        )],
        summary="No stage entries; cannot execute.",
    )
