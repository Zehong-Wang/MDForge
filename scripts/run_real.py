#!/usr/bin/env python
"""End-to-end MDForge run on SAMPL4 CB[7] + 1-adamantylazanium.

Configuration:
  - K=5 stage layout: 00_dock / 01_prep / 02_equilibrate / 03_production / 04_analysis
  - Co-designer expert prompts
  - Writer.revise sees full multi-trial history
  - M2 routing skips dock + equilibrate (multi-agent only on insight stages)
  - Budget metric = trials with production stage SUCCESS
  - Pre-eval + post-eval critique labels are ADVISORY ONLY. Only Layer-1
    (objective engineering bugs) and Layer-3 hard physical signals
    (status=diverged/timeout/exit≠0) can commit-abort. Multi-agent is
    interpretation, never prediction-based gating.
  - 3 experts: force_field / sampling / analysis-with-restraints. Restraint
    expertise is folded into the analysis prompt (shared thermodynamic-cycle
    reasoning surface).
  - Parallel multi-agent inference: each round of debate runs all experts
    concurrently via ThreadPoolExecutor; each thread carries its own
    claude_agent_sdk subprocess.

Components:
  - Writer:  Claude Opus via claude_agent_sdk (uses user's subscription)
  - 3 Experts (FF / Sampling / Analysis-with-restraints):
      Claude Sonnet via claude_agent_sdk
  - Layer 1: AST/static check (no LLM)
  - Layer 3: sandboxed staged execution via the conda env's python
  - Reputation: Beta(1,1) prior, per-trial pre/post update (slow loop)

Artifacts: <artifacts_root>/<timestamp>/
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Make src/ importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from src.llm import ClaudeSDKClient
from src.llm.usage_logger import UsageLogger, set_global_usage_logger
from src.orchestrator import Orchestrator, load_resume_state_from_orch_dir
from src.tasks import sampl4_cb7_adamantyl_amine, task_from_manifest
from src.verifiers.layer2.debate import Layer2Debate
from src.verifiers.layer2.experts.analysis import AnalysisExpert
from src.verifiers.layer2.experts.force_field import ForceFieldExpert
from src.verifiers.layer2.experts.sampling import SamplingExpert
# RestraintExpert dropped — restraint expertise folded into
# analysis expert. Rationale: restraint algebra (Boresch, APR
# standard-state correction) and free-energy estimator algebra share
# the same thermodynamic-cycle reasoning surface; collapsing them gives
# 3 experts of distinct expertise (FF / sampling / analysis-with-restraints)
# and saves one LLM call per debate round.
from src.verifiers.layer2.reputation import ExpertReputationTracker
from src.verifiers.layer3 import ExecutorConfig, Layer3Oracle
from src.verifiers.multi_molecule_benchmark import MultiMoleculeBenchmark
from src.writer import PipelineWriter
from src.writer_engineer import PipelineEngineer


# ---- Constants -----------------------------------------------------------

WRITER_MODEL = "claude-opus-4-7"
EXPERT_MODEL = "claude-sonnet-4-6"

# MD execution environment (Python + AmberTools + OpenMM). Defaults to a
# `pipeline/` conda env under the repo root; override with PRISM_PIPELINE.
PIPELINE_ROOT = os.environ.get("PRISM_PIPELINE", str(HERE.parent / "pipeline"))
CB7_PIPELINE_PYTHON = f"{PIPELINE_ROOT}/bin/python"

# How much wall-clock to allow each stage. Production stage 04 is the
# expensive one; capped at 2h so analysis stage 05 has room. Other stages
# either succeed in seconds or crash quickly (engineering bugs surface
# immediately), so a single 2h cap on everything is fine in practice.
STAGE_TIMEOUT_SECONDS = 2 * 60 * 60  # 2h per stage

# Writer's time hint: budget the WHOLE pipeline within ~2.5h (production
# is the lion's share). Tight, but realistic for SAMPL host-guest if the
# Writer chooses 8-10 windows × ~100ps per window on a single A40.
WRITER_TIME_BUDGET_MIN = 150  # 2.5h


def main() -> int:
    ts = time.strftime("%Y%m%d-%H%M%S")
    # artifacts_root is env-driven so the paper launcher can route
    # each (host, ablation) cell into its own directory tree.
    root_base = Path(os.environ.get("PRISM_RUN_ROOT", "/tmp/prism_run"))
    artifacts_root = root_base / ts
    artifacts_root.mkdir(parents=True, exist_ok=True)

    log_path = artifacts_root / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("run_real")
    log.info("=== MDForge run starting ===")
    log.info("artifacts_root = %s", artifacts_root)

    # ---- Make AmberTools + the right Python interpreter discoverable to
    # ANY subprocess spawned from this orchestrator (including the
    # Engineer's bundled-CLI Bash calls, which inherit our env). The
    # Layer3 stage executor ALSO needs these, but extra_env there is
    # already set below; mirroring them on os.environ is what lets the
    # Engineer's `python 01_prep.py` find antechamber/tleap/parmchk2.
    CB7_BIN = f"{PIPELINE_ROOT}/bin"
    os.environ["PATH"] = CB7_BIN + ":" + os.environ.get("PATH", "")
    os.environ["AMBERHOME"] = PIPELINE_ROOT
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("PRISM_GPU", "0"))
    os.environ.setdefault("OPENMM_PLATFORM", "CUDA")

    # The bundled claude CLI clamps Bash-tool timeouts to a
    # 600 s hard cap by default; long MD stage runs in the Engineer phase exceed
    # that and the CLI's execa timeout SIGKILLs the whole process group (exit -9).
    # Raise the cap so the engineer's Bash(timeout=...) calls are honored.
    os.environ["BASH_MAX_TIMEOUT_MS"] = str(3 * 60 * 60 * 1000)    # 3 h hard cap
    os.environ["BASH_DEFAULT_TIMEOUT_MS"] = str(45 * 60 * 1000)    # 45 min default

    # ---- Token usage logger ----
    # All LLM calls (writer, experts, engineer) will append one JSONL row
    # per API request to this file. Use scripts/aggregate_tokens.py to
    # summarize after the run.
    usage_log_path = artifacts_root / "token_usage.jsonl"
    set_global_usage_logger(UsageLogger(usage_log_path))
    log.info("token usage log: %s", usage_log_path)

    # ---- LLM clients --------------------------------------------------
    # Writer: give it WebSearch + WebFetch so it can do its Step 0
    # literature reconnaissance before choosing a method family.
    # Experts: text-only as before.
    writer_llm = ClaudeSDKClient(
        default_model=WRITER_MODEL,
        allowed_tools=["WebSearch", "WebFetch"],
    )
    # Experts also get WebSearch + WebFetch so they can cite literature
    # during Phase-0 design discussion AND during pre-/post-eval.
    expert_llm = ClaudeSDKClient(
        default_model=EXPERT_MODEL,
        allowed_tools=["WebSearch", "WebFetch"],
    )

    # ---- Writer -------------------------------------------------------
    writer = PipelineWriter(
        llm_client=writer_llm,
        max_tokens=16384,   # generous; multi-stage pipeline code can be long
        temperature=0.2,    # a touch of variation for revisions
    )

    # ---- Ablation flags ----------------------------------------------
    # PRISM_ABLATION ∈ {full, v1_final_only, v2_single_agent, v3_final_single}
    #   skip_intermediate  → kill design_discussion + pre_eval + per-stage post_eval
    #   single_agent_mode  → replace 3-expert debate with 1 GeneralistExpert × 1 round
    # The benchmark + post_eval_benchmark calls (final reward) ALWAYS run.
    ablation = os.environ.get("PRISM_ABLATION", "full")
    allowed_ablations = {"full", "v1_final_only", "v2_single_agent", "v3_final_single"}
    if ablation not in allowed_ablations:
        log.warning(
            "PRISM_ABLATION=%r is not one of %s; falling back to 'full'",
            ablation, sorted(allowed_ablations),
        )
        ablation = "full"
    skip_intermediate = ablation in ("v1_final_only", "v3_final_single")
    single_agent_mode = ablation in ("v2_single_agent", "v3_final_single")
    log.info(
        "ablation = %s  (skip_intermediate=%s, single_agent_mode=%s)",
        ablation, skip_intermediate, single_agent_mode,
    )

    # ---- 3 Experts + reputation + debate ------------------------------
    experts = [
        ForceFieldExpert(llm_client=expert_llm),
        SamplingExpert(llm_client=expert_llm),
        AnalysisExpert(llm_client=expert_llm),  # incl. restraint expertise
    ]
    reputation = ExpertReputationTracker(
        expert_names=[e.NAME for e in experts],
        persist_path=str(artifacts_root / "reputation.json"),
    )
    layer2 = Layer2Debate(
        experts=experts,
        reputation=reputation,
        n_rounds=2,
        single_agent_mode=single_agent_mode,
    )

    # ---- Layer 3: staged sandbox executor -----------------------------
    work_root = artifacts_root / "workdirs"
    work_root.mkdir(exist_ok=True)
    # Activate the pipeline conda env's bin directory on PATH so the
    # Writer's code can invoke AmberTools binaries (antechamber, tleap,
    # parmchk2, sander) as subprocesses. The python_executable below
    # already points to the pipeline env's Python interpreter (OpenMM /
    # OpenFF / parmed importable), but external binaries need PATH.
    import os as _os
    CB7_BIN = f"{PIPELINE_ROOT}/bin"
    layer3 = Layer3Oracle(ExecutorConfig(
        work_root=work_root,
        stage_timeout_seconds=STAGE_TIMEOUT_SECONDS,
        keep_artifacts=True,
        python_executable=CB7_PIPELINE_PYTHON,
        extra_env={
            # ONE GPU per molecule (writer_system.md pitfall #8 enforces
            # no intra-molecule GPU split). Different molecules / runs go
            # on different GPUs at the harness level, not inside one pipeline.
            "CUDA_VISIBLE_DEVICES": os.environ.get("PRISM_GPU", "0"),
            "OPENMM_PLATFORM": "CUDA",
            "OPENMM_CPU_THREADS": "4",
            # Make AmberTools binaries discoverable on subprocess PATH.
            "PATH": CB7_BIN + ":" + _os.environ.get("PATH", ""),
            "AMBERHOME": PIPELINE_ROOT,
        },
    ))

    # ---- PipelineEngineer (tool-using agent) --------------------------
    # CC-style engineer that drives the pipeline end-to-end in the sandbox,
    # using Read/Edit/Write/Bash to fix engineering bugs until stage 04
    # produces a ΔG. Persistent session via claude_agent_sdk.ClaudeSDKClient.
    engineer = PipelineEngineer(
        default_model=WRITER_MODEL,  # use Opus for engineering judgement
        max_turns=250,
        time_budget_seconds=60 * 60,  # 60 min wall-clock per trial
    )

    # ---- Multi-molecule benchmark verifier ----------------------------
    # After the Engineer's single-molecule trial, this applies the same
    # pipeline to 4 OTHER guests in parallel on GPUs 0-3.
    # The aggregate cross-molecule ΔG-vs-ITC errors are packaged as a
    # Layer-2 post-eval Critique that the NEXT Writer.revise sees —
    # forcing the verbal-RL loop to iterate on *cross-molecule* signal,
    # not single-molecule signal alone. This is the structural fix for
    # the "single-molecule over-fit" failure mode.
    #
    # atom / guest set / ns_per_window / per-task timeout are all
    # env-driven now so the paper launcher can dispatch the same script
    # across cb7 / oah / cbclip hosts and different benchmark splits.
    atom_id = os.environ.get("PRISM_ATOM", "sampl4_cb7")
    multi_benchmark = MultiMoleculeBenchmark(
        atom_id=atom_id,
        guest_indices=[
            int(x) for x in os.environ.get(
                "PRISM_BENCHMARK_GUESTS", "6,7,10,11"
            ).split(",")
            if x.strip()
        ],
        gpu_ids=[0, 1, 2, 3],
        ns_per_window=float(os.environ.get("PRISM_NS_PER_WINDOW", "1.0")),
        per_task_timeout_seconds=int(os.environ.get(
            "PRISM_BENCHMARK_TIMEOUT_SEC", str(90 * 60)
        )),
    )

    # ---- Orchestrator -------------------------------------------------
    orchestrator = Orchestrator(
        writer=writer,
        layer2=layer2,
        layer3=layer3,
        reputation=reputation,
        engineer=engineer,
        multi_molecule_benchmark=multi_benchmark,
        # Budget = "trials whose stage 04 actually succeeded"
        # (NOT "trials that started stage 04"). This is the corrected
        # accounting — a 3-second PME crash at production setup does not
        # consume budget.
        budget_n=int(os.environ.get("PRISM_BUDGET_N", "2")),
        confidence_threshold=0.6,
        artifacts_root=artifacts_root,
        skip_intermediate=skip_intermediate,
    )

    # Task selection from the SAMPL benchmark manifest.
    # PRISM_ATOM env var selects the (SAMPL,host) atom (default sampl4_cb7);
    # already read above so MultiMoleculeBenchmark sees the same value.
    # PRISM_GUEST_IDX selects the guest index within that atom (default 13).
    # See data/sampl_benchmark/manifest.json for the catalogue.
    guest_idx = int(os.environ.get("PRISM_GUEST_IDX", "13"))  # 13 = adz under sampl4_cb7
    try:
        task = task_from_manifest(atom_id=atom_id, guest_idx=guest_idx)
    except (KeyError, IndexError, FileNotFoundError) as exc:
        log.warning("manifest task load failed (%r) — falling back to sampl4_cb7_adamantyl_amine()", exc)
        task = sampl4_cb7_adamantyl_amine()
    log.info("task = %s, target = %s + %s (atom=%s)", task.task_id, task.target.host_id, task.target.guest_id, atom_id)
    log.info("experimental reference ΔG = %.3f ± %.3f kcal/mol (HELD ASIDE)",
             task.experimental_reference.delta_g_kcal_per_mol,
             task.experimental_reference.delta_g_uncertainty_kcal_per_mol)

    # Optional: resume from a prior orchestrator run by setting RESUME_FROM env var.
    # Format:  RESUME_FROM=<orch_dir>:<trial_idx>   (trial_idx optional, defaults to 1)
    # Example: RESUME_FROM=/tmp/prism_run/20260511-212458/sampl4-cb7-adz-20260511-212458:1
    resume_pipeline = None
    resume_critiques = None
    resume_history = None
    resume_env = os.environ.get("RESUME_FROM")
    if resume_env:
        if ":" in resume_env:
            orch_path_str, trial_str = resume_env.rsplit(":", 1)
            target_trial = int(trial_str)
        else:
            orch_path_str = resume_env
            target_trial = 1
        from pathlib import Path as _P
        orch_path = _P(orch_path_str)
        log.info("RESUMING from %s, target_trial=%d", orch_path, target_trial)
        resume_pipeline, resume_critiques, resume_history = load_resume_state_from_orch_dir(
            orch_path, target_trial=target_trial
        )
        # Patch the loaded pipeline's target with the live task target so
        # work-dir naming and downstream code use the right host_id/guest_id
        resume_pipeline = resume_pipeline.model_copy(update={"target": task.target})
        log.info(
            "RESUME state: pipeline rev=%d (%d code files), %d critiques, %d earlier-trial summaries",
            resume_pipeline.revision, len(resume_pipeline.code_files),
            len(resume_critiques), len(resume_history),
        )

    t0 = time.time()
    report = orchestrator.run(
        task,
        resume_pipeline=resume_pipeline,
        resume_critiques=resume_critiques,
        resume_history=resume_history,
    )
    dt = time.time() - t0

    log.info("=== run done in %.1f s ===", dt)
    log.info("converged = %s", report.converged)
    log.info("trials    = %d", len(report.trials))
    log.info("budget    = %d / %d production-completing", report.budget_used, report.budget_max)
    log.info("artifacts = %s", report.artifacts_dir)

    # Sentinel file so external monitor can detect completion.
    (artifacts_root / "DONE").write_text(
        f"converged={report.converged}\n"
        f"trials={len(report.trials)}\n"
        f"budget_used={report.budget_used}/{report.budget_max}\n"
        f"elapsed={dt:.1f}s\n"
        f"final_pipeline_rev={report.final_pipeline.revision if report.final_pipeline else 'n/a'}\n"
    )

    return 0 if report.converged else 1


if __name__ == "__main__":
    sys.exit(main())
