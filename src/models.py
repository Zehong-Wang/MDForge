"""Core data structures shared across the MDForge system.

The architecture has a single Pipeline-Writer agent (the Code agent) and a
verification ladder (Layer 1 constraint solver, Layer 2 multi-expert debate,
Layer 3 expensive oracle); these three layers map to the paper's per-stage
pipeline (Prep / Equilibration / Production / Analysis). Every module produces
or consumes one of the structures below.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VerdictLabel(str, Enum):
    """Outcome of a verification step."""

    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


class LayerName(str, Enum):
    """The three verification layers."""

    LAYER_1 = "layer_1"
    LAYER_2_PRE = "layer_2_pre"
    LAYER_2_POST = "layer_2_post"
    LAYER_3 = "layer_3"


class HostGuestSpec(BaseModel):
    """Identification of a host-guest system."""

    host_id: str
    guest_id: str
    host_smiles: str | None = None
    guest_smiles: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Pipeline(BaseModel):
    """An MD pipeline produced by the Writer agent.

    The Writer's output is *executable code*, not a structured config.
    `code_files` maps relative file paths to source. `entry_point` is the
    nominal entry point (informational; for staged execution the harness
    discovers stages by filename pattern, not by this field).

    Stage discovery convention: any file matching ``^\\d{2}_.*\\.py$``
    (e.g. ``00_dock.py``, ``01_setup.py``, ``05_analysis.py``) is a
    stage entry point. They are run in lexicographic order of filename.
    Other Python files (e.g. ``common.py``) are imported by stages but
    not run directly.
    """

    target: HostGuestSpec
    code_files: dict[str, str]
    entry_point: str
    revision: int = 0
    parent_revision: int | None = None
    rationale: str = ""

    def stage_files(self) -> list[str]:
        """Return ordered list of stage entry filenames (``^\\d{2}_.*\\.py$``).

        Order is lexicographic, which matches numeric order for two-digit
        prefixes. Non-stage Python files (e.g. helpers) are excluded.
        """
        import re

        stage_re = re.compile(r"^\d{2}_.*\.py$")
        return sorted(name for name in self.code_files if stage_re.match(name))


class Concern(BaseModel):
    """One specific issue raised by a verifier."""

    layer: LayerName
    expert: str | None = None
    severity: float = Field(ge=0.0, le=1.0)
    description: str
    suggested_focus: str | None = None


class ExpertVote(BaseModel):
    """One Layer-2 expert's verdict on a candidate pipeline.

    The expert plays a *co-design* role, not a pure verifier. The
    ``strategic_insight`` field is the load-bearing one — a narrative,
    multi-sentence assessment of the design from this expert's domain
    perspective, including alternative-method suggestions and literature
    context when relevant. ``concerns`` enumerates concrete fixable
    issues *after* the strategic assessment.
    """

    expert: str
    label: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    strategic_insight: str = ""
    """Narrative strategic-level assessment. What's the best approach for
    THIS specific system class from this expert's domain perspective?
    What would they choose differently? What's the load-bearing physical
    assumption? Cite domain literature precedent when applicable.
    REQUIRED for co-designer mode."""
    concerns: list[Concern] = Field(default_factory=list)
    reasoning: str = ""


class Critique(BaseModel):
    """Verifier output handed back to the Writer agent.

    A Critique is layer-typed: the Writer prompt is taught to react
    differently to Layer-1 (engineering), Layer-2 (methodology), and
    Layer-3 (empirical) feedback.
    """

    layer: LayerName
    label: VerdictLabel
    aggregate_confidence: float = Field(ge=0.0, le=1.0)
    concerns: list[Concern] = Field(default_factory=list)
    expert_votes: list[ExpertVote] = Field(default_factory=list)
    summary: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Layer3Output(BaseModel):
    """Raw output from the expensive simulation oracle.

    Produced by running the Writer's code in a sandbox. Not consumed
    directly by the Writer. The Layer-2 experts in their post-eval role
    translate this into a Critique.

    Status flags distinguish failure modes:
      - exited_normally: subprocess returned 0
      - timed_out: hit wall-clock limit
      - parse_failed: completed but result.json missing or malformed

    For staged execution (PRISM), one Layer3Output is produced per stage.
    `stage_id` carries the stage filename (e.g. ``"01_setup"``); `status`
    carries the writer-self-reported status (``"success"`` /
    ``"diverged"`` / ``"timeout"``); engineering errors (Python crash)
    are recorded via ``exited_normally=False`` + ``return_code != 0``.
    """

    stage_id: str | None = None
    status: str | None = None
    writer_notes: str | None = None
    delta_g_kcal_per_mol: float | None = None
    delta_g_uncertainty: float | None = None
    convergence_flags: dict[str, bool] = Field(default_factory=dict)
    replicate_consistency: float | None = None
    energy_components: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    wall_time_seconds: float | None = None
    artifacts_dir: str | None = None
    exited_normally: bool = False
    timed_out: bool = False
    parse_failed: bool = False
    return_code: int | None = None
    stdout_tail: str | None = None
    stderr_tail: str | None = None


class TaskState(BaseModel):
    """Mutable state of a single pipeline-design task as it iterates."""

    target: HostGuestSpec
    pipeline_history: list[Pipeline] = Field(default_factory=list)
    critique_history: list[Critique] = Field(default_factory=list)
    layer3_history: list[Layer3Output] = Field(default_factory=list)
    expensive_calls_used: int = 0
    converged: bool = False
    converged_pipeline_revision: int | None = None


class TrialSummary(BaseModel):
    """Compact summary of one prior trial.

    Used to feed `Writer.revise` a multi-trial history so it can learn
    across iterations, not just from the immediately-previous trial.

    Intentionally smaller than a full ``TrialRecord``: full code files
    of every prior trial would blow up the Writer's context. The
    rationale + critique summaries + failure point are enough for the
    Writer to understand what was tried, what failed, and what experts
    suggested.
    """

    revision: int
    rationale: str
    pre_eval_strategic_insights: list[str] = Field(default_factory=list)
    post_eval_strategic_insights_by_stage: dict[str, list[str]] = Field(default_factory=dict)
    final_critique_label: str | None = None
    final_critique_summary: str = ""
    failed_at_stage: int | None = None
    stage_outputs_summary: list[str] = Field(default_factory=list)
    """One line per stage: ``stage_id: status (wall_time s)``"""
    deepest_stage_reached: int | None = None
