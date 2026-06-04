"""Pipeline-Writer agent: proposes and revises MD pipelines.

The Writer is the only proposer in the architecture --- a single LLM-driven
agent that authors complete simulation code. Layer 1 statically analyses
that code, Layer 2 has a multi-expert debate over its methodology, and
Layer 3 runs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.llm.base import LLMClient, LLMMessage, LLMRole, NoLLMClient
from src.models import Critique, LayerName, Pipeline, TrialSummary
from src.parsing import ParsedWriterOutput, parse_writer_output

if TYPE_CHECKING:
    from src.tasks import Task

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
WRITER_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "writer_system.md"

DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0


@dataclass
class WriterDebugInfo:
    """Optional inspection blob attached to a Pipeline.

    Useful for logging the exact prompt/response that produced a revision.
    Not stored on Pipeline directly to keep the Pydantic model lean.
    """

    system_prompt: str
    user_prompt: str
    raw_response: str
    parsed: ParsedWriterOutput
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class PipelineWriter:
    """Single-agent pipeline proposer/reviser."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        system_prompt_path: Path = WRITER_SYSTEM_PROMPT_PATH,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self._llm: LLMClient = llm_client or NoLLMClient()
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt_path.read_text()
        self._last_debug: WriterDebugInfo | None = None

    def propose(
        self,
        task: "Task",
        time_budget_minutes: int = 60,
        design_discussion: Critique | None = None,
    ) -> Pipeline:
        """Author an initial Pipeline for the given task.

        ``design_discussion`` (optional): Phase-0 expert recommendations
        produced by ``Layer2Debate.design_discussion(task)`` — the
        Writer reads them as starting context for its method choice.
        Still ADVISORY (per P4): Writer is free to disagree.
        """
        user_prompt = self._render_propose_prompt(
            task, time_budget_minutes, design_discussion
        )
        response = self._llm.complete(
            messages=[LLMMessage(role=LLMRole.USER, content=user_prompt)],
            system=self._system_prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            caller_tag="writer.propose",
        )
        parsed = parse_writer_output(response.content)
        self._last_debug = WriterDebugInfo(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            raw_response=response.content,
            parsed=parsed,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return Pipeline(
            target=task.target,
            code_files=parsed.code_files,
            entry_point=parsed.entry_point,
            revision=0,
            parent_revision=None,
            rationale=parsed.rationale,
        )

    def revise(
        self,
        previous: Pipeline,
        critiques: list[Critique],
        task: "Task",
        time_budget_minutes: int = 60,
        trial_history: list[TrialSummary] | None = None,
    ) -> Pipeline:
        """Produce a revised Pipeline given critiques on the previous one.

        ``trial_history`` is the chronological list of summaries of
        ALL trials BEFORE ``previous``. Pass it so the Writer can see
        the full revision arc (not just one trial back). The
        immediately-previous trial's full code + critiques is conveyed
        via ``previous`` + ``critiques`` (full detail); earlier trials
        only get the compact ``TrialSummary`` form to keep context
        bounded.
        """
        if not critiques:
            raise ValueError("revise() called with no critiques")
        user_prompt = self._render_revise_prompt(
            previous, critiques, task, time_budget_minutes, trial_history or []
        )
        response = self._llm.complete(
            messages=[LLMMessage(role=LLMRole.USER, content=user_prompt)],
            system=self._system_prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            caller_tag=f"writer.revise.rev_{previous.revision + 1}",
        )
        parsed = parse_writer_output(response.content)
        self._last_debug = WriterDebugInfo(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            raw_response=response.content,
            parsed=parsed,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return Pipeline(
            target=task.target,
            code_files=parsed.code_files,
            entry_point=parsed.entry_point,
            revision=previous.revision + 1,
            parent_revision=previous.revision,
            rationale=parsed.rationale,
        )

    @property
    def last_debug(self) -> WriterDebugInfo | None:
        return self._last_debug

    def _render_propose_prompt(
        self,
        task: "Task",
        time_budget_minutes: int,
        design_discussion: Critique | None = None,
    ) -> str:
        return _propose_prompt_template(
            task, time_budget_minutes, design_discussion
        )

    def _render_revise_prompt(
        self,
        previous: Pipeline,
        critiques: list[Critique],
        task: "Task",
        time_budget_minutes: int,
        trial_history: list[TrialSummary],
    ) -> str:
        return _revise_prompt_template(
            previous, critiques, task, time_budget_minutes, trial_history
        )


# ---------------------------------------------------------------------------
# Prompt rendering helpers (kept module-level so they are unit-testable)
# ---------------------------------------------------------------------------


def _propose_prompt_template(
    task: "Task",
    time_budget_minutes: int,
    design_discussion: Critique | None = None,
) -> str:
    parts = [
        f"# Task `{task.task_id}`",
        "",
        task.description,
        "",
        "## Target",
        f"- Host: `{task.target.host_id}` (long name: "
        f"{task.target.metadata.get('host_long_name', 'n/a')})",
        f"- Guest: `{task.target.guest_id}` (long name: "
        f"{task.target.metadata.get('guest_long_name', 'n/a')})",
    ]
    if task.target.guest_smiles:
        parts.append(f"- Guest SMILES: `{task.target.guest_smiles}`")
    if task.target.host_smiles:
        parts.append(f"- Host SMILES: `{task.target.host_smiles}`")
    parts.append("")

    parts.append("## Conditions and constraints")
    for k, v in task.extra_context.items():
        parts.append(f"- {k}: {v}")
    parts.append(f"- wall-clock time budget: {time_budget_minutes} minutes")
    parts.append("")

    parts.append("## Auxiliary files staged in your working directory")
    file_lines = []
    for path in task.host_structure_files.values():
        file_lines.append(f"- `{path.name}` (host structure)")
    for path in task.guest_structure_files.values():
        file_lines.append(f"- `{path.name}` (guest structure)")
    if task.bound_complex_pdb:
        file_lines.append(f"- `{task.bound_complex_pdb.name}` (bound complex pose)")
    if file_lines:
        parts.extend(file_lines)
    else:
        parts.append("- (none)")
    parts.append("")
    parts.append(
        "Open these by basename. Do NOT attempt to download anything. "
        "Other Python tools (OpenMM, OpenFF Toolkit, openmmtools, alchemlyb, "
        "RDKit, MDAnalysis) are importable but offline."
    )
    parts.append("")

    if design_discussion is not None and design_discussion.expert_votes:
        parts.append("## Phase-0 expert design recommendations")
        parts.append(
            "Before you started, the multi-expert panel ran a Round-1+Round-2 "
            "design discussion specifically about THIS task. Each expert below "
            "independently recommended a method/family (with literature consulted "
            "via WebSearch). Read their `strategic_insight`s. You retain final "
            "decision authority (P4: their input is advisory) but you should "
            "engage with their recommendations explicitly in your `### RATIONALE` "
            "— either following them, or stating clearly why you diverge."
        )
        parts.append("")
        for v in design_discussion.expert_votes:
            parts.append(f"### {v.expert} — {v.label.value} (confidence {v.confidence:.2f})")
            if v.strategic_insight:
                parts.append(f"**Recommendation:** {v.strategic_insight}")
            if v.concerns:
                parts.append("Concrete concerns / pitfalls to avoid:")
                for c in v.concerns:
                    focus = f" [focus: {c.suggested_focus}]" if c.suggested_focus else ""
                    parts.append(f"- [severity {c.severity:.2f}]{focus} {c.description}")
            if v.reasoning:
                parts.append(f"Reasoning summary: {v.reasoning}")
            parts.append("")
        parts.append(
            f"Aggregate verdict: **{design_discussion.label.value}** "
            f"(panel confidence {design_discussion.aggregate_confidence:.2f}). "
            f"{design_discussion.summary}"
        )
        parts.append("")

    parts.append(
        "Now produce your design and code, following the output contract "
        "exactly (`### RATIONALE`, `### ENTRY`, one or more `### FILE: ...` blocks)."
    )
    return "\n".join(parts)


def _revise_prompt_template(
    previous: Pipeline,
    critiques: list[Critique],
    task: "Task",
    time_budget_minutes: int,
    trial_history: list[TrialSummary] | None = None,
) -> str:
    parts = [
        f"# Task `{task.task_id}` --- revision {previous.revision + 1}",
        "",
        task.description,
        "",
    ]

    # Multi-trial history: condensed summaries of all earlier attempts
    # (before `previous`). This is how the Writer carries learning
    # across the full revision arc, not just one trial back.
    earlier = list(trial_history or [])
    if earlier:
        parts.append("## Earlier attempts (chronological)")
        parts.append("")
        parts.append(
            "These are condensed summaries of all prior attempts BEFORE the most recent "
            "one shown below. Read them to understand the revision arc: what was tried, "
            "what failed where, and what expert co-designers recommended. Do not repeat "
            "the same mistake twice; do build on the strategic suggestions across trials."
        )
        parts.append("")
        for h in earlier:
            parts.append(f"### Rev {h.revision}")
            if h.rationale:
                # Keep concise to bound context
                short = h.rationale[:500] + ("…" if len(h.rationale) > 500 else "")
                parts.append(f"- Rationale: {short}")
            if h.stage_outputs_summary:
                parts.append(f"- Stage progress: {' → '.join(h.stage_outputs_summary)}")
            if h.failed_at_stage is not None:
                parts.append(f"- Failed at stage: {h.failed_at_stage}")
            if h.deepest_stage_reached is not None:
                parts.append(f"- Deepest stage reached: {h.deepest_stage_reached}")
            if h.final_critique_label:
                parts.append(f"- Final critique label: **{h.final_critique_label}**")
                if h.final_critique_summary:
                    parts.append(f"  - Summary: {h.final_critique_summary[:300]}")
            if h.pre_eval_strategic_insights:
                parts.append("- Pre-eval expert insights (key strategic positions):")
                for ins in h.pre_eval_strategic_insights:
                    if ins:
                        parts.append(f"  - {ins[:300]}")
            if h.post_eval_strategic_insights_by_stage:
                parts.append("- Post-eval expert insights (key strategic positions per stage):")
                for stage_id, inss in h.post_eval_strategic_insights_by_stage.items():
                    for ins in inss:
                        if ins:
                            parts.append(f"  - [{stage_id}] {ins[:300]}")
            parts.append("")
        parts.append("---")
        parts.append("")

    parts.append(
        "## Most recent attempt (revision {}) — full code shown below".format(previous.revision)
    )
    parts.append("")
    parts.append(f"Rationale: {previous.rationale or '(none recorded)'}")
    parts.append("")
    parts.append("Files in your most-recent attempt:")
    for filename, source in previous.code_files.items():
        parts.append(f"### Previous file `{filename}`")
        parts.append("```python")
        parts.append(source.rstrip())
        parts.append("```")
        parts.append("")

    parts.append("## Verifier feedback")
    parts.append("")
    parts.append(
        "Each block below is a separate verifier's report. Take all of them "
        "into account; do not ignore any. Layer-1 issues are mechanical/structural; "
        "Layer-2 are methodological; Layer-3 (post-eval) are interpretations of the "
        "actual simulation output."
    )
    parts.append("")
    for c in critiques:
        parts.append(f"### {_layer_human_name(c.layer)} verdict: {c.label.value}")
        if c.summary:
            parts.append(c.summary)
        if c.concerns:
            parts.append("")
            parts.append("Concerns:")
            for concern in c.concerns:
                attribution = f" (from {concern.expert})" if concern.expert else ""
                parts.append(
                    f"- [severity {concern.severity:.2f}{attribution}] "
                    f"{concern.description}"
                )
                if concern.suggested_focus:
                    parts.append(f"  - focus: {concern.suggested_focus}")
        parts.append("")

    parts.append("## Auxiliary files staged in your working directory")
    for path in task.host_structure_files.values():
        parts.append(f"- `{path.name}` (host structure)")
    for path in task.guest_structure_files.values():
        parts.append(f"- `{path.name}` (guest structure)")
    if task.bound_complex_pdb:
        parts.append(f"- `{task.bound_complex_pdb.name}` (bound complex pose)")
    parts.append(f"- wall-clock budget for the next attempt: {time_budget_minutes} minutes")
    parts.append("")

    parts.append(
        "Produce a revised pipeline addressing the concerns above. Use the same "
        "output contract: `### RATIONALE`, `### ENTRY`, one or more `### FILE: ...` "
        "blocks. You may keep, modify, or discard files from the previous attempt."
    )
    return "\n".join(parts)


def _layer_human_name(layer: LayerName) -> str:
    return {
        LayerName.LAYER_1: "Layer 1 (engineering / static analysis)",
        LayerName.LAYER_2_PRE: "Layer 2 (methodology pre-eval)",
        LayerName.LAYER_2_POST: "Layer 2 (post-eval, interprets simulation output)",
        LayerName.LAYER_3: "Layer 3 (raw simulation output)",
    }[layer]
