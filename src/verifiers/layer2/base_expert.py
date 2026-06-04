"""Shared machinery for Layer-2 expert verifier agents.

Each expert is an LLM-driven agent specialised on one strand of MD-pipeline
methodology. They share the same shape:

    pre_eval(pipeline, task, peer_votes=None) -> ExpertVote
    post_eval(pipeline, layer3_output, task, peer_votes=None) -> ExpertVote

Subclasses provide a `NAME` and a path to their domain-specific system
prompt; this base class handles prompt rendering, the LLM call, and JSON
vote parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from src.llm.base import LLMClient, LLMMessage, LLMRole, NoLLMClient
from src.models import ExpertVote, LayerName, Layer3Output, Pipeline
from src.verifiers.layer2.vote_parsing import parse_expert_vote

if TYPE_CHECKING:
    from src.tasks import Task

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts" / "experts"
SHARED_OUTPUT_FILENAME = "_shared_output.md"

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0


def _load_expert_system_prompt(prompts_dir: Path, filename: str) -> str:
    """Load `filename` and substitute the `{shared_output}` placeholder."""
    base = (prompts_dir / filename).read_text()
    if "{shared_output}" in base:
        shared = (prompts_dir / SHARED_OUTPUT_FILENAME).read_text()
        base = base.replace("{shared_output}", shared)
    return base


@dataclass
class ExpertDebugInfo:
    """Inspection blob for one pre_eval / post_eval call."""

    expert: str
    layer: LayerName
    system_prompt: str
    user_prompt: str
    raw_response: str
    parsed: ExpertVote
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class BaseExpert:
    """Base class for Layer-2 expert verifier agents."""

    NAME: ClassVar[str] = ""
    SYSTEM_PROMPT_FILENAME: ClassVar[str] = ""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompts_dir: Path = PROMPTS_DIR,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        if not self.NAME or not self.SYSTEM_PROMPT_FILENAME:
            raise TypeError(
                f"{type(self).__name__} must set both NAME and SYSTEM_PROMPT_FILENAME."
            )
        self._llm: LLMClient = llm_client or NoLLMClient()
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = _load_expert_system_prompt(
            prompts_dir, self.SYSTEM_PROMPT_FILENAME
        )
        self._last_debug: ExpertDebugInfo | None = None

    @property
    def last_debug(self) -> ExpertDebugInfo | None:
        return self._last_debug

    def design_recommendation(
        self,
        task: "Task",
        peer_votes: list[ExpertVote] | None = None,
        round_idx: int | None = None,
    ) -> ExpertVote:
        """Mode A: pre-pipeline design discussion.

        Asked at the very top of a task, BEFORE any Writer.propose has
        emitted code. Each expert recommends a method/family for the
        given host-guest system, optionally using ``WebSearch`` /
        ``WebFetch`` if their LLM client has those tools enabled.
        """
        prompt = render_design_recommendation_prompt(
            expert_name=self.NAME,
            task=task,
            peer_votes=peer_votes or [],
        )
        tag = f"phase0.{self.NAME}" + (f".r{round_idx}" if round_idx is not None else "")
        return self._call(prompt, layer=LayerName.LAYER_2_PRE, caller_tag=tag)

    def pre_eval(
        self,
        pipeline: Pipeline,
        task: "Task | None" = None,
        peer_votes: list[ExpertVote] | None = None,
        round_idx: int | None = None,
        trial_idx: int | None = None,
    ) -> ExpertVote:
        prompt = render_pre_eval_prompt(
            expert_name=self.NAME,
            pipeline=pipeline,
            task=task,
            peer_votes=peer_votes or [],
        )
        tag_parts = [f"pre.{self.NAME}"]
        if round_idx is not None:
            tag_parts.append(f"r{round_idx}")
        if trial_idx is not None:
            tag_parts.append(f"trial_{trial_idx}")
        return self._call(prompt, layer=LayerName.LAYER_2_PRE, caller_tag=".".join(tag_parts))

    def post_eval_benchmark(
        self,
        pipeline: Pipeline,
        benchmark_result,  # src.verifiers.multi_molecule_benchmark.BenchmarkResult
        task: "Task | None" = None,
        peer_votes: list[ExpertVote] | None = None,
        round_idx: int | None = None,
        trial_idx: int | None = None,
    ) -> ExpertVote:
        """Mode D: post-MD multi-molecule benchmark critique.

        After the pipeline has been validated by Engineer (single-mol code
        check) and then APPLIED to N benchmark molecules via
        MultiMoleculeBenchmark, the expert sees the cross-molecule ΔG
        results and critiques the pipeline from their domain perspective.
        This is the proper feedback signal for verbal-RL iteration:
        cross-molecule errors, not single-molecule trial output.
        """
        prompt = render_post_eval_benchmark_prompt(
            expert_name=self.NAME,
            pipeline=pipeline,
            benchmark_result=benchmark_result,
            task=task,
            peer_votes=peer_votes or [],
        )
        tag_parts = [f"post_bench.{self.NAME}"]
        if round_idx is not None:
            tag_parts.append(f"r{round_idx}")
        if trial_idx is not None:
            tag_parts.append(f"trial_{trial_idx}")
        return self._call(prompt, layer=LayerName.LAYER_2_POST, caller_tag=".".join(tag_parts))

    def post_eval(
        self,
        pipeline: Pipeline,
        layer3_output: Layer3Output,
        task: "Task | None" = None,
        peer_votes: list[ExpertVote] | None = None,
        round_idx: int | None = None,
        trial_idx: int | None = None,
        stage_id: str | None = None,
    ) -> ExpertVote:
        prompt = render_post_eval_prompt(
            expert_name=self.NAME,
            pipeline=pipeline,
            layer3_output=layer3_output,
            task=task,
            peer_votes=peer_votes or [],
        )
        tag_parts = [f"post.{self.NAME}"]
        if stage_id is not None:
            tag_parts.append(f"stage_{stage_id}")
        if round_idx is not None:
            tag_parts.append(f"r{round_idx}")
        if trial_idx is not None:
            tag_parts.append(f"trial_{trial_idx}")
        return self._call(prompt, layer=LayerName.LAYER_2_POST, caller_tag=".".join(tag_parts))

    def _call(self, user_prompt: str, layer: LayerName, caller_tag: str | None = None) -> ExpertVote:
        response = self._llm.complete(
            messages=[LLMMessage(role=LLMRole.USER, content=user_prompt)],
            system=self._system_prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            caller_tag=caller_tag,
        )
        vote = parse_expert_vote(response.content, expert=self.NAME, layer=layer)
        self._last_debug = ExpertDebugInfo(
            expert=self.NAME,
            layer=layer,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            raw_response=response.content,
            parsed=vote,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return vote


# ---------------------------------------------------------------------------
# Prompt rendering helpers (module-level so they are unit-testable)
# ---------------------------------------------------------------------------


def render_post_eval_benchmark_prompt(
    expert_name: str,
    pipeline: Pipeline,
    benchmark_result,
    task: "Task | None",
    peer_votes: list[ExpertVote],
) -> str:
    """Render the user prompt for **Mode D — multi-molecule benchmark post-eval**.

    The pipeline has been applied to N benchmark molecules; the expert sees
    per-molecule ΔG vs ITC + aggregate MAE/RMSE/bias/R² and critiques the
    pipeline from their domain lens.
    """
    parts = [
        "**Mode D — multi-molecule benchmark post-eval (Phase 4).**",
        "",
        "The pipeline (revision below) has been validated to run end-to-end "
        "by the Engineer, then APPLIED to a multi-molecule benchmark of "
        "host-guest systems. You see per-molecule ΔG predictions vs ITC "
        "ground truth across the benchmark, plus aggregate statistics. "
        "Critique the pipeline from your domain perspective: what does the "
        "**cross-molecule** error pattern imply about the pipeline's "
        "force-field / sampling / analysis choices? Your `strategic_insight` "
        "should drive the NEXT Writer.revise — it must be specific and "
        "actionable, not platitudes.",
        "",
        "# Task family",
    ]
    if task is not None:
        parts.append(task.description)
        parts.append(f"- host class: `{task.target.host_id}`")
    parts.append("")

    parts.append("# Pipeline rationale (from Writer, before benchmark)")
    parts.append(pipeline.rationale or "(none recorded)")
    parts.append("")

    parts.append("# Pipeline code (current revision)")
    for filename, source in pipeline.code_files.items():
        parts.append(f"## `{filename}`")
        parts.append("```python")
        parts.append(source.rstrip()[:4000])  # cap to keep prompt manageable
        if len(source) > 4000:
            parts.append(f"... ({len(source)-4000} chars truncated)")
        parts.append("```")
        parts.append("")

    parts.append("# Multi-molecule benchmark results")
    parts.append(f"- n_total = {len(benchmark_result.rows)}, "
                 f"n_valid = {benchmark_result.n_valid}")
    if benchmark_result.mae is not None:
        parts.append(f"- MAE  = {benchmark_result.mae:.2f} kcal/mol")
        parts.append(f"- RMSE = {benchmark_result.rmse:.2f} kcal/mol")
        parts.append(f"- bias = {benchmark_result.bias:+.2f} kcal/mol  (>0 = under-bind, <0 = over-bind)")
        if benchmark_result.r_squared is not None:
            parts.append(f"- R²   = {benchmark_result.r_squared:.3f}")
    parts.append(f"- wall-time = {benchmark_result.wall_time_seconds:.0f}s")
    parts.append("")
    parts.append("## Per-molecule")
    for r in benchmark_result.rows:
        if r.delta_g_predicted_kcal is not None:
            ec = r.energy_components or {}
            pull = ec.get("dG_pull_kcal")
            attach = ec.get("dG_attach_kcal")
            release = ec.get("dG_release_kcal")
            sigma_str = (
                f"{r.sigma_predicted_kcal:.2f}"
                if isinstance(r.sigma_predicted_kcal, (int, float))
                else "n/a"
            )
            err_str = (
                f"{r.error_kcal:+.2f}"
                if isinstance(r.error_kcal, (int, float))
                else "n/a"
            )
            itc_str = (
                f"{r.delta_g_experimental_kcal:+.2f}"
                if isinstance(r.delta_g_experimental_kcal, (int, float))
                else "n/a"
            )
            if all(isinstance(v, (int, float)) for v in (pull, attach, release)):
                parts.append(
                    f"- {r.guest_id} ({r.task_id}): "
                    f"pred = {r.delta_g_predicted_kcal:+.2f} ± {sigma_str}, "
                    f"ITC = {itc_str}, err = {err_str}; "
                    f"dG_attach = {attach:+.2f}, dG_pull = {pull:+.2f}, dG_release = {release:+.2f}"
                )
            else:
                parts.append(
                    f"- {r.guest_id} ({r.task_id}): "
                    f"pred = {r.delta_g_predicted_kcal:+.2f} ± {sigma_str}, "
                    f"ITC = {itc_str}, err = {err_str}"
                )
        else:
            parts.append(
                f"- {r.guest_id} ({r.task_id}): **FAILED** at {r.stage_reached or 'init'} "
                f"({r.failure_reason or 'unknown'}); ITC = {r.delta_g_experimental_kcal:+.2f}"
            )
    parts.append("")

    if peer_votes:
        parts.append("# Other experts' Round-1 benchmark critiques")
        for v in peer_votes:
            parts.append(f"## {v.expert}: {v.label.value} (conf {v.confidence:.2f})")
            if v.strategic_insight:
                parts.append(f"**Strategic insight:** {v.strategic_insight}")
            for c in v.concerns:
                focus = f" [focus: {c.suggested_focus}]" if c.suggested_focus else ""
                parts.append(f"- [severity {c.severity:.2f}]{focus} {c.description}")
            parts.append("")

    parts.append("# Your task")
    parts.append(
        f"As the **{expert_name}** co-designer, interpret the cross-molecule "
        "benchmark performance through your domain lens. The `strategic_insight` "
        "field should propose the **specific, surgical** pipeline change that "
        "would most improve next-iteration MAE. Avoid vague suggestions "
        "(\"sample longer\", \"tune the force field\"); instead name the "
        "specific file + region + change. Use `concerns` for granular issues. "
        "Output ONE JSON object matching the schema; no prose outside the JSON."
    )
    return "\n".join(parts)


def render_design_recommendation_prompt(
    expert_name: str,
    task: "Task",
    peer_votes: list[ExpertVote],
) -> str:
    """Render the user prompt for **Mode A — design recommendation**.

    No pipeline exists; the expert recommends a method based on the
    task spec alone, optionally consulting literature via WebSearch /
    WebFetch (if their LLM client has those tools).
    """
    parts = [
        "**Mode A — design recommendation** (Phase 0; pre-pipeline).",
        "",
        "No pipeline has been written yet. You are part of an up-front "
        "design-review panel: each domain expert independently recommends "
        "what method/family should be used for this specific host-guest "
        "task BEFORE any code is authored. Your recommendation will be "
        "given (alongside the other experts') to the Pipeline-Writer "
        "agent as context for its initial design.",
        "",
        "# Task",
    ]
    parts.append(task.description)
    parts.append("")
    parts.append(
        f"- target: host=`{task.target.host_id}` guest=`{task.target.guest_id}`"
    )
    if task.target.guest_smiles:
        parts.append(f"- guest SMILES: `{task.target.guest_smiles}`")
    if task.target.host_smiles:
        parts.append(f"- host SMILES: `{task.target.host_smiles}`")
    for k, v in task.extra_context.items():
        parts.append(f"- {k}: {v}")
    parts.append("")

    if peer_votes:
        parts.append("# Other experts' Round-1 design recommendations")
        parts.append(
            "These are independent first-round recommendations from peer "
            "domain experts. You may concur, dissent, or refine. "
            "Cross-domain commentary is welcome when their methodological "
            "choices substantively affect your domain."
        )
        parts.append("")
        for v in peer_votes:
            parts.append(f"## {v.expert}: {v.label.value} (confidence {v.confidence:.2f})")
            if v.strategic_insight:
                parts.append(f"**Strategic insight:** {v.strategic_insight}")
            if v.reasoning:
                parts.append(f"Reasoning: {v.reasoning}")
            if v.concerns:
                parts.append("Concrete concerns:")
                for c in v.concerns:
                    focus = f" [focus: {c.suggested_focus}]" if c.suggested_focus else ""
                    parts.append(f"- [severity {c.severity:.2f}]{focus} {c.description}")
            parts.append("")

    parts.append("# Your task")
    parts.append(
        f"As the **{expert_name}** co-designer, recommend the method "
        "family / force field choice / sampling strategy / analysis "
        "approach that you, as a senior expert in this domain, would "
        "use for this specific host-guest system. You have `WebSearch` "
        "and `WebFetch` available — consult the literature when you "
        "are uncertain or want to cite recent benchmark results.\n\n"
        "Your `strategic_insight` should be a concrete recommendation "
        "(e.g. \"use APR with Henriksen-Gilson SSC, 15–25 windows at "
        "1–2 ns each; cite Yin 2017 SAMPL5 and Henriksen 2015 JCTC\") "
        "— NOT just a critique of an unspecified design. Use `concerns` "
        "to flag pitfalls the Writer must avoid. Use `label=pass` if "
        "you can defensibly recommend a method, `uncertain` if the task "
        "is underspecified, `fail` if you think no clean method exists "
        "for this system class. Output ONE JSON object matching the "
        "schema; no prose outside the JSON."
    )
    return "\n".join(parts)


def render_pre_eval_prompt(
    expert_name: str,
    pipeline: Pipeline,
    task: "Task | None",
    peer_votes: list[ExpertVote],
) -> str:
    parts = [
        f"You are reviewing a proposed MD pipeline (revision {pipeline.revision}) "
        f"BEFORE it has been executed. Layer 3 has not run.",
        "",
        f"# Task",
    ]
    if task is not None:
        parts.append(task.description)
        parts.append("")
        parts.append(f"- target: host=`{task.target.host_id}` guest=`{task.target.guest_id}`")
        if task.target.guest_smiles:
            parts.append(f"- guest SMILES: `{task.target.guest_smiles}`")
        for k, v in task.extra_context.items():
            parts.append(f"- {k}: {v}")
    else:
        parts.append("(task spec not provided to this expert)")
    parts.append("")

    parts.append("# Pipeline rationale (from Writer)")
    parts.append(pipeline.rationale or "(no rationale recorded)")
    parts.append("")

    parts.append("# Pipeline code")
    for filename, source in pipeline.code_files.items():
        marker = " [entry point]" if filename == pipeline.entry_point else ""
        parts.append(f"## `{filename}`{marker}")
        parts.append("```python")
        parts.append(source.rstrip())
        parts.append("```")
        parts.append("")

    if peer_votes:
        parts.append("# Other experts' Round-1 strategic positions")
        parts.append(
            "These are independent first-round assessments from peer domain experts. "
            "Read their strategic_insight and concerns; you may concur, dissent, or "
            "refine. Cross-domain commentary is welcome when their choices substantively "
            "affect your domain."
        )
        parts.append("")
        for v in peer_votes:
            parts.append(f"## {v.expert}: {v.label.value} (confidence {v.confidence:.2f})")
            if v.strategic_insight:
                parts.append(f"**Strategic insight:** {v.strategic_insight}")
            if v.reasoning:
                parts.append(f"Reasoning: {v.reasoning}")
            if v.concerns:
                parts.append("Concrete concerns:")
                for c in v.concerns:
                    focus = f" [focus: {c.suggested_focus}]" if c.suggested_focus else ""
                    parts.append(f"- [severity {c.severity:.2f}]{focus} {c.description}")
            parts.append("")

    parts.append("# Your task")
    parts.append(
        f"As the **{expert_name}** co-designer, deliver your domain-level assessment "
        "of this proposed pipeline. Output ONE JSON object matching the schema in your "
        "system prompt. The `strategic_insight` field is the load-bearing one — fill "
        "it with substantive co-designer reasoning, not platitudes. Do not include "
        "any prose outside the JSON object."
    )
    return "\n".join(parts)


def render_post_eval_prompt(
    expert_name: str,
    pipeline: Pipeline,
    layer3_output: Layer3Output,
    task: "Task | None",
    peer_votes: list[ExpertVote],
) -> str:
    parts = [
        f"You are interpreting the raw output of the MD simulation that was "
        f"executed from revision {pipeline.revision} of the pipeline. Layer 3 has run.",
        "",
        "# Task",
    ]
    if task is not None:
        parts.append(task.description)
        parts.append("")
        parts.append(f"- target: host=`{task.target.host_id}` guest=`{task.target.guest_id}`")
        for k, v in task.extra_context.items():
            parts.append(f"- {k}: {v}")
    else:
        parts.append("(task spec not provided)")
    parts.append("")

    parts.append("# Pipeline rationale (from Writer, before execution)")
    parts.append(pipeline.rationale or "(no rationale recorded)")
    parts.append("")

    parts.append("# Layer-3 raw output")
    parts.append(f"- exited_normally: {layer3_output.exited_normally}")
    parts.append(f"- timed_out: {layer3_output.timed_out}")
    parts.append(f"- parse_failed: {layer3_output.parse_failed}")
    parts.append(f"- return_code: {layer3_output.return_code}")
    parts.append(f"- wall_time_seconds: {layer3_output.wall_time_seconds}")
    parts.append(f"- delta_g_kcal_per_mol: {layer3_output.delta_g_kcal_per_mol}")
    parts.append(f"- delta_g_uncertainty: {layer3_output.delta_g_uncertainty}")
    parts.append(f"- replicate_consistency: {layer3_output.replicate_consistency}")
    if layer3_output.convergence_flags:
        parts.append(f"- convergence_flags: {layer3_output.convergence_flags}")
    if layer3_output.energy_components:
        parts.append(f"- energy_components: {layer3_output.energy_components}")
    if layer3_output.diagnostics:
        parts.append(f"- diagnostics: {layer3_output.diagnostics}")
    parts.append("")
    if layer3_output.stdout_tail:
        parts.append("## stdout (tail)")
        parts.append("```")
        parts.append(layer3_output.stdout_tail.rstrip())
        parts.append("```")
        parts.append("")
    if layer3_output.stderr_tail:
        parts.append("## stderr (tail)")
        parts.append("```")
        parts.append(layer3_output.stderr_tail.rstrip())
        parts.append("```")
        parts.append("")

    parts.append("# Pipeline code (for reference)")
    for filename, source in pipeline.code_files.items():
        parts.append(f"## `{filename}`")
        parts.append("```python")
        parts.append(source.rstrip())
        parts.append("```")
        parts.append("")

    if peer_votes:
        parts.append("# Other experts' post-eval Round-1 strategic positions")
        for v in peer_votes:
            parts.append(f"## {v.expert}: {v.label.value} (confidence {v.confidence:.2f})")
            if v.strategic_insight:
                parts.append(f"**Strategic insight:** {v.strategic_insight}")
            if v.reasoning:
                parts.append(f"Reasoning: {v.reasoning}")
            if v.concerns:
                parts.append("Concrete concerns:")
                for c in v.concerns:
                    focus = f" [focus: {c.suggested_focus}]" if c.suggested_focus else ""
                    parts.append(f"- [severity {c.severity:.2f}]{focus} {c.description}")
            parts.append("")

    parts.append("# Your task")
    parts.append(
        f"As the **{expert_name}** co-designer, interpret the simulation output through "
        "your domain lens. Your `strategic_insight` should answer: did the observed "
        "behaviour validate or invalidate the design assumptions made before execution? "
        "What does the result imply about the load-bearing physical assumptions in your "
        "domain? Output ONE JSON object matching the schema in your system prompt; no "
        "prose outside the JSON object."
    )
    return "\n".join(parts)
