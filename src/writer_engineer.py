"""Engineer agent: a tool-using debugging session in the sandbox.

Bridges the gap between the Code agent (good at method choice) and the
execution oracle (good at clean evaluation), by giving a tool-enabled
agent a wall-clock budget to mechanically debug the pipeline until it
runs end-to-end and produces a finite binding free energy.

Workflow inside one `run_to_completion()` call:

1. The sandbox `work_dir` already contains the pipeline's `*.py` files and
   auxiliary input files (host/guest structures), staged by the harness.
2. We hand the persistent-session SDK client a brief that points it at the
   files and tells it to drive the stages (00 dock -> 01 prep -> 02
   equilibrate -> 03 production -> 04 analysis) until stage 04 emits a ΔG.
3. The agent uses `Read`, `Edit`, `Write`, `Bash` autonomously, then emits a
   short final summary. We read the resulting `*.py` files from `work_dir`
   and build a new `Pipeline` to hand back to the orchestrator.

After this call returns, the harness wipes any partial run artifacts (stage
result JSONs, intermediate files) and runs the *clean* pipeline in a fresh
sandbox for execution. The engineer's only contract is "the *.py files in
work_dir should now run end-to-end and produce a finite ΔG".

Engineering-only: the agent is NOT allowed to change the methodological
choice (APR vs DDM, force field family, etc.) — see engineer_system.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient as _SDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from src.llm.usage_logger import get_global_usage_logger
from src.models import Pipeline

if TYPE_CHECKING:
    from src.tasks import Task

_log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ENGINEER_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "engineer_system.md"

DEFAULT_MODEL = os.environ.get("MD_AGENT_ENGINEER_MODEL", "claude-opus-4-7")
# Tool access: full read/edit/write/Bash in the sandbox. No network.
ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash"]
# Per-call turn budget for the agent's tool-use loop. ~120 is generous for
# 20-40 edit-run cycles within 30 min.
DEFAULT_MAX_TURNS = 250
DEFAULT_TIME_BUDGET_SECONDS = 60 * 60  # 60 min for debug; production-grade timing happens in multi_task_runner with no cap

STAGE_FILE_RE = re.compile(r"^\d{2}_.*\.py$")
HELPER_FILE_RE = re.compile(r"^(?!\d{2}_).*\.py$")


@dataclass
class EngineerResult:
    """What happened in one engineering session — for logging / debugging."""

    final_pipeline: Pipeline
    summary_text: str = ""
    """Final assistant message — the engineer's own 3-line summary."""

    n_tool_calls: int = 0
    n_bash_calls: int = 0
    n_edit_calls: int = 0
    n_write_calls: int = 0
    wall_time_seconds: float = 0.0
    final_stage_statuses: dict[str, str] = field(default_factory=dict)
    """For each stage filename, the status read from `stage_NN_*_result.json`
    after the engineer is done (or "(missing)" if no result file was written)."""

    raw_transcript: list[str] = field(default_factory=list)
    """For inspection: a chronological list of text-form assistant messages
    and tool-use summaries."""


class PipelineEngineer:
    """Tool-using engineering agent that debugs a pipeline in a sandbox.

    Persistent-session: each `run_to_completion()` call holds a single
    `claude_agent_sdk.ClaudeSDKClient` for the whole tool-use loop, so the
    conversation prefix (incl. system prompt) is auto-cached and subprocess
    startup is paid once per call instead of per turn.
    """

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        time_budget_seconds: int = DEFAULT_TIME_BUDGET_SECONDS,
    ):
        self._default_model = default_model
        self._max_turns = max_turns
        self._time_budget_seconds = time_budget_seconds
        self._system_prompt = ENGINEER_SYSTEM_PROMPT_PATH.read_text()

    def run_to_completion(
        self,
        pipeline: Pipeline,
        work_dir: Path,
        task: "Task",
        trial_idx: int | None = None,
    ) -> EngineerResult:
        """Run a tool-using engineering session against the pipeline files
        already staged in `work_dir`. Drives all 5 stages end-to-end until
        stage 04 produces a ΔG (or the session times out / gets stuck).

        Caller is responsible for:
        - Writing `pipeline.code_files` into `work_dir` BEFORE the call.
        - Staging auxiliary files (host/guest structures) into `work_dir`.
        - Wiping any prior stage_NN_*_result.json files in `work_dir` so
          the engineer sees only its own results.

        Returns an `EngineerResult` whose `final_pipeline` reflects the
        ENGINEER-EDITED code files. The harness should hand
        `result.final_pipeline` to Layer 3 (in a *fresh* clean sandbox).
        """
        work_dir = work_dir.resolve()
        if not work_dir.is_dir():
            raise FileNotFoundError(f"work_dir does not exist: {work_dir}")

        brief = self._render_engineer_brief(pipeline, task)

        t0 = time.time()
        transcript, summary_text, tool_counts = asyncio.run(
            self._run_session(
                system_prompt=self._system_prompt,
                user_brief=brief,
                cwd=str(work_dir),
                trial_idx=trial_idx,
            )
        )
        wall = time.time() - t0

        # Read final state of stage files from disk (engineer may have
        # edited / overwritten them).
        final_code = self._read_pipeline_files(work_dir, pipeline)
        # Determine final stage statuses by reading any stage_*_result.json.
        final_statuses = self._read_stage_statuses(work_dir)

        final_pipeline = Pipeline(
            target=pipeline.target,
            code_files=final_code,
            entry_point=pipeline.entry_point,
            revision=pipeline.revision,  # same logical revision; just debugged
            parent_revision=pipeline.parent_revision,
            rationale=pipeline.rationale
            + "\n\n[engineer-debugged: "
            + summary_text[:300].replace("\n", " ").strip()
            + "]",
        )

        return EngineerResult(
            final_pipeline=final_pipeline,
            summary_text=summary_text,
            n_tool_calls=tool_counts["total"],
            n_bash_calls=tool_counts["Bash"],
            n_edit_calls=tool_counts["Edit"],
            n_write_calls=tool_counts["Write"],
            wall_time_seconds=wall,
            final_stage_statuses=final_statuses,
            raw_transcript=transcript,
        )

    # -- internal -----------------------------------------------------

    def _render_engineer_brief(self, pipeline: Pipeline, task: "Task") -> str:
        stage_files = sorted(
            name for name in pipeline.code_files if STAGE_FILE_RE.match(name)
        )
        helper_files = sorted(
            name for name in pipeline.code_files if HELPER_FILE_RE.match(name)
        )
        aux_basenames = []
        for path in task.host_structure_files.values():
            aux_basenames.append(path.name)
        for path in task.guest_structure_files.values():
            aux_basenames.append(path.name)
        if task.bound_complex_pdb:
            aux_basenames.append(task.bound_complex_pdb.name)

        lines = [
            f"# Engineer session for task `{task.task_id}` rev {pipeline.revision}",
            "",
            f"Target: host=`{task.target.host_id}` guest=`{task.target.guest_id}`.",
            "",
            "## Stage files in CWD (run with `python <file>`):",
        ]
        for f in stage_files:
            lines.append(f"- `{f}`")
        if helper_files:
            lines.append("")
            lines.append("## Helper files in CWD (imported by stages, do not run directly):")
            for f in helper_files:
                lines.append(f"- `{f}`")
        if aux_basenames:
            lines.append("")
            lines.append("## Auxiliary input files in CWD:")
            for b in aux_basenames:
                lines.append(f"- `{b}`")
        lines.append("")
        lines.append("## Pipeline rationale (from the strategic Writer — DO NOT change):")
        lines.append(pipeline.rationale or "(none)")
        lines.append("")
        lines.append(
            "## Your job\n"
            "Drive the pipeline end-to-end: run stages 00 → 01 → 02 → 03 → 04 "
            "in order. After each, inspect the corresponding "
            "`stage_NN_*_result.json` to check what actually happened. Fix "
            "bugs by editing the offending stage script and re-running ONLY "
            "that stage. **Hard goal: stage 04 must produce a finite ΔG.** "
            "Stage 03 is expensive (~15-45 min on the GPU); once it "
            "succeeds, don't re-run it — use its output files for stage 04. "
            "Time budget for this DEBUG session: ~60 minutes wall-clock. "
            "Production-grade timing (longer per-window MD) happens downstream "
            "in a separate validation pass; for this debug session, just verify "
            "the pipeline runs end-to-end with the smoke-test sampling level "
            "set via env var PRISM_PRODUCTION_NS_PER_WINDOW.\n\n"
            "Reminders: (a) DO NOT change the methodological design (force "
            "field, water model, sampling method, alchemical schedule, "
            "restraint family — those are the strategic Writer's choices); "
            "(b) DO NOT silence or relax QC gates to manufacture success "
            "— `status=\"diverged\"` with a finite ΔG is a valid outcome "
            "that the multi-agent post-eval will interpret. Your contract "
            "is mechanical correctness + a real ΔG, not a green dashboard.\n\n"
            "When you finish, emit the final summary (see system prompt), "
            "then stop calling tools."
        )
        return "\n".join(lines)

    async def _run_session(
        self,
        *,
        system_prompt: str,
        user_brief: str,
        cwd: str,
        trial_idx: int | None = None,
    ) -> tuple[list[str], str, dict[str, int]]:
        """Drive a single persistent-session tool-use loop.

        Returns (transcript, final_summary_text, tool_counts)."""
        # Observability: capture the bundled CLI's stderr so that if it
        # is SIGKILLed (exit code -9) we have the CLI-side diagnostics on disk.
        # Lines are buffered here and flushed to cli_stderr.log in the finally
        # block below (session end).
        cli_stderr_lines: list[str] = []
        opts = ClaudeAgentOptions(
            model=self._default_model,
            system_prompt=system_prompt,
            max_turns=self._max_turns,
            allowed_tools=ALLOWED_TOOLS,
            cwd=cwd,
            permission_mode="bypassPermissions",
            stderr=cli_stderr_lines.append,
        )

        transcript: list[str] = []
        last_text: str = ""
        tool_counts: dict[str, int] = {
            "total": 0,
            "Read": 0,
            "Write": 0,
            "Edit": 0,
            "Bash": 0,
        }

        client = _SDKClient(opts)
        try:
            await client.connect()
            await client.query(user_brief)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            last_text = block.text
                            transcript.append(f"[assistant text] {block.text[:500]}")
                        elif isinstance(block, ToolUseBlock):
                            tool_counts["total"] += 1
                            if block.name in tool_counts:
                                tool_counts[block.name] += 1
                            transcript.append(
                                f"[tool {block.name}] {str(block.input)[:300]}"
                            )
                elif isinstance(message, ResultMessage):
                    transcript.append(
                        f"[result] subtype={getattr(message, 'subtype', None)}"
                    )
                    # Log per-session usage stats (one ResultMessage per
                    # query() call; for the engineer's tool-use loop this
                    # represents the entire session's aggregate usage).
                    logger = get_global_usage_logger()
                    if logger is not None:
                        usage = getattr(message, "usage", None)
                        def _u(u, k):
                            return (
                                getattr(u, k, None)
                                or (u.get(k) if isinstance(u, dict) else None)
                            ) if u else None
                        tag = "engineer" + (f".trial_{trial_idx}" if trial_idx is not None else "")
                        logger.log(
                            caller=tag,
                            model=self._default_model,
                            input_tokens=_u(usage, "input_tokens"),
                            output_tokens=_u(usage, "output_tokens"),
                            cache_read_input_tokens=_u(usage, "cache_read_input_tokens"),
                            cache_creation_input_tokens=_u(usage, "cache_creation_input_tokens"),
                            latency_s=None,
                            extra={
                                "n_tool_calls": tool_counts["total"],
                                "n_bash": tool_counts["Bash"],
                                "n_edit": tool_counts["Edit"],
                                "n_write": tool_counts["Write"],
                                "n_read": tool_counts["Read"],
                            },
                        )
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass
            # Flush captured CLI stderr to disk for post-mortem inspection.
            if cli_stderr_lines:
                try:
                    (Path(cwd) / "cli_stderr.log").write_text(
                        "\n".join(cli_stderr_lines) + "\n"
                    )
                except Exception:
                    pass

        return transcript, last_text, tool_counts

    @staticmethod
    def _read_pipeline_files(work_dir: Path, original: Pipeline) -> dict[str, str]:
        """Read the FINAL state of all original code_files from disk.

        If a file was deleted by the engineer, we keep the original
        (defensive — engineer shouldn't delete stage files). If new helper
        files were created we pick them up too.
        """
        out: dict[str, str] = {}
        for name in original.code_files:
            p = work_dir / name
            if p.is_file():
                out[name] = p.read_text()
            else:
                out[name] = original.code_files[name]
        # Pick up any new helper *.py files the engineer wrote (not stage
        # files we don't know about — those follow the \d{2}_.*\.py pattern
        # but we only honour those in `original`).
        for p in work_dir.glob("*.py"):
            name = p.name
            if name in out:
                continue
            if STAGE_FILE_RE.match(name):
                # New stage file the engineer invented; include it.
                out[name] = p.read_text()
            else:
                # Helper file.
                out[name] = p.read_text()
        return out

    @staticmethod
    def _read_stage_statuses(work_dir: Path) -> dict[str, str]:
        """For each stage_NN_*_result.json in work_dir, record the status."""
        import json

        statuses: dict[str, str] = {}
        for p in sorted(work_dir.glob("stage_*_result.json")):
            try:
                d = json.loads(p.read_text())
                statuses[p.name] = str(d.get("status") or "(no status)")
            except Exception as e:
                statuses[p.name] = f"(parse error: {e})"
        return statuses
