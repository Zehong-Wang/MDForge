"""Layer 3 oracle: sandboxed execution of the Writer's pipeline code.

Two execution modes:

- **Atomic** (legacy, ``Layer3Oracle.run``): materialise every file, run
  the declared ``entry_point`` once, read a single ``result.json``.
- **Staged** (PRISM, ``Layer3Oracle.run_stage``): materialise every file
  once, then run a single stage entry (``00_dock.py``, ``01_setup.py``,
  ...) inside the same persistent working directory. Each stage writes
  ``stage_NN_result.json``. The harness invokes ``run_stage`` once per
  stage; cross-stage state lives on the filesystem of the work dir.

This is intentionally *not* a fixed MD pipeline. The Writer is supposed
to author the simulation code; this module only provides the safe,
introspectable execution environment.

Per-stage result schema:

    {
      "stage_id": "01_setup",
      "status": "success" | "diverged" | "timeout",
      "wall_time_seconds": float,
      "delta_g_kcal_per_mol": float | null,
      "delta_g_uncertainty_kcal_per_mol": float | null,
      "convergence_flags": {str: bool},
      "replicate_consistency": float | null,
      "energy_components": {str: float},
      "diagnostics": {str: any},
      "writer_notes": str
    }
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.models import Layer3Output, Pipeline

RESULT_JSON_FILENAME = "result.json"
DEFAULT_TIMEOUT_SECONDS = 60 * 60  # 1 hour, override per project
DEFAULT_STAGE_TIMEOUT_SECONDS = 24 * 60 * 60  # 24h, production-stage budget
STREAM_TAIL_CHARS = 5_000

STAGE_RE = re.compile(r"^(\d{2})_.*\.py$")


def stage_index(stage_filename: str) -> int:
    """Extract the two-digit prefix from ``00_dock.py`` -> 0."""
    m = STAGE_RE.match(stage_filename)
    if m is None:
        raise ValueError(f"Not a stage filename: {stage_filename!r}")
    return int(m.group(1))


def stage_result_filename(stage_filename: str) -> str:
    """Map ``00_dock.py`` -> ``stage_00_result.json``."""
    return f"stage_{stage_index(stage_filename):02d}_result.json"


@dataclass
class ExecutorConfig:
    """Tunable knobs for the sandboxed executor."""

    work_root: Path | None = None
    """Parent directory for per-run work dirs. Defaults to system tempdir."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    """Wall-clock cap (atomic-mode); exceeding it is recorded as `timed_out`."""

    stage_timeout_seconds: float = DEFAULT_STAGE_TIMEOUT_SECONDS
    """Wall-clock cap for a single stage in staged-mode execution."""

    keep_artifacts: bool = True
    """If False, the per-run work dir is deleted after parsing."""

    python_executable: str | None = None
    """Override the Python interpreter used to run the pipeline.

    For real MD runs this should point at the conda env that has OpenMM,
    AmberTools, OpenFF, etc. — by default the ``pipeline/`` env under the
    repo root, i.e. ``<repo_root>/pipeline/bin/python``.
    """

    extra_env: dict[str, str] | None = None
    """Extra environment variables overlaid on os.environ for the subprocess."""


class Layer3Oracle:
    """Run a Writer-produced pipeline in a sandbox; return Layer3Output.

    Two modes:

    - ``run(pipeline, auxiliary_files)`` — atomic: materialise + run +
      parse a single ``result.json``. Used by the legacy non-staged
      path and by tests.
    - ``prepare_workdir(pipeline, auxiliary_files) -> Path`` then
      ``run_stage(work_dir, pipeline, stage_filename)`` — staged: set
      up once, then invoke each stage entry separately. Cross-stage
      state lives on the work-dir filesystem.

    Both are synchronous and block for up to the configured timeout.
    """

    def __init__(self, config: ExecutorConfig | None = None):
        self._config = config or ExecutorConfig()

    # ---- Staged-mode interface (PRISM) ------------------------------------

    def prepare_workdir(
        self,
        pipeline: Pipeline,
        auxiliary_files: dict[str, Path] | None = None,
    ) -> Path:
        """Set up a fresh work dir for staged execution; return its path.

        Materialises every code file + every auxiliary file once. The
        caller then invokes ``run_stage`` per stage; the work dir
        persists across calls so cross-stage filesystem state (output
        PDBs, parm7, restart files) flows naturally.
        """
        work_dir = self._make_work_dir(pipeline)
        if auxiliary_files:
            self._stage_auxiliary_files(auxiliary_files, work_dir, pipeline.code_files)
        self._materialise_code(pipeline, work_dir)
        return work_dir

    def run_stage(
        self,
        work_dir: Path,
        pipeline: Pipeline,
        stage_filename: str,
    ) -> Layer3Output:
        """Run a single stage entry inside an already-prepared work dir.

        ``stage_filename`` must be one of the staged entries discovered
        by ``Pipeline.stage_files()`` (i.e., matches ``^\\d{2}_.*\\.py$``).

        Reads back ``stage_NN_result.json`` per the per-stage schema and
        packs into a Layer3Output with ``stage_id`` set.
        """
        if stage_filename not in pipeline.code_files:
            raise ValueError(
                f"Stage {stage_filename!r} is not in pipeline.code_files "
                f"(have: {sorted(pipeline.code_files)})."
            )
        if not STAGE_RE.match(stage_filename):
            raise ValueError(
                f"Stage filename {stage_filename!r} does not match the "
                f"\\d{{2}}_*.py convention."
            )

        result_filename = stage_result_filename(stage_filename)
        # Remove any stale per-stage result file from a previous attempt.
        stale = work_dir / result_filename
        if stale.exists():
            stale.unlink()

        return self._execute_python_script(
            work_dir=work_dir,
            script_path=stage_filename,
            result_filename=result_filename,
            timeout_seconds=self._config.stage_timeout_seconds,
            stage_id=stage_filename.removesuffix(".py"),
        )

    def read_stage_result_from_workdir(
        self,
        work_dir: Path,
        stage_filename: str,
    ) -> Layer3Output:
        """Construct a Layer3Output by reading whatever stage_NN_*_result.json
        the engineer agent (or anyone else) wrote into ``work_dir``, without
        re-running the stage.

        Used when an upstream tool-using agent already executed the stage
        and wrote its result to disk; the orchestrator only needs to parse
        and hand off to multi-agent post-eval.

        We are tolerant of two naming conventions:
        - canonical: ``stage_NN_result.json`` (the harness convention)
        - descriptive: ``stage_NN_<stage_id>_result.json`` (the convention
          some Writer-emitted scripts use)
        The most-recently-modified match wins.

        If no result file is found, the Layer3Output is marked
        ``parse_failed=True`` with ``status=None``; the engineer's session
        either didn't reach this stage or crashed without writing the file.
        """
        k = stage_index(stage_filename)
        stem = stage_filename.removesuffix(".py")
        # Look for stage_NN_result.json (canonical) AND stage_NN_<id>_result.json.
        candidates: list[Path] = list(work_dir.glob(f"stage_{k:02d}_result.json"))
        candidates += list(work_dir.glob(f"stage_{k:02d}_*_result.json"))
        # De-dup while preserving the most recent.
        seen: set[str] = set()
        unique: list[Path] = []
        for p in candidates:
            if p.name in seen:
                continue
            seen.add(p.name)
            unique.append(p)
        if not unique:
            return _build_output(
                payload=None,
                artifacts_dir=str(work_dir),
                wall_time=0.0,
                return_code=None,
                timed_out=False,
                parse_failed=True,
                stdout_tail=None,
                stderr_tail=None,
                stage_id_fallback=stem,
            )
        # Newest file wins.
        chosen = max(unique, key=lambda p: p.stat().st_mtime)
        parse_failed = False
        payload: dict | None = None
        try:
            payload = json.loads(chosen.read_text())
            if not isinstance(payload, dict):
                payload = None
                parse_failed = True
        except json.JSONDecodeError:
            parse_failed = True

        wall = 0.0
        if payload is not None:
            wall = _maybe_float(payload.get("wall_time_seconds")) or 0.0

        return _build_output(
            payload=payload,
            artifacts_dir=str(work_dir),
            wall_time=wall,
            return_code=0 if (payload and payload.get("status") == "success") else None,
            timed_out=(payload or {}).get("status") == "timeout",
            parse_failed=parse_failed,
            stdout_tail=None,
            stderr_tail=None,
            stage_id_fallback=stem,
        )

    # ---- Atomic-mode (legacy) ---------------------------------------------

    def run(
        self,
        pipeline: Pipeline,
        auxiliary_files: dict[str, Path] | None = None,
    ) -> Layer3Output:
        """Run the pipeline.

        Args:
            pipeline: the agent-authored Pipeline to execute.
            auxiliary_files: optional mapping of `relative_path -> source_path`
                of input files (host structure, guest structure, etc.) to be
                staged into the work directory *before* the pipeline's code is
                run. These are read-only inputs the agent can open by basename.
                If a name collides with a `code_files` entry, the code_files
                entry wins (the agent is allowed to override).
        """
        work_dir = self._make_work_dir(pipeline)
        try:
            if auxiliary_files:
                self._stage_auxiliary_files(auxiliary_files, work_dir, pipeline.code_files)
            self._materialise_code(pipeline, work_dir)
            return self._execute_and_parse(pipeline, work_dir)
        finally:
            if not self._config.keep_artifacts:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _stage_auxiliary_files(
        self,
        files: dict[str, Path],
        work_dir: Path,
        code_files: dict[str, str],
    ) -> None:
        for rel_path, src in files.items():
            if rel_path in code_files:
                # code_files takes precedence; the agent may have intentionally
                # overridden a staged input.
                continue
            target = (work_dir / rel_path).resolve()
            if not _is_within(target, work_dir.resolve()):
                raise ValueError(f"auxiliary_files entry escapes work dir: {rel_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    def _make_work_dir(self, pipeline: Pipeline) -> Path:
        root = self._config.work_root or Path(tempfile.gettempdir()) / "mdforge_layer3"
        root.mkdir(parents=True, exist_ok=True)
        run_id = f"{pipeline.target.host_id}-{pipeline.target.guest_id}-rev{pipeline.revision}-{uuid.uuid4().hex[:8]}"
        path = root / run_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _materialise_code(self, pipeline: Pipeline, work_dir: Path) -> None:
        for rel_path, source in pipeline.code_files.items():
            target = (work_dir / rel_path).resolve()
            # Reject path-traversal attempts: the resolved path must be
            # underneath work_dir.
            if not _is_within(target, work_dir.resolve()):
                raise ValueError(f"code_files entry escapes work dir: {rel_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)

    def _execute_and_parse(self, pipeline: Pipeline, work_dir: Path) -> Layer3Output:
        return self._execute_python_script(
            work_dir=work_dir,
            script_path=pipeline.entry_point,
            result_filename=RESULT_JSON_FILENAME,
            timeout_seconds=self._config.timeout_seconds,
            stage_id=None,
        )

    def _execute_python_script(
        self,
        *,
        work_dir: Path,
        script_path: str,
        result_filename: str,
        timeout_seconds: float,
        stage_id: str | None,
    ) -> Layer3Output:
        """Run ``python <script_path>`` inside ``work_dir`` with a timeout,
        capture stdout/stderr, parse ``<result_filename>``, return Layer3Output.

        Used by both atomic ``run`` and staged ``run_stage``.
        """
        python = self._config.python_executable or sys.executable
        env = os.environ.copy()
        if self._config.extra_env:
            env.update(self._config.extra_env)
        # Make subprocess output unbuffered so we get partial logs on timeout.
        env.setdefault("PYTHONUNBUFFERED", "1")

        cmd = [python, script_path]

        start = time.monotonic()
        timed_out = False
        return_code: int | None = None
        stdout_bytes = b""
        stderr_bytes = b""

        proc = subprocess.Popen(
            cmd,
            cwd=str(work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_seconds)
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(proc)
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                stdout_bytes = stdout_bytes or b""
                stderr_bytes = stderr_bytes or b""
            return_code = proc.returncode

        wall_time = time.monotonic() - start
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        result_path = work_dir / result_filename
        parsed_payload: dict | None = None
        parse_failed = False
        if result_path.exists():
            try:
                parsed_payload = json.loads(result_path.read_text())
                if not isinstance(parsed_payload, dict):
                    parse_failed = True
                    parsed_payload = None
            except json.JSONDecodeError:
                parse_failed = True
        else:
            parse_failed = True

        return _build_output(
            payload=parsed_payload,
            artifacts_dir=str(work_dir),
            wall_time=wall_time,
            return_code=return_code,
            timed_out=timed_out,
            parse_failed=parse_failed,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
            stage_id_fallback=stage_id,
        )


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _tail(text: str, n: int = STREAM_TAIL_CHARS) -> str | None:
    if not text:
        return None
    if len(text) <= n:
        return text
    return text[-n:]


def _build_output(
    payload: dict | None,
    artifacts_dir: str,
    wall_time: float,
    return_code: int | None,
    timed_out: bool,
    parse_failed: bool,
    stdout_tail: str | None,
    stderr_tail: str | None,
    stage_id_fallback: str | None = None,
) -> Layer3Output:
    delta_g: float | None = None
    delta_g_unc: float | None = None
    convergence_flags: dict[str, bool] = {}
    replicate_consistency: float | None = None
    energy_components: dict[str, float] = {}
    diagnostics: dict = {}
    stage_id: str | None = stage_id_fallback
    status: str | None = None
    writer_notes: str | None = None

    if payload is not None:
        # Pull out known keys; leftovers go under diagnostics so the
        # post-eval experts can still see them.
        known_keys = {
            "stage_id",
            "status",
            "writer_notes",
            "delta_g_kcal_per_mol",
            "delta_g_uncertainty_kcal_per_mol",
            "wall_time_seconds",
            "convergence_flags",
            "replicate_consistency",
            "energy_components",
            "diagnostics",
        }
        delta_g = _maybe_float(payload.get("delta_g_kcal_per_mol"))
        delta_g_unc = _maybe_float(payload.get("delta_g_uncertainty_kcal_per_mol"))
        convergence_flags = dict(payload.get("convergence_flags") or {})
        replicate_consistency = _maybe_float(payload.get("replicate_consistency"))
        energy_components = {}
        non_numeric_energy_components = {}
        for k, v in (payload.get("energy_components") or {}).items():
            if v is None:
                continue
            try:
                energy_components[k] = float(v)
            except (TypeError, ValueError):
                # Pipeline wrote a non-numeric diagnostic (string or dict, e.g.
                # "less-negative" or {"systematic_bias_TIP3P_cation": ...}).
                # Layer3Output schema requires dict[str, float] for energy_components,
                # so stash these under diagnostics rather than dropping them.
                non_numeric_energy_components[k] = v
        diagnostics = dict(payload.get("diagnostics") or {})
        if non_numeric_energy_components:
            diagnostics["non_numeric_energy_components"] = non_numeric_energy_components
        # Writer-self-reported metadata
        if payload.get("stage_id"):
            stage_id = str(payload["stage_id"])
        if payload.get("status"):
            status = str(payload["status"])
        if payload.get("writer_notes"):
            writer_notes = str(payload["writer_notes"])
        for k, v in payload.items():
            if k not in known_keys:
                diagnostics.setdefault(f"extra:{k}", v)

    return Layer3Output(
        stage_id=stage_id,
        status=status,
        writer_notes=writer_notes,
        delta_g_kcal_per_mol=delta_g,
        delta_g_uncertainty=delta_g_unc,
        convergence_flags=convergence_flags,
        replicate_consistency=replicate_consistency,
        energy_components=energy_components,
        diagnostics=diagnostics,
        wall_time_seconds=wall_time,
        artifacts_dir=artifacts_dir,
        exited_normally=(return_code == 0 and not timed_out),
        timed_out=timed_out,
        parse_failed=parse_failed,
        return_code=return_code,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _maybe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
