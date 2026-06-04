"""Multi-molecule benchmark verifier.

After the Engineer has driven a pipeline to completion on the design
molecule (single-trial single-molecule), this module APPLIES the same
pipeline to N other benchmark molecules in parallel on N GPUs, collects
per-molecule ΔG predictions, compares to experimental ITC references,
and returns a Layer-2 ``Critique`` that captures the cross-molecule
performance. That Critique is then fed to the next ``Writer.revise``
trial so the pipeline iterates on **cross-molecule** signal — not just
the single-molecule trial signal — which was found to drive over-fitting
in earlier experiments.

Design:
- Inputs: ``Pipeline`` object + a list of ``Task`` objects (one per
  benchmark molecule) + a list of GPU ids (one per task).
- Each task is given a sandbox dir with the pipeline's code + the task's
  host/guest/complex files + a per-task ``task_metadata.json``.
- Stages 01-04 are launched in parallel, one subprocess per GPU.
- After all complete (or time out), per-task ΔG is extracted and compared
  to ``task.experimental_reference.delta_g_kcal_per_mol``.
- Aggregate MAE / RMSE / bias / R² is computed.
- A ``Critique`` is returned with per-task errors as concerns + an
  aggregate strategic_insight describing the systematic failure modes
  (e.g. wildly varying dG_pull → window schedule issue; certain guests
  fail at antechamber → mol2 validation issue).

P4 invariant: This is **post-eval critique signal**, advisory only.
It is appended to the trial's post_critiques list so Writer.revise reads
it, but it does NOT gate trial execution.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.models import Concern, Critique, ExpertVote, LayerName, Pipeline, VerdictLabel
from src.tasks import Task, task_from_manifest

_log = logging.getLogger(__name__)

# Default sampling for the benchmark stage — keep equal or shorter than
# the engineer's smoke MD so the benchmark doesn't dominate wall-clock.
DEFAULT_NS_PER_WINDOW = 0.05
DEFAULT_BENCHMARK_TIMEOUT_SECONDS = 60 * 60  # per-task hard cap

# MD execution environment (Python + AmberTools + OpenMM). Defaults to a
# `pipeline/` conda env under the repo root; override with PRISM_PIPELINE.
PIPELINE_ROOT = os.environ.get(
    "PRISM_PIPELINE", str(Path(__file__).resolve().parents[2] / "pipeline")
)


@dataclass
class BenchmarkRow:
    """One row of the multi-molecule benchmark result table."""
    atom_id: str
    guest_idx: int
    guest_id: str
    task_id: str
    sandbox: str
    delta_g_predicted_kcal: float | None = None
    sigma_predicted_kcal: float | None = None
    delta_g_experimental_kcal: float | None = None
    sigma_experimental_kcal: float | None = None
    error_kcal: float | None = None
    status: str | None = None
    stage_reached: str | None = None
    failure_reason: str | None = None
    energy_components: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Top-level multi-molecule benchmark result."""
    rows: list[BenchmarkRow]
    n_valid: int
    mae: float | None = None
    rmse: float | None = None
    bias: float | None = None
    r_squared: float | None = None
    kendall_tau: float | None = None
    spearman_rho: float | None = None
    wall_time_seconds: float = 0.0


class MultiMoleculeBenchmark:
    """Apply a pipeline to N benchmark molecules in parallel and score it."""

    def __init__(
        self,
        atom_id: str,
        guest_indices: list[int],
        gpu_ids: list[int],
        cb7_bin: str = f"{PIPELINE_ROOT}/bin",
        ns_per_window: float = DEFAULT_NS_PER_WINDOW,
        per_task_timeout_seconds: float = DEFAULT_BENCHMARK_TIMEOUT_SECONDS,
    ):
        if len(gpu_ids) < len(guest_indices):
            raise ValueError(
                f"need at least {len(guest_indices)} GPU ids; got {gpu_ids}"
            )
        self.atom_id = atom_id
        self.guest_indices = list(guest_indices)
        self.gpu_ids = list(gpu_ids)[: len(guest_indices)]
        self.cb7_bin = cb7_bin
        self.ns_per_window = ns_per_window
        self.per_task_timeout = per_task_timeout_seconds

    def evaluate(
        self,
        pipeline: Pipeline,
        out_dir: Path,
    ) -> tuple[BenchmarkResult, Critique]:
        """Run the pipeline on each task in parallel; return aggregate result
        + Layer-2 post-eval Critique for the orchestrator to thread back to
        the next Writer.revise."""
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # Stage all sandboxes upfront (sequential — cheap)
        sandboxes = []
        for gi, gpu in zip(self.guest_indices, self.gpu_ids):
            task = task_from_manifest(self.atom_id, gi)
            sb = out_dir / f"task_{gi:02d}_{task.target.guest_id}"
            sb.mkdir(parents=True, exist_ok=True)
            self._stage_pipeline_code(pipeline, sb)
            self._stage_aux(task, sb)
            sandboxes.append((gi, gpu, sb, task))

        # Launch all in parallel
        procs = []
        for i, (gi, gpu, sb, task) in enumerate(sandboxes):
            log_path = sb / "benchmark_run.log"
            proc = self._launch(sb, gpu, i, log_path)
            procs.append((gi, gpu, sb, task, proc, log_path))
            _log.info(
                "multi_molecule_benchmark: launched task_%02d %s on GPU %d (PID %d)",
                gi, task.target.guest_id, gpu, proc.pid,
            )

        # Wait for all
        for gi, gpu, sb, task, proc, _ in procs:
            try:
                rc = proc.wait(timeout=self.per_task_timeout)
                _log.info(
                    "multi_molecule_benchmark: task_%02d %s exit=%d",
                    gi, task.target.guest_id, rc,
                )
            except subprocess.TimeoutExpired:
                _log.warning(
                    "multi_molecule_benchmark: task_%02d %s TIMEOUT after %ds; killing",
                    gi, task.target.guest_id, int(self.per_task_timeout),
                )
                try:
                    proc.kill()
                except Exception:
                    pass

        # Collect results
        rows: list[BenchmarkRow] = []
        for gi, gpu, sb, task, proc, _ in procs:
            row = BenchmarkRow(
                atom_id=self.atom_id,
                guest_idx=gi,
                guest_id=task.target.guest_id,
                task_id=task.task_id,
                sandbox=str(sb),
                delta_g_experimental_kcal=task.experimental_reference.delta_g_kcal_per_mol,
                sigma_experimental_kcal=task.experimental_reference.delta_g_uncertainty_kcal_per_mol,
            )
            self._populate_row_from_sandbox(row, sb)
            rows.append(row)

        result = self._aggregate(rows, time.time() - t0)
        critique = self._to_critique(result, rows)

        # Persist summary.json next to the sandboxes
        summary = {
            "atom_id": self.atom_id,
            "guest_indices": self.guest_indices,
            "gpus": self.gpu_ids,
            "n_valid": result.n_valid,
            "mae_kcal": result.mae,
            "rmse_kcal": result.rmse,
            "bias_kcal": result.bias,
            "r_squared": result.r_squared,
            "kendall_tau": result.kendall_tau,
            "spearman_rho": result.spearman_rho,
            "wall_time_seconds": result.wall_time_seconds,
            "rows": [_row_to_dict(r) for r in rows],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        _log.info(
            "multi_molecule_benchmark: %d/%d valid; MAE=%s RMSE=%s bias=%s R²=%s wall=%.0fs",
            result.n_valid, len(rows),
            f"{result.mae:.2f}" if result.mae is not None else "N/A",
            f"{result.rmse:.2f}" if result.rmse is not None else "N/A",
            f"{result.bias:+.2f}" if result.bias is not None else "N/A",
            f"{result.r_squared:.3f}" if result.r_squared is not None else "N/A",
            result.wall_time_seconds,
        )
        return result, critique

    # ---- helpers ---------------------------------------------------------

    # GPU-index fix: the benchmark masks each task to ONE GPU via
    # CUDA_VISIBLE_DEVICES, so the only valid OpenMM DeviceIndex inside a
    # task process is "0". Some Engineer-written pipelines wrongly set
    # DeviceIndex to the absolute GPU id (read straight off
    # CUDA_VISIBLE_DEVICES), which raises
    #   openmm.OpenMMException: Illegal value for DeviceIndex: <n>
    # on every GPU except 0 — silently collapsing a 4-guest benchmark to
    # n_valid=1. We can't trust the generated code, so inject a runtime
    # monkey-patch at the top of each staged stage file that forces
    # DeviceIndex → "0" at the OpenMM API boundary (covers
    # Platform.setPropertyDefaultValue and properties-dict forms).
    _GPU_FIX_SNIPPET = (
        "# === MDForge auto-injected GPU-index fix ===\n"
        "try:\n"
        "    import openmm as _prism_mm\n"
        "    _prism_o_spdv = _prism_mm.Platform.setPropertyDefaultValue\n"
        "    def _prism_p_spdv(self, name, value):\n"
        "        if str(name) == 'DeviceIndex':\n"
        "            value = '0'\n"
        "        return _prism_o_spdv(self, name, value)\n"
        "    _prism_mm.Platform.setPropertyDefaultValue = _prism_p_spdv\n"
        "    def _prism_scrub(x):\n"
        "        if isinstance(x, dict) and 'DeviceIndex' in x:\n"
        "            x = dict(x); x['DeviceIndex'] = '0'\n"
        "        return x\n"
        "    _prism_o_ctx = _prism_mm.Context.__init__\n"
        "    def _prism_p_ctx(self, *a, **k):\n"
        "        a = tuple(_prism_scrub(z) for z in a)\n"
        "        if 'platformProperties' in k:\n"
        "            k['platformProperties'] = _prism_scrub(k['platformProperties'])\n"
        "        return _prism_o_ctx(self, *a, **k)\n"
        "    _prism_mm.Context.__init__ = _prism_p_ctx\n"
        "    import openmm.app as _prism_app\n"
        "    _prism_o_sim = _prism_app.Simulation.__init__\n"
        "    def _prism_p_sim(self, *a, **k):\n"
        "        a = tuple(_prism_scrub(z) for z in a)\n"
        "        if 'platformProperties' in k:\n"
        "            k['platformProperties'] = _prism_scrub(k['platformProperties'])\n"
        "        return _prism_o_sim(self, *a, **k)\n"
        "    _prism_app.Simulation.__init__ = _prism_p_sim\n"
        "except Exception:\n"
        "    pass\n"
        "# === end MDForge GPU-fix ===\n"
    )

    @classmethod
    def _inject_gpu_fix(cls, src_text: str) -> str:
        """Prepend the DeviceIndex monkey-patch, after any shebang and
        `from __future__` imports (which must stay first in the file)."""
        lines = src_text.splitlines(keepends=True)
        insert_at = 0
        for i, ln in enumerate(lines):
            s = ln.strip()
            if i == 0 and s.startswith("#!"):
                insert_at = 1
            elif s.startswith("from __future__"):
                insert_at = i + 1
        return "".join(lines[:insert_at]) + cls._GPU_FIX_SNIPPET + "".join(lines[insert_at:])

    @classmethod
    def _stage_pipeline_code(cls, pipeline: Pipeline, dst: Path) -> None:
        """Write Pipeline.code_files into dst (skip leftover 00_*.py).

        Each `.py` stage file is prepended with the GPU-index fix.
        """
        for name, src_text in pipeline.code_files.items():
            if name.startswith("00_") and name.endswith(".py"):
                continue
            if name.endswith(".py"):
                src_text = cls._inject_gpu_fix(src_text)
            (dst / name).write_text(src_text)

    @staticmethod
    def _stage_aux(task: Task, dst: Path) -> None:
        """Same canonical-name staging as _collect_auxiliary_files in
        orchestrator.py: original-basename + canonical names + task_metadata.json."""
        for p in task.host_structure_files.values():
            shutil.copy2(p, dst / p.name)
            ext = p.suffix.lstrip(".")
            if ext in ("mol2", "pdb", "sdf"):
                shutil.copy2(p, dst / f"host.{ext}")
        for p in task.guest_structure_files.values():
            shutil.copy2(p, dst / p.name)
            ext = p.suffix.lstrip(".")
            if ext in ("mol2", "sdf", "pdb"):
                shutil.copy2(p, dst / f"guest.{ext}")
        if task.bound_complex_pdb is not None:
            shutil.copy2(task.bound_complex_pdb, dst / task.bound_complex_pdb.name)
            shutil.copy2(task.bound_complex_pdb, dst / "complex.pdb")
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
            "experimental_dG_kcal_per_mol": task.experimental_reference.delta_g_kcal_per_mol,
            "experimental_sigma_kcal_per_mol": task.experimental_reference.delta_g_uncertainty_kcal_per_mol,
        }
        (dst / "task_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    def _launch(
        self,
        sandbox: Path,
        gpu_id: int,
        task_idx: int,
        log_path: Path,
    ) -> subprocess.Popen:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PRISM_TASK_IDX"] = str(task_idx)
        env["PRISM_REPLICA_SEED"] = str(42 + task_idx * 1009)
        env["PRISM_PRODUCTION_NS_PER_WINDOW"] = f"{self.ns_per_window:g}"
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = self.cb7_bin + ":" + env.get("PATH", "")
        env["AMBERHOME"] = PIPELINE_ROOT
        env["OPENMM_PLATFORM"] = "CUDA"
        python = f"{self.cb7_bin}/python"
        cmd = [
            "bash", "-c",
            f"set -e; "
            f"echo '=== STAGE 01 ===' && {python} 01_prep.py 2>&1 && "
            f"echo '=== STAGE 02 ===' && {python} 02_equilibrate.py 2>&1 && "
            f"echo '=== STAGE 03 ===' && {python} 03_production.py 2>&1 && "
            f"echo '=== STAGE 04 ===' && {python} 04_analysis.py 2>&1",
        ]
        log_f = open(log_path, "w")
        return subprocess.Popen(
            cmd,
            cwd=str(sandbox),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    @staticmethod
    def _populate_row_from_sandbox(row: BenchmarkRow, sandbox: Path) -> None:
        """Read whatever the deepest stage_NN_*_result.json produced."""
        for k in (4, 3, 2, 1):
            cands = sorted(sandbox.glob(f"stage_{k:02d}_*result.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not cands:
                cands = sorted(sandbox.glob(f"stage_{k:02d}*result.json"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
            if cands:
                row.stage_reached = cands[0].name
                try:
                    d = json.loads(cands[0].read_text())
                except Exception as e:
                    row.failure_reason = f"json_parse_error: {e}"
                    return
                row.status = d.get("status")
                row.energy_components = d.get("energy_components") or {}
                ec = row.energy_components
                if k == 4:
                    dG = (
                        d.get("delta_g_kcal_per_mol")
                        or ec.get("delta_g_bind_with_rocklin_kcal")
                        or ec.get("delta_g_bind_raw_kcal")
                        or ec.get("delta_g_bind_kcal")
                    )
                    sigma = (
                        d.get("delta_g_uncertainty_kcal_per_mol")
                        or ec.get("estimated_total_uncertainty_kcal")
                        or ec.get("statistical_uncertainty_kcal")
                    )
                    if isinstance(dG, (int, float)) and (dG == dG):  # not NaN
                        row.delta_g_predicted_kcal = float(dG)
                        if isinstance(sigma, (int, float)):
                            row.sigma_predicted_kcal = float(sigma)
                        if row.delta_g_experimental_kcal is not None:
                            row.error_kcal = float(dG) - row.delta_g_experimental_kcal
                else:
                    row.failure_reason = f"stopped_at_stage_{k:02d}"
                return
        row.failure_reason = "no_stage_results"

    @staticmethod
    def _aggregate(rows: list[BenchmarkRow], wall_time: float) -> BenchmarkResult:
        valid = [
            r for r in rows
            if r.delta_g_predicted_kcal is not None
            and r.delta_g_experimental_kcal is not None
            and r.delta_g_experimental_kcal == r.delta_g_experimental_kcal  # not NaN
            and r.delta_g_predicted_kcal == r.delta_g_predicted_kcal
        ]
        if not valid:
            return BenchmarkResult(rows=rows, n_valid=0, wall_time_seconds=wall_time)
        errs = [r.error_kcal for r in valid]
        mae = sum(abs(e) for e in errs) / len(errs)
        rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
        bias = sum(errs) / len(errs)
        r2 = None
        if len(valid) >= 3:
            xs = [r.delta_g_experimental_kcal for r in valid]
            ys = [r.delta_g_predicted_kcal for r in valid]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            ssxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            ssxx = sum((x - mx) ** 2 for x in xs)
            ssyy = sum((y - my) ** 2 for y in ys)
            if ssxx * ssyy > 0:
                r2 = (ssxy * ssxy) / (ssxx * ssyy)
        # Rank correlations are the primary ranking metric used by
        # the paper's best-rev selection criterion (Kendall τ → Spearman
        # ρ → |bias| → MAE). Both require ≥2 valid points; scipy raises
        # if N<2 so we guard. Wrapped in try/except so a malformed array
        # (e.g. all-tied) silently degrades to None rather than crashing
        # the orchestrator at the end of a multi-hour trial.
        kendall_tau: float | None = None
        spearman_rho: float | None = None
        if len(valid) >= 2:
            try:
                from scipy.stats import kendalltau, spearmanr
                xs_corr = [r.delta_g_experimental_kcal for r in valid]
                ys_corr = [r.delta_g_predicted_kcal for r in valid]
                tau_res = kendalltau(xs_corr, ys_corr)
                rho_res = spearmanr(xs_corr, ys_corr)
                # scipy returns a NamedTuple-like; .statistic / .correlation
                # depending on version. Use [0] for back-compat.
                tau_val = float(tau_res[0])
                rho_val = float(rho_res[0])
                if tau_val == tau_val:  # not NaN
                    kendall_tau = tau_val
                if rho_val == rho_val:
                    spearman_rho = rho_val
            except Exception as exc:
                _log.warning(
                    "multi_molecule_benchmark: rank-correlation computation "
                    "failed (%r); leaving kendall_tau/spearman_rho = None",
                    exc,
                )
        return BenchmarkResult(
            rows=rows, n_valid=len(valid),
            mae=mae, rmse=rmse, bias=bias, r_squared=r2,
            kendall_tau=kendall_tau, spearman_rho=spearman_rho,
            wall_time_seconds=wall_time,
        )

    @staticmethod
    def _to_critique(result: BenchmarkResult, rows: list[BenchmarkRow]) -> Critique:
        """Convert the multi-molecule benchmark result into a Layer-2 critique
        that the Writer.revise will see on the next trial."""
        # Build the per-row description as a single text block
        n_total = len(rows)
        n_valid = result.n_valid
        n_failed = n_total - n_valid

        lines = [
            f"Multi-molecule benchmark on {n_total} {rows[0].atom_id if rows else '?'} "
            f"guests (1 ns/window, parallel on {n_total} GPUs):"
        ]
        for r in rows:
            if r.delta_g_predicted_kcal is not None:
                pull = r.energy_components.get("dG_pull_kcal", "?")
                sigma_s = (
                    f"{r.sigma_predicted_kcal:.2f}"
                    if isinstance(r.sigma_predicted_kcal, (int, float)) else "n/a"
                )
                err_s = (
                    f"{r.error_kcal:+7.2f}"
                    if isinstance(r.error_kcal, (int, float)) else "n/a"
                )
                itc_s = (
                    f"{r.delta_g_experimental_kcal:+7.2f}"
                    if isinstance(r.delta_g_experimental_kcal, (int, float)) else "n/a"
                )
                pull_s = pull if isinstance(pull, str) else (
                    f"{pull:+.2f}" if isinstance(pull, (int, float)) else "n/a"
                )
                lines.append(
                    f"  {r.guest_id:>4} ({r.task_id}): pred = {r.delta_g_predicted_kcal:+7.2f} kcal/mol "
                    f"(σ={sigma_s}), ITC = {itc_s}, err = {err_s}, dG_pull = {pull_s}"
                )
            else:
                lines.append(
                    f"  {r.guest_id:>4} ({r.task_id}): FAILED at {r.stage_reached or 'init'} "
                    f"({r.failure_reason or 'unknown'}); ITC = {r.delta_g_experimental_kcal:+.2f}"
                )

        if n_valid > 0:
            lines += [
                "",
                f"Aggregate ({n_valid}/{n_total} valid): "
                f"MAE={result.mae:.2f}, RMSE={result.rmse:.2f}, bias={result.bias:+.2f}"
                + (f", R²={result.r_squared:.3f}" if result.r_squared is not None else "")
                + " kcal/mol.",
            ]
        else:
            lines += [
                "",
                "**All benchmark molecules FAILED.** Pipeline does not transfer.",
            ]

        # Identify systematic failure modes by inspecting energy components
        # variance across valid rows
        diagnostics = MultiMoleculeBenchmark._diagnose(rows)
        if diagnostics:
            lines += ["", "Suspected systematic failure modes:"]
            lines += [f"  - {d}" for d in diagnostics]

        description = "\n".join(lines)

        # Construct concerns
        concerns: list[Concern] = []
        if n_failed > 0:
            concerns.append(Concern(
                layer=LayerName.LAYER_2_POST,
                expert="multi_molecule_benchmark",
                severity=min(1.0, 0.5 + 0.5 * (n_failed / n_total)),
                description=f"{n_failed}/{n_total} benchmark molecules failed end-to-end.",
                suggested_focus="01_prep.py (parameterisation robustness across guest variations)",
            ))
        if result.mae is not None and result.mae > 10:
            concerns.append(Concern(
                layer=LayerName.LAYER_2_POST,
                expert="multi_molecule_benchmark",
                severity=min(1.0, result.mae / 20.0),
                description=f"Cross-molecule MAE = {result.mae:.2f} kcal/mol — pipeline is not "
                            f"yet competitive (literature SAMPL CB7 best: ~1-2 kcal/mol; "
                            f"defensible: ~3-5 kcal/mol).",
                suggested_focus="04_analysis.py — investigate dG_pull cross-molecule variance / window-exclusion logic",
            ))
        for d in diagnostics:
            concerns.append(Concern(
                layer=LayerName.LAYER_2_POST,
                expert="multi_molecule_benchmark",
                severity=0.7,
                description=d,
                suggested_focus=None,
            ))

        # Construct the single expert vote
        # label = pass / fail / uncertain based on MAE
        if n_valid == 0:
            label = VerdictLabel.FAIL
            conf = 0.95
            si = ("All benchmark molecules failed end-to-end. The pipeline does not transfer "
                  "beyond the design molecule. The next revision MUST address the parameterisation "
                  "robustness issues identified above before re-attempting cross-molecule evaluation.")
        elif result.mae is None or result.mae > 10:
            label = VerdictLabel.FAIL
            conf = 0.9
            si = (f"Cross-molecule MAE = {result.mae:.2f} kcal/mol on {n_valid}/{n_total} valid guests. "
                  "This is well above competitive autonomous-MD-pipeline performance for CB7-cation "
                  "binding (literature ~3 kcal/mol). The next revision should focus on the systematic "
                  "failure modes identified above — particularly cross-molecule variability of dG_pull, "
                  "which signals data-driven analysis-stage logic that doesn't generalise.")
        elif result.mae > 5:
            label = VerdictLabel.UNCERTAIN
            conf = 0.7
            si = (f"Cross-molecule MAE = {result.mae:.2f} kcal/mol on {n_valid}/{n_total} valid guests. "
                  "Pipeline transfers reasonably but has room to improve. Suggested next revision: "
                  "tighten remaining cross-molecule outliers without breaking transferability.")
        else:
            label = VerdictLabel.PASS
            conf = 0.85
            si = (f"Cross-molecule MAE = {result.mae:.2f} kcal/mol on {n_valid}/{n_total} valid guests "
                  "— competitive with literature autonomous methods. Pipeline is methodologically "
                  "sound and transfers; minor refinements possible but not urgent.")

        vote = ExpertVote(
            expert="multi_molecule_benchmark",
            label=label,
            confidence=conf,
            strategic_insight=si,
            concerns=concerns,
            reasoning=description[:1500],
        )

        return Critique(
            layer=LayerName.LAYER_2_POST,
            label=label,
            aggregate_confidence=conf,
            concerns=concerns,
            expert_votes=[vote],
            summary=(
                f"Multi-molecule benchmark: {n_valid}/{n_total} valid; "
                + (f"MAE={result.mae:.2f}" if result.mae is not None else "no valid points")
                + (f", RMSE={result.rmse:.2f}" if result.rmse is not None else "")
                + (f", bias={result.bias:+.2f}" if result.bias is not None else "")
                + " kcal/mol."
            ),
        )

    @staticmethod
    def _diagnose(rows: list[BenchmarkRow]) -> list[str]:
        """Inspect cross-row energy components for systematic issues."""
        out = []
        # dG_pull spread
        pulls = [r.energy_components.get("dG_pull_kcal") for r in rows
                 if r.energy_components.get("dG_pull_kcal") is not None]
        pulls = [float(p) for p in pulls if isinstance(p, (int, float))]
        if len(pulls) >= 3:
            mn, mx = min(pulls), max(pulls)
            if mx - mn > 30:
                out.append(
                    f"dG_pull spread across molecules is {mn:+.1f} to {mx:+.1f} kcal/mol "
                    f"(range {mx-mn:.1f}); literature variability for CB7-cation pull-leg is "
                    f"typically <10 kcal/mol — strongly suggests data-driven analysis logic "
                    f"that doesn't generalise."
                )
            elif any(p < -20 or p > 60 for p in pulls):
                out.append(
                    f"dG_pull values are outside literature range (-15 to +35) on some guests: {pulls}"
                )
        # Sign flips
        signs = set()
        for p in pulls:
            signs.add("+" if p > 0 else "-" if p < 0 else "0")
        if "+" in signs and "-" in signs and len(pulls) >= 3:
            out.append(
                "dG_pull SIGN FLIPS across guests — physically implausible since the pull-leg "
                "always describes guest UNBINDING (always +). Pipeline likely has a windows-"
                "exclusion bug that integrates over wrong sign region for some guests."
            )
        # Failure pattern
        n_failed = sum(1 for r in rows if r.delta_g_predicted_kcal is None)
        if n_failed > 0:
            # group by failure_reason
            from collections import Counter
            reasons = Counter(r.failure_reason for r in rows if r.delta_g_predicted_kcal is None)
            for reason, count in reasons.most_common():
                if reason and ("sqm" in reason.lower() or "odd" in reason.lower()
                              or "antechamber" in reason.lower() or "mol2" in reason.lower()):
                    out.append(
                        f"{count} guest(s) failed at antechamber/sqm with '{reason}' — "
                        f"this is a guest-mol2-protonation issue (declared net_charge inconsistent "
                        f"with mol2 atom inventory). Stage 01 needs guest mol2 sanity check "
                        f"(integer net-charge → expected closed-shell electron count) BEFORE "
                        f"invoking antechamber."
                    )
        return out


def _row_to_dict(r: BenchmarkRow) -> dict:
    return {
        "atom_id": r.atom_id,
        "guest_idx": r.guest_idx,
        "guest_id": r.guest_id,
        "task_id": r.task_id,
        "sandbox": r.sandbox,
        "delta_g_predicted_kcal": r.delta_g_predicted_kcal,
        "sigma_predicted_kcal": r.sigma_predicted_kcal,
        "delta_g_experimental_kcal": r.delta_g_experimental_kcal,
        "sigma_experimental_kcal": r.sigma_experimental_kcal,
        "error_kcal": r.error_kcal,
        "status": r.status,
        "stage_reached": r.stage_reached,
        "failure_reason": r.failure_reason,
        "energy_components": r.energy_components,
    }
