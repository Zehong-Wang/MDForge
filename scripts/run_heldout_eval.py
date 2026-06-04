#!/usr/bin/env python
"""Held-out evaluation pass for the paper run.

Given the best-rev pipeline produced by the 5-trial verbal-RL inner
loop, this script applies it to every **held-out** guest for the host
in question (the complement of the 4-guest benchmark set picked by
``run_paper_one_host.py``), parallelising over 4 GPUs in batches.

It reuses the existing ``MultiMoleculeBenchmark.evaluate(...)``
subprocess-based per-guest runner so the held-out pass uses **the
identical execution path** as the in-the-loop benchmark (same
sandbox staging, same `01_prep → 04_analysis` invocation, same env
plumbing, same per-stage result parsing, same aggregate computation).
Only the guest-index list differs.

For long held-out sets (>4 guests) the script splits indices into
ceil(N/4) batches of ≤4 guests each, runs each batch through
``MultiMoleculeBenchmark.evaluate`` on the same 4 GPUs (0..3), then
re-aggregates rows across all batches into a single ``heldout_summary.json``
including MAE, RMSE, bias, R², Pearson r, Kendall τ, Spearman ρ.

CLI inputs (env vars):

- ``PAPER_HOST``              — atom_id, e.g. ``sampl4_cb7`` (required)
- ``PAPER_PIPELINE_DIR``      — best-rev pipeline ``.../pipeline/code``
                                 directory containing the stage source files
                                 (required)
- ``PAPER_HELDOUT_INDICES``   — comma-separated guest indices to evaluate
                                 (required)
- ``PAPER_OUTPUT_DIR``        — destination directory for sandboxes and
                                 ``heldout_summary.json`` (required)
- ``PRISM_NS_PER_WINDOW``     — sampling depth per window (default ``1.0``;
                                 held-out uses the same depth as training)
- ``PRISM_BENCHMARK_TIMEOUT_SEC`` — per-task hard cap (default 90 min)
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from src.models import HostGuestSpec, Pipeline  # noqa: E402
from src.verifiers.multi_molecule_benchmark import (  # noqa: E402
    BenchmarkRow,
    MultiMoleculeBenchmark,
)


def _env_req(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise SystemExit(f"FATAL: {name} env var is required")
    return val


def _load_pipeline_from_code_dir(code_dir: Path, host_id: str) -> Pipeline:
    """Read every file under ``code_dir`` into a Pipeline.code_files dict.

    Drops `__pycache__` and any leftover `00_*.py` docking-stage files
    (older trial trees may still carry them).
    """
    if not code_dir.exists() or not code_dir.is_dir():
        raise SystemExit(f"FATAL: pipeline code dir not found: {code_dir}")
    files: dict[str, str] = {}
    for p in sorted(code_dir.rglob("*")):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(code_dir))
        if rel.startswith("00_") and rel.endswith(".py"):
            continue
        try:
            files[rel] = p.read_text()
        except UnicodeDecodeError:
            print(f"[heldout] skipping non-text file {rel}", file=sys.stderr)
    if not files:
        raise SystemExit(f"FATAL: no source files in {code_dir}")
    # Pick a defensible entry_point: prefer the lowest-numbered stage,
    # fall back to first file. The benchmark runner doesn't use this
    # field at all (stage discovery is by filename pattern) — only
    # included for model validity.
    stage_re = __import__("re").compile(r"^\d{2}_.*\.py$")
    stages = sorted(n for n in files if stage_re.match(n))
    entry = stages[0] if stages else next(iter(files))
    return Pipeline(
        target=HostGuestSpec(host_id=host_id, guest_id="heldout"),
        code_files=files,
        entry_point=entry,
        revision=0,
        rationale=f"Loaded from {code_dir} for held-out evaluation.",
    )


def _aggregate_all(rows: list[BenchmarkRow]) -> dict:
    """Cross-batch aggregate. Mirrors MultiMoleculeBenchmark._aggregate but
    also computes Pearson r (which the in-loop aggregate omits)."""
    valid = [
        r for r in rows
        if r.delta_g_predicted_kcal is not None
        and r.delta_g_experimental_kcal is not None
        and r.delta_g_predicted_kcal == r.delta_g_predicted_kcal
        and r.delta_g_experimental_kcal == r.delta_g_experimental_kcal
    ]
    n_valid = len(valid)
    out: dict = {
        "n_total": len(rows),
        "n_valid": n_valid,
        "mae_kcal": None,
        "rmse_kcal": None,
        "bias_kcal": None,
        "r_squared": None,
        "pearson_r": None,
        "kendall_tau": None,
        "spearman_rho": None,
    }
    if n_valid == 0:
        return out
    errs = [r.error_kcal for r in valid]
    out["mae_kcal"] = sum(abs(e) for e in errs) / n_valid
    out["rmse_kcal"] = math.sqrt(sum(e * e for e in errs) / n_valid)
    out["bias_kcal"] = sum(errs) / n_valid
    xs = [r.delta_g_experimental_kcal for r in valid]
    ys = [r.delta_g_predicted_kcal for r in valid]
    if n_valid >= 3:
        mx = sum(xs) / n_valid
        my = sum(ys) / n_valid
        ssxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        ssxx = sum((x - mx) ** 2 for x in xs)
        ssyy = sum((y - my) ** 2 for y in ys)
        if ssxx > 0 and ssyy > 0:
            out["pearson_r"] = ssxy / math.sqrt(ssxx * ssyy)
            out["r_squared"] = out["pearson_r"] ** 2
    if n_valid >= 2:
        try:
            from scipy.stats import kendalltau, spearmanr
            tau = kendalltau(xs, ys)[0]
            rho = spearmanr(xs, ys)[0]
            if tau == tau:
                out["kendall_tau"] = float(tau)
            if rho == rho:
                out["spearman_rho"] = float(rho)
        except Exception as exc:
            print(f"[heldout] scipy rank-correlation failed: {exc!r}", file=sys.stderr)
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


def main() -> int:
    host = _env_req("PAPER_HOST")
    pipeline_dir = Path(_env_req("PAPER_PIPELINE_DIR")).resolve()
    heldout_csv = _env_req("PAPER_HELDOUT_INDICES")
    output_dir = Path(_env_req("PAPER_OUTPUT_DIR")).resolve()
    ns_per_window = float(os.environ.get("PRISM_NS_PER_WINDOW", "1.0"))
    per_task_timeout = int(os.environ.get("PRISM_BENCHMARK_TIMEOUT_SEC", str(90 * 60)))

    indices = [int(x) for x in heldout_csv.split(",") if x.strip()]
    if not indices:
        raise SystemExit("FATAL: PAPER_HELDOUT_INDICES is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[heldout] host={host}")
    print(f"[heldout] pipeline_dir={pipeline_dir}")
    print(f"[heldout] held_out indices ({len(indices)}): {indices}")
    print(f"[heldout] output_dir={output_dir}")
    print(f"[heldout] ns_per_window={ns_per_window} per_task_timeout={per_task_timeout}s")

    pipeline = _load_pipeline_from_code_dir(pipeline_dir, host_id=host.split("_")[-1])

    # Split into batches of 4 (4 GPUs available).
    batch_size = 4
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    print(f"[heldout] {len(batches)} batches of ≤{batch_size} guests each")

    all_rows: list[BenchmarkRow] = []
    t_total0 = time.time()
    for bi, batch_indices in enumerate(batches):
        n = len(batch_indices)
        gpu_ids = list(range(n))  # contiguous 0..n-1
        bench = MultiMoleculeBenchmark(
            atom_id=host,
            guest_indices=batch_indices,
            gpu_ids=gpu_ids,
            ns_per_window=ns_per_window,
            per_task_timeout_seconds=per_task_timeout,
        )
        batch_out_dir = output_dir / f"batch_{bi:02d}"
        batch_out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[heldout] batch {bi+1}/{len(batches)}: guests={batch_indices} "
            f"gpus={gpu_ids} → {batch_out_dir}"
        )
        t0 = time.time()
        result, _critique = bench.evaluate(pipeline=pipeline, out_dir=batch_out_dir)
        dt = time.time() - t0
        print(
            f"[heldout] batch {bi+1} done: {result.n_valid}/{len(result.rows)} valid; "
            f"MAE={result.mae if result.mae is not None else 'N/A'} "
            f"τ={result.kendall_tau if result.kendall_tau is not None else 'N/A'} "
            f"wall={dt:.0f}s"
        )
        all_rows.extend(result.rows)

    agg = _aggregate_all(all_rows)
    summary = {
        "host": host,
        "pipeline_dir": str(pipeline_dir),
        "held_out_indices": indices,
        "ns_per_window": ns_per_window,
        "per_task_timeout_seconds": per_task_timeout,
        "n_batches": len(batches),
        "wall_time_seconds": time.time() - t_total0,
        "aggregate": agg,
        "rows": [_row_to_dict(r) for r in all_rows],
    }
    summary_path = output_dir / "heldout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[heldout] wrote {summary_path}")
    print(
        f"[heldout] FINAL: n_valid={agg['n_valid']}/{agg['n_total']}, "
        f"MAE={agg['mae_kcal']}, RMSE={agg['rmse_kcal']}, bias={agg['bias_kcal']}, "
        f"R²={agg['r_squared']}, Pearson r={agg['pearson_r']}, "
        f"Kendall τ={agg['kendall_tau']}, Spearman ρ={agg['spearman_rho']}"
    )
    return 0 if agg["n_valid"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
