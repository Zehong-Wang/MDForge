#!/usr/bin/env python
"""Paper run: dispatch one (host × ablation) cell through `run_real.py`.

A thin launcher. It runs in *two phases*:

1.  **Pre-run setup** — load the SAMPL manifest, read every guest's
    `measurement.yaml`, sort guests by experimental ΔG, pick the four
    equally-spaced quintile-position guests as the **benchmark
    (train/eval) set**, pick the **lead** as the benchmark guest with
    ΔG closest to the median of the four, and mark the rest as
    **held-out**. Persist the split to ``SETUP.json`` inside the
    artefact directory so every downstream consumer (held-out eval,
    summary subagent) can see the apples-to-apples cell split.

2.  **Dispatch** — export `PRISM_ATOM`, `PRISM_GUEST_IDX`,
    `PRISM_BENCHMARK_GUESTS`, `PRISM_BUDGET_N`, `PRISM_RUN_ROOT`,
    `PRISM_ABLATION`, then ``subprocess.run`` the existing
    ``scripts/run_real.py`` and wait for its `DONE` sentinel. On
    success, write `CELL_DONE_<host>_<ablation>` at the paper run
    root so the master bash orchestrator (`run_paper_all_hosts.sh`)
    can detect cell completion without re-parsing logs.

CLI inputs (all via env vars; no argparse so the master script can
just `export` them):

- ``PAPER_HOST`` ∈ ``sampl4_cb7 | sampl4_oah | sampl5_cbclip``  (required)
- ``PAPER_ABLATION`` ∈ ``full | v1_final_only | v2_single_agent
                       | v3_final_single``                       (default ``full``)
- ``PAPER_RUN_TS``  — run-tree timestamp produced by the master
                      bash script                                  (required)

All other run-tunables (``PRISM_BUDGET_N``, ``PRISM_NS_PER_WINDOW``,
``PRISM_BENCHMARK_TIMEOUT_SEC`` …) inherit from the calling environment
unchanged.

> **Benchmark/lead/held-out split**: the benchmark/lead/held-out
> split is computed *fresh per host* from experimental ΔG quintile
> positions; the hardcoded `[6,7,10,11]` defaults of `run_real.py` are
> overridden by this script for every paper cell.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from src.tasks import load_manifest, task_from_manifest  # noqa: E402


VALID_HOSTS = ("sampl4_cb7", "sampl4_oah", "sampl5_cbclip")
VALID_ABLATIONS = (
    "full",
    "v1_final_only",
    "v2_single_agent",
    "v3_final_single",
)


def _read_env(name: str, *, required: bool = False, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"FATAL: {name} env var is required")
    return val or ""


def _benchmark_indices(n: int) -> list[int]:
    """4 quintile positions (rounded ints) over [0, n-1] when n guests
    are sorted by ΔG_exp: use ``N//5``, ``2*N//5``,
    ``3*N//5``, ``4*N//5``; equivalent to ``numpy.linspace(0.2, 0.8, 4)
    * (N-1)`` rounded. We use the simple ``k * N // 5`` form because it
    matches the spec text exactly and works with N=5 (minimum to have
    4 distinct positions)."""
    if n < 5:
        raise SystemExit(
            f"FATAL: need at least 5 guests with measurement.yaml for "
            f"quintile-positioned benchmark; got n={n}"
        )
    positions = [(k * n) // 5 for k in (1, 2, 3, 4)]
    # Defensive dedup-while-preserving-order in case n is small and two
    # positions collapse to the same index.
    seen = set()
    unique = []
    for p in positions:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    if len(unique) < 4:
        # Shift later positions forward until we have 4 distinct indices.
        cursor = unique[-1] + 1
        while len(unique) < 4 and cursor < n:
            if cursor not in seen:
                unique.append(cursor)
                seen.add(cursor)
            cursor += 1
    if len(unique) < 4:
        raise SystemExit(
            f"FATAL: could not assemble 4 distinct quintile positions for n={n}"
        )
    return unique[:4]


def _compute_split(host: str) -> dict:
    """Return the split dict {benchmark_indices, lead_idx, held_out_indices,
    benchmark_dg_exp, lead_dg_exp} for *host* by reading the manifest."""
    manifest = load_manifest()
    if host not in manifest["atoms"]:
        raise SystemExit(
            f"FATAL: host {host!r} not in manifest atoms "
            f"{sorted(manifest['atoms'])}"
        )
    guests = manifest["atoms"][host]["guests"]
    # Build (guest_idx, ΔG_exp_kcal) for every guest that has a
    # parseable measurement.yaml. We piggy-back on task_from_manifest
    # so the parsing matches the rest of the codebase verbatim.
    pairs: list[tuple[int, float]] = []
    for gi in range(len(guests)):
        try:
            task = task_from_manifest(host, gi)
        except Exception as exc:
            print(
                f"[paper] skipping {host} guest {gi}: task_from_manifest "
                f"failed ({exc!r})",
                file=sys.stderr,
            )
            continue
        dg = task.experimental_reference.delta_g_kcal_per_mol
        if dg != dg:  # NaN
            print(
                f"[paper] skipping {host} guest {gi} ({task.target.guest_id}):"
                " ΔG_exp is NaN (no measurement.yaml or unparseable)",
                file=sys.stderr,
            )
            continue
        pairs.append((gi, float(dg)))

    if len(pairs) < 5:
        raise SystemExit(
            f"FATAL: {host} has only {len(pairs)} guests with parseable "
            "experimental ΔG; cannot form 4-guest benchmark + held-out split."
        )

    # Sort by ΔG_exp and pick quintile positions.
    pairs_sorted = sorted(pairs, key=lambda p: p[1])
    qpos = _benchmark_indices(len(pairs_sorted))
    benchmark = [pairs_sorted[p] for p in qpos]  # 4 (idx, dG)
    benchmark_idxs = [p[0] for p in benchmark]
    benchmark_dgs = [p[1] for p in benchmark]

    # Lead = benchmark guest whose ΔG is closest to the median of the
    # 4 benchmark ΔGs: median falls between idx-1 and
    # idx-2 of the sorted 4 → pick whichever of the two is closer.
    median_dg = (benchmark_dgs[1] + benchmark_dgs[2]) / 2.0
    cand1, cand2 = benchmark[1], benchmark[2]
    if abs(cand1[1] - median_dg) <= abs(cand2[1] - median_dg):
        lead_idx = cand1[0]
        lead_dg = cand1[1]
    else:
        lead_idx = cand2[0]
        lead_dg = cand2[1]

    held_out = [p[0] for p in pairs if p[0] not in set(benchmark_idxs)]

    return {
        "host": host,
        "n_guests_with_dg": len(pairs),
        "benchmark_indices": benchmark_idxs,
        "lead_idx": lead_idx,
        "held_out_indices": held_out,
        "benchmark_dg_exp_kcal": benchmark_dgs,
        "lead_dg_exp_kcal": lead_dg,
        "median_dg_used_kcal": median_dg,
    }


def main() -> int:
    host = _read_env("PAPER_HOST", required=True)
    ablation = _read_env("PAPER_ABLATION", default="full")
    run_ts = _read_env("PAPER_RUN_TS", required=True)

    if host not in VALID_HOSTS:
        raise SystemExit(f"FATAL: PAPER_HOST must be one of {VALID_HOSTS}, got {host!r}")
    if ablation not in VALID_ABLATIONS:
        raise SystemExit(
            f"FATAL: PAPER_ABLATION must be one of {VALID_ABLATIONS}, got {ablation!r}"
        )

    # Per-cell artefact root: /tmp NVMe for MD throughput; rsync mirror
    # to /scratch365 is owned by run_paper_all_hosts.sh.
    paper_run_root = Path(f"/tmp/prism_paper_run/{run_ts}")
    paper_run_root.mkdir(parents=True, exist_ok=True)
    cell_root = paper_run_root / f"{host}-{ablation}"
    cell_root.mkdir(parents=True, exist_ok=True)

    print(f"[paper] HOST={host}  ABLATION={ablation}  TS={run_ts}")
    print(f"[paper] cell_root = {cell_root}")

    split = _compute_split(host)
    split["ablation"] = ablation
    split["cell_root"] = str(cell_root)
    setup_path = cell_root / "SETUP.json"
    setup_path.write_text(json.dumps(split, indent=2))
    print(f"[paper] split: benchmark_indices={split['benchmark_indices']}, "
          f"lead_idx={split['lead_idx']}, n_held_out={len(split['held_out_indices'])}")
    print(f"[paper] wrote SETUP.json -> {setup_path}")

    # ---- Build env for run_real.py subprocess --------------------------
    env = os.environ.copy()
    env["PRISM_ATOM"] = host
    env["PRISM_GUEST_IDX"] = str(split["lead_idx"])
    env["PRISM_BENCHMARK_GUESTS"] = ",".join(str(i) for i in split["benchmark_indices"])
    env["PRISM_BUDGET_N"] = env.get("PRISM_BUDGET_N", "5")
    env["PRISM_RUN_ROOT"] = str(cell_root)
    env["PRISM_ABLATION"] = ablation
    # CUDA / AMBERHOME / OPENMM_PLATFORM inherit from env unchanged.

    # ---- Dispatch run_real.py and wait for DONE sentinel ---------------
    run_real = HERE / "run_real.py"
    log_path = cell_root / "run_real.stdout.log"
    print(f"[paper] dispatching {run_real} → log {log_path}")
    t0 = time.time()
    with open(log_path, "w") as logf:
        rc = subprocess.run(
            [sys.executable, str(run_real)],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    dt = time.time() - t0
    print(f"[paper] run_real.py exit={rc} wall={dt:.0f}s")

    # ---- Find the DONE sentinel written by run_real.py ------------------
    # run_real puts artefacts at PRISM_RUN_ROOT/<ts>/DONE. The <ts> is
    # the inner run timestamp it generates. We glob for it so the master
    # bash script can rely on a single cell-level done sentinel.
    inner_dones = list(cell_root.glob("*/DONE"))
    cell_done_path = paper_run_root / f"CELL_DONE_{host}_{ablation}"
    if inner_dones:
        latest = max(inner_dones, key=lambda p: p.stat().st_mtime)
        cell_done_path.write_text(
            f"host={host}\nablation={ablation}\nexit={rc}\n"
            f"wall_seconds={dt:.0f}\nrun_real_done={latest}\n"
            f"setup={setup_path}\n"
        )
        print(f"[paper] wrote {cell_done_path} (inner DONE = {latest})")
    else:
        # Even if run_real exited non-zero (e.g. budget exhausted without
        # convergence), if it produced no DONE we surface that here.
        cell_done_path.write_text(
            f"host={host}\nablation={ablation}\nexit={rc}\nwall_seconds={dt:.0f}\n"
            f"run_real_done=MISSING (no DONE sentinel under {cell_root})\n"
            f"setup={setup_path}\n"
        )
        print(f"[paper] WARNING: no inner DONE found under {cell_root}; "
              f"wrote {cell_done_path} anyway", file=sys.stderr)

    return rc


if __name__ == "__main__":
    sys.exit(main())
