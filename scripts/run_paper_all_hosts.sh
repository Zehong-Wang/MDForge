#!/usr/bin/env bash
# Master serial orchestrator for the full 4-variant × 3-host paper run
# (12 cells): 4 variants × 3 hosts.
#
# Design constraints:
#   * STRICTLY SERIAL — 4 GPUs are saturated by the inner benchmark of
#     each cell; running two cells concurrently would thrash. One cell
#     at a time, full stop.
#   * /tmp NVMe writes (MD I/O), then 30-min rsync mirror to
#     persistent storage (survives /tmp scrubs and machine reboots).
#   * `nohup`-safe and logout-safe: all output redirected to log files;
#     no terminal-dependent commands; no `tty`; no interactive prompts.
#
# Usage (run under nohup; survives logout):
#   cd <repo-root>
#   nohup bash scripts/run_paper_all_hosts.sh > /tmp/run_paper_all_hosts.boot.log 2>&1 &
#
# The actual run logs live at $SCRATCH/master.log and per-cell log
# files; $TMP is the live /tmp tree (where MD I/O happens), $SCRATCH
# is the rsync-mirrored persistent home (under the repo root by default).

set -u  # -e omitted on purpose: a single failing cell must NOT abort the
        # remaining 11 cells; we handle per-cell exit codes manually.

# ---- Setup -----------------------------------------------------------
TS="$(date +%Y%m%d-%H%M%S)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${PAPER_SCRATCH:-${REPO_ROOT}/runs/paper_run/${TS}}"
# SCRATCH is a writable persistent location, under the repo root by default
# (override via PAPER_SCRATCH). Heavy MD trajectories are excluded from the
# rsync mirror so the persistent footprint stays small.
TMP="/tmp/prism_paper_run/${TS}"
mkdir -p "${SCRATCH}" "${TMP}"

export PAPER_RUN_TS="${TS}"

MASTER_LOG="${SCRATCH}/master.log"
STATUS_MD="${SCRATCH}/STATUS.md"

PYBIN="${PYBIN:-${REPO_ROOT}/pipeline/bin/python}"
PAPER_ONE_HOST="${REPO_ROOT}/scripts/run_paper_one_host.py"
HELDOUT_EVAL="${REPO_ROOT}/scripts/run_heldout_eval.py"

# Redirect everything else from this point on to the master log.
exec >> "${MASTER_LOG}" 2>&1

echo "================================================================"
echo "PAPER RUN MASTER ORCHESTRATOR"
echo "TS         = ${TS}"
echo "SCRATCH    = ${SCRATCH}"
echo "TMP        = ${TMP}"
echo "PYBIN      = ${PYBIN}"
echo "MASTER_LOG = ${MASTER_LOG}"
echo "STATUS_MD  = ${STATUS_MD}"
echo "started at $(date -Iseconds)"
echo "================================================================"

# Sanity-check the helpers exist before we kick off the long run.
for f in "${PAPER_ONE_HOST}" "${HELDOUT_EVAL}"; do
  if [[ ! -f "${f}" ]]; then
    echo "FATAL: missing helper script ${f}" >&2
    exit 1
  fi
done

# Seed STATUS.md so a tail -f works from minute zero.
cat > "${STATUS_MD}" <<EOF
# Paper run STATUS — ${TS}

| # | cell | exit | best-rev | held-out τ | held-out MAE | wall (s) | completed at |
|---|------|------|----------|-----------|--------------|----------|--------------|
EOF

# ---- Background rsync loop ------------------------------------------
# Every 30 min, mirror /tmp run tree → $SCRATCH. Logs go to rsync.log
# in scratch. The loop terminates only when this master script kills it.
# `sleep` is the first cmd inside the loop body so we do not fire a
# rsync immediately on launch (no point — /tmp is still empty).
(
  while sleep 1800; do
    rsync -aL --delete \
      --exclude='master.log' --exclude='STATUS.md' \
      --exclude='rsync.log' --exclude='rsync.pid' \
      --exclude='ALL_DONE' \
      --exclude='*.dcd' --exclude='*.trj' --exclude='*.nc' \
      --exclude='*.xtc' --exclude='*.trr' --exclude='*.tng' \
      --exclude='*.rst' --exclude='*.rst7' --exclude='*.crd' \
      --exclude='*.mdcrd' --exclude='*.netcdf' \
      "${TMP}/" "${SCRATCH}/" 2>>"${SCRATCH}/rsync.log"
    echo "rsync at $(date -Iseconds)" >> "${SCRATCH}/rsync.log"
  done
) &
RSYNC_PID=$!
echo "${RSYNC_PID}" > "${SCRATCH}/rsync.pid"
echo "[master] background rsync loop launched (PID ${RSYNC_PID})"

# Make sure we kill the rsync loop on any exit path.
cleanup_rsync() {
  if [[ -n "${RSYNC_PID:-}" ]] && kill -0 "${RSYNC_PID}" 2>/dev/null; then
    echo "[master] killing rsync loop (PID ${RSYNC_PID})"
    kill "${RSYNC_PID}" 2>/dev/null || true
  fi
}
trap cleanup_rsync EXIT INT TERM

# ---- Cell list (12 cells, host-grouped) -----------------------------
CELLS=(
  "sampl4_cb7:full"
  "sampl4_cb7:v1_final_only"
  "sampl4_cb7:v2_single_agent"
  "sampl4_cb7:v3_final_single"
  "sampl4_oah:full"
  "sampl4_oah:v1_final_only"
  "sampl4_oah:v2_single_agent"
  "sampl4_oah:v3_final_single"
  "sampl5_cbclip:full"
  "sampl5_cbclip:v1_final_only"
  "sampl5_cbclip:v2_single_agent"
  "sampl5_cbclip:v3_final_single"
)

# ---- Per-cell driver -------------------------------------------------
# Best-rev ranking: read every trial's benchmark_trial_NN/summary.json,
# require n_valid==4, then sort by
#   (kendall_tau desc, spearman_rho desc, |bias| asc, mae asc)
# and emit the absolute path to the winning trial's pipeline/code dir
# (the held-out evaluator's PAPER_PIPELINE_DIR input).
pick_best_rev() {
  local cell_root="$1"
  "${PYBIN}" - <<PYEOF
import json, sys
from pathlib import Path

cell_root = Path("${cell_root}")
candidates = []
for summary in cell_root.rglob("benchmark_trial_*/summary.json"):
    try:
        d = json.loads(summary.read_text())
    except Exception as exc:
        print(f"[pick_best_rev] skip {summary}: {exc!r}", file=sys.stderr)
        continue
    if d.get("n_valid") != 4:
        continue
    tau = d.get("kendall_tau")
    rho = d.get("spearman_rho")
    bias = d.get("bias_kcal")
    mae = d.get("mae_kcal")
    # Rank tuple: tau ↑, rho ↑, |bias| ↓, mae ↓. We sort ascending so
    # invert the "higher is better" terms by negating.
    rank = (
        -(tau if tau is not None else -1e9),
        -(rho if rho is not None else -1e9),
        abs(bias) if bias is not None else 1e9,
        mae if mae is not None else 1e9,
    )
    # The trial's pipeline code dir is two levels up from summary.json
    # (benchmark_trial_NN/summary.json → trial_NN/pipeline/code/). The
    # exact layout matches orchestrator._save_pipeline:
    #   <cell_root>/<task-ts>/trial_NN/pipeline/code/
    # The benchmark output dir lives at the same level as workdir:
    #   <cell_root>/<task-ts>/benchmark_trial_NN/summary.json
    # We resolve trial dir by reading the rev number from summary path.
    bench_dir = summary.parent  # benchmark_trial_NN
    name = bench_dir.name  # benchmark_trial_03
    trial_num = name.split("_")[-1]  # "03"
    trial_dir = bench_dir.parent / f"trial_{trial_num}"
    code_dir = trial_dir / "pipeline" / "code"
    if not code_dir.exists():
        # benchmark_trial_NN lives under workdirs/, but the
        # trial_NN/pipeline/code lives under <artifact_dir>/<task-ts>/, which
        # is a sibling of workdirs/. So we need to rglob from one level up
        # (the artifact_dir, not workdirs/).
        alt = list(bench_dir.parent.parent.rglob(f"trial_{trial_num}/pipeline/code"))
        if alt:
            code_dir = alt[0]
        else:
            # Last resort: glob anywhere under cell_root.
            alt2 = list(cell_root.rglob(f"trial_{trial_num}/pipeline/code"))
            if alt2:
                code_dir = alt2[0]
            else:
                print(f"[pick_best_rev] no code dir for {summary}", file=sys.stderr)
                continue
    candidates.append((rank, summary, code_dir, tau, rho, bias, mae))

if not candidates:
    print("NONE")
    sys.exit(0)
candidates.sort(key=lambda x: x[0])
_rank, _summary, code_dir, tau, rho, bias, mae = candidates[0]
print(str(code_dir))
print(f"tau={tau} rho={rho} bias={bias} mae={mae}", file=sys.stderr)
PYEOF
}

run_one_cell() {
  local cell="$1"
  local host="${cell%%:*}"
  local ablation="${cell##*:}"
  local cell_root="${TMP}/${host}-${ablation}"
  local cell_log="${SCRATCH}/${host}-${ablation}.log"

  echo "----------------------------------------------------------------"
  echo "[master] CELL START  host=${host}  ablation=${ablation}"
  echo "[master] cell_root = ${cell_root}"
  echo "[master] cell_log  = ${cell_log}"
  echo "[master] start at $(date -Iseconds)"

  local t0
  t0=$(date +%s)

  # ---- Phase A: 5-trial verbal-RL ----------------------------------
  PAPER_HOST="${host}" \
  PAPER_ABLATION="${ablation}" \
  PAPER_RUN_TS="${TS}" \
    "${PYBIN}" "${PAPER_ONE_HOST}" \
    >> "${cell_log}" 2>&1
  local rl_exit=$?
  echo "[master] run_paper_one_host.py exit=${rl_exit}"

  # ---- Phase B: pick best-rev pipeline -----------------------------
  local best_code_dir
  best_code_dir=$(pick_best_rev "${cell_root}" 2>>"${cell_log}")
  echo "[master] pick_best_rev → ${best_code_dir}"

  local heldout_dir="${cell_root}/heldout"
  local tau="N/A"
  local mae="N/A"
  local heldout_exit=255

  if [[ "${best_code_dir}" == "NONE" || -z "${best_code_dir}" ]]; then
    echo "[master] NO best-rev pipeline (all trials had n_valid<4); skipping held-out"
  else
    # Look up held-out indices from SETUP.json
    local setup_json="${cell_root}/SETUP.json"
    if [[ ! -f "${setup_json}" ]]; then
      echo "[master] WARNING: ${setup_json} missing — cannot run held-out"
    else
      local heldout_csv
      heldout_csv=$("${PYBIN}" -c "import json,sys; \
print(','.join(str(x) for x in json.load(open('${setup_json}'))['held_out_indices']))")
      mkdir -p "${heldout_dir}"
      PAPER_HOST="${host}" \
      PAPER_PIPELINE_DIR="${best_code_dir}" \
      PAPER_HELDOUT_INDICES="${heldout_csv}" \
      PAPER_OUTPUT_DIR="${heldout_dir}" \
        "${PYBIN}" "${HELDOUT_EVAL}" \
        >> "${cell_log}" 2>&1
      heldout_exit=$?
      echo "[master] run_heldout_eval.py exit=${heldout_exit}"
      # Extract τ / MAE for the STATUS.md row.
      if [[ -f "${heldout_dir}/heldout_summary.json" ]]; then
        tau=$("${PYBIN}" -c "import json; \
d=json.load(open('${heldout_dir}/heldout_summary.json'))['aggregate']; \
print(d.get('kendall_tau','N/A'))")
        mae=$("${PYBIN}" -c "import json; \
d=json.load(open('${heldout_dir}/heldout_summary.json'))['aggregate']; \
print(d.get('mae_kcal','N/A'))")
      fi
    fi
  fi

  local t1
  t1=$(date +%s)
  local wall=$((t1 - t0))

  echo "[master] CELL DONE  host=${host}  ablation=${ablation}  wall=${wall}s  rl_exit=${rl_exit}  heldout_exit=${heldout_exit}"
  echo "| $(($(grep -c '^|' "${STATUS_MD}") - 1)) | ${host}-${ablation} | rl=${rl_exit} ho=${heldout_exit} | ${best_code_dir} | ${tau} | ${mae} | ${wall} | $(date -Iseconds) |" >> "${STATUS_MD}"
}

# ---- Main loop -------------------------------------------------------
for cell in "${CELLS[@]}"; do
  run_one_cell "${cell}"
done

# ---- Final mirror + cleanup -----------------------------------------
echo "[master] all 12 cells dispatched; doing final rsync"
rsync -aL --delete \
  --exclude='master.log' --exclude='STATUS.md' \
  --exclude='rsync.log' --exclude='rsync.pid' --exclude='ALL_DONE' \
  --exclude='*.dcd' --exclude='*.trj' --exclude='*.nc' \
  --exclude='*.xtc' --exclude='*.trr' --exclude='*.tng' \
  --exclude='*.rst' --exclude='*.rst7' --exclude='*.crd' \
  --exclude='*.mdcrd' --exclude='*.netcdf' \
  "${TMP}/" "${SCRATCH}/" 2>>"${SCRATCH}/rsync.log" || true
echo "rsync final at $(date -Iseconds)" >> "${SCRATCH}/rsync.log"

cleanup_rsync
trap - EXIT INT TERM

date > "${SCRATCH}/ALL_DONE"
echo "[master] wrote ${SCRATCH}/ALL_DONE"
echo "[master] finished at $(date -Iseconds)"
