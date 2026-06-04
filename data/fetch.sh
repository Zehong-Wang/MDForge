#!/usr/bin/env bash
# Fetch and assemble the SAMPL host-guest data MDForge needs. Idempotent:
# safe to re-run. Covers all three paper hosts (CB[7], OAH, CBClip) plus
# the rest of the 8-atom / 68-pair unified benchmark.
#
#   CB[7], OAH, OAMe, OA, TEMOA, CB8  ->  cloned from Taproom
#   CBClip                            ->  downloaded from the SAMPL5 repo
#                                         by build_sampl_benchmark.py
#
# Override the Python used for the build step with PYTHON=/path/to/python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"
PYTHON="${PYTHON:-python3}"

# ---- 1. Clone the source repositories -------------------------------
if [ ! -d "SAMPL4" ]; then
    echo "Cloning samplchallenges/SAMPL4 (reference inputs)..."
    git clone --depth 1 https://github.com/samplchallenges/SAMPL4.git
else
    echo "SAMPL4/ already present; skipping."
fi

if [ ! -d "host-guest-benchmarks" ]; then
    echo "Cloning slochower/host-guest-benchmarks (Taproom)..."
    git clone --depth 1 https://github.com/slochower/host-guest-benchmarks.git
else
    echo "host-guest-benchmarks/ already present; skipping."
fi

# ---- 2. Build the unified benchmark ---------------------------------
# Assembles data/sampl_benchmark/{manifest.json, <atom>/...} from Taproom
# and downloads the SAMPL5 CBClip atom from GitHub (needs network).
echo "Building unified SAMPL benchmark (downloads CBClip)..."
"${PYTHON}" "${REPO_ROOT}/scripts/build_sampl_benchmark.py"

echo
echo "Done. MDForge reads from:"
echo "  - data/sampl_benchmark/manifest.json            (unified, all 8 atoms)"
echo "  - data/sampl_benchmark/{sampl4_cb7,sampl4_oah,sampl5_cbclip}/  (paper hosts)"
echo "  - data/host-guest-benchmarks/taproom/systems/   (Taproom source)"
echo "  - data/SAMPL4/host-guest/sampl4_hostguest/CB7/  (reference)"
