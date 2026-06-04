#!/usr/bin/env bash
# Create the `pipeline/` conda env that MDForge uses to EXECUTE agent-written
# MD code (AmberTools + OpenMM + OpenFF + free-energy analysis). This is the
# simulation backend, separate from the env you run the agent in.
#
# Usage:
#   bash scripts/setup_pipeline.sh                     # create ./pipeline
#   PREFIX=/path/to/env bash scripts/setup_pipeline.sh # custom location
#
# If you build it somewhere other than <repo_root>/pipeline, point MDForge at
# it with:  export PRISM_PIPELINE=/path/to/env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-${REPO_ROOT}/pipeline}"
ENV_FILE="${REPO_ROOT}/environment.yml"

# Prefer mamba (much faster solver) over conda when available.
if command -v mamba >/dev/null 2>&1; then
    SOLVER=mamba
elif command -v conda >/dev/null 2>&1; then
    SOLVER=conda
else
    echo "ERROR: neither mamba nor conda is on PATH." >&2
    echo "Install Miniforge first: https://github.com/conda-forge/miniforge" >&2
    exit 1
fi

if [ -x "${PREFIX}/bin/python" ]; then
    echo "Env already exists at ${PREFIX}; skipping creation (delete it to rebuild)."
else
    echo "Creating MD env at ${PREFIX} with ${SOLVER} (this takes a while)..."
    "${SOLVER}" env create -p "${PREFIX}" -f "${ENV_FILE}"
fi

echo "Verifying Python packages..."
"${PREFIX}/bin/python" - <<'PY'
import importlib
for m in ["openmm", "openff.toolkit", "openmmtools", "parmed",
          "pymbar", "alchemlyb", "mdtraj", "rdkit"]:
    importlib.import_module(m)
    print(f"  ok: {m}")
PY

echo "Verifying AmberTools binaries..."
for b in antechamber tleap parmchk2 sqm sander; do
    if [ -x "${PREFIX}/bin/${b}" ]; then
        echo "  ok: ${b}"
    else
        echo "  MISSING: ${b}" >&2
        exit 1
    fi
done

echo
echo "Done. MDForge uses this env by default (it lives at <repo_root>/pipeline)."
if [ "${PREFIX}" != "${REPO_ROOT}/pipeline" ]; then
    echo "Since it is not at the default location, set:"
    echo "  export PRISM_PIPELINE=${PREFIX}"
fi
