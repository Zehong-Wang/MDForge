"""Task specifications: what the Writer agent is asked to do.

A `Task` bundles together (a) the host-guest target, (b) the local
paths of the structure files the agent may use, (c) the experimental
ground-truth ΔG kept aside for evaluation, and (d) any extra context
(temperature, pH, charge state) the agent should know.

Tasks are loaded from `data/sampl_benchmark/manifest.json`, which lists
the 8 (SAMPL_N, host) atoms that have complete bound-pose coverage = 68
host-guest pairs. Each task carries a pre-docked `complex.pdb` so the
pipeline does NOT do docking — it consumes the bound complex as input.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.models import HostGuestSpec

# Project root = the parent of `src/`. Resolved from this file's location
# so the package keeps working when installed in editable mode.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPL_BENCHMARK_DIR = PROJECT_ROOT / "data" / "sampl_benchmark"
SAMPL_MANIFEST_PATH = SAMPL_BENCHMARK_DIR / "manifest.json"

# Legacy taproom path kept for backward compat with existing call sites.
TAPROOM_CB7 = PROJECT_ROOT / "data" / "host-guest-benchmarks" / "taproom" / "systems" / "cb7"


class ExperimentalReference(BaseModel):
    """Ground-truth measurement, kept out of the agent's prompt."""

    delta_g_kcal_per_mol: float
    delta_g_uncertainty_kcal_per_mol: float
    technique: str
    pH: float | None = None
    temperature_kelvin: float | None = None
    doi: str | None = None
    raw_value: str | None = None  # e.g. "-58.99 kJ/mol" verbatim from source


class Task(BaseModel):
    """A single MD pipeline-design task.

    The pipeline is K=4 stages: 01_prep → 02_equilibrate → 03_production
    → 04_analysis. Docking is OUT of scope — the bound complex starting
    structure (`bound_complex_pdb`) is mandatory input, taken from the
    SAMPL benchmark manifest. The Writer's stage 01 reads it directly
    rather than producing it.
    """

    task_id: str
    target: HostGuestSpec
    description: str
    host_structure_files: dict[str, Path]
    """Filename -> absolute path. The agent may copy these into its code_files."""
    guest_structure_files: dict[str, Path]
    bound_complex_pdb: Path
    """REQUIRED: the docked / experimentally-determined bound complex pose.
    PRISM does not do docking; pose is benchmark input."""
    extra_context: dict = Field(default_factory=dict)
    """Free-form: pH, charge state, temperature, hints, etc."""
    experimental_reference: ExperimentalReference
    """Held aside for evaluation. NOT to be shown to the Writer."""

    def host_files_as_text(self) -> dict[str, str]:
        return {p.name: p.read_text() for p in self.host_structure_files.values()}

    def guest_files_as_text(self) -> dict[str, str]:
        return {p.name: p.read_text() for p in self.guest_structure_files.values()}


def _kj_to_kcal(kj_per_mol: float) -> float:
    return kj_per_mol / 4.184


# ---------------------------------------------------------------------------
# SAMPL4 CB7 task definitions
# ---------------------------------------------------------------------------


def sampl4_cb7_adamantyl_amine() -> Task:
    """SAMPL4 CB7 + 1-adamantylazanium (taproom code: `adz`).

    Chosen as the primary demonstration target because:
      - Rigid bicyclic guest (sampling converges fast).
      - Single charge state at experimental pH (no protonation ambiguity).
      - Strong binder (~-14 kcal/mol) so the signal is large vs. uncertainty.
      - Curated structures + restraints already exist in taproom.
    """
    cb7_root = TAPROOM_CB7
    guest_dir = cb7_root / "adz"
    if not guest_dir.exists():
        raise FileNotFoundError(
            f"taproom CB7/adz directory missing at {guest_dir}. "
            "Did you clone host-guest-benchmarks into data/ (see data/fetch.sh)?"
        )

    host_files = {
        p.name: p for p in [
            cb7_root / "cb7.mol2",
            cb7_root / "cb7.pdb",
            cb7_root / "cb7.sdf",
        ]
        if p.exists()
    }
    guest_files = {
        p.name: p for p in [
            guest_dir / "adz.mol2",
            guest_dir / "adz.sdf",
        ]
        if p.exists()
    }
    complex_pdb = guest_dir / "cb7-adz-p.pdb"
    if not complex_pdb.exists():
        complex_pdb = None

    return Task(
        task_id="sampl4-cb7-adz",
        target=HostGuestSpec(
            host_id="cb7",
            guest_id="adz",
            host_smiles=None,
            guest_smiles="C1C2CC3(CC1CC(C2)C3)[NH3+]",  # 1-adamantylazanium
            metadata={
                "host_long_name": "cucurbit[7]uril",
                "guest_long_name": "1-adamantylazanium (1-adamantanaminium)",
                "sampl4_compound_id": "13",
                "taproom_code": "adz",
            },
        ),
        description=(
            "Design and run an MD binding free energy calculation for "
            "1-adamantylazanium binding to cucurbit[7]uril (CB7) in aqueous "
            "solution at 298.15 K, pH ≈ 7. Output an absolute ΔG_bind "
            "(kcal/mol) and an uncertainty estimate. The host is neutral; the "
            "guest carries a +1 charge. Structure files are provided but the "
            "agent must decide force field, water model, sampling protocol, "
            "restraints, and analysis methodology."
        ),
        host_structure_files=host_files,
        guest_structure_files=guest_files,
        bound_complex_pdb=complex_pdb,
        extra_context={
            "temperature_kelvin": 298.15,
            "pH": 7.0,
            "host_net_charge": 0,
            "guest_net_charge": 1,
            "stoichiometry": "1:1",
        },
        experimental_reference=ExperimentalReference(
            delta_g_kcal_per_mol=_kj_to_kcal(-58.99),
            delta_g_uncertainty_kcal_per_mol=_kj_to_kcal(0.42),  # taproom value
            technique="ITC",
            pH=7.0,
            temperature_kelvin=298.15,
            doi="10.1007/s10822-014-9735-1",
            raw_value="-58.99 kJ/mol",
        ),
    )


SAMPL4_CB7_TASKS: dict[str, callable] = {
    "sampl4-cb7-adz": sampl4_cb7_adamantyl_amine,
}


# ---------------------------------------------------------------------------
# Manifest-driven task loader (post-docking-removal)
# ---------------------------------------------------------------------------

def _parse_dG_kj_mol(raw: str) -> float:
    """Parse strings like '-58.99 kJ/mol' (Taproom) or '-29.50 kJ/mol' to float."""
    return float(str(raw).split()[0])


def load_manifest(manifest_path: Path = SAMPL_MANIFEST_PATH) -> dict:
    """Load the SAMPL benchmark manifest produced by
    `scripts/build_sampl_benchmark.py`. Returns the raw dict."""
    return json.loads(Path(manifest_path).read_text())


def task_from_manifest(
    atom_id: str,
    guest_idx: int,
    manifest_path: Path = SAMPL_MANIFEST_PATH,
) -> Task:
    """Build a Task for a single (SAMPL atom, guest) pair from the manifest.

    `atom_id` is e.g. 'sampl4_cb7' or 'sampl5_cbclip'. `guest_idx` indexes
    into `manifest['atoms'][atom_id]['guests']`. The bound complex pdb is
    pulled from the manifest's `complex_pdb` field for that guest, marked
    REQUIRED — pipelines will not do docking.
    """
    import yaml as _yaml

    manifest = load_manifest(manifest_path)
    if atom_id not in manifest["atoms"]:
        raise KeyError(
            f"Unknown atom '{atom_id}'. Available: {sorted(manifest['atoms'])}"
        )
    atom = manifest["atoms"][atom_id]
    if guest_idx < 0 or guest_idx >= len(atom["guests"]):
        raise IndexError(
            f"guest_idx {guest_idx} out of range for {atom_id} "
            f"({len(atom['guests'])} guests)"
        )
    g = atom["guests"][guest_idx]

    def _abs(rel: str) -> Path:
        return PROJECT_ROOT / rel

    host_files = {
        Path(rel).name: _abs(rel)
        for rel in atom["host_files"].values()
    }
    files = g["files"]
    guest_files = {}
    for key in ("guest_mol2", "guest_sdf", "guest_pdb"):
        if key in files:
            p = _abs(files[key])
            guest_files[p.name] = p
    if "complex_pdb" not in files:
        raise ValueError(
            f"{atom_id}/{g.get('sampl_guest_id')} has no complex_pdb in manifest"
        )
    bound_pdb = _abs(files["complex_pdb"])

    # Read guest charge from Taproom guest.yaml if available; otherwise
    # leave as None and let the pipeline auto-detect from mol2 charges.
    guest_charge: int | None = None
    host_charge: int | None = None
    n_sym: int | None = None
    if "guest_yaml" in files:
        try:
            gy = _yaml.safe_load(_abs(files["guest_yaml"]).read_text())
            # Taproom stores e.g. "net_charge: +1e" or "+1" or 1
            nc = gy.get("net_charge")
            if nc is not None:
                s = str(nc).replace("e", "").replace("+", "").strip()
                guest_charge = int(s) if s.lstrip("-").isdigit() else None
        except Exception:
            pass
    # Host charges per host class (these are well-known constants)
    host_charge_table = {
        "cb7": 0, "cb8": 0, "oah": -8, "oam": -8,
        "acd": 0, "bcd": 0, "cbclip": 0,
    }
    host_charge = host_charge_table.get(atom["host_id"], 0)
    # Symmetry correction count: cucurbiturils have n_sym=2 (two equivalent
    # portals); octa-acid + cyclodextrin systems are typically 1.
    n_sym_table = {"cb7": 2, "cb8": 2}
    n_sym = n_sym_table.get(atom["host_id"], 1)

    # Experimental ΔG: read measurement.yaml if present (Taproom atoms).
    # Fall back to None for atoms where it isn't (e.g., user can supply later).
    exp_ref: ExperimentalReference | None = None
    pH_val: float | None = None
    T_K: float | None = None
    if "measurement_yaml" in files:
        meas = _yaml.safe_load(_abs(files["measurement_yaml"]).read_text())
        m = meas.get("measurement", {})
        s = meas.get("state", {})
        prov = meas.get("provenance", {})
        T_str = str(s.get("temperature", "298 K")).split()[0]
        T_K = float(T_str)
        pH_val = (
            float(s["pH"]) if s.get("pH") is not None else None
        )
        exp_ref = ExperimentalReference(
            delta_g_kcal_per_mol=_kj_to_kcal(_parse_dG_kj_mol(m["delta_G"])),
            delta_g_uncertainty_kcal_per_mol=_kj_to_kcal(
                _parse_dG_kj_mol(m.get("delta_G_uncertainty", "0.0 kJ/mol"))
            ),
            technique=str(m.get("technique", "?")),
            pH=pH_val,
            temperature_kelvin=T_K,
            doi=str(prov.get("doi", "")) or None,
            raw_value=str(m["delta_G"]),
        )
    if exp_ref is None:
        # SAMPL5 CBClip path doesn't ship per-guest measurement yamls in the
        # SimulationFiles drop. Fill a placeholder; the orchestrator's
        # held-out comparison will skip if uncertainty=NaN.
        exp_ref = ExperimentalReference(
            delta_g_kcal_per_mol=float("nan"),
            delta_g_uncertainty_kcal_per_mol=float("nan"),
            technique="not_yet_loaded",
        )

    sampl_gid = g.get("sampl_guest_id", f"idx{guest_idx}")
    short = g.get("taproom_guest_id") or sampl_gid
    task_id = f"{atom_id}-{sampl_gid}".lower()

    description = (
        f"Design and run a binding free energy calculation for "
        f"{short} (SAMPL{atom['sampl_n']} {atom['host_long_name']} guest "
        f"{sampl_gid}) at the published experimental conditions. "
        f"Output an absolute ΔG_bind in kcal/mol with an uncertainty estimate. "
        f"The bound complex starting structure is provided as `complex.pdb` "
        f"in the working directory; you do NOT have to dock — the pipeline "
        f"begins at parameterization."
    )

    return Task(
        task_id=task_id,
        target=HostGuestSpec(
            host_id=atom["host_id"],
            guest_id=short,
            host_smiles=None,
            guest_smiles=None,
            metadata={
                "host_long_name": atom["host_long_name"],
                "sampl_n": atom["sampl_n"],
                "sampl_guest_id": sampl_gid,
                "atom_id": atom_id,
                "manifest_source": atom.get("source"),
            },
        ),
        description=description,
        host_structure_files=host_files,
        guest_structure_files=guest_files,
        bound_complex_pdb=bound_pdb,
        extra_context={
            "temperature_kelvin": T_K,
            "pH": pH_val,
            "atom_id": atom_id,
            "host_net_charge": host_charge,
            "guest_net_charge": guest_charge,
            "n_sym": n_sym,
        },
        experimental_reference=exp_ref,
    )


def get_task(task_id: str) -> Task:
    if task_id not in SAMPL4_CB7_TASKS:
        raise KeyError(
            f"Unknown task_id '{task_id}'. Available: {sorted(SAMPL4_CB7_TASKS)}"
        )
    return SAMPL4_CB7_TASKS[task_id]()
