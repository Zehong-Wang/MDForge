#!/usr/bin/env python
"""Build a unified SAMPL host-guest benchmark dataset under
data/sampl_benchmark/.

Includes the 8 (SAMPL, host) atoms with complete bound-pose coverage
(= 68 host-guest pairs):

  1. SAMPL4-CB7   (14, Taproom)
  2. SAMPL4-OAH    (9, Taproom)
  3. SAMPL5-CBClip (10, official SAMPL5 GitHub — needs download)
  4. SAMPL5-OAH    (6, Taproom)
  5. SAMPL5-OAMe   (6, Taproom)
  6. SAMPL6-OA     (8, Taproom)
  7. SAMPL6-TEMOA  (8, Taproom)
  8. SAMPL8-CB8    (7, Taproom)

For Taproom-sourced atoms, the script SYMLINKS into the local
host-guest-benchmarks/taproom/systems/<host>/<guest>/ dirs (no copy).
For SAMPL5 CBClip, the script downloads each guest's prmtop+rst7+gro+top
+ molecule mol2/pdb/sd from the official SAMPL5 GitHub repo, then
extracts host.pdb, guest.pdb, complex.pdb via ParmEd.

Final layout:

  data/sampl_benchmark/
    manifest.json                 (top-level index of all 68 pairs)
    sampl4_cb7/                   (one dir per atom)
      cb7.mol2 | cb7.pdb | cb7.sdf            <- host
      C13-adz/                                  <- one dir per guest
        adz.mol2 | adz.sdf | cb7-adz-p.pdb    <- guest + bound complex
        guest.yaml | measurement.yaml
      ... 14 dirs ...
    sampl4_oah/...
    sampl5_cbclip/
      CBClip.mol2 | CBClip.pdb
      G1/
        CBC-G1.mol2 | CBC-G1.pdb | CBC-G1.sd
        CBC-G1.prmtop | CBC-G1.rst7 | CBC-G1.gro | CBC-G1.top
        complex.pdb (extracted)
      ... 10 dirs ...

manifest.json schema:
  {
    "atoms": {
      "sampl4_cb7": {
        "sampl_n": 4, "host_id": "cb7", "host_long_name": "Cucurbit[7]uril",
        "host_files": {"mol2": "...", "pdb": "..."},
        "guests": [
          {"guest_id": "C13", "alias": "adz", "smiles": "...",
           "files": {"mol2": "...", "sdf": "...", "complex_pdb": "..."},
           "experimental": {"delta_g_kcal": -14.10, "uncertainty_kcal": 0.10,
                            "technique": "ITC", "T_K": 298.15, "doi": "..."},
           "provenance": "taproom"},
          ...
        ]
      },
      ...
    }
  }
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
TAPROOM = ROOT / "data" / "host-guest-benchmarks" / "taproom" / "systems"
OUT = ROOT / "data" / "sampl_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Atom definitions
# ---------------------------------------------------------------------------

# (Taproom guest_id -> SAMPL guest label) per atom, derived from
# guest.yaml `data_set` provenance + cross-reference.
TAPROOM_ATOMS = {
    "sampl4_cb7": {
        "sampl_n": 4,
        "host_id": "cb7",
        "host_long_name": "Cucurbit[7]uril",
        "taproom_host_dir": "cb7",
        "guests": [
            ("phm", "C1"),  ("dpm", "C2"),  ("hxm", "C3"),  ("axm", "C4"),
            ("cha", "C5"),  ("chm", "C6"),  ("c6m", "C7"),  ("c7m", "C8"),
            ("c8m", "C9"),  ("prm", "C10"), ("bhm", "C11"), ("hpm", "C12"),
            ("adz", "C13"), ("haz", "C14"),
        ],
    },
    "sampl4_oah": {
        "sampl_n": 4,
        "host_id": "oah",
        "host_long_name": "Octa-acid (OAH)",
        "taproom_host_dir": "oah",
        "guests": [
            ("ben", "O1"), ("mbn", "O2"), ("ebn", "O3"), ("c4b", "O4"),
            ("c3b", "O5"), ("chx", "O6"), ("mhc", "O7"), ("c5c", "O8"),
            ("c7c", "O9"),
        ],
    },
    "sampl5_oah": {
        "sampl_n": 5,
        "host_id": "oah",
        "host_long_name": "Octa-acid (OAH)",
        "taproom_host_dir": "oah",
        "guests": [
            ("hxy", "G1"), ("cbn", "G2"), ("hxa", "G3"),
            ("bra", "G4"),  # taproom yaml mis-labels this as O4 but it is SAMPL5 G4
            ("trz", "G5"), ("nbn", "G6"),
        ],
    },
    "sampl5_oam": {
        "sampl_n": 5,
        "host_id": "oam",
        "host_long_name": "Tetra-endo-methyl octa-acid (OAMe / TEMOA)",
        "taproom_host_dir": "oam",
        "guests": [
            ("hxy", "G1"), ("cbn", "G2"), ("hxa", "G3"),
            ("bra", "G4"),
            ("trz", "G5"), ("nbn", "G6"),
        ],
    },
    "sampl6_oa": {
        "sampl_n": 6,
        "host_id": "oah",
        "host_long_name": "Octa-acid (OA)",
        "taproom_host_dir": "oah",
        "guests": [
            ("cpa", "G0"), ("t2h", "G1"), ("prl", "G2"), ("hx5", "G3"),
            ("crl", "G4"), ("m4p", "G5"), ("mpe", "G6"), ("dpe", "G7"),
        ],
    },
    "sampl6_temoa": {
        "sampl_n": 6,
        "host_id": "oam",
        "host_long_name": "Tetra-endo-methyl octa-acid (TEMOA)",
        "taproom_host_dir": "oam",
        "guests": [
            ("cpa", "G0"), ("t2h", "G1"), ("prl", "G2"), ("hx5", "G3"),
            ("crl", "G4"), ("m4p", "G5"), ("mpe", "G6"), ("dpe", "G7"),
        ],
    },
    "sampl8_cb8": {
        "sampl_n": 8,
        "host_id": "cb8",
        "host_long_name": "Cucurbit[8]uril (drug-of-abuse panel)",
        "taproom_host_dir": "cb8",
        "guests": [
            ("mth", "G1"), ("fen", "G2"), ("mor", "G3"), ("hmo", "G4"),
            ("ket", "G5"), ("pcp", "G6"), ("con", "G7"),
        ],
    },
}

# Official SAMPL5 CBClip atom — fetched from samplchallenges/SAMPL5
SAMPL5_CBCLIP = {
    "sampl_n": 5,
    "host_id": "cbclip",
    "host_long_name": "CBClip (Isaacs glycoluril molecular clip)",
    "n_guests": 10,
    "github_root": (
        "https://raw.githubusercontent.com/samplchallenges/SAMPL5/main"
        "/host-guest/269_data_909449"
    ),
}

# Experimental ΔG (kcal/mol) for CBClip + 10 guests, taken from
# Yin et al. 2017 JCAMD ("Overview of the SAMPL5 host-guest challenge:
# Are we doing better?") Table 1. Conditions: 20 mM sodium phosphate
# buffer, pH 7.4, T = 298 K. Uncertainties are SEM from experimental
# replicates of Ka. Techniques vary by guest (NMR / UV-Vis / fluorescence).
SAMPL5_CBCLIP_EXP = {
    "G1":  {"dG_kcal": -5.83,  "sigma_kcal": 0.03, "technique": "NMR"},
    "G2":  {"dG_kcal": -2.51,  "sigma_kcal": 0.07, "technique": "NMR"},
    "G3":  {"dG_kcal": -4.02,  "sigma_kcal": 0.03, "technique": "NMR"},
    "G4":  {"dG_kcal": -7.24,  "sigma_kcal": 0.03, "technique": "UV-Vis"},
    "G5":  {"dG_kcal": -8.53,  "sigma_kcal": 0.07, "technique": "UV-Vis"},
    "G6":  {"dG_kcal": -8.64,  "sigma_kcal": 0.05, "technique": "UV-Vis"},
    "G7":  {"dG_kcal": -5.17,  "sigma_kcal": 0.02, "technique": "UV-Vis"},
    "G8":  {"dG_kcal": -6.17,  "sigma_kcal": 0.04, "technique": "UV-Vis"},
    "G9":  {"dG_kcal": -7.39,  "sigma_kcal": 0.02, "technique": "UV-Vis"},
    "G10": {"dG_kcal": -10.35, "sigma_kcal": 0.03, "technique": "fluorescence"},
}
SAMPL5_CBCLIP_EXP_DOI = "10.1007/s10822-016-9974-4"  # Yin et al. 2017 JCAMD
SAMPL5_CBCLIP_EXP_T_K = 298.0
SAMPL5_CBCLIP_EXP_PH = 7.4


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _copy(target: Path, dst: Path) -> None:
    """Copy `target` -> `dst`, replacing any existing file."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    shutil.copy2(target, dst)


def build_taproom_atom(atom_id: str, spec: dict) -> dict:
    """Symlink Taproom files into the unified atom dir, return atom record
    for the manifest."""
    atom_dir = OUT / atom_id
    atom_dir.mkdir(parents=True, exist_ok=True)

    host_dir = TAPROOM / spec["taproom_host_dir"]
    if not host_dir.is_dir():
        raise FileNotFoundError(f"Taproom host dir missing: {host_dir}")

    # Symlink host files (mol2 / pdb / sdf / svg)
    host_files = {}
    for ext in ("mol2", "pdb", "sdf"):
        src = host_dir / f"{spec['taproom_host_dir']}.{ext}"
        if src.is_file():
            dst = atom_dir / src.name
            _copy(src, dst)
            host_files[ext] = str(dst.relative_to(ROOT))

    record = {
        "atom_id": atom_id,
        "sampl_n": spec["sampl_n"],
        "host_id": spec["host_id"],
        "host_long_name": spec["host_long_name"],
        "source": "taproom",
        "host_files": host_files,
        "guests": [],
    }

    for tap_id, sampl_gid in spec["guests"]:
        guest_src = host_dir / tap_id
        if not guest_src.is_dir():
            print(f"  WARN: missing taproom guest dir {guest_src}", file=sys.stderr)
            continue
        guest_dst = atom_dir / f"{sampl_gid}-{tap_id}"
        guest_dst.mkdir(exist_ok=True)
        files: dict[str, str] = {}
        # Files we expect: <tap_id>.mol2, <tap_id>.sdf,
        # <host>-<tap_id>-p.pdb, guest.yaml, measurement.yaml
        for child in guest_src.iterdir():
            if child.is_file():
                dst = guest_dst / child.name
                _copy(child, dst)
                # Categorise file kind for manifest
                n = child.name.lower()
                if n.endswith(".mol2"):
                    files["guest_mol2"] = str(dst.relative_to(ROOT))
                elif n.endswith(".sdf"):
                    files["guest_sdf"] = str(dst.relative_to(ROOT))
                elif n.endswith("-p.pdb"):
                    files["complex_pdb"] = str(dst.relative_to(ROOT))
                elif n == "measurement.yaml":
                    files["measurement_yaml"] = str(dst.relative_to(ROOT))
                elif n == "guest.yaml":
                    files["guest_yaml"] = str(dst.relative_to(ROOT))
        record["guests"].append({
            "sampl_guest_id": sampl_gid,
            "taproom_guest_id": tap_id,
            "files": files,
        })
    return record


def build_sampl5_cbclip_atom() -> dict:
    """Download SAMPL5 CBClip files (host + 10 guests) directly from GitHub.

    Each guest has:
      - mol2 + pdb + sd in MoleculeFilesNew/CBClipAndGuests/CBC-G{N}.{mol2,pdb,sd}
        (G1-G7 have all three; G8 has pdb+sd; G9, G10 only sd)
      - prmtop + rst7 + gro + top + cms + lmp + input in
        SimulationFilesNew/CBC-G{N}/
    Plus host files CBClip.{mol2,pdb,sd} in MoleculeFilesNew/CBClipAndGuests/.

    The bound complex coords live in prmtop+rst7. We can extract a complex.pdb
    via parmed: parmed.load_file(prmtop, xyz=rst7).save("complex.pdb").
    """
    spec = SAMPL5_CBCLIP
    atom_id = "sampl5_cbclip"
    atom_dir = OUT / atom_id
    atom_dir.mkdir(parents=True, exist_ok=True)

    base = spec["github_root"]
    host_root = f"{base}/MoleculeFilesNew/CBClipAndGuests"
    sim_root = f"{base}/SimulationFilesNew"

    def _download(url: str, dst: Path) -> bool:
        if dst.is_file() and dst.stat().st_size > 0:
            return True
        try:
            print(f"  GET {url} -> {dst.name}", file=sys.stderr)
            urllib.request.urlretrieve(url, dst)
            return dst.stat().st_size > 0
        except Exception as e:
            print(f"  FAIL ({e})", file=sys.stderr)
            if dst.exists():
                dst.unlink()
            return False

    # Host files
    host_files = {}
    for ext in ("mol2", "pdb", "sd"):
        url = f"{host_root}/CBClip.{ext}"
        dst = atom_dir / f"CBClip.{ext}"
        if _download(url, dst):
            kind = {"mol2": "mol2", "pdb": "pdb", "sd": "sdf"}[ext]
            host_files[kind] = str(dst.relative_to(ROOT))

    record = {
        "atom_id": atom_id,
        "sampl_n": 5,
        "host_id": "cbclip",
        "host_long_name": spec["host_long_name"],
        "source": "official_sampl5",
        "host_files": host_files,
        "guests": [],
    }

    for n in range(1, spec["n_guests"] + 1):
        gid = f"G{n}"
        guest_dir = atom_dir / gid
        guest_dir.mkdir(exist_ok=True)
        files: dict[str, str] = {}

        # Guest molecule files (variable availability per guest)
        for ext in ("mol2", "pdb", "sd"):
            url = f"{host_root}/CBC-{gid}.{ext}"
            dst = guest_dir / f"CBC-{gid}.{ext}"
            if _download(url, dst):
                kind = {"mol2": "guest_mol2", "pdb": "guest_pdb", "sd": "guest_sdf"}[ext]
                files[kind] = str(dst.relative_to(ROOT))

        # Simulation files (bound complex topology + coords) — all 10 guests have these
        for ext in ("prmtop", "rst7", "gro", "top"):
            url = f"{sim_root}/CBC-{gid}/CBC-{gid}.{ext}"
            dst = guest_dir / f"CBC-{gid}.{ext}"
            if _download(url, dst):
                files[f"sim_{ext}"] = str(dst.relative_to(ROOT))

        # Extract complex.pdb from prmtop+rst7 if both present
        prmtop_path = guest_dir / f"CBC-{gid}.prmtop"
        rst7_path = guest_dir / f"CBC-{gid}.rst7"
        if prmtop_path.is_file() and rst7_path.is_file():
            complex_pdb = guest_dir / "complex.pdb"
            try:
                import parmed
                p = parmed.load_file(str(prmtop_path), xyz=str(rst7_path))
                p.save(str(complex_pdb), overwrite=True)
                files["complex_pdb"] = str(complex_pdb.relative_to(ROOT))
            except Exception as e:
                print(f"  WARN: parmed extract failed for {gid}: {e}", file=sys.stderr)

        # Write a Taproom-shaped measurement.yaml so task_from_manifest()
        # picks up the experimental ΔG via the same reader path used for
        # Taproom-sourced atoms.
        if gid in SAMPL5_CBCLIP_EXP:
            exp = SAMPL5_CBCLIP_EXP[gid]
            kJ_per_mol = exp["dG_kcal"] * 4.184
            sigma_kJ = exp["sigma_kcal"] * 4.184
            meas_yaml = guest_dir / "measurement.yaml"
            meas_yaml.write_text(
                "# Auto-generated from Yin et al. 2017 JCAMD Table 1\n"
                "# (SAMPL5 host-guest overview, CBClip + 10 guests).\n"
                f"state:\n"
                f"  temperature: {SAMPL5_CBCLIP_EXP_T_K} K\n"
                f"  pressure: 1 atm\n"
                f"  pH: {SAMPL5_CBCLIP_EXP_PH}\n"
                f"\n"
                f"substance:\n"
                f"  name: \"CBC-{gid}\"\n"
                f"\n"
                f"measurement:\n"
                f"  technique: {exp['technique']}\n"
                f"  delta_G: {kJ_per_mol:.4f} kJ/mol\n"
                f"  delta_G_uncertainty: {sigma_kJ:.4f} kJ/mol\n"
                f"  stoichiometry: \"1:1\"\n"
                f"\n"
                f"provenance:\n"
                f"  comment: \"Yin et al. 2017 JCAMD Table 1; conditions: 20 mM sodium phosphate buffer pH 7.4, 298 K\"\n"
                f"  doi: {SAMPL5_CBCLIP_EXP_DOI}\n"
                f"\n"
                f"unusual: False\n"
            )
            files["measurement_yaml"] = str(meas_yaml.relative_to(ROOT))

        record["guests"].append({
            "sampl_guest_id": gid,
            "files": files,
        })
    return record


def main() -> int:
    manifest = {"atoms": {}}

    print("== Taproom-sourced atoms ==")
    for atom_id, spec in TAPROOM_ATOMS.items():
        print(f"  building {atom_id} ({len(spec['guests'])} guests)")
        manifest["atoms"][atom_id] = build_taproom_atom(atom_id, spec)

    print("\n== SAMPL5 CBClip (downloading from official GitHub) ==")
    manifest["atoms"]["sampl5_cbclip"] = build_sampl5_cbclip_atom()

    # Stats
    total_pairs = sum(len(a["guests"]) for a in manifest["atoms"].values())
    print(f"\n== Done: {len(manifest['atoms'])} atoms, {total_pairs} host-guest pairs ==")
    for aid, a in manifest["atoms"].items():
        print(f"  {aid:<16} : {len(a['guests']):>2} guests  ({a['source']})")

    out_path = OUT / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
