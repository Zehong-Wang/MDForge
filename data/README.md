# Data

External datasets used by MDForge. None of it is vendored into the repository
— run `bash data/fetch.sh` to acquire and assemble it (this directory is
gitignored except for this README and `fetch.sh`).

## Layout (after running `bash fetch.sh`)

```
data/
  SAMPL4/                       Original SAMPL4 challenge inputs (reference)
  host-guest-benchmarks/        Taproom systems (CB7, OAH, OA, TEMOA, CB8, ...)
  sampl_benchmark/              Unified benchmark assembled by the build script
    manifest.json               Top-level index of all 8 atoms / 68 pairs
    sampl4_cb7/ sampl4_oah/ sampl5_cbclip/ ...   one dir per atom
```

## What MDForge uses

`src/tasks.py` reads `sampl_benchmark/manifest.json` and, per guest, the
structures and held-aside experimental ΔG (`measurement.yaml`: technique, pH,
DOI). The three hosts evaluated in the paper are **CB[7]** (SAMPL4), **OAH**
(SAMPL4), and **CBClip** (SAMPL5). CB[7]/OAH come from Taproom; CBClip is
downloaded from the official SAMPL5 repo by `scripts/build_sampl_benchmark.py`.

## Sources

  - [samplchallenges/SAMPL4](https://github.com/samplchallenges/SAMPL4)
  - [samplchallenges/SAMPL5](https://github.com/samplchallenges/SAMPL5) (CBClip)
  - [slochower/host-guest-benchmarks](https://github.com/slochower/host-guest-benchmarks) (Taproom)
