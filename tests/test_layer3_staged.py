"""Tests for the staged Layer 3 execution path (PRISM Phase 2 substrate).

The atomic-mode `run` is exercised by `test_layer3.py`; this file targets
the new `prepare_workdir` + `run_stage` interface that the orchestrator
uses for K-stage PRISM execution.

Invariants tested here:
  - Stage discovery is by filename pattern, in lex order
  - `prepare_workdir` materialises auxiliary files + code files once
  - `run_stage` reads `stage_NN_result.json` and packs it into Layer3Output
    with `stage_id` set
  - Cross-stage filesystem state persists between calls
  - A stage that crashes (Python exception) is reported with
    `exited_normally=False`, `parse_failed=True`, `return_code != 0`,
    and `status` left None (the writer self-reported nothing)
  - A stage that times out is reported with `timed_out=True`
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from src.models import HostGuestSpec, Pipeline
from src.verifiers.layer3 import (
    ExecutorConfig,
    Layer3Oracle,
    stage_index,
    stage_result_filename,
)


# ------------------------------------------------------------------------
# Stage filename helpers


def test_stage_index_extracts_numeric_prefix():
    assert stage_index("00_dock.py") == 0
    assert stage_index("05_analysis.py") == 5
    assert stage_index("12_extra.py") == 12


def test_stage_index_rejects_non_stage_filename():
    with pytest.raises(ValueError):
        stage_index("common.py")
    with pytest.raises(ValueError):
        stage_index("dock.py")


def test_stage_result_filename_convention():
    assert stage_result_filename("00_dock.py") == "stage_00_result.json"
    assert stage_result_filename("05_analysis.py") == "stage_05_result.json"


# ------------------------------------------------------------------------
# Pipeline stage discovery


def _trivial_pipeline(code_files: dict[str, str]) -> Pipeline:
    return Pipeline(
        target=HostGuestSpec(host_id="test", guest_id="g"),
        code_files=code_files,
        entry_point=next(iter(code_files)),
        rationale="test",
    )


def test_pipeline_stage_files_lex_ordered():
    p = _trivial_pipeline({
        "common.py": "pass",
        "00_dock.py": "pass",
        "01_setup.py": "pass",
        "05_analysis.py": "pass",
        "03_equilibrate.py": "pass",
        "02_solvate.py": "pass",
        "NotAStage.py": "pass",
        "04_production.py": "pass",
    })
    assert p.stage_files() == [
        "00_dock.py",
        "01_setup.py",
        "02_solvate.py",
        "03_equilibrate.py",
        "04_production.py",
        "05_analysis.py",
    ]


def test_pipeline_stage_files_excludes_helpers():
    p = _trivial_pipeline({
        "common.py": "pass",
        "00_dock.py": "pass",
        "helpers.py": "pass",
        "_private.py": "pass",
    })
    assert p.stage_files() == ["00_dock.py"]


# ------------------------------------------------------------------------
# Staged execution


def _result_writer_stage(stage_id: str, status: str = "success", extras: str = "") -> str:
    return f'''
import json, time, os
t0 = time.time()
{extras}
with open("stage_{stage_id.split("_")[0]}_result.json", "w") as f:
    json.dump({{
        "stage_id": "{stage_id}",
        "status": "{status}",
        "wall_time_seconds": time.time() - t0,
        "delta_g_kcal_per_mol": None,
        "diagnostics": {{"k": {int(stage_id.split("_")[0])}}},
        "writer_notes": "test stage {stage_id}",
    }}, f)
'''


def test_run_stage_happy_path(tmp_path: Path):
    pipeline = _trivial_pipeline({
        "00_dock.py": _result_writer_stage("00_dock"),
    })
    oracle = Layer3Oracle(ExecutorConfig(work_root=tmp_path, stage_timeout_seconds=30))
    work_dir = oracle.prepare_workdir(pipeline)
    out = oracle.run_stage(work_dir, pipeline, "00_dock.py")
    assert out.exited_normally is True
    assert out.parse_failed is False
    assert out.stage_id == "00_dock"
    assert out.status == "success"
    assert out.writer_notes == "test stage 00_dock"
    assert out.diagnostics == {"k": 0}


def test_run_stage_cross_stage_filesystem_state(tmp_path: Path):
    """Stage 01 should be able to read Stage 00's output file."""
    pipeline = _trivial_pipeline({
        "00_dock.py": _result_writer_stage(
            "00_dock", extras='open("checkpoint.txt", "w").write("from-00")'
        ),
        "01_setup.py": _result_writer_stage(
            "01_setup",
            extras=(
                'with open("checkpoint.txt") as f:\n'
                '    cp = f.read()\n'
                'assert cp == "from-00", cp\n'
            ),
        ),
    })
    oracle = Layer3Oracle(ExecutorConfig(work_root=tmp_path, stage_timeout_seconds=30))
    work_dir = oracle.prepare_workdir(pipeline)
    out0 = oracle.run_stage(work_dir, pipeline, "00_dock.py")
    out1 = oracle.run_stage(work_dir, pipeline, "01_setup.py")
    assert out0.exited_normally and out1.exited_normally
    # And the checkpoint file is still there
    assert (work_dir / "checkpoint.txt").read_text() == "from-00"


def test_run_stage_python_crash(tmp_path: Path):
    """Engineering crash (Python exception) is reported correctly."""
    pipeline = _trivial_pipeline({
        "00_dock.py": "raise RuntimeError('intentional crash')",
    })
    oracle = Layer3Oracle(ExecutorConfig(work_root=tmp_path, stage_timeout_seconds=30))
    work_dir = oracle.prepare_workdir(pipeline)
    out = oracle.run_stage(work_dir, pipeline, "00_dock.py")
    assert out.exited_normally is False
    assert out.return_code is not None and out.return_code != 0
    assert out.parse_failed is True   # never wrote stage_00_result.json
    # status is None because the writer code never ran the json.dump
    assert out.status is None
    # But stage_id falls back to the filename-derived value
    assert out.stage_id == "00_dock"
    assert "intentional crash" in (out.stderr_tail or "")


def test_run_stage_timeout(tmp_path: Path):
    pipeline = _trivial_pipeline({
        "00_dock.py": "import time\ntime.sleep(60)",
    })
    oracle = Layer3Oracle(ExecutorConfig(work_root=tmp_path, stage_timeout_seconds=2))
    work_dir = oracle.prepare_workdir(pipeline)
    t0 = time.time()
    out = oracle.run_stage(work_dir, pipeline, "00_dock.py")
    elapsed = time.time() - t0
    assert out.timed_out is True
    assert elapsed < 30  # killed quickly, not waited 60s
    assert out.parse_failed is True


def test_run_stage_rejects_unknown_filename(tmp_path: Path):
    pipeline = _trivial_pipeline({"00_dock.py": "pass"})
    oracle = Layer3Oracle(ExecutorConfig(work_root=tmp_path))
    work_dir = oracle.prepare_workdir(pipeline)
    with pytest.raises(ValueError):
        oracle.run_stage(work_dir, pipeline, "nonexistent.py")


def test_run_stage_rejects_non_stage_filename(tmp_path: Path):
    pipeline = _trivial_pipeline({"common.py": "pass"})
    oracle = Layer3Oracle(ExecutorConfig(work_root=tmp_path))
    work_dir = oracle.prepare_workdir(pipeline)
    with pytest.raises(ValueError):
        oracle.run_stage(work_dir, pipeline, "common.py")


def test_prepare_workdir_stages_auxiliary_files(tmp_path: Path):
    aux_src = tmp_path / "_inputs"
    aux_src.mkdir()
    host_pdb = aux_src / "cb7.pdb"
    host_pdb.write_text("HEADER fake host structure\n")
    pipeline = _trivial_pipeline({
        "00_dock.py": _result_writer_stage(
            "00_dock",
            extras='assert open("cb7.pdb").read().startswith("HEADER")',
        ),
    })
    oracle = Layer3Oracle(ExecutorConfig(work_root=tmp_path, stage_timeout_seconds=30))
    work_dir = oracle.prepare_workdir(pipeline, auxiliary_files={"cb7.pdb": host_pdb})
    assert (work_dir / "cb7.pdb").exists()
    out = oracle.run_stage(work_dir, pipeline, "00_dock.py")
    assert out.exited_normally
