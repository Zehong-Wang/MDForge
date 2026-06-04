"""Tests for the Layer-3 sandboxed executor."""

from __future__ import annotations

from src.models import HostGuestSpec, Pipeline
from src.verifiers.layer3 import ExecutorConfig, Layer3Oracle


def _pipeline(code_files: dict[str, str], entry_point: str = "run.py") -> Pipeline:
    return Pipeline(
        target=HostGuestSpec(host_id="CB7", guest_id="g1"),
        code_files=code_files,
        entry_point=entry_point,
    )


def _oracle(tmp_path, **kwargs) -> Layer3Oracle:
    cfg = ExecutorConfig(work_root=tmp_path, **kwargs)
    return Layer3Oracle(cfg)


def test_runs_well_formed_script_and_parses_result(tmp_path):
    src = """\
import json

with open("result.json", "w") as fh:
    json.dump({
        "delta_g_kcal_per_mol": -8.2,
        "delta_g_uncertainty_kcal_per_mol": 0.4,
        "convergence_flags": {"replicate_consistent": True},
        "energy_components": {"electrostatic": -10.0, "vdw": -3.0},
        "diagnostics": {"note": "demo"},
    }, fh)
print("done")
"""
    out = _oracle(tmp_path).run(_pipeline({"run.py": src}))
    assert out.exited_normally is True
    assert out.timed_out is False
    assert out.parse_failed is False
    assert out.return_code == 0
    assert out.delta_g_kcal_per_mol == -8.2
    assert out.delta_g_uncertainty == 0.4
    assert out.convergence_flags == {"replicate_consistent": True}
    assert out.energy_components == {"electrostatic": -10.0, "vdw": -3.0}
    assert out.diagnostics == {"note": "demo"}
    assert out.stdout_tail and "done" in out.stdout_tail


def test_handles_script_crash(tmp_path):
    src = """\
raise RuntimeError("boom")
"""
    out = _oracle(tmp_path).run(_pipeline({"run.py": src}))
    assert out.exited_normally is False
    assert out.return_code != 0
    assert out.parse_failed is True
    assert out.delta_g_kcal_per_mol is None
    assert out.stderr_tail and "RuntimeError" in out.stderr_tail


def test_handles_timeout(tmp_path):
    src = """\
import time
time.sleep(10)
"""
    out = _oracle(tmp_path, timeout_seconds=1.0).run(_pipeline({"run.py": src}))
    assert out.timed_out is True
    assert out.exited_normally is False
    assert out.delta_g_kcal_per_mol is None


def test_parse_failed_when_result_missing(tmp_path):
    src = """\
print("ran but produced nothing")
"""
    out = _oracle(tmp_path).run(_pipeline({"run.py": src}))
    assert out.exited_normally is True
    assert out.parse_failed is True
    assert out.delta_g_kcal_per_mol is None


def test_parse_failed_when_result_malformed(tmp_path):
    src = """\
with open("result.json", "w") as fh:
    fh.write("this is not json {")
"""
    out = _oracle(tmp_path).run(_pipeline({"run.py": src}))
    assert out.exited_normally is True
    assert out.parse_failed is True
    assert out.delta_g_kcal_per_mol is None


def test_unknown_payload_keys_go_into_diagnostics(tmp_path):
    src = """\
import json
with open("result.json", "w") as fh:
    json.dump({
        "delta_g_kcal_per_mol": -5.0,
        "custom_field": [1, 2, 3],
        "another": "value",
    }, fh)
"""
    out = _oracle(tmp_path).run(_pipeline({"run.py": src}))
    assert out.exited_normally is True
    assert out.delta_g_kcal_per_mol == -5.0
    assert "extra:custom_field" in out.diagnostics
    assert out.diagnostics["extra:custom_field"] == [1, 2, 3]
    assert out.diagnostics["extra:another"] == "value"


def test_multi_file_codebase(tmp_path):
    main_src = """\
import json
import helpers
result = helpers.compute()
with open("result.json", "w") as fh:
    json.dump({"delta_g_kcal_per_mol": result}, fh)
"""
    helper_src = """\
def compute():
    return -7.0
"""
    out = _oracle(tmp_path).run(
        _pipeline({"run.py": main_src, "helpers.py": helper_src})
    )
    assert out.exited_normally is True
    assert out.delta_g_kcal_per_mol == -7.0


def test_path_traversal_rejected(tmp_path):
    src = "print('hi')\n"
    pipeline = _pipeline(
        {"run.py": src, "../escape.py": "print('nope')"},
    )
    oracle = _oracle(tmp_path)
    try:
        oracle.run(pipeline)
    except ValueError as exc:
        assert "escapes work dir" in str(exc)
    else:
        raise AssertionError("expected path-traversal ValueError")


def test_artifacts_dir_returned_and_contains_files(tmp_path):
    src = """\
import json
with open("result.json", "w") as fh:
    json.dump({"delta_g_kcal_per_mol": -1.0}, fh)
with open("trajectory.dummy", "w") as fh:
    fh.write("not a real trajectory")
"""
    out = _oracle(tmp_path).run(_pipeline({"run.py": src}))
    from pathlib import Path
    assert out.artifacts_dir is not None
    assert (Path(out.artifacts_dir) / "result.json").exists()
    assert (Path(out.artifacts_dir) / "trajectory.dummy").exists()
