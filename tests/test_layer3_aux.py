"""Tests for Layer-3 auxiliary-file staging."""

from __future__ import annotations

from pathlib import Path

from src.models import HostGuestSpec, Pipeline
from src.verifiers.layer3 import ExecutorConfig, Layer3Oracle


def _pipeline(code_files, entry_point="run.py"):
    return Pipeline(
        target=HostGuestSpec(host_id="cb7", guest_id="adz"),
        code_files=code_files,
        entry_point=entry_point,
    )


def test_auxiliary_file_is_staged_and_readable(tmp_path):
    aux = tmp_path / "host.mol2"
    aux.write_text("@<TRIPOS>MOLECULE\ncb7\n# fake mol2\n")

    src = """\
import json
with open('host.mol2') as fh:
    contents = fh.read()
with open('result.json', 'w') as fh:
    json.dump({'delta_g_kcal_per_mol': -1.0, 'diagnostics': {'host_chars': len(contents)}}, fh)
"""
    cfg = ExecutorConfig(work_root=tmp_path / "runs")
    out = Layer3Oracle(cfg).run(
        _pipeline({"run.py": src}),
        auxiliary_files={"host.mol2": aux},
    )
    assert out.exited_normally is True
    assert out.delta_g_kcal_per_mol == -1.0
    assert out.diagnostics.get("extra:diagnostics", {}).get("host_chars") or \
        out.diagnostics.get("host_chars") is not None or \
        out.diagnostics  # diagnostics nesting handled below
    # We accept either nesting; assert at least the file was readable.
    assert "host_chars" in str(out.diagnostics)


def test_code_files_override_auxiliary_files(tmp_path):
    aux = tmp_path / "shared.txt"
    aux.write_text("from-auxiliary")

    src = """\
import json
with open('shared.txt') as fh:
    s = fh.read()
with open('result.json', 'w') as fh:
    json.dump({'delta_g_kcal_per_mol': 0.0, 'diagnostics': {'shared': s}}, fh)
"""
    cfg = ExecutorConfig(work_root=tmp_path / "runs")
    out = Layer3Oracle(cfg).run(
        _pipeline({"run.py": src, "shared.txt": "from-code-files"}),
        auxiliary_files={"shared.txt": aux},
    )
    assert out.exited_normally is True
    # code_files entry wins over auxiliary_files when names collide.
    assert "from-code-files" in str(out.diagnostics)


def test_auxiliary_file_traversal_rejected(tmp_path):
    aux = tmp_path / "ok.txt"
    aux.write_text("ok")
    cfg = ExecutorConfig(work_root=tmp_path / "runs")
    oracle = Layer3Oracle(cfg)
    try:
        oracle.run(
            _pipeline({"run.py": "print('hi')\n"}),
            auxiliary_files={"../escape.txt": aux},
        )
    except ValueError as exc:
        assert "escapes work dir" in str(exc)
    else:
        raise AssertionError("expected path-traversal ValueError")
