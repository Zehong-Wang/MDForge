"""Tests for the Layer-1 code static-analysis verifier."""

from __future__ import annotations

from src.models import HostGuestSpec, LayerName, Pipeline, VerdictLabel
from src.verifiers.layer1 import verify_layer1


def _pipeline_with_code(files: dict[str, str], entry_point: str = "run.py") -> Pipeline:
    return Pipeline(
        target=HostGuestSpec(host_id="CB7", guest_id="g1"),
        code_files=files,
        entry_point=entry_point,
    )


WELL_FORMED_ENTRY = """\
import json
import openmm

def main():
    # imagine a simulation here
    with open("result.json", "w") as fh:
        json.dump({"delta_g_kcal_per_mol": -8.0}, fh)

if __name__ == "__main__":
    main()
"""


def test_pass_on_well_formed_pipeline():
    p = _pipeline_with_code({"run.py": WELL_FORMED_ENTRY})
    c = verify_layer1(p)
    assert c.layer is LayerName.LAYER_1
    assert c.label is VerdictLabel.PASS
    assert c.concerns == []


def test_fail_on_missing_entry_point_declaration():
    p = _pipeline_with_code({"run.py": WELL_FORMED_ENTRY}, entry_point="")
    c = verify_layer1(p)
    assert c.label is VerdictLabel.FAIL
    assert any("entry_point" in x.description for x in c.concerns)


def test_fail_on_entry_point_not_in_code_files():
    p = _pipeline_with_code({"run.py": WELL_FORMED_ENTRY}, entry_point="main.py")
    c = verify_layer1(p)
    assert c.label is VerdictLabel.FAIL
    assert any("not present in code_files" in x.description for x in c.concerns)


def test_fail_on_empty_entry_point():
    p = _pipeline_with_code({"run.py": ""})
    c = verify_layer1(p)
    assert c.label is VerdictLabel.FAIL


def test_fail_on_syntax_error():
    p = _pipeline_with_code({"run.py": "def broken(:\n    pass\n"})
    c = verify_layer1(p)
    assert c.label is VerdictLabel.FAIL
    assert any("Syntax error" in x.description for x in c.concerns)


def test_uncertain_when_entry_point_has_no_imports():
    src = """\
def main():
    pass

if __name__ == "__main__":
    main()
"""
    p = _pipeline_with_code({"run.py": src})
    c = verify_layer1(p)
    assert c.label is VerdictLabel.UNCERTAIN
    assert any("import" in x.description.lower() for x in c.concerns)


def test_uncertain_when_no_main_block():
    src = """\
import openmm
import json

def helper():
    return 1
"""
    p = _pipeline_with_code({"run.py": src})
    c = verify_layer1(p)
    # No main block AND no result.json mention: both raise concerns,
    # the latter is severity 0.7 which is UNCERTAIN.
    assert c.label is VerdictLabel.UNCERTAIN


def test_uncertain_when_no_result_json_mentioned():
    src = """\
import openmm
import json

def main():
    print("done")

if __name__ == "__main__":
    main()
"""
    p = _pipeline_with_code({"run.py": src})
    c = verify_layer1(p)
    assert c.label is VerdictLabel.UNCERTAIN
    assert any("result.json" in x.description for x in c.concerns)


def test_pass_when_helper_file_writes_result_json():
    main_src = """\
import openmm
import helpers

if __name__ == "__main__":
    helpers.go()
"""
    helper_src = """\
import json

def go():
    with open("result.json", "w") as fh:
        json.dump({"delta_g_kcal_per_mol": -8.0}, fh)
"""
    p = _pipeline_with_code({"run.py": main_src, "helpers.py": helper_src})
    c = verify_layer1(p)
    assert c.label is VerdictLabel.PASS


def test_helper_file_with_syntax_error_fails():
    main_src = WELL_FORMED_ENTRY
    helper_src = "def broken(:\n    pass\n"
    p = _pipeline_with_code({"run.py": main_src, "helpers.py": helper_src})
    c = verify_layer1(p)
    assert c.label is VerdictLabel.FAIL


def test_non_python_helpers_are_ignored():
    main_src = WELL_FORMED_ENTRY
    p = _pipeline_with_code({
        "run.py": main_src,
        "config.yaml": "force_field: GAFF2\nthis: is not python:\n",
    })
    c = verify_layer1(p)
    assert c.label is VerdictLabel.PASS
