"""Tests for task spec module."""

from __future__ import annotations

import pytest

from src.tasks import get_task, sampl4_cb7_adamantyl_amine


def test_adz_task_loads_with_real_files():
    task = sampl4_cb7_adamantyl_amine()
    assert task.task_id == "sampl4-cb7-adz"
    assert task.target.host_id == "cb7"
    assert task.target.guest_id == "adz"
    assert task.target.guest_smiles is not None
    assert "[NH3+]" in task.target.guest_smiles


def test_adz_host_files_present():
    task = sampl4_cb7_adamantyl_amine()
    # We expect at least one host structure file to be present.
    assert len(task.host_structure_files) > 0
    for path in task.host_structure_files.values():
        assert path.exists()


def test_adz_guest_files_present():
    task = sampl4_cb7_adamantyl_amine()
    assert len(task.guest_structure_files) > 0
    for path in task.guest_structure_files.values():
        assert path.exists()


def test_experimental_reference_in_kcal():
    task = sampl4_cb7_adamantyl_amine()
    ref = task.experimental_reference
    # -58.99 kJ/mol = -14.10 kcal/mol (within rounding)
    assert -14.5 < ref.delta_g_kcal_per_mol < -13.5
    assert ref.technique == "ITC"
    assert ref.doi == "10.1007/s10822-014-9735-1"


def test_get_task_lookup():
    task = get_task("sampl4-cb7-adz")
    assert task.task_id == "sampl4-cb7-adz"


def test_unknown_task_id_raises():
    with pytest.raises(KeyError):
        get_task("does-not-exist")


def test_host_files_as_text_returns_strings():
    task = sampl4_cb7_adamantyl_amine()
    files = task.host_files_as_text()
    assert all(isinstance(v, str) for v in files.values())
    assert all(len(v) > 0 for v in files.values())


def test_extra_context_has_charge_state():
    task = sampl4_cb7_adamantyl_amine()
    assert task.extra_context["guest_net_charge"] == 1
    assert task.extra_context["host_net_charge"] == 0
    assert task.extra_context["temperature_kelvin"] == 298.15
