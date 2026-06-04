"""Tests for the Writer-output parser."""

from __future__ import annotations

import pytest

from src.parsing import ParseError, parse_writer_output


WELL_FORMED = """\
### RATIONALE
We will run a short alchemical FEP using OpenMM and openmmtools.

### ENTRY
run.py

### FILE: run.py
```python
import json
print('hi')
with open('result.json', 'w') as fh:
    json.dump({'delta_g_kcal_per_mol': -8.0}, fh)
```

### FILE: helpers.py
```python
def go():
    return 1
```
"""


def test_parses_well_formed_output():
    parsed = parse_writer_output(WELL_FORMED)
    assert parsed.entry_point == "run.py"
    assert "alchemical FEP" in parsed.rationale
    assert set(parsed.code_files) == {"run.py", "helpers.py"}
    assert "delta_g_kcal_per_mol" in parsed.code_files["run.py"]
    assert parsed.code_files["helpers.py"].strip().startswith("def go")


def test_parser_tolerates_unlabelled_fences():
    text = """\
### ENTRY
run.py

### FILE: run.py
```
print('hi')
```
"""
    parsed = parse_writer_output(text)
    assert parsed.code_files["run.py"].strip() == "print('hi')"


def test_parser_tolerates_py_lang_tag():
    text = """\
### ENTRY
run.py

### FILE: run.py
```py
print('hi')
```
"""
    parsed = parse_writer_output(text)
    assert "print('hi')" in parsed.code_files["run.py"]


def test_parser_handles_entry_inside_fence():
    text = """\
### ENTRY
```
run.py
```

### FILE: run.py
```python
print('hi')
```
"""
    parsed = parse_writer_output(text)
    assert parsed.entry_point == "run.py"


def test_missing_entry_block_raises():
    text = """\
### FILE: run.py
```python
print('hi')
```
"""
    with pytest.raises(ParseError, match="ENTRY"):
        parse_writer_output(text)


def test_missing_file_blocks_raises():
    text = """\
### ENTRY
run.py
"""
    with pytest.raises(ParseError, match="FILE"):
        parse_writer_output(text)


def test_entry_not_in_files_raises():
    text = """\
### ENTRY
main.py

### FILE: run.py
```python
print('hi')
```
"""
    with pytest.raises(ParseError, match="not present"):
        parse_writer_output(text)


def test_file_block_without_fence_raises():
    text = """\
### ENTRY
run.py

### FILE: run.py
print('hi') without a fence
"""
    with pytest.raises(ParseError, match="fenced code block"):
        parse_writer_output(text)


def test_absolute_path_rejected():
    text = """\
### ENTRY
/tmp/run.py

### FILE: /tmp/run.py
```python
print('hi')
```
"""
    with pytest.raises(ParseError, match="Absolute"):
        parse_writer_output(text)


def test_dotdot_path_rejected():
    text = """\
### ENTRY
../run.py

### FILE: ../run.py
```python
print('hi')
```
"""
    with pytest.raises(ParseError, match=r"\.\."):
        parse_writer_output(text)


def test_multiple_files_in_arbitrary_order():
    text = """\
### FILE: helpers.py
```python
def x(): return 1
```

### RATIONALE
short.

### ENTRY
run.py

### FILE: run.py
```python
import helpers
print(helpers.x())
```
"""
    parsed = parse_writer_output(text)
    assert parsed.entry_point == "run.py"
    assert set(parsed.code_files) == {"run.py", "helpers.py"}
    assert parsed.rationale == "short."


def test_no_headers_raises():
    with pytest.raises(ParseError, match="no recognised"):
        parse_writer_output("Here is some prose without any headers.")
