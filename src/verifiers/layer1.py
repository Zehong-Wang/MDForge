"""Layer 1 verifier: code static analysis on the Writer's pipeline.

This layer answers the epistemic question "is this a valid implementation?"
The Writer's output is Python source code; Layer 1 checks that the code
parses, has the structural elements that an MD-pipeline script needs to
have, and is at least plausibly runnable. It is intentionally cheap and
syntactic.

What is NOT checked here (these are Layer 2's job):
  - Whether the chosen force field, water model, or sampling protocol is
    appropriate for the host-guest system.
  - Whether the analysis methodology will produce a meaningful free energy.
  - Whether the parameter values are realistic.

What IS checked:
  - The pipeline declares an `entry_point` and that file is in `code_files`.
  - Every Python file parses (no SyntaxError).
  - The entry-point file imports something (sanity).
  - The entry-point file has executable code (a top-level call, a `main`,
    or an `if __name__ == "__main__":` block).
  - The entry-point file mentions writing a `result.json` somewhere
    (soft check, since the executor expects this convention).
"""

from __future__ import annotations

import ast

from src.models import Concern, Critique, LayerName, Pipeline, VerdictLabel

# The executor will look for this filename in the work directory.
# Documented here too so writer prompts and Layer-1 checks stay in sync.
RESULT_JSON_FILENAME = "result.json"


def _parse_python(source: str, filename: str) -> tuple[ast.AST | None, Concern | None]:
    try:
        tree = ast.parse(source, filename=filename)
        return tree, None
    except SyntaxError as exc:
        return None, Concern(
            layer=LayerName.LAYER_1,
            severity=1.0,
            description=f"Syntax error in {filename}: {exc.msg} (line {exc.lineno}).",
            suggested_focus=filename,
        )


def _check_entry_point(pipeline: Pipeline) -> list[Concern]:
    if not pipeline.entry_point:
        return [Concern(
            layer=LayerName.LAYER_1,
            severity=1.0,
            description="Pipeline does not declare an entry_point.",
            suggested_focus="entry_point",
        )]
    if pipeline.entry_point not in pipeline.code_files:
        return [Concern(
            layer=LayerName.LAYER_1,
            severity=1.0,
            description=(
                f"Entry point '{pipeline.entry_point}' is not present in code_files "
                f"(have: {sorted(pipeline.code_files.keys())})."
            ),
            suggested_focus="entry_point",
        )]
    if not pipeline.code_files[pipeline.entry_point].strip():
        return [Concern(
            layer=LayerName.LAYER_1,
            severity=1.0,
            description=f"Entry point '{pipeline.entry_point}' is empty.",
            suggested_focus=pipeline.entry_point,
        )]
    return []


def _check_all_files_parse(pipeline: Pipeline) -> tuple[list[Concern], dict[str, ast.AST]]:
    concerns: list[Concern] = []
    parsed: dict[str, ast.AST] = {}
    for name, source in pipeline.code_files.items():
        if not name.endswith(".py"):
            continue  # ignore non-Python helper files
        tree, err = _parse_python(source, name)
        if err is not None:
            concerns.append(err)
        elif tree is not None:
            parsed[name] = tree
    return concerns, parsed


def _has_main_block(tree: ast.AST) -> bool:
    """Return True if the AST has either a top-level call/expression statement,
    a `main()`-style invocation, or an `if __name__ == '__main__':` block."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare):
                if (
                    isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                ):
                    return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
    return False


def _has_imports(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
    return False


def _mentions_result_json(source: str) -> bool:
    return RESULT_JSON_FILENAME in source


def verify_layer1(pipeline: Pipeline) -> Critique:
    """Run all Layer-1 static checks and return a Critique."""
    concerns: list[Concern] = []

    # 1. Entry-point declaration.
    ep_concerns = _check_entry_point(pipeline)
    concerns += ep_concerns
    # If the entry point is broken, we cannot meaningfully run the rest.
    if any(c.severity >= 0.9 for c in ep_concerns):
        return _make_critique(concerns)

    # 2. All Python files must parse.
    parse_concerns, parsed = _check_all_files_parse(pipeline)
    concerns += parse_concerns
    if any(c.severity >= 0.9 for c in parse_concerns):
        return _make_critique(concerns)

    # 3. Entry-point structural checks.
    ep_tree = parsed.get(pipeline.entry_point)
    if ep_tree is not None:
        if not _has_imports(ep_tree):
            concerns.append(Concern(
                layer=LayerName.LAYER_1,
                severity=0.6,
                description=(
                    f"Entry point '{pipeline.entry_point}' contains no import statements; "
                    "an MD pipeline almost certainly needs at least one (openmm, openff, etc.)."
                ),
                suggested_focus=pipeline.entry_point,
            ))
        if not _has_main_block(ep_tree):
            concerns.append(Concern(
                layer=LayerName.LAYER_1,
                severity=0.4,
                description=(
                    f"Entry point '{pipeline.entry_point}' has no obvious main "
                    "execution block (a top-level call or `if __name__ == \"__main__\"` block)."
                ),
                suggested_focus=pipeline.entry_point,
            ))

    # 4. Result-file convention (soft check).
    ep_source = pipeline.code_files.get(pipeline.entry_point, "")
    if not _mentions_result_json(ep_source):
        # Allow the entry point to delegate writing to another file.
        whole_codebase = "\n".join(pipeline.code_files.values())
        if not _mentions_result_json(whole_codebase):
            concerns.append(Concern(
                layer=LayerName.LAYER_1,
                severity=0.7,
                description=(
                    f"No file mentions '{RESULT_JSON_FILENAME}'. The Layer-3 executor "
                    "reads results from this filename; missing it likely means the "
                    "code never reports a ΔG to the harness."
                ),
                suggested_focus=pipeline.entry_point,
            ))

    return _make_critique(concerns)


def _make_critique(concerns: list[Concern]) -> Critique:
    if not concerns:
        return Critique(
            layer=LayerName.LAYER_1,
            label=VerdictLabel.PASS,
            aggregate_confidence=1.0,
            concerns=[],
            summary="All Layer-1 checks passed.",
        )
    max_sev = max(c.severity for c in concerns)
    if max_sev >= 0.9:
        label = VerdictLabel.FAIL
    elif max_sev >= 0.5:
        label = VerdictLabel.UNCERTAIN
    else:
        label = VerdictLabel.PASS
    confidence = max(0.0, 1.0 - max_sev / 2.0)
    return Critique(
        layer=LayerName.LAYER_1,
        label=label,
        aggregate_confidence=confidence,
        concerns=concerns,
        summary=f"Layer 1 raised {len(concerns)} concern(s); max severity {max_sev:.2f}.",
    )
