"""Tests for the PipelineWriter agent (using a fake LLM client)."""

from __future__ import annotations

from src.llm.base import LLMMessage, LLMResponse
from src.models import Concern, Critique, LayerName, VerdictLabel
from src.tasks import sampl4_cb7_adamantyl_amine
from src.writer import PipelineWriter


class _FakeLLM:
    """Records what was asked and replays a canned response."""

    def __init__(self, response_content: str):
        self._response = response_content
        self.last_messages: list[LLMMessage] | None = None
        self.last_system: str | None = None
        self.last_kwargs: dict | None = None

    def complete(
        self,
        messages,
        *,
        model=None,
        max_tokens=4096,
        temperature=0.0,
        system=None,
        caller_tag=None,
    ):
        self.last_messages = list(messages)
        self.last_system = system
        self.last_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return LLMResponse(
            content=self._response,
            model=model or "fake-model",
            input_tokens=10,
            output_tokens=20,
        )


CANNED_PROPOSE_RESPONSE = """\
### RATIONALE
Short alchemical FEP using OpenMM with GAFF2 + TIP3P.

### ENTRY
run.py

### FILE: run.py
```python
import json
with open('result.json', 'w') as fh:
    json.dump({'delta_g_kcal_per_mol': -14.0}, fh)
```
"""


CANNED_REVISE_RESPONSE = """\
### RATIONALE
Switched to OpenFF 2.x parameters for the guest after the methodology critic flagged GAFF2.

### ENTRY
run.py

### FILE: run.py
```python
import json
with open('result.json', 'w') as fh:
    json.dump({'delta_g_kcal_per_mol': -14.5}, fh)
```
"""


def test_propose_returns_pipeline_with_parsed_fields():
    fake = _FakeLLM(CANNED_PROPOSE_RESPONSE)
    writer = PipelineWriter(llm_client=fake)
    task = sampl4_cb7_adamantyl_amine()

    pipeline = writer.propose(task)

    assert pipeline.target == task.target
    assert pipeline.entry_point == "run.py"
    assert "run.py" in pipeline.code_files
    assert pipeline.revision == 0
    assert pipeline.parent_revision is None
    assert "FEP" in pipeline.rationale


def test_propose_sends_system_prompt_and_task_context():
    fake = _FakeLLM(CANNED_PROPOSE_RESPONSE)
    writer = PipelineWriter(llm_client=fake)
    task = sampl4_cb7_adamantyl_amine()
    writer.propose(task, time_budget_minutes=42)

    assert fake.last_system is not None
    assert "Pipeline-Writer agent" in fake.last_system
    assert "result.json" in fake.last_system  # output contract is taught

    # User message should mention the task ID, target, conditions and budget.
    assert len(fake.last_messages) == 1
    user_msg = fake.last_messages[0].content
    assert "sampl4-cb7-adz" in user_msg
    assert "cb7" in user_msg
    assert "adz" in user_msg
    assert "298.15" in user_msg or "298" in user_msg
    assert "42 minutes" in user_msg
    # Auxiliary file basenames should be listed.
    assert "cb7.mol2" in user_msg
    assert "adz.mol2" in user_msg


def test_revise_increments_revision_and_includes_critiques():
    fake = _FakeLLM(CANNED_REVISE_RESPONSE)
    writer = PipelineWriter(llm_client=fake)
    task = sampl4_cb7_adamantyl_amine()
    initial = writer.propose(task)

    fake._response = CANNED_REVISE_RESPONSE  # next call returns the revise canned
    critique = Critique(
        layer=LayerName.LAYER_2_PRE,
        label=VerdictLabel.FAIL,
        aggregate_confidence=0.9,
        concerns=[
            Concern(
                layer=LayerName.LAYER_2_PRE,
                expert="force_field",
                severity=0.9,
                description="GAFF2 is unreliable for this aromatic + ammonium guest; consider OpenFF.",
                suggested_focus="force field selection",
            ),
        ],
        summary="Force-field choice flagged.",
    )

    revised = writer.revise(initial, [critique], task)

    assert revised.revision == 1
    assert revised.parent_revision == 0
    assert revised.entry_point == "run.py"
    # Revise prompt should embed the critique text.
    user_msg = fake.last_messages[0].content
    assert "force_field" in user_msg
    assert "OpenFF" in user_msg
    assert "previous" in user_msg.lower()
    # Revise prompt should embed the previous code so the agent knows what to edit.
    assert "delta_g_kcal_per_mol" in user_msg


def test_writer_records_debug_info():
    fake = _FakeLLM(CANNED_PROPOSE_RESPONSE)
    writer = PipelineWriter(llm_client=fake)
    task = sampl4_cb7_adamantyl_amine()
    writer.propose(task)

    debug = writer.last_debug
    assert debug is not None
    assert "Pipeline-Writer agent" in debug.system_prompt
    assert "sampl4-cb7-adz" in debug.user_prompt
    assert debug.parsed.entry_point == "run.py"
    assert debug.input_tokens == 10
    assert debug.output_tokens == 20
