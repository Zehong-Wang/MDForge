"""Force-field domain expert."""

from __future__ import annotations

from src.verifiers.layer2.base_expert import BaseExpert


class ForceFieldExpert(BaseExpert):
    NAME = "force_field"
    SYSTEM_PROMPT_FILENAME = "force_field.md"
