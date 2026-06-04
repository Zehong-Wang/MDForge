"""Restraint design domain expert."""

from __future__ import annotations

from src.verifiers.layer2.base_expert import BaseExpert


class RestraintExpert(BaseExpert):
    NAME = "restraint"
    SYSTEM_PROMPT_FILENAME = "restraint.md"
