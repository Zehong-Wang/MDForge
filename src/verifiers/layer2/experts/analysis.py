"""Convergence / error-analysis domain expert."""

from __future__ import annotations

from src.verifiers.layer2.base_expert import BaseExpert


class AnalysisExpert(BaseExpert):
    NAME = "analysis"
    SYSTEM_PROMPT_FILENAME = "analysis.md"
