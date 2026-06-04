"""Generalist domain expert (ablation V2 / V3).

Wraps a single LLM agent whose system prompt merges the force-field,
sampling, and analysis-with-restraints expertise into one role. Used
by `Layer2Debate(single_agent_mode=True)` to ablate the multi-agent
debate component while keeping the same domain expertise surface as
the 3-expert baseline. This isolates the contribution of **debate**
(round-2 cross-visibility revision + ≥2-agent vote aggregation) from
the contribution of having a single LLM read each prompt.

Method signatures and JSON output schema are inherited from BaseExpert;
the only customisation is NAME (used in critique attribution / artifact
filenames) and SYSTEM_PROMPT_FILENAME (which points at the merged
prompt under `prompts/experts/generalist.md`).
"""

from __future__ import annotations

from src.verifiers.layer2.base_expert import BaseExpert


class GeneralistExpert(BaseExpert):
    NAME = "generalist"
    SYSTEM_PROMPT_FILENAME = "generalist.md"
