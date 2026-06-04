"""Expert verifier agents for the multi-expert debate.

Each expert carries one strand of community-distilled domain knowledge.
The default panel is the three experts force-field, sampling, and
analysis (J=3); restraint expertise is folded into the analysis expert.
A standalone restraint expert is also available for configurations that
want to give restraints their own jurisdiction.

Every expert implements two roles:
  - pre_eval(pipeline, task, peer_votes=None) -> ExpertVote
      Read the proposed Pipeline before it has been simulated.
      Produce a methodological judgement.
  - post_eval(pipeline, output, task, peer_votes=None) -> ExpertVote
      Read the raw simulation output and translate it into
      actionable critique within the expert's domain.

Comparing pre_eval and post_eval verdicts is what feeds the bandit
reputation tracker.
"""

from src.verifiers.layer2.experts.analysis import AnalysisExpert
from src.verifiers.layer2.experts.force_field import ForceFieldExpert
from src.verifiers.layer2.experts.restraint import RestraintExpert
from src.verifiers.layer2.experts.sampling import SamplingExpert

# Default panel matches the deployed configuration (J=3): restraint
# expertise lives inside the analysis expert. RestraintExpert remains
# importable for setups that want a dedicated restraint jurisdiction.
DEFAULT_EXPERT_CLASSES = (
    ForceFieldExpert,
    SamplingExpert,
    AnalysisExpert,
)

__all__ = [
    "AnalysisExpert",
    "DEFAULT_EXPERT_CLASSES",
    "ForceFieldExpert",
    "RestraintExpert",
    "SamplingExpert",
]
