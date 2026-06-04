"""Parser for the JSON output of Layer-2 expert agents.

Each expert is instructed to emit a single JSON object, optionally inside
a ```json fenced block. The schema (co-designer mode):

    {
      "label": "pass" | "fail" | "uncertain",
      "confidence": <float in [0, 1]>,
      "strategic_insight": <str, the load-bearing co-designer narrative>,
      "concerns": [
        {
          "severity": <float in [0, 1]>,
          "description": <str>,
          "suggested_focus": <str | null>
        },
        ...
      ],
      "reasoning": <str, may be empty>
    }

For backward compatibility with the older verifier-mode schema, the
`strategic_insight` field is OPTIONAL at parse time (defaults to empty
string if missing). New co-designer expert prompts require it; tests
that exercise the verifier-mode schema continue to pass.

This module turns raw LLM text into an ExpertVote. It is permissive:
trailing prose, leading prose, and ```json fences are all OK.
"""

from __future__ import annotations

import json
import re

from src.models import Concern, ExpertVote, LayerName, VerdictLabel

FENCE_RE = re.compile(r"```(?:json)?\s*\n(?P<body>.*?)\n```", re.DOTALL | re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class VoteParseError(ValueError):
    """Raised when a Layer-2 expert's response cannot be parsed."""


def parse_expert_vote(
    text: str,
    expert: str,
    layer: LayerName,
) -> ExpertVote:
    payload = _extract_json_object(text)
    return _payload_to_vote(payload, expert=expert, layer=layer, raw_text=text)


def _extract_json_object(text: str) -> dict:
    candidates: list[str] = []
    for m in FENCE_RE.finditer(text):
        candidates.append(m.group("body"))
    # Fallback: take the first {...} blob even outside fences.
    if not candidates:
        m = JSON_OBJECT_RE.search(text)
        if m is not None:
            candidates.append(m.group(0))

    last_err: Exception | None = None
    for cand in candidates:
        # `strict=False` tolerates literal control characters (e.g. raw \n)
        # inside JSON string values. LLM outputs frequently include such
        # characters in long narrative fields like `strategic_insight`.
        try:
            obj = json.loads(cand, strict=False)
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
        if isinstance(obj, dict):
            return obj
        last_err = ValueError(f"JSON value is not an object: {type(obj).__name__}")
    if last_err is not None:
        raise VoteParseError(f"Could not parse expert vote JSON: {last_err}")
    raise VoteParseError("No JSON object found in expert response.")


def _payload_to_vote(
    payload: dict,
    expert: str,
    layer: LayerName,
    raw_text: str,
) -> ExpertVote:
    try:
        label = VerdictLabel(str(payload["label"]).strip().lower())
    except KeyError as exc:
        raise VoteParseError("expert response missing 'label'") from exc
    except ValueError as exc:
        raise VoteParseError(
            f"expert response 'label' is not pass/fail/uncertain: {payload.get('label')!r}"
        ) from exc

    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError) as exc:
        raise VoteParseError(f"'confidence' is not a number: {payload.get('confidence')!r}") from exc
    confidence = max(0.0, min(1.0, confidence))

    concerns: list[Concern] = []
    raw_concerns = payload.get("concerns") or []
    if not isinstance(raw_concerns, list):
        raise VoteParseError("'concerns' must be a list")
    for item in raw_concerns:
        if not isinstance(item, dict):
            raise VoteParseError(f"concern entry is not an object: {item!r}")
        try:
            severity = float(item.get("severity", 0.5))
        except (TypeError, ValueError) as exc:
            raise VoteParseError(f"concern.severity not a number: {item.get('severity')!r}") from exc
        severity = max(0.0, min(1.0, severity))
        description = str(item.get("description", "")).strip()
        if not description:
            # Skip empty concerns silently rather than failing the whole vote.
            continue
        focus = item.get("suggested_focus")
        if focus is not None:
            focus = str(focus).strip() or None
        concerns.append(Concern(
            layer=layer,
            expert=expert,
            severity=severity,
            description=description,
            suggested_focus=focus,
        ))

    reasoning = str(payload.get("reasoning", "")).strip()
    strategic_insight = str(payload.get("strategic_insight", "")).strip()

    return ExpertVote(
        expert=expert,
        label=label,
        confidence=confidence,
        strategic_insight=strategic_insight,
        concerns=concerns,
        reasoning=reasoning,
    )
