"""Per-call LLM token-usage logger.

Writes one JSONL row per Anthropic API call to a configurable path. Each
row captures input/output tokens AND the prompt-cache stats
(cache_creation_input_tokens, cache_read_input_tokens) so cost analysis
can correctly account for cache discount (~10×) and cache-write surcharge
(~1.25×).

Aggregation happens out-of-band via `scripts/aggregate_tokens.py`.

Zero overhead per call: appending one JSON line is O(1) disk I/O, the
token counts are already on the SDK's ResultMessage, and the caller_tag
is a plain string. No extra API calls.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Anthropic published list prices (USD / million tokens).
# Update when Anthropic changes pricing. Keys are MODEL STEM (without any
# date suffix or vendor prefix). The lookup is by `model.startswith(...)`.
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-4-7":   {"in": 15.00, "out": 75.00, "cache_w": 18.75, "cache_r": 1.50},
    "claude-opus-4-6":   {"in": 15.00, "out": 75.00, "cache_w": 18.75, "cache_r": 1.50},
    "claude-opus-4-5":   {"in": 15.00, "out": 75.00, "cache_w": 18.75, "cache_r": 1.50},
    "claude-opus-4":     {"in": 15.00, "out": 75.00, "cache_w": 18.75, "cache_r": 1.50},
    "claude-sonnet-4-6": {"in":  3.00, "out": 15.00, "cache_w":  3.75, "cache_r": 0.30},
    "claude-sonnet-4-5": {"in":  3.00, "out": 15.00, "cache_w":  3.75, "cache_r": 0.30},
    "claude-sonnet-4":   {"in":  3.00, "out": 15.00, "cache_w":  3.75, "cache_r": 0.30},
    "claude-haiku-4-5":  {"in":  1.00, "out":  5.00, "cache_w":  1.25, "cache_r": 0.10},
    "claude-haiku-4":    {"in":  1.00, "out":  5.00, "cache_w":  1.25, "cache_r": 0.10},
}


def model_pricing(model: str | None) -> dict[str, float] | None:
    """Look up pricing for a model name, tolerant of date suffixes."""
    if not model:
        return None
    for stem, p in PRICING_USD_PER_MTOK.items():
        if model.startswith(stem):
            return p
    return None


def estimate_cost_usd(
    *,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> float | None:
    """Compute USD cost for one call given its usage record. None if
    pricing unknown."""
    p = model_pricing(model)
    if p is None:
        return None
    cost = 0.0
    cost += (input_tokens or 0) * p["in"] / 1e6
    cost += (output_tokens or 0) * p["out"] / 1e6
    cost += (cache_creation_input_tokens or 0) * p["cache_w"] / 1e6
    cost += (cache_read_input_tokens or 0) * p["cache_r"] / 1e6
    return cost


class UsageLogger:
    """Append-only JSONL token-usage recorder."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        caller: str,
        model: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
        latency_s: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append a single JSONL row. Thread-safe (lock-guarded write)."""
        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "caller": caller,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "latency_s": latency_s,
        }
        cost = estimate_cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        if cost is not None:
            row["estimated_cost_usd"] = round(cost, 6)
        if extra:
            row.update(extra)
        line = json.dumps(row, default=str)
        with self._lock:
            with self.log_path.open("a") as f:
                f.write(line + "\n")


# Module-level singleton — set once per run by run_real.py / the
# orchestrator. ClaudeSDKClient / PipelineEngineer pick it up if
# present; otherwise nothing is logged.
_global_logger: UsageLogger | None = None


def set_global_usage_logger(logger: UsageLogger | None) -> None:
    """Wire a logger into module-level state so clients can pick it up
    without explicit plumbing."""
    global _global_logger
    _global_logger = logger


def get_global_usage_logger() -> UsageLogger | None:
    return _global_logger
