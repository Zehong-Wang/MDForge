#!/usr/bin/env python
"""Aggregate per-call token usage JSONL into a per-phase + per-model summary.

Usage:
    python scripts/aggregate_tokens.py <token_usage.jsonl>

For each phase (`writer.propose`, `writer.revise`, `phase0`, `pre`, `post`,
`engineer`, …), prints:

- call count, total input/output/cache_read/cache_create tokens
- estimated USD cost (assuming current Anthropic list prices, see
  src.llm.usage_logger.PRICING_USD_PER_MTOK)
- cache-hit ratio (cache_read / total_input)

A final TOTAL row gives the whole-run cost.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from src.llm.usage_logger import estimate_cost_usd


def phase_of(caller: str) -> str:
    """Map a caller_tag to a phase bucket."""
    if not caller:
        return "unknown"
    return caller.split(".", 1)[0]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: aggregate_tokens.py <token_usage.jsonl>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 2

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping bad line: {e}", file=sys.stderr)
            continue

    by_phase: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "in": 0, "out": 0, "cr": 0, "cw": 0, "cost": 0.0, "latency": 0.0}
    )

    def _ensure_int(x):
        if x is None:
            return 0
        try:
            return int(x)
        except (TypeError, ValueError):
            return 0

    for r in rows:
        phase = phase_of(r.get("caller", "unknown"))
        d = by_phase[phase]
        d["calls"] += 1
        d["in"] += _ensure_int(r.get("input_tokens"))
        d["out"] += _ensure_int(r.get("output_tokens"))
        d["cr"] += _ensure_int(r.get("cache_read_input_tokens"))
        d["cw"] += _ensure_int(r.get("cache_creation_input_tokens"))
        # Prefer logged cost; recompute if missing.
        c = r.get("estimated_cost_usd")
        if c is None:
            c = estimate_cost_usd(
                model=r.get("model"),
                input_tokens=_ensure_int(r.get("input_tokens")),
                output_tokens=_ensure_int(r.get("output_tokens")),
                cache_creation_input_tokens=_ensure_int(r.get("cache_creation_input_tokens")),
                cache_read_input_tokens=_ensure_int(r.get("cache_read_input_tokens")),
            )
        d["cost"] += float(c or 0.0)
        lat = r.get("latency_s")
        d["latency"] += float(lat or 0.0)

    phases = sorted(by_phase.keys())
    if not phases:
        print("(no usage records)")
        return 0

    # Print table.
    headers = ("phase", "calls", "input", "output", "cache_r", "cache_w", "hit%", "cost $", "lat s")
    rows_out = []
    totals = {"calls": 0, "in": 0, "out": 0, "cr": 0, "cw": 0, "cost": 0.0, "latency": 0.0}
    for ph in phases:
        d = by_phase[ph]
        total_in = d["in"] + d["cr"] + d["cw"]
        hit = (d["cr"] / total_in * 100.0) if total_in else 0.0
        rows_out.append((
            ph, str(d["calls"]),
            f"{d['in']:,}", f"{d['out']:,}",
            f"{d['cr']:,}", f"{d['cw']:,}",
            f"{hit:5.1f}",
            f"{d['cost']:7.4f}",
            f"{d['latency']:6.1f}",
        ))
        for k in totals:
            totals[k] += d[k]
    total_in = totals["in"] + totals["cr"] + totals["cw"]
    hit_t = (totals["cr"] / total_in * 100.0) if total_in else 0.0
    rows_out.append((
        "TOTAL", str(totals["calls"]),
        f"{totals['in']:,}", f"{totals['out']:,}",
        f"{totals['cr']:,}", f"{totals['cw']:,}",
        f"{hit_t:5.1f}",
        f"{totals['cost']:7.4f}",
        f"{totals['latency']:6.1f}",
    ))

    # Column widths.
    widths = [max(len(h), max(len(r[i]) for r in rows_out)) for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for r in rows_out:
        print(fmt.format(*r))

    return 0


if __name__ == "__main__":
    sys.exit(main())
