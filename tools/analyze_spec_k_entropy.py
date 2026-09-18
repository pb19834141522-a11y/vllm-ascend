#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Summarize Spec-K entropy JSONL files and replay candidate thresholds."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _distribution(values: Iterable[float]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered) if ordered else None,
        "quantiles": {
            f"p{q * 100:g}": _quantile(ordered, q) for q in QUANTILES
        },
    }


def _candidate_top_k(ppl: float, base_top_k: int, thresholds: list[float]) -> int:
    return base_top_k - len(thresholds) + sum(ppl >= value for value in thresholds)


def analyze(paths: list[Path], thresholds: list[float] | None) -> dict:
    metadata: list[dict] = []
    tokens: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("record_type") == "metadata":
                    metadata.append(record)
                elif record.get("record_type") == "token":
                    tokens.append(record)

    base_top_ks = {int(record["base_top_k"]) for record in metadata}
    if len(base_top_ks) != 1:
        raise ValueError(f"Expected exactly one base_top_k, found {sorted(base_top_ks)}")
    base_top_k = base_top_ks.pop()
    if thresholds is not None:
        if len(thresholds) >= base_top_k:
            raise ValueError("Candidate thresholds must contain fewer values than base_top_k.")
        if any(value <= 0 for value in thresholds):
            raise ValueError("Candidate thresholds must be positive.")
        if any(left < right for left, right in zip(thresholds, thresholds[1:])):
            raise ValueError("Candidate thresholds must be in non-increasing order.")

    selected = [record for record in tokens if record["selected_for_verification"]]
    recorded_budgets = [int(record["expert_budget"]) for record in tokens]
    selected_recorded_budgets = [
        int(record["expert_budget"]) for record in selected
    ]
    by_position: dict[int, list[float]] = defaultdict(list)
    for record in tokens:
        by_position[int(record["draft_position"])].append(float(record["ppl"]))

    result = {
        "files": [str(path) for path in paths],
        "base_top_k": base_top_k,
        "all_entropy": _distribution(float(record["entropy"]) for record in tokens),
        "all_ppl": _distribution(float(record["ppl"]) for record in tokens),
        "selected_entropy": _distribution(
            float(record["entropy"]) for record in selected
        ),
        "selected_ppl": _distribution(float(record["ppl"]) for record in selected),
        "recorded_avg_expert_budget": (
            sum(recorded_budgets) / len(recorded_budgets)
            if recorded_budgets
            else None
        ),
        "selected_recorded_avg_expert_budget": (
            sum(selected_recorded_budgets) / len(selected_recorded_budgets)
            if selected_recorded_budgets
            else None
        ),
        "recorded_expert_budget_counts": dict(
            sorted(Counter(recorded_budgets).items())
        ),
        "selected_recorded_expert_budget_counts": dict(
            sorted(Counter(selected_recorded_budgets).items())
        ),
        "ppl_by_draft_position": {
            str(position): _distribution(values)
            for position, values in sorted(by_position.items())
        },
    }
    if thresholds is not None:
        candidate_budgets = [
            _candidate_top_k(float(record["ppl"]), base_top_k, thresholds)
            for record in tokens
        ]
        selected_candidate_budgets = [
            _candidate_top_k(float(record["ppl"]), base_top_k, thresholds)
            for record in selected
        ]
        result["candidate_thresholds"] = thresholds
        result["candidate_avg_expert_budget"] = (
            sum(candidate_budgets) / len(candidate_budgets)
            if candidate_budgets
            else None
        )
        result["selected_candidate_avg_expert_budget"] = (
            sum(selected_candidate_budgets) / len(selected_candidate_budgets)
            if selected_candidate_budgets
            else None
        )
        result["candidate_expert_budget_counts"] = dict(
            sorted(Counter(candidate_budgets).items())
        )
        result["selected_candidate_expert_budget_counts"] = dict(
            sorted(Counter(selected_candidate_budgets).items())
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="A diagnostics JSONL file or a directory containing spec_k_entropy_*.jsonl.",
    )
    parser.add_argument(
        "--thresholds",
        help="Optional comma-separated PPL thresholds to replay, for example 128,38,38,11.8311808.",
    )
    args = parser.parse_args()

    paths = (
        sorted(args.path.glob("spec_k_entropy_*.jsonl"))
        if args.path.is_dir()
        else [args.path]
    )
    if not paths:
        parser.error(f"No Spec-K entropy JSONL files found under {args.path}")
    thresholds = (
        [float(value) for value in args.thresholds.split(",")]
        if args.thresholds
        else None
    )
    print(json.dumps(analyze(paths, thresholds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
