#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Summarize Spec-K entropy JSONL files and replay candidate thresholds."""

from __future__ import annotations

import argparse
import json
import math
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _as_numpy(values: array | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _distribution(values: array | np.ndarray) -> dict:
    numeric = _as_numpy(values)
    quantiles = np.quantile(numeric, QUANTILES) if numeric.size else ()
    return {
        "count": int(numeric.size),
        "mean": float(numeric.mean()) if numeric.size else None,
        "quantiles": {
            f"p{q * 100:g}": float(value)
            for q, value in zip(QUANTILES, quantiles)
        },
    }


def analyze(paths: list[Path], thresholds: list[float] | None) -> dict:
    metadata: list[dict] = []
    all_raw_entropy = array("d")
    all_corrected_entropy = array("d")
    all_delta = array("d")
    selected_raw_entropy = array("d")
    selected_corrected_entropy = array("d")
    selected_delta = array("d")
    recorded_budget_counts: Counter[int] = Counter()
    selected_recorded_budget_counts: Counter[int] = Counter()
    recorded_budget_sum = 0
    selected_recorded_budget_sum = 0
    recorded_budget_total = 0
    selected_recorded_budget_total = 0
    by_position: dict[int, dict[str, array]] = defaultdict(
        lambda: {
            "raw_entropy": array("d"),
            "corrected_entropy": array("d"),
            "delta": array("d"),
        }
    )
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("record_type") == "metadata":
                    metadata.append(record)
                elif record.get("record_type") == "token":
                    corrected = float(
                        record.get("corrected_entropy", record["entropy"])
                    )
                    raw = float(record.get("raw_entropy", corrected))
                    delta = float(
                        record.get("markov_entropy_delta", corrected - raw)
                    )
                    budget = int(record["expert_budget"])
                    position = int(record["draft_position"])
                    all_raw_entropy.append(raw)
                    all_corrected_entropy.append(corrected)
                    all_delta.append(delta)
                    recorded_budget_counts[budget] += 1
                    recorded_budget_sum += budget
                    recorded_budget_total += 1
                    by_position[position]["raw_entropy"].append(raw)
                    by_position[position]["corrected_entropy"].append(corrected)
                    by_position[position]["delta"].append(delta)
                    if record["selected_for_verification"]:
                        selected_raw_entropy.append(raw)
                        selected_corrected_entropy.append(corrected)
                        selected_delta.append(delta)
                        selected_recorded_budget_counts[budget] += 1
                        selected_recorded_budget_sum += budget
                        selected_recorded_budget_total += 1

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

    result = {
        "files": [str(path) for path in paths],
        "base_top_k": base_top_k,
        "all_raw_entropy": _distribution(all_raw_entropy),
        "all_corrected_entropy": _distribution(all_corrected_entropy),
        "all_markov_entropy_delta": _distribution(all_delta),
        "all_raw_ppl": _distribution(np.exp(_as_numpy(all_raw_entropy))),
        "all_corrected_ppl": _distribution(
            np.exp(_as_numpy(all_corrected_entropy))
        ),
        "selected_raw_entropy": _distribution(selected_raw_entropy),
        "selected_corrected_entropy": _distribution(selected_corrected_entropy),
        "selected_markov_entropy_delta": _distribution(selected_delta),
        "selected_raw_ppl": _distribution(
            np.exp(_as_numpy(selected_raw_entropy))
        ),
        "selected_corrected_ppl": _distribution(
            np.exp(_as_numpy(selected_corrected_entropy))
        ),
        "markov_effect": {
            "lowered_entropy_tokens": sum(value < 0.0 for value in all_delta),
            "unchanged_entropy_tokens": sum(value == 0.0 for value in all_delta),
            "raised_entropy_tokens": sum(value > 0.0 for value in all_delta),
            "lowered_entropy_fraction": (
                sum(value < 0.0 for value in all_delta) / len(all_delta)
                if all_delta
                else None
            ),
        },
        "recorded_avg_expert_budget": (
            recorded_budget_sum / recorded_budget_total
            if recorded_budget_total
            else None
        ),
        "selected_recorded_avg_expert_budget": (
            selected_recorded_budget_sum / selected_recorded_budget_total
            if selected_recorded_budget_total
            else None
        ),
        "recorded_expert_budget_counts": dict(
            sorted(recorded_budget_counts.items())
        ),
        "selected_recorded_expert_budget_counts": dict(
            sorted(selected_recorded_budget_counts.items())
        ),
        "markov_entropy_by_draft_position": {
            str(position): {
                key: _distribution(values)
                for key, values in distributions.items()
            }
            for position, distributions in sorted(by_position.items())
        },
    }
    # Compatibility aliases: Spec-K always consumes post-Markov logits.
    result["all_entropy"] = result["all_corrected_entropy"]
    result["all_ppl"] = result["all_corrected_ppl"]
    result["selected_entropy"] = result["selected_corrected_entropy"]
    result["selected_ppl"] = result["selected_corrected_ppl"]
    if thresholds is not None:
        def candidate_summary(values: array) -> tuple[float | None, dict[int, int]]:
            entropy = _as_numpy(values)
            if not entropy.size:
                return None, {}
            budgets = np.full(
                entropy.shape,
                base_top_k - len(thresholds),
                dtype=np.int16,
            )
            for threshold in thresholds:
                budgets += entropy >= math.log(threshold)
            unique, counts = np.unique(budgets, return_counts=True)
            return (
                float(budgets.mean()),
                {
                    int(budget): int(count)
                    for budget, count in zip(unique, counts)
                },
            )

        candidate_avg, candidate_counts = candidate_summary(
            all_corrected_entropy
        )
        selected_candidate_avg, selected_candidate_counts = candidate_summary(
            selected_corrected_entropy
        )
        result["candidate_thresholds"] = thresholds
        result["candidate_avg_expert_budget"] = candidate_avg
        result["selected_candidate_avg_expert_budget"] = selected_candidate_avg
        result["candidate_expert_budget_counts"] = candidate_counts
        result["selected_candidate_expert_budget_counts"] = (
            selected_candidate_counts
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
