"""Analyze Stage-2 benchmark artifacts and apply the MR13 sharding gate."""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def _matching(rows: Iterable[Mapping[str, Any]], **criteria: Any) -> list[Mapping[str, Any]]:
    return [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]


def _best(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    values = list(rows)
    return max(values, key=lambda row: float(row["median_samples_per_second"]), default=None)


def _rate(row: Mapping[str, Any] | None) -> float | None:
    return None if row is None else float(row["median_samples_per_second"])


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def analyze(payloads: list[Mapping[str, Any]], material_benefit: float = 0.10) -> dict[str, Any]:
    """Diagnose measured bottleneck signals without claiming unmeasured causality."""
    rows = [row for payload in payloads for row in payload.get("aggregates", [])]
    shuffled = _matching(rows, pipeline="materialized", access="shuffled")
    best = _best(shuffled)
    ceiling = _best(_matching(rows, pipeline="device_only"))

    worker_scaling = []
    configurations = {(row.get("compression"), row.get("chunk_samples"), row.get("layout")) for row in shuffled}
    for compression, chunk_samples, layout in sorted(configurations, key=str):
        matches = _matching(
            shuffled,
            compression=compression,
            chunk_samples=chunk_samples,
            layout=layout,
        )
        zero = _best(row for row in matches if int(row.get("workers", -1)) == 0)
        multi = _best(row for row in matches if int(row.get("workers", 0)) > 0)
        if zero and multi:
            worker_scaling.append(
                {
                    "compression": compression,
                    "chunk_samples": chunk_samples,
                    "layout": layout,
                    "zero_workers_samples_per_second": _rate(zero),
                    "best_multiworker_count": multi["workers"],
                    "best_multiworker_samples_per_second": _rate(multi),
                    "ratio": _ratio(_rate(multi), _rate(zero)),
                }
            )

    compression_pairs = []
    for row in _matching(shuffled, compression="none"):
        compressed = _best(
            _matching(
                shuffled,
                compression="lzf",
                chunk_samples=row.get("chunk_samples"),
                layout=row.get("layout"),
                workers=row.get("workers"),
            )
        )
        if compressed:
            compression_pairs.append(
                {
                    "workers": row.get("workers"),
                    "chunk_samples": row.get("chunk_samples"),
                    "layout": row.get("layout"),
                    "lzf_vs_none_ratio": _ratio(_rate(compressed), _rate(row)),
                }
            )

    layout_pairs = []
    shard_pairs = []
    for row in _matching(shuffled, layout="full"):
        separate = _best(
            _matching(
                shuffled,
                layout="separate",
                compression=row.get("compression"),
                chunk_samples=row.get("chunk_samples"),
                workers=row.get("workers"),
            )
        )
        if separate:
            layout_pairs.append(
                {
                    "workers": row.get("workers"),
                    "compression": row.get("compression"),
                    "chunk_samples": row.get("chunk_samples"),
                    "separate_vs_full_ratio": _ratio(_rate(separate), _rate(row)),
                }
            )
        sharded = _best(
            _matching(
                shuffled,
                layout="sharded",
                compression=row.get("compression"),
                chunk_samples=row.get("chunk_samples"),
                workers=row.get("workers"),
            )
        )
        if sharded:
            shard_pairs.append(
                {
                    "workers": row.get("workers"),
                    "compression": row.get("compression"),
                    "chunk_samples": row.get("chunk_samples"),
                    "sharded_vs_full_ratio": _ratio(_rate(sharded), _rate(row)),
                }
            )

    repetitions = int(best.get("repetitions", 0)) if best else 0
    variability = float(best.get("stdev_samples_per_second", 0.0)) / float(best["median_samples_per_second"]) if best and float(best["median_samples_per_second"]) > 0 and repetitions > 1 else None
    ceiling_fraction = _ratio(_rate(best), _rate(ceiling))
    strongest_worker_ratio = max((item["ratio"] for item in worker_scaling), default=None)
    stable = repetitions >= 3 and variability is not None and variability <= 0.05
    contention_isolated = stable and strongest_worker_ratio is not None and strongest_worker_ratio < 1.0 - material_benefit
    shard_benefit = max((item["sharded_vs_full_ratio"] for item in shard_pairs), default=None)
    recommend_mr14 = contention_isolated and shard_benefit is not None and shard_benefit >= 1.0 + material_benefit

    signals = []
    if ceiling_fraction is not None:
        signals.append(f"Best materialized shuffled throughput reached {ceiling_fraction:.1%} of the device-only ceiling.")
    if strongest_worker_ratio is not None:
        signals.append(f"Best measured multi-worker scaling versus zero workers was {strongest_worker_ratio:.2f}x.")
    signals.append("Compression impact was measured." if compression_pairs else "Compression was not compared in this artifact set.")
    signals.append("Physical split layout impact was measured." if layout_pairs else "Physical split layout was not compared in this artifact set.")
    if repetitions < 3:
        signals.append("Fewer than three repetitions prevent a stable bottleneck claim.")
    if not shard_pairs:
        signals.append("No sharded candidate was measured against the single-file baseline.")

    if recommend_mr14:
        status = "go"
        rationale = "Stable measurements isolate a material single-file penalty and a sharded candidate demonstrates benefit."
    else:
        status = "no-go"
        rationale = "The evidence does not demonstrate a stable, material sharding benefit; retain the single-file format and stop before MR14."

    return {
        "analysis_version": 1,
        "thresholds": {"material_benefit_fraction": material_benefit, "minimum_repetitions": 3, "maximum_rate_stdev_fraction": 0.05},
        "best_materialized_shuffled": dict(best) if best else None,
        "device_only_ceiling_samples_per_second": _rate(ceiling),
        "device_only_ceiling_fraction": ceiling_fraction,
        "worker_scaling": worker_scaling,
        "compression_comparisons": compression_pairs,
        "layout_comparisons": layout_pairs,
        "shard_comparisons": shard_pairs,
        "signals": signals,
        "decision": {"status": status, "mr14_recommended": recommend_mr14, "rationale": rationale},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="MR12 JSON result artifacts")
    parser.add_argument("--material-benefit", type=float, default=0.10, help="Minimum fractional benefit required for format complexity")
    parser.add_argument("--output", type=Path, help="Optional JSON analysis output")
    return parser.parse_args()


def main() -> None:
    """Load artifacts, print diagnostics, and optionally save the decision."""
    args = _parse_args()
    if not math.isfinite(args.material_benefit) or not 0 < args.material_benefit < 1:
        raise ValueError("material benefit must be between zero and one")
    payloads = [json.loads(path.read_text()) for path in args.results]
    result = analyze(payloads, args.material_benefit)
    print("MR13 bottleneck analysis")
    for signal in result["signals"]:
        print(f"- {signal}")
    decision = result["decision"]
    print(f"Decision: {decision['status'].upper()} for MR14")
    print(decision["rationale"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
