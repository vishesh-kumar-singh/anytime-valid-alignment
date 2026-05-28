#!/usr/bin/env python3
"""Analyze disagreement-gated hybrid selection from BoN-suite style JSONL outputs.

This script is post-hoc: it uses per-example method rows and shared candidate-level
statistics (`candidate_means`, `candidate_variances`) to estimate whether a hybrid
policy (low disagreement -> method A, high disagreement -> method B) improves over
single-method baselines.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MethodRow:
    method: str
    is_correct: bool
    reward_mean: float
    reward_variance: float


@dataclass
class ExampleGroup:
    example_index: int
    candidate_means: list[float]
    candidate_variances: list[float]
    rows: dict[str, MethodRow]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _group_examples(records: list[dict[str, Any]]) -> list[ExampleGroup]:
    groups: dict[int, ExampleGroup] = {}
    for r in records:
        idx = int(r["example_index"])
        g = groups.get(idx)
        if g is None:
            g = ExampleGroup(
                example_index=idx,
                candidate_means=list(r.get("candidate_means", [])),
                candidate_variances=list(r.get("candidate_variances", [])),
                rows={},
            )
            groups[idx] = g

        method = str(r["method"])
        g.rows[method] = MethodRow(
            method=method,
            is_correct=bool(r["is_correct"]),
            reward_mean=float(r["reward_mean"]),
            reward_variance=float(r["reward_variance"]),
        )

        # Prefer first non-empty candidate arrays; fall back if missing.
        if (not g.candidate_means) and r.get("candidate_means"):
            g.candidate_means = list(r["candidate_means"])
        if (not g.candidate_variances) and r.get("candidate_variances"):
            g.candidate_variances = list(r["candidate_variances"])

    return [groups[k] for k in sorted(groups.keys())]


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def disagreement_score(candidate_means: list[float], candidate_variances: list[float], top_k: int) -> float:
    if not candidate_means or not candidate_variances:
        return float("nan")

    n = min(len(candidate_means), len(candidate_variances))
    k = max(1, min(top_k, n))

    # Focus disagreement where selector actually competes: top-k by mean utility.
    top_ids = sorted(range(n), key=lambda i: candidate_means[i], reverse=True)[:k]
    vals = [candidate_variances[i] for i in top_ids]
    return sum(vals) / float(len(vals))


def _metrics(rows: list[MethodRow]) -> dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "avg_reward": float("nan"),
            "avg_variance": float("nan"),
            "top10_reward_wrong_rate": float("nan"),
            "top10_reward_count": 0,
        }
    n = len(rows)
    acc = sum(1 for r in rows if r.is_correct) / n
    avg_r = sum(r.reward_mean for r in rows) / n
    avg_v = sum(r.reward_variance for r in rows) / n

    sorted_rows = sorted(rows, key=lambda r: r.reward_mean, reverse=True)
    k = max(1, int(round(0.10 * n)))
    top = sorted_rows[:k]
    top_wrong_rate = sum(1 for r in top if not r.is_correct) / len(top)

    return {
        "n": n,
        "accuracy": acc,
        "avg_reward": avg_r,
        "avg_variance": avg_v,
        "top10_reward_wrong_rate": top_wrong_rate,
        "top10_reward_count": len(top),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Disagreement-gated hybrid policy analysis")
    p.add_argument("--input", required=True, help="Input JSONL from bon-only/main run")
    p.add_argument("--low-method", default="vanilla_bon", help="Method used on low disagreement prompts")
    p.add_argument("--high-method", default="bon_sharpe", help="Method used on high disagreement prompts")
    p.add_argument("--top-k", type=int, default=3, help="Top-k candidates by mean for disagreement score")
    p.add_argument("--threshold-quantile", type=float, default=0.75, help="Disagreement quantile threshold")
    p.add_argument(
        "--threshold-value",
        type=float,
        default=None,
        help="Optional fixed disagreement threshold. If set, quantile is ignored.",
    )
    p.add_argument("--output-prefix", required=True, help="Prefix path for output .json/.txt")
    args = p.parse_args()

    in_path = Path(args.input)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(in_path)
    groups = _group_examples(records)

    scores: list[float] = []
    per_ex_score: dict[int, float] = {}
    for g in groups:
        s = disagreement_score(g.candidate_means, g.candidate_variances, args.top_k)
        per_ex_score[g.example_index] = s
        if not math.isnan(s):
            scores.append(s)

    if args.threshold_value is not None:
        thr = float(args.threshold_value)
        threshold_source = "fixed"
    else:
        thr = _quantile(scores, args.threshold_quantile)
        threshold_source = "quantile"

    low_rows: list[MethodRow] = []
    high_rows: list[MethodRow] = []
    hybrid_rows: list[MethodRow] = []
    vanilla_rows: list[MethodRow] = []
    high_method_rows: list[MethodRow] = []

    skipped = 0
    for g in groups:
        if args.low_method not in g.rows or args.high_method not in g.rows:
            skipped += 1
            continue

        low_r = g.rows[args.low_method]
        high_r = g.rows[args.high_method]
        s = per_ex_score.get(g.example_index, float("nan"))

        vanilla_rows.append(low_r)
        high_method_rows.append(high_r)

        if math.isnan(s):
            # Conservative fallback: if disagreement missing, keep low method.
            chosen = low_r
            low_rows.append(low_r)
        elif s >= thr:
            chosen = high_r
            high_rows.append(high_r)
        else:
            chosen = low_r
            low_rows.append(low_r)

        hybrid_rows.append(chosen)

    result = {
        "input": str(in_path),
        "low_method": args.low_method,
        "high_method": args.high_method,
        "top_k": args.top_k,
        "threshold_quantile": args.threshold_quantile,
        "threshold_value": thr,
        "threshold_source": threshold_source,
        "n_examples": len(groups),
        "n_scored": len(scores),
        "n_skipped_missing_methods": skipped,
        "n_low_bucket": len(low_rows),
        "n_high_bucket": len(high_rows),
        "metrics": {
            args.low_method: _metrics(vanilla_rows),
            args.high_method: _metrics(high_method_rows),
            "hybrid": _metrics(hybrid_rows),
            "bucket_low_examples": _metrics(low_rows),
            "bucket_high_examples": _metrics(high_rows),
        },
    }

    out_json = out_prefix.with_suffix(".json")
    out_txt = out_prefix.with_suffix(".txt")

    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m_low = result["metrics"][args.low_method]
    m_high = result["metrics"][args.high_method]
    m_hyb = result["metrics"]["hybrid"]
    lines = [
        "Disagreement-Gated Hybrid Analysis",
        "=" * 42,
        f"Input: {in_path}",
        f"Low method: {args.low_method}",
        f"High method: {args.high_method}",
        f"Disagreement top-k: {args.top_k}",
        f"Threshold quantile: {args.threshold_quantile}",
        f"Threshold value: {thr:.6f}",
        f"Examples: total={len(groups)} scored={len(scores)} skipped={skipped}",
        f"Buckets: low={len(low_rows)} high={len(high_rows)}",
        "",
        "Accuracy / Top10RewardWrongRate",
        f"- {args.low_method:16s}: acc={m_low['accuracy']:.4f}  top10_wrong={m_low['top10_reward_wrong_rate']:.4f}",
        f"- {args.high_method:16s}: acc={m_high['accuracy']:.4f}  top10_wrong={m_high['top10_reward_wrong_rate']:.4f}",
        f"- {'hybrid':16s}: acc={m_hyb['accuracy']:.4f}  top10_wrong={m_hyb['top10_reward_wrong_rate']:.4f}",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved {out_json}")
    print(f"Saved {out_txt}")


if __name__ == "__main__":
    main()
