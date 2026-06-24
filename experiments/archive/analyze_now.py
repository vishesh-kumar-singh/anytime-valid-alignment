"""Offline analyses that need no regeneration and no new model.

Runs on the already-saved grading cache (``data/cache/math500_final.npz``):

  1. Two-ceiling gap decomposition  — where each question lives (unsolvable /
     plurality-correct / minority-correct) and what can move it. Grounds the
     argument in FUTURE_WORK.md ("Where the plateau actually comes from").

  2. N4 · Adaptive-consistency early stopping (Aggarwal et al., arXiv:2305.11860)
     — replay the existing 24-sample sets, stop sampling once the answer
     distribution is confident, and report accuracy vs. mean samples used. Pure
     offline simulation: quantifies the compute wasted at a fixed N=24.

Outputs a human-readable summary and a machine-readable json to
``runs/analysis/``.

Usage:
    python experiments/analyze_now.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "cache" / "math500_final.npz"
OUT = REPO / "runs" / "analysis"


def load():
    d = np.load(CACHE, allow_pickle=True)
    # corr: (Q,N) bool correctness ; cid: (Q,N) answer-cluster id (-1 = no answer)
    # scores: (Q, R, N) per-RM scores ; rm_names: (R,)
    return d["corr"], d["cid"], d["scores"], [str(x) for x in d["rm_names"]]


def gap_decomposition(corr, cid):
    Q, N = corr.shape
    oracle_mask = corr.any(axis=1)
    unsolvable = int((~oracle_mask).sum())

    plurality_correct = 0
    minority_correct = 0
    for i in np.where(oracle_mask)[0]:
        ids, c = cid[i], corr[i]
        valid = [j for j in range(N) if ids[j] >= 0]
        if not valid:
            minority_correct += 1
            continue
        cnt = Counter(ids[j] for j in valid)
        plurality = max(cnt.items(), key=lambda kv: kv[1])[0]
        memb = [j for j in range(N) if ids[j] == plurality]
        if c[memb].any():
            plurality_correct += 1
        else:
            minority_correct += 1

    return {
        "n_questions": Q,
        "pass@1": float(corr.mean()),
        "oracle@N": float(oracle_mask.mean()),
        "unsolvable": {"count": unsolvable, "frac": unsolvable / Q},
        "plurality_correct": {"count": plurality_correct, "frac": plurality_correct / Q},
        "minority_correct": {"count": minority_correct, "frac": minority_correct / Q},
        "selection_headroom_frac": minority_correct / Q,
    }


def _beta_prob_top_beats_second(n1: int, n2: int) -> float:
    """P(rate_1 > rate_2) under Beta(n+1) posteriors, normal approx.

    Practical stand-in for the Dirichlet stick-breaking stop test of
    Adaptive-Consistency; transparent and monotone in the count margin.
    """
    a1, a2 = n1 + 1, n2 + 1
    m1, m2 = a1 / (a1 + a2 + 0.0), a2 / (a1 + a2 + 0.0)
    # Variance of two independent Betas, then normal approx of the difference.
    def var(a, b):
        s = a + b
        return a * b / (s * s * (s + 1))
    sd = (var(a1, a2) + var(a2, a1)) ** 0.5 + 1e-9
    z = (m1 - m2) / sd
    # standard normal CDF
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def adaptive_stopping(corr, cid, thresholds=(0.90, 0.95, 0.99), min_samples=3):
    Q, N = corr.shape
    out = {}
    # Fixed-N=24 majority baseline (matched correctness rule).
    out["fixed_N24_majority"] = {
        "acc": _majority_acc(corr, cid, N),
        "mean_samples": float(N),
    }
    for thr in thresholds:
        accs, used = [], []
        for i in range(Q):
            ids, c = cid[i], corr[i]
            counts: Counter = Counter()
            stop_m = N
            for m in range(1, N + 1):
                if ids[m - 1] >= 0:
                    counts[ids[m - 1]] += 1
                if m >= min_samples and len(counts) >= 1:
                    top = counts.most_common(2)
                    n1 = top[0][1]
                    n2 = top[1][1] if len(top) > 1 else 0
                    if _beta_prob_top_beats_second(n1, n2) >= thr:
                        stop_m = m
                        break
            used.append(stop_m)
            # accuracy of the answer chosen at the stop point
            sub_ids = ids[:stop_m]
            sub_c = c[:stop_m]
            valid = [j for j in range(stop_m) if sub_ids[j] >= 0]
            if not valid:
                accs.append(False)
                continue
            cnt = Counter(sub_ids[j] for j in valid)
            win = max(cnt.items(), key=lambda kv: kv[1])[0]
            memb = [j for j in range(stop_m) if sub_ids[j] == win]
            accs.append(bool(sub_c[memb].any()))
        out[f"adaptive_thr={thr}"] = {
            "acc": float(np.mean(accs)),
            "mean_samples": float(np.mean(used)),
        }
    return out


def _majority_acc(corr, cid, N):
    accs = []
    for i in range(corr.shape[0]):
        ids, c = cid[i, :N], corr[i, :N]
        valid = [j for j in range(N) if ids[j] >= 0]
        if not valid:
            accs.append(False)
            continue
        cnt = Counter(ids[j] for j in valid)
        win = max(cnt.items(), key=lambda kv: kv[1])[0]
        memb = [j for j in range(N) if ids[j] == win]
        accs.append(bool(c[memb].any()))
    return float(np.mean(accs))


def main():
    corr, cid, scores, rm_names = load()
    gap = gap_decomposition(corr, cid)
    adapt = adaptive_stopping(corr, cid)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"gap_decomposition": gap, "adaptive_stopping": adapt, "rm_names": rm_names}
    (OUT / "analyze_now.json").write_text(json.dumps(payload, indent=2))

    lines = []
    p = lines.append
    p("=" * 64)
    p("TWO-CEILING GAP DECOMPOSITION (MATH500, N=24)")
    p("=" * 64)
    p(f"pass@1   : {gap['pass@1']*100:5.1f}%")
    p(f"oracle@N : {gap['oracle@N']*100:5.1f}%  (perfect-verifier ceiling)")
    p("")
    p(f"  unsolvable        : {gap['unsolvable']['count']:3d} "
      f"({gap['unsolvable']['frac']*100:4.1f}%)  -> only generation can fix")
    p(f"  plurality-correct : {gap['plurality_correct']['count']:3d} "
      f"({gap['plurality_correct']['frac']*100:4.1f}%)  -> majority vote already gets these")
    p(f"  minority-correct  : {gap['minority_correct']['count']:3d} "
      f"({gap['minority_correct']['frac']*100:4.1f}%)  -> needs a better verifier (votes can't)")
    p("")
    p(f"  => entire selection headroom = {gap['selection_headroom_frac']*100:.1f} pts, "
      f"all in the minority-correct questions")
    p("")
    p("=" * 64)
    p("N4 · ADAPTIVE-CONSISTENCY EARLY STOPPING (offline replay)")
    p("=" * 64)
    p(f"{'rule':<22}{'acc':>8}{'mean samples':>16}")
    for k, v in adapt.items():
        p(f"{k:<22}{v['acc']*100:7.1f}%{v['mean_samples']:16.1f}")
    p("")
    p("Same/near-same accuracy as fixed N=24 at a fraction of the samples =")
    p("wasted compute the current pipeline spends at fixed N.")

    text = "\n".join(lines)
    (OUT / "analyze_now.txt").write_text(text + "\n")
    print(text)
    print(f"\nsaved -> {OUT/'analyze_now.txt'}  and  {OUT/'analyze_now.json'}")


if __name__ == "__main__":
    main()
