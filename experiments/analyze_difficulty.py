"""Difficulty- and subject-stratified analysis of MATH500.

Answers the question "is MATH500 the right benchmark to evaluate selection /
verification?" by breaking the single 79.4% number down by problem difficulty
(level 1-5) and subject. It joins the MATH-500 ``level``/``subject`` metadata
(by dataset order = ``example_index``) with the cached grading in
``data/cache/math500_final.npz``.

For each stratum it reports, at N=24:
  pass@1, majority@24, Skywork Best-of-N, oracle@24,
  ONA (= SkyBoN / oracle), and the minority-correct fraction.

Requires the ``datasets`` library (run in the project env):
    conda run -n langchain python experiments/analyze_difficulty.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data_utils.math500 import load_math500
CACHE = REPO / "data" / "cache" / "math500_final.npz"
OUT = REPO / "runs" / "analysis"
SKY = 3  # Skywork index in rm_names (the single strongest general RM)


def _maj_correct(corr, cid, N, i):
    ids, c = cid[i], corr[i]
    valid = [j for j in range(N) if ids[j] >= 0]
    if not valid:
        return False
    win = max(Counter(ids[j] for j in valid).items(), key=lambda kv: kv[1])[0]
    return bool(c[[j for j in range(N) if ids[j] == win]].any())


def _minority_frac(corr, cid, N, idx):
    minc = solv = 0
    for i in idx:
        if not corr[i].any():
            continue
        solv += 1
        if not _maj_correct(corr, cid, N, i):
            minc += 1
    return minc / max(solv, 1)


def _stratum(corr, cid, scores, N, idx):
    idx = np.asarray(idx)
    n = len(idx)
    sel = scores[idx][:, SKY, :].argmax(1)
    sky = float(corr[idx][np.arange(n), sel].mean())
    orc = float(corr[idx].any(1).mean())
    return {
        "n": int(n),
        "pass@1": float(corr[idx].mean()),
        "majority@24": float(np.mean([_maj_correct(corr, cid, N, i) for i in idx])),
        "skywork_bon@24": sky,
        "oracle@24": orc,
        "ona": (sky / orc) if orc > 0 else 0.0,
        "minority_correct_frac": _minority_frac(corr, cid, N, idx),
    }


def main():
    ex = load_math500()
    levels = [int(str(e["level"]).split()[-1]) if str(e["level"]).strip() else -1 for e in ex]
    subjects = [e["subject"] for e in ex]

    d = np.load(CACHE, allow_pickle=True)
    corr, cid, scores = d["corr"], d["cid"], d["scores"]
    Q, N = corr.shape
    levels = np.array(levels[:Q])

    by_level = {f"level_{lv}": _stratum(corr, cid, scores, N, np.where(levels == lv)[0])
                for lv in sorted(set(levels))}
    by_subject = {sb: _stratum(corr, cid, scores, N, [i for i in range(Q) if subjects[i] == sb])
                  for sb in sorted(set(subjects))}
    overall = _stratum(corr, cid, scores, N, np.arange(Q))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analyze_difficulty.json").write_text(
        json.dumps({"by_level": by_level, "by_subject": by_subject, "all": overall}, indent=2)
    )

    def row(name, s):
        return (f"{name:<26}{s['n']:>4}{s['pass@1']*100:>8.1f}{s['majority@24']*100:>9.1f}"
                f"{s['skywork_bon@24']*100:>9.1f}{s['oracle@24']*100:>9.1f}"
                f"{s['ona']*100:>9.1f}{s['minority_correct_frac']*100:>9.1f}")

    lines = ["MATH500 @N=24  (Skywork BoN as the selector)",
             f"{'stratum':<26}{'n':>4}{'pass@1':>8}{'major':>9}{'SkyBoN':>9}"
             f"{'oracle':>9}{'ONA':>9}{'minor%':>9}", "-" * 83, "BY LEVEL"]
    lines += [row(f"  {k.replace('_',' ')}", v) for k, v in by_level.items()]
    lines.append("BY SUBJECT")
    lines += [row(f"  {k}", v) for k, v in by_subject.items()]
    lines += ["-" * 83, row("ALL", overall)]

    text = "\n".join(lines)
    (OUT / "analyze_difficulty.txt").write_text(text + "\n")
    print(text)
    print(f"\nsaved -> {OUT/'analyze_difficulty.txt'}  and  {OUT/'analyze_difficulty.json'}")


if __name__ == "__main__":
    main()
