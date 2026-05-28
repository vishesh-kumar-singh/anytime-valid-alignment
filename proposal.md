# Research Proposal: Ensemble Methods for Test-Time Alignment

## What Was Done (Prior UGP Work)

The previous work ([noisy-reward-alignment.pdf](file:///home/vishesh/My%20Stuff/Research/Sayank%20Ray/Test-Time-Alignment/noisy-reward-alignment.pdf)) built an inference-time alignment pipeline with:

### Core Pipeline
1. **Generator**: Qwen2.5-7B-Instruct generates N=8 candidates per prompt
2. **Ensemble RM**: 3 reward models (OASST-DeBERTa, Llama-RM, ArmoRM) score each candidate → (μ, σ²)
3. **Selection policies**: Pick candidates using uncertainty-aware utilities instead of raw argmax

### Methods Implemented
| Method | What It Does | Result |
|--------|-------------|--------|
| Vanilla BoN | argmax μ | Baseline: 88.5% → 92.2% (after normalization) |
| Sharpe BoN | argmax μ/σ | ~same accuracy, 30-35% lower variance selected |
| Linear BoN | argmax (μ − c·σ) | Slight drop |
| SoftBoN variants | Probabilistic selection | Mixed |
| Vanilla SeqBoN | Sequential filtering, r ≥ γ | +4.5% over parallel BoN |
| Sharpe SeqBoN | Sequential, μ/σ ≥ γ | Best on MATH500 (72.2%) |
| Hybrid routing | Disagreement-gated switching | Matched best, interpretable |

### Key Finding That Worked
**Robust normalization** (per-model z-score + tanh bounding) was the single biggest gain: +11pp over min-max on GSM8K. Without it, nothing else works well.

---

## Why It Wasn't Enough for Publication

> [!IMPORTANT]
> The core issue: **accuracy improvements from uncertainty-aware selection were marginal** (~0.1% on full GSM8K-1319). The variance reduction story (30-35% lower σ²) is interesting but not a standalone contribution.

Specific weaknesses:

1. **Tiny accuracy deltas**: Sharpe BoN vs Vanilla BoN = 91.7% vs 92.2% — vanilla actually wins on full GSM8K
2. **No principled theory**: Why should μ/σ be the right objective? No regret bounds, no sample complexity analysis
3. **Fixed N=8**: Never demonstrated scaling behavior — does the gap grow with N? (This is where reward hacking should kick in)
4. **Outcome-level RMs only**: No process reward models (step-level scoring), which are known to be stronger for math reasoning
5. **Only 2 benchmarks**: GSM8K (too easy for modern models) and MATH500 — no harder benchmarks tested
6. **Ensemble diversity not studied**: Are the 3 RMs actually diverse? If they all learn similar biases, disagreement σ² is meaningless

---

## What Can Be Done: Research Directions

### Direction 1: Reward Hacking Mitigation at Scale (High Impact, Feasible)

**Thesis**: As N grows, vanilla BoN's accuracy should plateau or *drop* due to reward hacking, while uncertainty-aware methods should maintain gains. The prior work never tested this.

**Experiments**:
- Sweep N ∈ {4, 8, 16, 32, 64, 128} for vanilla BoN, Sharpe BoN, and Linear BoN
- Plot accuracy vs N curves — if Sharpe BoN crosses above vanilla at large N, that's a publishable result
- Track reward-hacking proxy (high-μ, high-σ² selections) as N increases
- Test on both GSM8K and MATH500

**Why it could work**: The limits-of-BoN paper (arXiv:2503.21878) shows theoretically that reward hacking ∝ e^reward at large N. Uncertainty-aware methods should act as a natural regularizer.

**Effort**: Medium (reuse existing pipeline, just sweep N)

---

### Direction 2: Process Reward Model (PRM) Integration (High Impact, More Work)

**Thesis**: Outcome-level RMs score the *final answer*. PRMs score *each reasoning step*. Combining step-level uncertainty with outcome-level uncertainty should give much stronger signals.

**Approach**:
- Use an open PRM (e.g., `math-shepherd-mistral-7b-prm` or Qwen PRM)
- For each candidate, get step-level scores from PRM + outcome score from ensemble RM
- Define step-variance: variance of PRM scores across reasoning steps
- New selection: penalize candidates where the PRM is confident on early steps but uncertain on later ones (indicates the reasoning "went off track")

**Why it could work**: PRMs are known to outperform ORMs on math benchmarks. Adding uncertainty on top of PRM is novel.

**Effort**: High (need PRM integration, new scoring logic)

---

### Direction 3: Adaptive N — When to Stop Generating (Medium Impact, Feasible)

**Thesis**: Instead of fixed N=8, decide *per-prompt* how many candidates to generate based on ensemble agreement. If the first few candidates all agree, stop early. If they disagree, generate more.

**Approach**:
- After generating k candidates, compute rolling ensemble σ²
- Stopping rule: stop if σ² < threshold for the top candidate AND top candidate's μ is above a quality floor
- This gives a **compute-accuracy Pareto curve** — same accuracy with fewer RM calls on easy prompts

**Why it could work**: Easy prompts (GSM8K grade-school math) don't need N=8 candidates. Hard prompts (MATH500 competition problems) might benefit from N=32. Adaptive N makes the compute budget problem-aware.

**Effort**: Low-Medium (modify the generation loop, add stopping logic)

---

### Direction 4: Ensemble Diversity Analysis & Optimization (Medium Impact, Foundational)

**Thesis**: The current 3-RM ensemble was chosen ad-hoc. Are these models actually diverse? What happens with 5, 7, 10 RMs? Is there an optimal ensemble composition?

**Experiments**:
- Compute pairwise rank correlation between all available open RMs on GSM8K/MATH500
- Measure effective ensemble size (how many independent signals do K models give?)
- Try deliberately diverse ensembles: mix model families (DeBERTa, Llama, Mistral), mix training data, mix model sizes
- Plot accuracy vs ensemble size K with fixed N

**Why it matters**: If the 3 RMs are highly correlated, σ² is not really "uncertainty" — it's just noise. Understanding ensemble diversity is foundational for all other directions.

**Effort**: Medium (scoring with many RMs, correlation analysis)

---

### Direction 5: Harder Benchmarks + Scaling the Generator (Medium Impact, Necessary)

**Thesis**: GSM8K is nearly saturated by modern 7B models (~92% baseline). We need benchmarks where there's room to improve.

**Benchmarks to add**:
- **GPQA-Diamond**: Graduate-level science questions (harder, used in RF-SeqBoN paper)
- **AIME 2024**: Competition math, very hard, low baseline accuracy
- **LiveCodeBench**: Code generation with test-case verification (different domain)

**Generator scaling**:
- Compare Qwen2.5-7B vs Qwen2.5-14B vs Qwen2.5-72B (4-bit)
- Key question: does the uncertainty-aware gap grow or shrink with stronger generators?

**Effort**: Low-Medium (add data loaders, run experiments)

---

### Direction 6: MAB-Guided Reasoning Generation (High Impact, High Novelty)

**Thesis**: Standard parallel Best-of-N generates candidates blindly. Even sequential filtering only acts as an acceptance gate. By casting the generation of reasoning steps as an exploration-exploitation problem (a multi-armed bandit), we can actively guide the generation process.

**Approach**:
- Treat intermediate reasoning prefixes as "arms".
- Use UCB (Upper Confidence Bound) or Thompson Sampling to dynamically allocate the test-time compute budget.
- **Exploit**: Expand on prefixes with high mean reward/value.
- **Explore**: Expand on highly uncertain but potentially optimal new reasoning paths.
- Stop when a complete, high-confidence path is found or the budget is exhausted.

**Why it could work**: This moves test-time alignment from passive post-hoc filtering to active, uncertainty-guided tree search during generation, potentially vastly improving sample efficiency.

**Effort**: High (requires integrating tree-search/MAB logic into the auto-regressive generation loop)

---

## Recommended Roadmap

| Phase | Timeline | What to Do |
|-------|----------|-----------|
| **Phase 1** | Week 1-2 | Direction 4 (ensemble diversity) + Direction 1 (N-sweep) — foundational + high impact |
| **Phase 2** | Week 3-4 | Direction 3 (adaptive N) + Direction 5 (harder benchmarks) |
| **Phase 3** | Week 5-8 | Direction 2 (PRM integration) — the big bet |
| **Throughout** | Ongoing | Build theory / regret analysis for whichever direction shows strongest empirical signal |

## Open Questions for Discussion with Prof

1. **Which direction does the prof find most interesting?** — PRM integration is novel but heavy; N-sweep reward hacking is a safer bet
2. **Theory angle**: Should we aim for regret bounds on Sharpe-BoN, or keep it empirical?
3. **Compute budget**: How much GPU time do we have? PRM + large N sweeps are expensive
4. **Publication target**: Workshop paper (empirical, 4 pages) vs full paper (needs theory)?
