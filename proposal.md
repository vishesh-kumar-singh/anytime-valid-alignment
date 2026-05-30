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
- **NEW Hypothesis (Cross-RM Rank Verification)**: Instead of scalar math (μ/σ), use ordinal rank consensus. Take the #1 candidate from RM A. If RM B also ranks it in its top-K (e.g., top 3 of 8), accept it. If not, evaluate the #2 candidate from RM A. This relies purely on rank agreement rather than poorly calibrated score distributions.
- Plot accuracy vs N curves — if Sharpe BoN or Rank Verification crosses above vanilla at large N, that's a publishable result
- Track reward-hacking proxy (high-μ, high-σ² selections) as N increases
- Test on both GSM8K and MATH500

**Why it could work**: The limits-of-BoN paper (arXiv:2503.21878) shows theoretically that reward hacking ∝ e^reward at large N. Uncertainty-aware methods and rank-consensus heuristics should act as natural regularizers against over-optimizing a single RM's biases.

**Effort**: Medium (reuse existing pipeline, just sweep N and add rank-checking logic)

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

### Direction 7: Reward Model Calibration & Debiasing (High Impact, Feasible)

**Thesis**: RMs trained on preference data inherit systematic biases — verbosity bias (longer = higher score), sycophancy bias (agreeable = higher), formatting bias (markdown/bullets = higher). Rather than ensembling away these biases, we can directly *measure and remove* them.

**Approaches**:
- **Length-conditional scoring**: Regress out the effect of response length from RM scores. Fit a simple `score ~ f(length)` curve on a held-out set, then use the *residual* as the debiased score. This directly attacks verbosity hacking — the single most documented RM failure mode.
- **Contrastive debiasing**: For each candidate, generate a "minimal perturbation" that changes style but preserves content (e.g., strip formatting, shorten). The true quality signal is the component of the score that *doesn't change* under style perturbations — a causal intervention on spurious features.
- **Prompt-conditioned normalization**: Instead of global z-scoring, normalize RM scores *per prompt*. Easy prompts produce uniformly high-scoring candidates; hard prompts produce low-scoring ones. Without per-prompt normalization, the RM's bias toward easy prompts dominates the selection.

**Why it could work**: Verbosity and formatting biases are well-documented in the RM literature (Singhal et al., 2023; Park et al., 2024). Removing them is low-hanging fruit that directly improves selection quality without needing additional models.

**Effort**: Low-Medium (length debiasing is ~10 lines of code; contrastive debiasing needs an extra LLM call per candidate)

---

### Direction 8: Self-Consistency as a Free "Reward Model" (High Impact, Feasible)

**Thesis**: The generator itself contains alignment information. For tasks with verifiable answers (math, code), self-consistency (majority vote over final answers) provides a reward signal that is *orthogonal* to RM scoring and inherently immune to RM biases.

**Approaches**:
- **Majority-vote filtering before RM scoring**: Cluster candidates by final answer. Only apply RM scoring *within* the majority cluster. This uses self-consistency as a hard filter — if 6/8 candidates say "42", the RM can only pick among those 6, preventing it from selecting a high-scoring but wrong answer.
- **RM-consistency agreement signal**: Score with the RM, but also check if the RM's top pick agrees with the majority vote. When they disagree, that's a signal the RM is being hacked — flag it and fall back to majority vote.
- **Weighted hybrid**: `final_score = α · RM_score + (1-α) · consistency_score`, where consistency_score measures how many other candidates agree with this one's conclusion. The α can be *adaptive* — lower α when the RM shows high variance across candidates (indicating uncertainty/hackability).

**Why it could work**: Self-consistency is already known to boost accuracy for reasoning tasks (Wang et al., 2022). Combining it with RM scoring gives two independent signals — one from an external judge, one from the generator's own confidence. Their disagreement is itself an informative signal.

**Effort**: Low (majority vote is trivial to implement; the hybrid scoring is a simple weighted sum)

---

### Direction 9: Single-Model RM Uncertainty Estimation (Medium Impact, Feasible)

**Thesis**: Current uncertainty comes from *disagreement* across ensemble members. But you can extract uncertainty from a *single* RM, making uncertainty-aware selection possible without an ensemble at all.

**Approaches**:
- **MC Dropout on the RM**: Enable dropout at inference time, run the RM K times on the same candidate, get a score distribution. This gives you uncertainty without needing multiple models — much cheaper than an ensemble.
- **Last-layer Laplace approximation**: Fit a Gaussian posterior on the RM's final linear layer weights. This yields a *principled* Bayesian uncertainty estimate per prediction with minimal computational overhead (only the last layer is treated stochastically).
- **Token-level reward attribution**: Instead of treating the RM as a black box producing a scalar, decompose the score via gradient attribution to identify *which tokens* contributed most. If the high-scoring tokens are stylistic (e.g., "Certainly!", bullet points) rather than substantive, that's a hacking indicator — penalize accordingly.

**Why it could work**: MC Dropout uncertainty is well-validated in the ML literature (Gal & Ghahramani, 2016). Applying it to RMs at test time is unexplored and avoids the cost of maintaining multiple RM weights.

**Effort**: Medium (MC Dropout is easy; Laplace approximation requires some implementation; token attribution needs gradient access)

---

### Direction 10: Verification-Based Rejection (High Impact, Medium Effort)

**Thesis**: RMs give a *score*, but scores can be hacked. Verification — checking if the answer is actually correct — is much harder to hack. Use verification as a post-BoN rejection filter.

**Approaches**:
- **Self-verification loop**: After BoN selects a candidate, ask the *generator* to verify the solution: "Check if this solution is correct step by step." If the generator finds an error, reject and pick the next-best candidate. This costs one extra forward pass but catches cases where the RM was fooled by fluent but wrong reasoning.
- **Inverse problem verification**: For math, check if the answer satisfies the original constraints (plug the answer back in). For code, run test cases. This is *domain-specific* but completely immune to RM bias.
- **Cross-candidate consistency checking**: Take the top-K candidates from RM scoring. Check if they're *mutually consistent* — do they reach the same conclusion via different paths? Inconsistent top candidates strongly indicate the RM is hacked on at least some of them.

**Why it could work**: Verification is asymmetrically easier than generation. A model that can't solve a hard problem can often still *check* a proposed solution. This is the P vs NP intuition applied to alignment.

**Effort**: Medium (self-verification needs one extra LLM call; inverse verification is domain-specific; cross-consistency is cheap)

---

### Direction 11: Selection-Time Regularization (High Impact, Feasible)

**Thesis**: Even if the RM is biased, you can regularize *how you use it* during selection to prevent reward hacking, without modifying the RM itself.

**Approaches**:
- **KL-constrained selection**: Add a penalty for how "unlikely" the candidate is under the base model: `score = RM(x) - β · (-log p_base(x))`. This is implicit KL regularization — the same principle RLHF uses during training, but applied at inference time. It prevents selecting out-of-distribution candidates that exploit RM blind spots.
- **Pessimistic selection (Worst-Case BoN)**: Model the RM as having additive noise ε ~ N(0, σ²) and pick the candidate that maximizes the *worst-case* score: `argmax_i [R(x_i) - λ · uncertainty(x_i)]`. This avoids candidates that might only score high due to RM noise/bias — a conservative, robustness-first approach.
- **Rank-based selection**: Convert RM scores to ranks, discarding cardinal scale information entirely. The candidate ranked #1 gets score N, ranked #2 gets N-1, etc. Rank aggregation is inherently more robust to outlier scores and scale miscalibration than raw score comparison.

**Why it could work**: KL-constrained selection is theoretically grounded — it mirrors the KL penalty in RLHF's objective function (Stiennon et al., 2020). Applying it at test time is novel and directly connects BoN to the reward hacking literature. The pessimistic approach also has a clean connection to distributionally robust optimization.

**Effort**: Low-Medium (KL-constrained needs base-model log-probs, which vLLM can provide; rank-based is trivial)

---

### Direction 12: Difficulty-Aware Selection Policy (Medium Impact, Feasible)

**Thesis**: RMs perform differently on easy vs. hard problems, but current methods use the same selection strategy everywhere. The selection policy should adapt to prompt difficulty.

**Approaches**:
- **Difficulty-conditional RM trust**: Estimate prompt difficulty (e.g., by generator perplexity, or by how much candidates' final answers disagree). On *easy* prompts, trust the RM more. On *hard* prompts where the RM is more likely to be fooled, fall back to self-consistency or conservative selection.
- **RM confidence thresholding**: If the RM score spread across candidates is very small (all candidates score ~similarly), the RM ranking is essentially noise — fall back to a simpler heuristic like shortest-correct-answer or majority vote. Only use RM selection when the RM has a clear top pick.
- **Stratified evaluation**: Analyze and report results stratified by difficulty (e.g., by number of reasoning steps, or by baseline accuracy). This reveals *where* uncertainty-aware methods help most — likely on medium-difficulty problems where the RM is uncertain but not random.

**Why it could work**: Prior BoN results average across easy and hard problems, masking heterogeneous effects. If uncertainty-aware methods help a lot on hard problems but hurt slightly on easy ones, stratification reveals a much cleaner story.

**Effort**: Low (difficulty estimation is cheap; stratified evaluation is just post-hoc analysis)

---

### Direction 13: Pairwise Comparison Instead of Pointwise Scoring (Medium Impact, Medium Effort)

**Thesis**: Most RMs are *trained* on pairwise preference data but *used* pointwise (assign a scalar per response). This train-use mismatch degrades reliability. Using the RM closer to its training objective should yield more robust selections.

**Approaches**:
- **Tournament-style BoN**: Instead of scoring all N candidates and taking argmax, run a single-elimination or round-robin pairwise tournament. Compare candidates head-to-head using the RM as a preference classifier. The tournament winner is the final selection. This is O(N log N) RM calls instead of O(N), but each call is closer to the RM's training distribution.
- **Bradley-Terry aggregation**: Run all O(N²) pairwise comparisons, fit a Bradley-Terry model, and select based on the fitted strength parameters. This is more robust than raw pointwise scores because pairwise judgments are inherently less susceptible to scale bias and outlier scores.
- **LLM-as-judge pairwise**: Use the generator itself (or a stronger LLM) to do pairwise comparisons. This is the "LLM-as-judge" paradigm, which is known to have different (and often complementary) biases compared to trained RMs.

**Why it could work**: Pairwise preference accuracy is typically much higher than pointwise score reliability for most RMs. The computational overhead is modest for small N (N=8 → 28 pairwise comparisons, or 3 rounds of elimination).

**Effort**: Medium (requires changing the selection interface from scores to comparisons; Bradley-Terry fitting is straightforward)

---

### Direction 14: Meta-Learning the Selection Policy (High Impact, Medium Effort)

**Thesis**: Selection policies (Sharpe, Linear, etc.) are currently hand-designed. Instead, *learn* the optimal selection policy from data, letting the model discover which signals matter and when.

**Approaches**:
- **Learned selector on validation set**: Treat the selection policy as a lightweight model. On a validation set with ground-truth answers, learn the optimal combination of RM score, uncertainty, length, self-consistency, etc. This can be as simple as logistic regression: `P(correct) = σ(w₁·μ + w₂·σ + w₃·length + w₄·consistency + ...)`, fitted to maximize selection accuracy.
- **Per-prompt policy routing**: Train a classifier that, given prompt features + candidate set statistics, predicts *which* selection policy (vanilla BoN, Sharpe, majority vote, etc.) will give the best result. This is a meta-learning approach — the router learns when the RM can be trusted vs. when to fall back to simpler heuristics.
- **Bandit-style online policy selection**: Treat the choice of selection policy as a multi-armed bandit problem. Maintain running estimates of each policy's accuracy and use UCB/Thompson Sampling to select the policy per prompt. This adapts online without a separate validation set.

**Why it could work**: Different selection strategies likely have complementary strengths. Sharpe BoN may excel on uncertain problems while vanilla BoN is fine on easy ones. A meta-learner can capture this structure rather than requiring manual analysis.

**Effort**: Medium (learned selector needs a validation set with ground truth; bandit approach needs online feedback)

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
