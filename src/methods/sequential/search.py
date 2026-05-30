"""
Sequential Reward Filtering (RF-SeqBoN) — from the paper:
"On the Limits of Test-Time Compute: Sequential Reward Filtering for Better Inference"

Two variants implemented:
  1. Vanilla RF-SeqBoN: threshold on scalar reward r(a_i) >= gamma
  2. Sharpe RF-SeqBoN: threshold on mu/sigma^2 >= gamma
  3. Linear Uncertainty RF-SeqBoN: threshold on mu - c*sigma >= gamma
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import math

@dataclass
class SeqBoNResult:
    """Result of sequential reward-filtered generation."""
    final_answer: str           # The best answer among all N generated candidates
    accepted_answers: List[str] # All answers that passed the filter
    all_answers: List[str]      # All N sampled answers (accepted + rejected)
    all_rewards: List[float]    # Reward/mu for each sampled answer
    all_variances: List[float]  # Variance for each sampled answer (0 for vanilla)
    n_accepted: int             # Number of answers that passed the filter
    n_total: int                # Total samples generated


GenerateFn = Callable[[str, int], List[str]]
RewardFn = Callable[[str, List[str]], Tuple[List[float], List[float]]]
ScalarRewardFn = Callable[[str, List[str]], List[float]]


def _build_history_prompt(
    base_prompt: str, 
    accepted_answers: List[str], 
    history_budget: int = 2,
    dataset: str = ""
) -> str:
    """
    Build structured sequential prompt with history budget and self-critique.
    """
    if not accepted_answers:
        return base_prompt

    recent = accepted_answers[-history_budget:]
    
    prompt_parts = [base_prompt, "\n\n### Previous Attempts:"]
    for i, ans in enumerate(recent):
        prompt_parts.append(f"\n--- Attempt {i+1} ---\n{ans}")
        
    prompt_parts.append("\n\n### Instructions:")
    prompt_parts.append("The above attempts may contain errors or be incomplete. "
                        "Please analyze them, correct any mistakes via self-critique, "
                        "and write a final improved answer.")
    
    if dataset == "gsm8k":
        prompt_parts.append(
            "Keep the reasoning concise and put the final numerical answer after ####."
        )
    elif dataset == "math500":
        prompt_parts.append(
            "Keep the reasoning concise. Final line only: \\boxed{answer}. "
            "Do not add any text after that final boxed line."
        )
        
    return "".join(prompt_parts)


# ---------------------------------------------------------------------------
# 1. Vanilla RF-SeqBoN
# ---------------------------------------------------------------------------

def rf_seqbon_vanilla(
    prompt: str,
    generate_fn: GenerateFn,
    reward_fn: ScalarRewardFn,
    n: int,
    gamma: float,
    history_budget: int = 2,
    dataset: str = ""
) -> SeqBoNResult:
    accepted_answers: List[str] = []
    all_answers: List[str] = []
    all_rewards: List[float] = []
    all_variances: List[float] = []

    for i in range(n):
        history = _build_history_prompt(prompt, accepted_answers, history_budget, dataset)
        candidates = generate_fn(history, 1)
        answer = candidates[0]
        all_answers.append(answer)

        # Score
        scores = reward_fn(prompt, [answer])
        reward = scores[0]
        all_rewards.append(reward)
        all_variances.append(0.0)

        if reward >= gamma:
            accepted_answers.append(answer)

    # FINAL SELECTION: Max reward across ALL generated candidates
    best_idx = max(range(len(all_rewards)), key=lambda i: all_rewards[i])
    final_answer = all_answers[best_idx]

    return SeqBoNResult(
        final_answer=final_answer,
        accepted_answers=accepted_answers,
        all_answers=all_answers,
        all_rewards=all_rewards,
        all_variances=all_variances,
        n_accepted=len(accepted_answers),
        n_total=n,
    )


# ---------------------------------------------------------------------------
# 2. Sharpe Ratio RF-SeqBoN
# ---------------------------------------------------------------------------

def rf_seqbon_sharpe_ratio(
    prompt: str,
    generate_fn: GenerateFn,
    reward_fn: RewardFn,
    n: int,
    gamma: float,
    eps: float = 1e-8,
    history_budget: int = 2,
    dataset: str = ""
) -> SeqBoNResult:
    accepted_answers: List[str] = []
    all_answers: List[str] = []
    all_rewards: List[float] = []
    all_variances: List[float] = []
    all_scores: List[float] = []

    for i in range(n):
        history = _build_history_prompt(prompt, accepted_answers, history_budget, dataset)
        candidates = generate_fn(history, 1)
        answer = candidates[0]
        all_answers.append(answer)

        means, variances = reward_fn(prompt, [answer])
        mu = means[0]
        sigma_sq = variances[0]
        sigma = max(sigma_sq, 0.0) ** 0.5
        all_rewards.append(mu)
        all_variances.append(sigma_sq)

        sharpe = mu / (sigma + eps)
        all_scores.append(sharpe)

        if sharpe >= gamma:
            accepted_answers.append(answer)

    # FINAL SELECTION: Max Sharpe score across ALL generated candidates
    best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
    final_answer = all_answers[best_idx]

    return SeqBoNResult(
        final_answer=final_answer,
        accepted_answers=accepted_answers,
        all_answers=all_answers,
        all_rewards=all_rewards,
        all_variances=all_variances,
        n_accepted=len(accepted_answers),
        n_total=n,
    )


# ---------------------------------------------------------------------------
# 3. Linear Uncertainty RF-SeqBoN (mu - c*sigma)
# ---------------------------------------------------------------------------

def rf_seqbon_linear_uncertainty(
    prompt: str,
    generate_fn: GenerateFn,
    reward_fn: RewardFn,
    n: int,
    gamma: float,
    c: float = 1.0,
    history_budget: int = 2,
    dataset: str = ""
) -> SeqBoNResult:
    """SeqBoN thresholding on mu - c * sigma >= gamma"""
    accepted_answers: List[str] = []
    all_answers: List[str] = []
    all_rewards: List[float] = []
    all_variances: List[float] = []
    all_scores: List[float] = []

    for i in range(n):
        history = _build_history_prompt(prompt, accepted_answers, history_budget, dataset)
        candidates = generate_fn(history, 1)
        answer = candidates[0]
        all_answers.append(answer)

        means, variances = reward_fn(prompt, [answer])
        mu = means[0]
        sigma_sq = variances[0]
        all_rewards.append(mu)
        all_variances.append(sigma_sq)

        sigma = math.sqrt(max(0, sigma_sq))
        score = mu - c * sigma
        all_scores.append(score)

        if score >= gamma:
            accepted_answers.append(answer)

    # FINAL SELECTION: Max (mu - c * sigma) score across ALL generated candidates
    best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
    final_answer = all_answers[best_idx]

    return SeqBoNResult(
        final_answer=final_answer,
        accepted_answers=accepted_answers,
        all_answers=all_answers,
        all_rewards=all_rewards,
        all_variances=all_variances,
        n_accepted=len(accepted_answers),
        n_total=n,
    )


__all__ = [
    "SeqBoNResult",
    "rf_seqbon_vanilla",
    "rf_seqbon_sharpe_ratio",
    "rf_seqbon_linear_uncertainty",
    "rf_seqbon_variance_aware",
    "rf_seqbon_adaptive_threshold",
]


def rf_seqbon_variance_aware(
    prompt: str,
    generate_fn: GenerateFn,
    reward_fn: RewardFn,
    n: int,
    gamma: float,
    eps: float = 1e-8,
    history_budget: int = 2,
    dataset: str = "",
) -> SeqBoNResult:
    """Backward-compatible alias for the old API name.

    This preserves existing imports/tests that still call
    `rf_seqbon_variance_aware` while delegating to the Sharpe variant.
    """
    return rf_seqbon_sharpe_ratio(
        prompt=prompt,
        generate_fn=generate_fn,
        reward_fn=reward_fn,
        n=n,
        gamma=gamma,
        eps=eps,
        history_budget=history_budget,
        dataset=dataset,
    )


# ---------------------------------------------------------------------------
# 4. Adaptive Threshold RF-SeqBoN (burn-in calibrated gamma)
# ---------------------------------------------------------------------------

def rf_seqbon_adaptive_threshold(
    prompt: str,
    generate_fn: GenerateFn,
    reward_fn: RewardFn,
    n: int,
    n_burnin: int = 3,
    gamma_percentile: float = 50.0,
    use_sharpe: bool = True,
    eps: float = 1e-8,
    history_budget: int = 2,
    dataset: str = "",
) -> SeqBoNResult:
    """Adaptive-threshold RF-SeqBoN with burn-in calibration.

    Instead of a fixed gamma threshold, we:
      1. Generate n_burnin candidates as a "burn-in" batch.
      2. Compute their Sharpe scores (or raw rewards).
      3. Set gamma as the q-th percentile of those scores.
      4. Run the remaining (n - n_burnin) rounds with this calibrated gamma.

    This automatically calibrates the threshold per-prompt:
      - Hard prompts get a lower threshold (more accepting)
      - Easy prompts get a higher threshold (more selective)

    Args:
        prompt: The input prompt.
        generate_fn: Function(history, k) -> List[str].
        reward_fn: Function(prompt, answers) -> (means, variances).
        n: Total number of sequential rounds.
        n_burnin: Number of initial candidates for threshold calibration.
        gamma_percentile: Percentile of burn-in scores to use as threshold.
        use_sharpe: If True, threshold on Sharpe scores; else on raw rewards.
        eps: Stability constant.
        history_budget: Max recent accepted answers in prompt history.
        dataset: Dataset name for prompt formatting.

    Returns:
        SeqBoNResult with all generated candidates and the selected answer.
    """
    accepted_answers: List[str] = []
    all_answers: List[str] = []
    all_rewards: List[float] = []
    all_variances: List[float] = []
    all_scores: List[float] = []

    # Ensure at least 1 burn-in and at least 1 sequential round after
    n_burnin = max(1, min(n_burnin, n - 1))

    # Phase 1: Burn-in — generate n_burnin candidates to calibrate gamma
    burnin_scores: List[float] = []

    for i in range(n_burnin):
        history = _build_history_prompt(prompt, accepted_answers, history_budget, dataset)
        candidates = generate_fn(history, 1)
        answer = candidates[0]
        all_answers.append(answer)

        means, variances = reward_fn(prompt, [answer])
        mu = means[0]
        sigma_sq = variances[0]
        sigma = max(sigma_sq, 0.0) ** 0.5
        all_rewards.append(mu)
        all_variances.append(sigma_sq)

        if use_sharpe:
            score = mu / (sigma + eps)
        else:
            score = mu
        all_scores.append(score)
        burnin_scores.append(score)

        # During burn-in, accept all answers to build context
        accepted_answers.append(answer)

    # Compute adaptive threshold from burn-in distribution
    import numpy as _np
    adaptive_gamma = float(_np.percentile(burnin_scores, gamma_percentile))

    # Phase 2: Sequential filtering with calibrated gamma
    for i in range(n_burnin, n):
        history = _build_history_prompt(prompt, accepted_answers, history_budget, dataset)
        candidates = generate_fn(history, 1)
        answer = candidates[0]
        all_answers.append(answer)

        means, variances = reward_fn(prompt, [answer])
        mu = means[0]
        sigma_sq = variances[0]
        sigma = max(sigma_sq, 0.0) ** 0.5
        all_rewards.append(mu)
        all_variances.append(sigma_sq)

        if use_sharpe:
            score = mu / (sigma + eps)
        else:
            score = mu
        all_scores.append(score)

        if score >= adaptive_gamma:
            accepted_answers.append(answer)

    # FINAL SELECTION: Max score across ALL generated candidates
    best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
    final_answer = all_answers[best_idx]

    return SeqBoNResult(
        final_answer=final_answer,
        accepted_answers=accepted_answers,
        all_answers=all_answers,
        all_rewards=all_rewards,
        all_variances=all_variances,
        n_accepted=len(accepted_answers),
        n_total=n,
    )

