"""
Unit tests for all pipeline components.
Run: python -m pytest tests/ -v
"""

import pytest
from src.policies.selection import greedy_argmax, risk_adjusted_argmax, sharpe_ratio_argmax, softbon_select
from src.policies.controller import (
    BudgetConfig, decide_path, summarize_uncertainty,
    thompson_sampling_select, lcb_select, cvar_select,
)
from src.sequential.search import rf_seqbon_vanilla, rf_seqbon_variance_aware
from src.reward_modeling.inference import RewardStats
from src.data_utils.gsm8k import extract_gsm8k_answer, check_answer_correct


# =============================================================================
# Selection policy tests
# =============================================================================

class TestVanillaBon:
    def test_greedy_picks_max(self):
        assert greedy_argmax([0.1, 0.9, 0.5]) == 1

    def test_greedy_empty_raises(self):
        with pytest.raises(ValueError):
            greedy_argmax([])


class TestRiskAdjustedBon:
    def test_prefers_lower_variance(self):
        # Same mean, different variance → picks lower variance
        idx, scores = risk_adjusted_argmax([1.0, 1.0], [0.0, 1.0], lam=0.5)
        assert idx == 0  # 1.0 - 0.5*0.0 > 1.0 - 0.5*1.0

    def test_high_mean_wins_if_lambda_small(self):
        idx, scores = risk_adjusted_argmax([0.5, 1.0], [0.0, 0.5], lam=0.01)
        assert idx == 1  # 1.0 - 0.01*0.5 = 0.995 > 0.5

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            risk_adjusted_argmax([1.0], [0.0], lam=-1)


class TestSharpeRatioBon:
    def test_prefers_high_signal_to_noise(self):
        # Candidate 0: mu=1.0, sigma=sqrt(0.1) -> 3.16
        # Candidate 1: mu=2.0, sigma=sqrt(1.0) -> 2.00
        idx, scores = sharpe_ratio_argmax([1.0, 2.0], [0.1, 1.0])
        assert idx == 0

    def test_handles_zero_variance(self):
        # eps prevents division by zero
        idx, scores = sharpe_ratio_argmax([1.0, 0.5], [0.0, 0.0])
        assert idx == 0  # Higher mean wins when variance is equal (zero)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sharpe_ratio_argmax([], [])


class TestSoftBoN:
    def test_softbon_probability_distribution(self):
        idx, probs = softbon_select([1.0, 2.0, 0.5], temperature=1.0)
        assert 0 <= idx < 3
        assert abs(sum(probs) - 1.0) < 1e-6


# =============================================================================
# Sequential search tests (RF-SeqBoN)
# =============================================================================

class TestVanillaSeqBoN:
    def test_accepts_above_threshold(self):
        answers = ["good_answer", "bad_answer", "great_answer"]
        rewards = {"good_answer": 0.8, "bad_answer": 0.2, "great_answer": 0.9}
        call_idx = [0]

        def gen_fn(hist, k):
            ans = answers[call_idx[0]]
            call_idx[0] += 1
            return [ans]

        def reward_fn(prompt, ans_list):
            return [rewards[a] for a in ans_list]

        result = rf_seqbon_vanilla("test", gen_fn, reward_fn, n=3, gamma=0.5)
        assert result.n_accepted == 2  # good and great pass, bad doesn't
        assert result.n_total == 3
        assert result.final_answer == "great_answer"

    def test_fallback_when_none_accepted(self):
        def gen_fn(hist, k):
            return ["bad"]

        def reward_fn(prompt, ans_list):
            return [0.1]

        result = rf_seqbon_vanilla("test", gen_fn, reward_fn, n=3, gamma=0.9)
        assert result.n_accepted == 0
        assert result.final_answer == "bad"  # Fallback to best


class TestVarianceAwareSeqBoN:
    def test_rejects_high_variance(self):
        # High mean but high variance → mu/sigma^2 is low → rejected
        answers = ["uncertain", "certain"]
        call_idx = [0]

        def gen_fn(hist, k):
            ans = answers[min(call_idx[0], len(answers) - 1)]
            call_idx[0] += 1
            return [ans]

        def reward_fn(prompt, ans_list):
            stats = {
                "uncertain": ([0.9], [10.0]),  # mu/sigma^2 = 0.09
                "certain": ([0.5], [0.01]),     # mu/sigma^2 = 50.0
            }
            means, vars_ = stats[ans_list[0]]
            return means, vars_

        result = rf_seqbon_variance_aware("test", gen_fn, reward_fn, n=2, gamma=1.0)
        assert result.n_accepted == 1  # Only "certain" passes
        assert result.final_answer == "certain"


# =============================================================================
# Controller tests
# =============================================================================

class TestController:
    def test_low_uncertainty_routes_to_bon(self):
        cfg = BudgetConfig(u_low=0.1, u_high=0.3)
        assert decide_path([0.01, 0.02, 0.03], cfg) == "bon"

    def test_medium_uncertainty_routes_to_sequential(self):
        cfg = BudgetConfig(u_low=0.1, u_high=0.3)
        assert decide_path([0.15, 0.2, 0.25], cfg) == "sequential"

    def test_high_uncertainty_routes_to_early_stop(self):
        cfg = BudgetConfig(u_low=0.1, u_high=0.3)
        assert decide_path([0.5, 0.6, 0.7], cfg) == "early_stop"


# =============================================================================
# GSM8K answer extraction tests
# =============================================================================

class TestGSM8K:
    def test_extract_standard_format(self):
        assert extract_gsm8k_answer("Step 1... Step 2... #### 42") == "42"

    def test_extract_with_commas(self):
        assert extract_gsm8k_answer("#### 1,234") == "1234"

    def test_extract_negative(self):
        assert extract_gsm8k_answer("#### -5") == "-5"

    def test_answer_correct_basic(self):
        assert check_answer_correct("#### 42", "#### 42")
        assert not check_answer_correct("#### 41", "#### 42")

    def test_answer_correct_float_int(self):
        assert check_answer_correct("#### 5.0", "#### 5")

    def test_answer_correct_with_text(self):
        assert check_answer_correct(
            "Let me solve step by step...\n#### 100",
            "The answer is\n#### 100"
        )


# =============================================================================
# Advanced controller strategy tests
# =============================================================================

class TestThompsonSampling:
    def test_returns_valid_index(self):
        idx = thompson_sampling_select([0.5, 1.0, 0.8], [0.1, 0.1, 0.1])
        assert 0 <= idx < 3

    def test_deterministic_with_zero_variance(self):
        # With zero variance, sampling is deterministic → picks argmax mean
        idx = thompson_sampling_select([0.1, 0.9, 0.5], [0.0, 0.0, 0.0])
        assert idx == 1

    def test_single_candidate(self):
        assert thompson_sampling_select([0.5], [0.1]) == 0


class TestLCB:
    def test_prefers_high_mean_low_var(self):
        # Candidate 0: 1.0 - 1.96*sqrt(0.01) = 0.804
        # Candidate 1: 0.8 - 1.96*sqrt(1.0) = -1.16
        idx = lcb_select([1.0, 0.8], [0.01, 1.0])
        assert idx == 0

    def test_returns_valid_index(self):
        idx = lcb_select([0.5, 1.0, 0.8], [0.1, 0.2, 0.05])
        assert 0 <= idx < 3


class TestCVaR:
    def test_returns_valid_index(self):
        idx = cvar_select([0.5, 1.0, 0.8], [0.1, 0.2, 0.05])
        assert 0 <= idx < 3

    def test_prefers_low_variance(self):
        # Same mean, very different variance
        idx = cvar_select([1.0, 1.0], [0.001, 10.0])
        assert idx == 0


# =============================================================================
# MATH500 data utils tests
# =============================================================================

class TestMATH500:
    def test_extract_boxed_answer(self):
        from src.data_utils.math500 import extract_math_answer
        assert extract_math_answer("The answer is $\\boxed{42}$.") == "42"

    def test_extract_boxed_fraction(self):
        from src.data_utils.math500 import extract_math_answer
        assert extract_math_answer("$\\boxed{\\frac{1}{2}}$") == "\\frac{1}{2}"

    def test_extract_hash_format(self):
        from src.data_utils.math500 import extract_math_answer
        assert extract_math_answer("#### 42") == "42"

    def test_check_correct_numeric(self):
        from src.data_utils.math500 import check_math_correct
        assert check_math_correct("The answer is 42", "42")

    def test_check_correct_symbolic(self):
        from src.data_utils.math500 import check_math_correct
        assert check_math_correct("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}")


# =============================================================================
# Experiment runner scorer tests
# =============================================================================

class TestScorerInterface:
    def test_trivial_scorer_creates(self):
        """Test that TrivialEnsembleScorer can be instantiated (no GPU needed)."""
        from src.experiments.run_risk_aware_inference import TrivialEnsembleScorer
        # Just check the class exists and has the right methods
        assert hasattr(TrivialEnsembleScorer, 'score')
        assert hasattr(TrivialEnsembleScorer, 'score_scalar')

    def test_robust_scorer_creates(self):
        """Test that RobustURMScorer class exists."""
        from src.experiments.run_risk_aware_inference import RobustURMScorer
        assert hasattr(RobustURMScorer, 'score')
        assert hasattr(RobustURMScorer, 'score_scalar')

    def test_build_scorer_trivial(self):
        """Test factory rejects robust mode without ensemble dir."""
        from src.experiments.run_risk_aware_inference import ExperimentConfig, build_scorer
        cfg = ExperimentConfig(rm_mode="robust", urm_ensemble_dir=None)
        with pytest.raises(ValueError, match="--urm-ensemble-dir required"):
            build_scorer(cfg)


# =============================================================================
# Dataset loading tests
# =============================================================================

class TestDatasetLoading:
    def test_load_dataset_unknown_raises(self):
        from src.experiments.run_risk_aware_inference import load_dataset_examples
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_examples("nonexistent_dataset")

    def test_check_correctness_alpaca(self):
        from src.experiments.run_risk_aware_inference import check_correctness
        # AlpacaEval has no correctness → always False
        assert check_correctness("any answer", "any ref", "alpaca_eval") is False
