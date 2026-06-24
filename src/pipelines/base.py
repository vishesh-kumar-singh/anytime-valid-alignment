"""Abstract base class for hypothesis experiments.

Every pipeline module defines a single class inheriting from ``EvaluationPipeline``
and implements ``evaluate_methods()``.  The base class provides a standard
``run()`` template that handles N-scaling, difficulty stratification, and saving.
"""

from __future__ import annotations

import glob
import json
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..core.data import check_correctness
from ..core.method_runners import run_vanilla_bon
from ..core.normalize import normalize_member_scores
from ..core.results import ResultsManager


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

# Default plot styles for known methods
METHOD_STYLE = {
    "vanilla_bon (ensemble)": {"color": "#e74c3c", "marker": "o", "ls": "-"},
    "bon_sharpe":             {"color": "#2ecc71", "marker": "s", "ls": "-"},
    "bon_linear":             {"color": "#3498db", "marker": "D", "ls": "-"},
    "soft_bon":               {"color": "#e67e22", "marker": "^", "ls": "--"},
    "softbon_sharpe":         {"color": "#9b59b6", "marker": "v", "ls": "--"},
    "softbon_linear":         {"color": "#1abc9c", "marker": "X", "ls": "--"},
    "majority_vote":          {"color": "#f1c40f", "marker": "*", "ls": "-"},
    "rm_within_majority":     {"color": "#34495e", "marker": "p", "ls": "-"},
    "rm_consistency_agree":   {"color": "#95a5a6", "marker": "h", "ls": "-"},
    "mob_vanilla":            {"color": "#f39c12", "marker": "h", "ls": "--"},
    "mob_sharpe":             {"color": "#16a085", "marker": "H", "ls": "--"},
    "wbon_inv_var":           {"color": "#8e44ad", "marker": "d", "ls": "--"},
    "wbon_sharpe":            {"color": "#d35400", "marker": "p", "ls": "--"},
}


def _normalize_math_answer(ans: str) -> str:
    """Light canonicalisation so equivalent LaTeX forms cluster together."""
    s = ans.strip().rstrip(".")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\sqrt2", "\\sqrt{2}").replace("\\sqrt3", "\\sqrt{3}")
    return s


import functools

@functools.lru_cache(maxsize=None)
def _extract_answer(text: str, dataset: str) -> Optional[str]:
    """Extract final answer from a response for self-consistency clustering.

    For MATH500 this uses brace-balanced ``\\boxed{}`` extraction (the previous
    regex only handled one level of nesting and silently failed on answers like
    ``\\boxed{\\frac{\\sqrt{3}}{2}}`` — ~18% of questions), then falls back to a
    trailing ``$...$`` group or the last number.  Answers are normalised so
    that trivially-equivalent forms map to the same cluster.
    """
    if dataset == "gsm8k":
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
        if match:
            return match.group(1).replace(",", "").strip()
        return None

    if dataset == "math500":
        # Brace-balanced last \boxed{...}
        start = text.rfind("\\boxed{")
        if start != -1:
            i = start + len("\\boxed{")
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    if depth == 0:
                        return _normalize_math_answer(text[i:j])
                    depth -= 1
        # Fallback: last inline-math span
        spans = re.findall(r"\$(.+?)\$", text)
        if spans:
            return _normalize_math_answer(spans[-1])
        # Fallback: last number
        nums = re.findall(r"-?[\d,]+\.?\d*", text)
        if nums:
            return nums[-1].replace(",", "")

            
    if dataset == "mmlu":
        match = re.search(r'(?:answer is|Answer:|\*\*Answer\*\*:?)\s*\(?([A-D])\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        match = re.search(r'\(([A-D])\)', text, re.IGNORECASE)
        if match:
            matches = re.findall(r'\(([A-D])\)', text, re.IGNORECASE)
            return matches[-1].upper()
            
        matches = re.findall(r'\b([A-D])\b', text)
        if matches:
            return matches[-1].upper()
        return None

    return None


class EvaluationPipeline(ABC):
    """Base class for all experiment hypotheses.

    Subclasses must set the three class-level attributes and implement
    ``evaluate_methods()``.

    The ``run()`` method is a concrete template that:
    1. Loads data
    2. Sweeps N values, calling ``evaluate_methods()`` per example
    3. Always includes individual RM baselines for comparison
    4. Computes difficulty-stratified results at max(N)
    5. Returns a comprehensive results dict

    Attributes:
        name: Machine-readable identifier, e.g. ``h1_uncertainty_aware``.
        category: ``"parallel"`` or ``"sequential"``.
        description: One-line human-readable hypothesis statement.
    """

    name: str = ""
    category: str = "parallel"
    description: str = ""

    # ── Helpers for loading precomputed data ──────────────────────────

    @staticmethod
    def load_candidates(candidates_path: str) -> List[dict]:
        """Load the precomputed candidates JSONL file."""
        examples: List[dict] = []
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    @staticmethod
    def load_scores(score_paths: str) -> tuple[List[str], Dict[int, List[List[float]]]]:
        """Load RM score files matching a glob pattern."""
        if "," in score_paths:
            score_files = [p.strip() for p in score_paths.split(",") if p.strip()]
        else:
            score_files = sorted(glob.glob(score_paths))

        if not score_files:
            raise FileNotFoundError(f"No score files found matching: {score_paths}")

        rm_names: List[str] = []
        all_rm_scores: Dict[int, List[List[float]]] = {}

        for sf in score_files:
            rm_name = Path(sf).stem.split("_scores_")[-1]
            rm_names.append(rm_name)
            with open(sf, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    idx = row["example_index"]
                    if idx not in all_rm_scores:
                        all_rm_scores[idx] = []
                    all_rm_scores[idx].append(row["scores"])

        return rm_names, all_rm_scores

    @staticmethod
    def detect_dataset(candidates_path: str) -> str:
        """Infer dataset name from candidates filename."""
        name = Path(candidates_path).name.lower()
        if "gsm8k" in name:
            return "gsm8k"
        if "math500" in name:
            return "math500"
        if "mmlu" in name:
            return "mmlu"
        if "alpaca" in name:
            return "alpaca"
        return "unknown"

    @staticmethod
    def ensemble_stats(
        member_scores: List[List[float]],
        norm_mode: str = "robust_zscore",
        norm_eps: float = 1e-6,
        bound_tanh: bool = False,
    ) -> tuple[List[float], List[float], List[List[float]]]:
        """Compute ensemble mean/variance from member scores."""
        normalized = normalize_member_scores(member_scores, norm_mode, norm_eps, bound_tanh)
        arr = np.asarray(normalized, dtype=float)
        means = np.mean(arr, axis=0).tolist()
        variances = np.var(arr, axis=0, ddof=0).tolist()
        return means, variances, normalized

    # ── Abstract interface ────────────────────────────────────────────

    @abstractmethod
    def evaluate_methods(
        self,
        candidates: List[str],
        means: List[float],
        variances: List[float],
        member_scores: List[List[float]],
        rm_names: List[str],
        dataset: str,
        rng: np.random.Generator,
    ) -> Dict[str, str]:
        """Evaluate this hypothesis's methods on a single example.

        Args:
            candidates: List of candidate response texts.
            means: Ensemble mean scores per candidate.
            variances: Ensemble variance per candidate.
            member_scores: Raw per-RM scores, shape (K, N).
            rm_names: Names of the K reward models.
            dataset: Dataset name for answer extraction.
            rng: Random number generator for stochastic methods.

        Returns:
            Dict mapping method_name -> selected_candidate_text.
        """

    # ── Concrete template run ─────────────────────────────────────────

    def run(
        self,
        candidates_path: str = "",
        score_paths: str = "",
        eval_n: str = "4,8,16,24",
        variance_norm_mode: str = "robust_zscore",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the full evaluation: N-scaling."""
        examples = self.load_candidates(candidates_path)
        rm_names, all_rm_scores = self.load_scores(score_paths)
        dataset = self.detect_dataset(candidates_path)
        n_values = [int(x.strip()) for x in eval_n.split(",")]
        max_n = max(n_values)

        print(f"[{self.name}] Running on {dataset}: N = {n_values}")
        print(f"[{self.name}] {len(examples)} examples, {len(rm_names)} RMs: {rm_names}")

        results_by_n: Dict[str, Dict[str, float]] = {}
        oracle_by_n: Dict[str, float] = {}
        normalized_by_n: Dict[str, Dict[str, float]] = {}

        for n in sorted(n_values):
            print(f"  Evaluating N={n}...")
            correct: Dict[str, int] = defaultdict(int)
            total: Dict[str, int] = defaultdict(int)
            n_valid = 0      # questions actually evaluated at this N
            n_solvable = 0   # questions with >=1 correct candidate (oracle@N hits)

            rng = np.random.default_rng(42)

            for ex in examples:
                idx = ex["example_index"]
                if idx not in all_rm_scores or len(all_rm_scores[idx]) != len(rm_names):
                    continue

                candidates = ex["candidates"][:n]
                gold = ex["ground_truth"]
                member_scores_n = [s[:n] for s in all_rm_scores[idx]]

                # Oracle@N for this question: is the answer reachable at all?
                # (Denominator of the oracle-normalized score below.)
                n_valid += 1
                if any(check_correctness(c, gold, dataset) for c in candidates):
                    n_solvable += 1

                means, variances, _norm = self.ensemble_stats(
                    member_scores_n, norm_mode=variance_norm_mode
                )

                # --- Individual RM baselines (always included) ---
                for j, rm in enumerate(rm_names):
                    m_rm = run_vanilla_bon(candidates, member_scores_n[j], variances)
                    method_name = f"vanilla_bon ({rm})"
                    total[method_name] += 1
                    if check_correctness(m_rm.selected_answer, gold, dataset):
                        correct[method_name] += 1

                # --- Ensemble baseline ---
                m_ens = run_vanilla_bon(candidates, means, variances)
                total["vanilla_bon (ensemble)"] += 1
                if check_correctness(m_ens.selected_answer, gold, dataset):
                    correct["vanilla_bon (ensemble)"] += 1

                # --- Hypothesis-specific methods ---
                selections = self.evaluate_methods(
                    candidates, means, variances, member_scores_n,
                    rm_names, dataset, rng,
                )
                for method_name, selected_text in selections.items():
                    total[method_name] += 1
                    if check_correctness(selected_text, gold, dataset):
                        correct[method_name] += 1

            results_by_n[str(n)] = {m: correct[m] / total[m] for m in total if total[m] > 0}
            # Oracle@N and the oracle-normalized accuracy (a.k.a. coverage
            # utilisation): of the questions that *can* be answered correctly
            # (>=1 correct candidate exists), what fraction did this method get?
            # ONA = accuracy / Oracle@N, in [0, 1]; a perfect selector scores 1.0.
            oracle_by_n[str(n)] = (n_solvable / n_valid) if n_valid > 0 else 0.0
            normalized_by_n[str(n)] = {
                m: (correct[m] / n_solvable if n_solvable > 0 else 0.0)
                for m in total if total[m] > 0
            }

        return {
            "hypothesis": self.name,
            "dataset": dataset,
            "n_values": n_values,
            "n_examples": len(examples),
            "rm_names": rm_names,
            "variance_norm_mode": variance_norm_mode,
            "results_by_n": results_by_n,
            "oracle_by_n": oracle_by_n,
            "normalized_by_n": normalized_by_n,
            "overall": results_by_n.get(str(max_n), {}),
            "overall_normalized": normalized_by_n.get(str(max_n), {}),
        }

    # ── Plotting ──────────────────────────────────────────────────────

    def plot(self, results: Dict[str, Any], manager: ResultsManager) -> List[Path]:
        """Generate standard plots: N-scaling."""
        dataset = results["dataset"].upper()
        n_values = results["n_values"]
        results_by_n = results["results_by_n"]
        paths: List[Path] = []

        # Determine which methods to plot (skip individual RMs for cleanliness)
        all_methods = list(results_by_n[str(n_values[-1])].keys())
        plot_methods = [m for m in all_methods if not m.startswith("vanilla_bon (") or m == "vanilla_bon (ensemble)"]

        # --- Accuracy vs N ---
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in plot_methods:
            style = METHOD_STYLE.get(method, {"color": "#7f8c8d", "marker": "o", "ls": "-"})
            xs, ys = [], []
            for n in sorted(n_values):
                val = results_by_n.get(str(n), {}).get(method)
                if val is not None:
                    xs.append(n)
                    ys.append(val)
            if xs:
                ax.plot(xs, ys, label=method, color=style["color"],
                        marker=style["marker"], linestyle=style["ls"],
                        markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xlabel("Number of Candidates (N)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs N — {dataset}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.legend(loc="best", framealpha=0.9, fontsize=8)
        fig.tight_layout()
        paths.append(manager.save_plot(fig, "accuracy_vs_n"))
        plt.close(fig)

        return paths

    # ── Standard save ─────────────────────────────────────────────────

    def make_experiment_name(self, dataset: str) -> str:
        """Generate a stable experiment name (no timestamp — overwrites on re-run)."""
        return f"{self.name}_{dataset}"

    def save(
        self,
        results: Dict[str, Any],
        dataset: str,
        config: Optional[Any] = None,
        experiment_name: Optional[str] = None,
    ) -> ResultsManager:
        """Save results using the standard directory layout.

        Uses a stable directory name so re-runs overwrite previous results.
        """
        exp_name = experiment_name or self.make_experiment_name(dataset)
        output_dir = Path("runs") / exp_name
        manager = ResultsManager(output_dir)

        if config is not None:
            manager.save_config(config)

        manager.save_results(results)
        plot_paths = []

        summary_text = self._build_summary(results, plot_paths)
        manager.save_summary(summary_text)

        print(f"\n{'=' * 60}")
        print(f"  Results saved to: {output_dir}")
        print(f"  ├── results.json")
        print(f"  ├── summary.txt")
        if plot_paths:
            print(f"  └── plots/ ({len(plot_paths)} plots)")
        print(f"{'=' * 60}\n")

        return manager

    def _build_summary(self, results: Dict[str, Any], plot_paths: List[Path]) -> str:
        """Build a human-readable summary string."""
        lines = [
            f"Pipeline: {self.name}",
            f"Description: {self.description}",
            f"Category: {self.category}",
            f"Timestamp: {datetime.now().isoformat()}",
            "",
        ]

        # N-sweep results
        if "results_by_n" in results:
            oracle_by_n = results.get("oracle_by_n", {})
            normalized_by_n = results.get("normalized_by_n", {})
            lines.append("Results by N  (Acc = accuracy;  ONA = oracle-normalized = Acc / Oracle@N):")
            lines.append("-" * 70)
            for n_val, methods in sorted(results["results_by_n"].items(), key=lambda x: int(x[0])):
                orc = oracle_by_n.get(n_val)
                header = f"\n  N = {n_val}"
                if orc is not None:
                    header += f"   (Oracle@N = {orc:.1%})"
                lines.append(header + ":")
                norm = normalized_by_n.get(n_val, {})
                for method, acc in methods.items():
                    ona = norm.get(method)
                    avg_budget = results.get("average_budget", {}).get(method)
                    
                    line_str = f"    {method:<40s}  Acc {acc:>6.1%}"
                    if ona is not None:
                        line_str += f"   ONA {ona:>6.1%}"
                    if avg_budget is not None:
                        line_str += f"   BUG {avg_budget['mean']:>5.2f}+-{avg_budget['std']:<4.2f}"
                    
                    lines.append(line_str)
            lines.append("")

        if plot_paths:
            lines.append("Generated Plots:")
            for p in plot_paths:
                lines.append(f"  - {p.name}")

        return "\n".join(lines)
