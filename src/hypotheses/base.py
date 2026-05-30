"""Abstract base class for hypothesis experiments.

Every hypothesis module defines a single class inheriting from ``Hypothesis``
and implements ``run()`` and ``plot()``.  The base class provides a standard
``save()`` method that writes results to the canonical directory layout.
"""

from __future__ import annotations

import glob
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.normalize import normalize_member_scores
from ..core.results import ResultsManager


class Hypothesis(ABC):
    """Base class for all experiment hypotheses.

    Subclasses must set the three class-level attributes and implement
    ``run()`` and ``plot()``.

    Attributes:
        name: Machine-readable identifier, e.g. ``h1_reward_hacking_at_scale``.
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
        """Load RM score files matching a glob pattern.

        Args:
            score_paths: Glob pattern or comma-separated paths to score JSONL files.

        Returns:
            ``(rm_names, all_rm_scores)`` where ``all_rm_scores[example_idx]``
            is a list of K score vectors (one per RM).
        """
        # Resolve paths
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
        return "unknown"

    @staticmethod
    def ensemble_stats(
        member_scores: List[List[float]],
        norm_mode: str = "robust_zscore",
        norm_eps: float = 1e-6,
        bound_tanh: bool = False,
    ) -> tuple[List[float], List[float], List[List[float]]]:
        """Compute ensemble mean/variance from member scores.

        Args:
            member_scores: ``(K, N)`` raw scores.
            norm_mode: Normalization strategy.
            norm_eps: Stability epsilon.
            bound_tanh: Apply tanh bounding.

        Returns:
            ``(means, variances, normalized_member_scores)``
        """
        normalized = normalize_member_scores(member_scores, norm_mode, norm_eps, bound_tanh)
        arr = np.asarray(normalized, dtype=float)
        means = np.mean(arr, axis=0).tolist()
        variances = np.var(arr, axis=0, ddof=0).tolist()
        return means, variances, normalized

    # ── Abstract interface ────────────────────────────────────────────

    @abstractmethod
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the hypothesis experiment.

        Each hypothesis defines its own keyword arguments (e.g.
        ``candidates_path``, ``score_paths``, ``eval_n``).

        Returns:
            Structured results dict that will be saved to ``results.json``.
        """

    @abstractmethod
    def plot(self, results: Dict[str, Any], manager: ResultsManager) -> List[Path]:
        """Generate all plots from *results*.

        Args:
            results: The dict returned by ``run()``.
            manager: ``ResultsManager`` for saving plots.

        Returns:
            List of paths to saved plot files.
        """

    # ── Standard save ─────────────────────────────────────────────────

    def make_experiment_name(self, dataset: str) -> str:
        """Generate a timestamped experiment name."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{dataset}_{ts}"

    def save(
        self,
        results: Dict[str, Any],
        dataset: str,
        config: Optional[Any] = None,
        experiment_name: Optional[str] = None,
    ) -> ResultsManager:
        """Save results using the standard directory layout.

        Creates ``runs/{category}/{experiment_name}/`` with config, results,
        summary, and plots.

        Returns:
            The ``ResultsManager`` used for saving.
        """
        exp_name = experiment_name or self.make_experiment_name(dataset)
        output_dir = Path("runs") / self.category / exp_name
        manager = ResultsManager(output_dir)

        # Save config if provided
        if config is not None:
            manager.save_config(config)

        # Save structured results
        manager.save_results(results)

        # Generate and save plots
        plot_paths = self.plot(results, manager)

        # Build and save summary
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
            f"Hypothesis: {self.name}",
            f"Description: {self.description}",
            f"Category: {self.category}",
            f"Timestamp: {datetime.now().isoformat()}",
            "",
        ]

        # Add method-level accuracy table if present
        if "accuracy_by_method" in results:
            lines.append("Method Accuracies:")
            lines.append("-" * 50)
            for method, acc in results["accuracy_by_method"].items():
                lines.append(f"  {method:<35s}  {acc:>6.1%}")
            lines.append("")

        # Add N-sweep results if present
        if "results_by_n" in results:
            lines.append("Results by N:")
            lines.append("-" * 70)
            for n_val, methods in sorted(results["results_by_n"].items(), key=lambda x: int(x[0])):
                lines.append(f"\n  N = {n_val}:")
                for method, acc in methods.items():
                    lines.append(f"    {method:<35s}  {acc:>6.1%}")
            lines.append("")

        # List generated plots
        if plot_paths:
            lines.append("Generated Plots:")
            for p in plot_paths:
                lines.append(f"  - {p.name}")

        return "\n".join(lines)
