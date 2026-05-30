"""Result management for experiment outputs.

Provides:
  - ``MethodResult``   — per-example, per-method outcome record
  - ``ResultsManager`` — standardised I/O for runs/{category}/{experiment}/
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MethodResult:
    """Outcome of applying a single selection method to one example."""

    method: str
    selected_answer: str
    reward_mean: float
    reward_variance: float
    score: float
    n_candidates: int
    n_rm_calls: int
    latency_sec: float


class ResultsManager:
    """Manages the output directory for a single experiment run.

    Directory layout::

        runs/{category}/{experiment_name}/
        ├── config.json
        ├── results.json
        ├── raw_results.jsonl
        ├── summary.txt
        └── plots/
            ├── accuracy_vs_n.png
            └── ...

    Usage::

        mgr = ResultsManager("runs/parallel/h1_gsm8k_20260530_120000")
        mgr.save_config(cfg)
        mgr.append_row(row_dict)
        mgr.save_results({"accuracy": 0.92, ...})
        mgr.save_summary("Method | Accuracy\\n...")
        mgr.save_plot(fig, "accuracy_vs_n")
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._plots_dir = self.output_dir / "plots"

    # ── Config ────────────────────────────────────────────────────────

    def save_config(self, config: Any) -> Path:
        """Serialise experiment config to ``config.json``."""
        path = self.output_dir / "config.json"
        cfg_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config)
        with path.open("w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2, default=str)
        return path

    # ── Structured results ────────────────────────────────────────────

    def save_results(self, results: Dict[str, Any]) -> Path:
        """Save the final structured results dict to ``results.json``."""
        path = self.output_dir / "results.json"
        results["_saved_at"] = datetime.now().isoformat()
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        return path

    def load_results(self) -> Dict[str, Any]:
        """Load previously saved ``results.json``."""
        path = self.output_dir / "results.json"
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ── Raw per-example rows ──────────────────────────────────────────

    def append_row(self, row: Dict[str, Any]) -> None:
        """Append a single per-example result row to ``raw_results.jsonl``."""
        path = self.output_dir / "raw_results.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def append_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Append multiple rows in a single file-open."""
        path = self.output_dir / "raw_results.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def load_raw_results(self) -> List[Dict[str, Any]]:
        """Read all rows from ``raw_results.jsonl``."""
        path = self.output_dir / "raw_results.jsonl"
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    # ── Human-readable summary ────────────────────────────────────────

    def save_summary(self, text: str) -> Path:
        """Write a human-readable summary to ``summary.txt``."""
        path = self.output_dir / "summary.txt"
        with path.open("w", encoding="utf-8") as f:
            f.write(text)
        return path

    # ── Plots ─────────────────────────────────────────────────────────

    def save_plot(self, fig: Any, name: str, dpi: int = 300) -> Path:
        """Save a matplotlib figure to ``plots/{name}.png``.

        Args:
            fig: A ``matplotlib.figure.Figure`` object.
            name: Filename stem (without extension).
            dpi: Resolution.

        Returns:
            Path to the saved image.
        """
        self._plots_dir.mkdir(parents=True, exist_ok=True)
        path = self._plots_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    @property
    def plots_dir(self) -> Path:
        """Return the path to the plots subdirectory."""
        self._plots_dir.mkdir(parents=True, exist_ok=True)
        return self._plots_dir
