"""
Experiment Logger — persistent, structured logging for all runs.

Writes a human-readable + machine-parseable log file under logs/ that tracks:
  - Experiment configuration (model, dataset, N, RM ensemble, etc.)
  - Per-example progress with running accuracy
  - Periodic checkpoints (every K examples) with intermediate results
  - Final summary table
  - Errors and crashes with full context
  - Wall-clock timing for each phase

Log files are timestamped and never overwritten, so you always have
a complete history of every experiment attempt.
"""

import json
import logging
import os
import platform
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """Structured experiment logger that writes to both console and file."""

    def __init__(self, log_dir: str = "logs", experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{timestamp}"
        self.log_file = self.log_dir / f"{self.run_id}.log"
        self.metrics_file = self.log_dir / f"{self.run_id}_metrics.jsonl"

        # Setup Python logger
        self.logger = logging.getLogger(self.run_id)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # File handler — everything
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        self.logger.addHandler(fh)

        # Console handler — INFO and above
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        self.logger.addHandler(ch)

        # Tracking state
        self.start_time: Optional[float] = None
        self.total_examples: int = 0
        self.completed_examples: int = 0
        self.correct_by_method: Dict[str, int] = {}
        self.total_by_method: Dict[str, int] = {}
        self.checkpoint_interval: int = 10  # log checkpoint every N examples

        self.logger.info(f"{'=' * 70}")
        self.logger.info(f"  Experiment Logger initialized: {self.run_id}")
        self.logger.info(f"  Log file: {self.log_file}")
        self.logger.info(f"  Metrics:  {self.metrics_file}")
        self.logger.info(f"{'=' * 70}")

    def log_environment(self):
        """Log system environment info."""
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"

        env_info = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu_name,
            "gpu_memory": gpu_mem,
            "hostname": platform.node(),
            "timestamp": datetime.now().isoformat(),
        }
        self.logger.info(f"Environment: GPU={gpu_name} ({gpu_mem}), PyTorch={torch.__version__}")
        self._write_metric("environment", env_info)

    def log_config(self, cfg) -> None:
        """Log experiment configuration."""
        cfg_dict = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else vars(cfg)
        self.logger.info(f"{'─' * 70}")
        self.logger.info("  EXPERIMENT CONFIGURATION")
        self.logger.info(f"{'─' * 70}")
        for key, val in cfg_dict.items():
            self.logger.info(f"  {key}: {val}")
        self.logger.info(f"{'─' * 70}")
        self._write_metric("config", cfg_dict)

    def log_phase_start(self, phase: str, details: str = "") -> None:
        """Log the start of a major phase (model loading, dataset loading, inference)."""
        msg = f"▶ PHASE: {phase}"
        if details:
            msg += f" — {details}"
        self.logger.info(msg)
        self._write_metric("phase_start", {"phase": phase, "details": details})

    def log_phase_end(self, phase: str, duration_sec: float) -> None:
        """Log the end of a major phase."""
        self.logger.info(f"✓ PHASE COMPLETE: {phase} ({duration_sec:.1f}s)")
        self._write_metric("phase_end", {"phase": phase, "duration_sec": duration_sec})

    def log_inference_start(self, total_examples: int, bon_n: int) -> None:
        """Log the start of the inference loop."""
        self.total_examples = total_examples
        self.completed_examples = 0
        self.start_time = time.time()
        self.logger.info(f"{'=' * 70}")
        self.logger.info(f"  INFERENCE START: {total_examples} examples, N={bon_n}")
        self.logger.info(f"{'=' * 70}")
        self._write_metric("inference_start", {
            "total_examples": total_examples,
            "bon_n": bon_n,
        })

    def log_example_done(self, example_idx: int, method_results: List[dict]) -> None:
        """Log completion of a single example across all methods."""
        self.completed_examples += 1
        elapsed = time.time() - self.start_time if self.start_time else 0
        pct = (self.completed_examples / self.total_examples * 100) if self.total_examples else 0

        # Track running accuracy per method
        for r in method_results:
            method = r.get("method", "unknown")
            self.total_by_method[method] = self.total_by_method.get(method, 0) + 1
            if r.get("is_correct"):
                self.correct_by_method[method] = self.correct_by_method.get(method, 0) + 1

        # Estimate remaining time
        if self.completed_examples > 0 and elapsed > 0:
            rate = elapsed / self.completed_examples
            remaining = rate * (self.total_examples - self.completed_examples)
            eta_str = _format_duration(remaining)
        else:
            eta_str = "?"

        # Log brief progress every example (DEBUG level, only goes to file)
        self.logger.debug(
            f"  [{self.completed_examples}/{self.total_examples}] "
            f"({pct:.0f}%) Example {example_idx} done | "
            f"Elapsed: {_format_duration(elapsed)} | ETA: {eta_str}"
        )

        # Log detailed checkpoint every N examples (INFO level, goes to console too)
        if self.completed_examples % self.checkpoint_interval == 0 or self.completed_examples == self.total_examples:
            self._log_checkpoint(elapsed, eta_str)

    def _log_checkpoint(self, elapsed: float, eta_str: str) -> None:
        """Log a detailed checkpoint with running accuracy for all methods."""
        pct = (self.completed_examples / self.total_examples * 100) if self.total_examples else 0

        self.logger.info(f"")
        self.logger.info(f"  ── CHECKPOINT [{self.completed_examples}/{self.total_examples}] ({pct:.0f}%) ──")
        self.logger.info(f"  Elapsed: {_format_duration(elapsed)} | ETA: {eta_str}")

        for method in sorted(self.total_by_method.keys()):
            total = self.total_by_method[method]
            correct = self.correct_by_method.get(method, 0)
            acc = correct / total if total > 0 else 0
            self.logger.info(f"    {method:<32s}  {acc:>6.1%}  ({correct}/{total})")

        self.logger.info(f"  {'─' * 50}")

        checkpoint_data = {
            "completed": self.completed_examples,
            "total": self.total_examples,
            "elapsed_sec": elapsed,
            "eta": eta_str,
            "running_accuracy": {
                m: self.correct_by_method.get(m, 0) / self.total_by_method[m]
                for m in self.total_by_method
            },
        }
        self._write_metric("checkpoint", checkpoint_data)

    def log_inference_end(self, total_rows: int) -> None:
        """Log the end of inference."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info(f"{'=' * 70}")
        self.logger.info(f"  INFERENCE COMPLETE")
        self.logger.info(f"  Total examples: {self.completed_examples}")
        self.logger.info(f"  Total rows:     {total_rows}")
        self.logger.info(f"  Wall time:      {_format_duration(elapsed)}")
        self.logger.info(f"{'=' * 70}")
        self._write_metric("inference_end", {
            "total_examples": self.completed_examples,
            "total_rows": total_rows,
            "wall_time_sec": elapsed,
        })

    def log_summary(self, summary: dict) -> None:
        """Log the final summary dict."""
        self._write_metric("summary", summary)
        self.logger.info("Final summary saved to metrics file.")

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error with full context."""
        import traceback
        tb = traceback.format_exc()
        self.logger.error(f"ERROR during {context}: {error}")
        self.logger.error(tb)
        self._write_metric("error", {
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": tb,
        })

    def log_output_saved(self, path: str) -> None:
        """Log that an output file was saved."""
        self.logger.info(f"  Output saved: {path}")
        self._write_metric("output_saved", {"path": path})

    def _write_metric(self, event: str, data: Any) -> None:
        """Append a structured metric line to the JSONL metrics file."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data,
        }
        try:
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Never crash on logging failure


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string like '1h 23m 45s'."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"
