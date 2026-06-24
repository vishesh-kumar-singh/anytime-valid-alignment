"""Pipeline module registry.

Each pipeline is a self-contained experiment module implementing
the ``EvaluationPipeline`` interface from ``src.pipelines.base``.

Usage::

    from src.pipelines import get_pipeline, list_pipelines

    # List available pipelines
    for pid, desc in list_pipelines().items():
        print(f"{pid}: {desc}")

    # Run one
    p = get_pipeline("sprt_rwsc")
    results = p.run(candidates_path="...", score_paths="...")
"""

from __future__ import annotations

from typing import Dict, Type

from .base import EvaluationPipeline

# Lazy imports to avoid loading all pipeline code at import time.
_REGISTRY: Dict[str, str] = {
    "rwsc": "src.pipelines.rwsc.RWSCPipeline",


    "sprt_rwsc": "src.pipelines.sprt.SPRTPipeline",
    "seq_baselines": "src.pipelines.seq_baselines.SeqBaselinesPipeline",
    "baselines": "src.pipelines.parallel_baselines.ParallelBaselinesPipeline",
}


def _import_class(dotted_path: str) -> Type[EvaluationPipeline]:
    """Import a class from a dotted path like ``pkg.mod.ClassName``."""
    module_path, cls_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def get_pipeline(pipeline_id: str) -> EvaluationPipeline:
    """Instantiate a pipeline by its short ID (e.g. ``sprt_rwsc``)."""
    if pipeline_id not in _REGISTRY:
        raise KeyError(
            f"Unknown pipeline: {pipeline_id!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    cls = _import_class(_REGISTRY[pipeline_id])
    return cls()


def list_pipelines() -> Dict[str, str]:
    """Return ``{id: description}`` for all registered pipelines."""
    result: Dict[str, str] = {}
    for pid, dotted_path in _REGISTRY.items():
        try:
            cls = _import_class(dotted_path)
            result[pid] = cls.description
        except Exception as e:
            result[pid] = f"(error loading: {e})"
    return result
