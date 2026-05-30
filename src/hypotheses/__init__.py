"""Hypothesis module registry.

Each hypothesis is a self-contained experiment module implementing
the ``Hypothesis`` interface from ``src.hypotheses.base``.

Usage::

    from src.hypotheses import get_hypothesis, list_hypotheses

    # List available hypotheses
    for hid, desc in list_hypotheses().items():
        print(f"{hid}: {desc}")

    # Run one
    h = get_hypothesis("h5")
    results = h.run(candidates_path="...", score_paths="...")
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Hypothesis

# Lazy imports to avoid loading all hypothesis code at import time.
_REGISTRY: Dict[str, str] = {
    "h1": "src.hypotheses.h1_reward_hacking_at_scale.H1RewardHackingAtScale",
    "h6": "src.hypotheses.h6_self_consistency.H6SelfConsistency",
    "h9": "src.hypotheses.h9_difficulty_stratified.H9DifficultyStratified",
}


def _import_class(dotted_path: str) -> Type[Hypothesis]:
    """Import a class from a dotted path like ``pkg.mod.ClassName``."""
    module_path, cls_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def get_hypothesis(hypothesis_id: str) -> Hypothesis:
    """Instantiate a hypothesis by its short ID (e.g. ``h1``)."""
    if hypothesis_id not in _REGISTRY:
        raise KeyError(
            f"Unknown hypothesis: {hypothesis_id!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    cls = _import_class(_REGISTRY[hypothesis_id])
    return cls()


def list_hypotheses() -> Dict[str, str]:
    """Return ``{id: description}`` for all registered hypotheses."""
    result: Dict[str, str] = {}
    for hid, dotted_path in _REGISTRY.items():
        try:
            cls = _import_class(dotted_path)
            result[hid] = cls.description
        except Exception as e:
            result[hid] = f"(error loading: {e})"
    return result
