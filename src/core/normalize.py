"""Cross-model variance normalization for RM ensembles.

When multiple reward models score the same set of candidates, their raw
score scales often differ by orders of magnitude.  Normalization ensures
that ensemble mean/variance statistics are meaningful.

Supported modes:
  - ``none``          — raw scores, no transformation
  - ``zscore``        — per-model (score − mean) / std
  - ``robust_zscore`` — per-model (score − median) / (1.4826 × MAD)
  - ``minmax``        — per-model (score − min) / (max − min)

Optionally applies ``tanh`` bounding to map to ``[-1, 1]`` after normalization.
"""

from __future__ import annotations

from typing import List

import numpy as np


def normalize_member_scores(
    member_scores: List[List[float]],
    mode: str = "none",
    eps: float = 1e-6,
    bound_tanh: bool = False,
) -> List[List[float]]:
    """Normalize each RM member's candidate scores independently.

    Args:
        member_scores: Shape ``(K, N)`` where K = models, N = candidates.
        mode: Normalization strategy — ``none | zscore | robust_zscore | minmax``.
        eps: Stability term for scale denominators.
        bound_tanh: Map normalised values to ``[-1, 1]`` via ``tanh``.

    Returns:
        Normalized scores with the same ``(K, N)`` shape.
    """
    norm_mode = mode.lower().strip()
    arr = np.asarray(member_scores, dtype=float)

    if norm_mode == "none":
        return arr.tolist()

    if arr.ndim != 2:
        raise ValueError(f"member_scores must be rank-2 (K, N), got shape={arr.shape}")

    normalized = np.zeros_like(arr, dtype=float)

    for k in range(arr.shape[0]):
        row = arr[k]

        if norm_mode == "zscore":
            center = float(np.mean(row))
            scale = float(np.std(row, ddof=0))
            if scale < eps:
                normalized[k] = np.zeros_like(row)
            else:
                normalized[k] = (row - center) / (scale + eps)

        elif norm_mode == "robust_zscore":
            center = float(np.median(row))
            mad = float(np.median(np.abs(row - center)))
            robust_scale = 1.4826 * mad
            if robust_scale < eps:
                std = float(np.std(row, ddof=0))
                if std < eps:
                    normalized[k] = np.zeros_like(row)
                else:
                    normalized[k] = (row - center) / (std + eps)
            else:
                normalized[k] = (row - center) / (robust_scale + eps)

        elif norm_mode == "minmax":
            lo = float(np.min(row))
            hi = float(np.max(row))
            span = hi - lo
            if span < eps:
                normalized[k] = np.zeros_like(row)
            else:
                normalized[k] = (row - lo) / (span + eps)

        else:
            raise ValueError(
                f"Unknown normalization mode: {mode!r}. "
                "Use none, zscore, robust_zscore, or minmax."
            )

    if bound_tanh:
        normalized = np.tanh(normalized)

    return normalized.tolist()
