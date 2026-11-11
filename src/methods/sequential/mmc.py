"""
Martingale Majority Certificate (MMC) stopping rule.
Official implementation adapted directly from Cordero-Encinar & Duncan (2025):
"Certified Self-Consistency: Statistical Guarantees and Test-Time Training for Reliable Reasoning in LLMs"
(https://github.com/paulaoak/certified_self_consistency)
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional, Tuple, List
import numpy as np


# ---------- Math helpers ----------
def logsumexp(log_terms: List[float]) -> float:
    """Stable log-sum-exp."""
    if not log_terms:
        return float("-inf")
    m = max(log_terms)
    if math.isinf(m):
        return m
    s = sum(math.exp(x - m) for x in log_terms)
    return m + math.log(s)


def make_trunc_beta_grid(
    a: float = 0.5, b: float = 0.5, K: int = 41, lo: float = 0.5001, hi: float = 0.9999
) -> Tuple[List[float], List[float]]:
    """
    Discrete grid q_k in (lo, hi) with weights proportional to Beta(a,b) pdf, normalized.
    Truncated to (1/2, 1] because we test one-sided q>1/2.
    """
    qs = [lo + (hi - lo) * k / (K - 1) for k in range(K)]
    raw = [(q ** (a - 1.0)) * ((1.0 - q) ** (b - 1.0)) for q in qs]
    Z = sum(raw)
    ws = [r / Z for r in raw]
    return qs, ws


# ---------- Bernoulli mixture e-process (Dirac / Point Prior) ----------
@dataclass
class BernoulliMixtureEProcess:
    """
    E-process for testing H0: q <= 1/2 vs H1: q > 1/2 on a Bernoulli subsequence.
    Uses a dirac/point prior.
    """
    s: int = 0
    f: int = 0
    qdirac: float = 0.0
    log_e: float = 0.0  # log E-value (starts at log 1)

    def update(self, y: int) -> float:
        """Update with y in {0,1}. Returns new log e-value. Ignores other inputs."""
        aux = (self.s + 1/2) / (self.s + self.f + 1)
        self.qdirac = np.clip(aux, 0.50001, 0.99999)
        if y not in (0, 1):
            return self.log_e
        self.s += y
        self.f += (1 - y)
        m = self.s + self.f
        # e_n = 2^m * qdirac^s * (1-qdirac)^f
        self.log_e = m * math.log(2.0) + self.s * math.log(self.qdirac) + self.f * math.log1p(-self.qdirac)
        return self.log_e

    def num_updates(self) -> int:
        return self.s + self.f

    def reset(self):
        self.s = 0
        self.f = 0
        self.log_e = 0.0


# ---------- Bernoulli mixture e-process (Beta Grid Prior) ----------
@dataclass
class BernoulliMixtureEProcessBeta:
    """
    E-process for testing H0: q <= 1/2 vs H1: q > 1/2 on a Bernoulli subsequence.
    Uses a discrete Beta(a,b) mixture truncated to (1/2,1].
    """
    qs: List[float]
    ws: List[float]
    s: int = 0
    f: int = 0
    log_e: float = 0.0

    def update(self, y: int) -> float:
        """Update with y in {0,1}. Returns new log e-value. Ignores other inputs."""
        if y not in (0, 1):
            return self.log_e
        self.s += y
        self.f += (1 - y)
        m = self.s + self.f
        # e_n = 2^m * sum_k w_k * q_k^s * (1-q_k)^f
        log_terms = [
            math.log(w) + self.s * math.log(q) + self.f * math.log1p(-q)
            for (q, w) in zip(self.qs, self.ws)
        ]
        self.log_e = m * math.log(2.0) + logsumexp(log_terms)
        return self.log_e

    def num_updates(self) -> int:
        return self.s + self.f

    def reset(self):
        self.s = 0
        self.f = 0
        self.log_e = 0.0


# ---------- Plurality decider with Point Prior (Official MMC) ----------
class PluralityEAndDeciderPrior:
    """
    Two e-processes with predictable selection:
      * Pairwise: leader vs runner-up on {i, j} subsequence.
      * Other:   leader vs Other (anything not in a tiny contender set C) on {i, Other}.
    Stop only when BOTH log E-values individually exceed log(1/delta) and each
    subsequence has at least a few updates (min updates guards). Optional check
    requires the current leader after the vote equals the previously certified i.
    """
    def __init__(
        self,
        labels: Optional[Iterable[Any]] = None,
        M: int = 2,
        delta: float = 0.1,
        N_max: int = 32,
        min_pair_updates: int = 1,
        min_other_updates: int = 1,
        require_current_leader_match: bool = False,
    ):
        self.counts = Counter()
        self.labels = list(labels) if labels is not None else []
        self.M = int(M)
        self.delta = float(delta)
        self.N_max = int(N_max)
        self.log_bar = math.log(1.0 / self.delta)
        self.min_pair_updates = int(min_pair_updates)
        self.min_other_updates = int(min_other_updates)
        self.require_current_leader_match = bool(require_current_leader_match)

        self.pair_E = BernoulliMixtureEProcess()
        self.other_E = BernoulliMixtureEProcess()

        self.prev_C: List[Any] = []
        self.prev_i: Any = None
        self.prev_j: Any = None

        self.n = 0
        self.stopped = False
        self.abstained = False
        self.winner = None

    def _top_M_from_counts(self) -> List[Any]:
        items = list(self.counts.items())
        for lab in self.labels:
            if lab not in self.counts:
                items.append((lab, 0))
        items.sort(key=lambda kv: (-kv[1], str(kv[0])))
        return [k for k, _ in items[: max(1, self.M)]]

    def _choose_predictable_sets(self) -> Tuple[List[Any], Any, Any]:
        C = self._top_M_from_counts()
        if not C:
            return [], None, None
        C_sorted = sorted(C, key=lambda k: (-self.counts[k], str(k)))
        i = C_sorted[0]
        j = C_sorted[1] if len(C_sorted) >= 2 else None
        return C, i, j

    def on_vote(self, label: Any) -> Dict[str, Any]:
        """Ingest one vote and update the decision state; returns diagnostics."""
        if self.stopped or self.abstained:
            return self.status()

        # 1) Predictable selection from t-1 (to ensure martingale property)
        C_tm1, i_tm1, j_tm1 = self._choose_predictable_sets()
        self.prev_C, self.prev_i, self.prev_j = C_tm1, i_tm1, j_tm1

        # 2) Observe vote & update counts
        self.counts[label] += 1
        if label not in self.labels:
            self.labels.append(label)
        self.n += 1

        # 3) Update e-processes on their subsequences
        # Pairwise on {i, j}
        logE_pair = self.pair_E.log_e
        if i_tm1 is not None and j_tm1 is not None:
            if label == i_tm1:
                logE_pair = self.pair_E.update(1)
            elif label == j_tm1:
                logE_pair = self.pair_E.update(0)

        # Other on {i, Other}
        logE_other = self.other_E.log_e
        if i_tm1 is not None:
            in_C_prev = label in (self.prev_C or [])
            if label == i_tm1:
                logE_other = self.other_E.update(1)
            elif not in_C_prev:
                logE_other = self.other_E.update(0)

        # 4) AND rule + min updates + (optional) current-leader check
        cross_pair = (logE_pair >= self.log_bar) and (self.pair_E.num_updates() >= self.min_pair_updates)
        cross_other = (logE_other >= self.log_bar) and (self.other_E.num_updates() >= self.min_other_updates)
        crossed = cross_pair and cross_other

        if crossed and (self.prev_i is not None) and self.require_current_leader_match:
            current_leader = max(self.counts, key=lambda k: (self.counts[k], -ord(str(k)[0])))
            crossed = (current_leader == self.prev_i)

        if crossed:
            self.stopped = True
            self.winner = self.prev_i
        elif self.n >= self.N_max and not self.stopped:
            self.abstained = True

        return {
            "stopped": self.stopped,
            "abstained": self.abstained,
            "winner": self.winner,
            "n": self.n,
            "prev_C": list(self.prev_C),
            "prev_i": self.prev_i,
            "prev_j": self.prev_j,
            "logE_pair": logE_pair,
            "logE_other": logE_other,
            "log_bar": self.log_bar,
            "pair_updates": self.pair_E.num_updates(),
            "other_updates": self.other_E.num_updates(),
            "cross_pair": cross_pair,
            "cross_other": cross_other,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "stopped": self.stopped,
            "abstained": self.abstained,
            "winner": self.winner,
            "n": self.n,
            "prev_C": list(self.prev_C),
            "prev_i": self.prev_i,
            "prev_j": self.prev_j,
            "logE_pair": self.pair_E.log_e,
            "logE_other": self.other_E.log_e,
            "log_bar": self.log_bar,
            "pair_updates": self.pair_E.num_updates(),
            "other_updates": self.other_E.num_updates(),
            "counts": dict(self.counts),
        }

    def reset(self, delta: Optional[float] = None, N_max: Optional[int] = None):
        if delta is not None:
            self.delta = float(delta)
            self.log_bar = math.log(1.0 / self.delta)
        if N_max is not None:
            self.N_max = int(N_max)
        self.counts.clear()
        self.pair_E.reset()
        self.other_E.reset()
        self.prev_C = []
        self.prev_i = None
        self.prev_j = None
        self.n = 0
        self.stopped = False
        self.abstained = False
        self.winner = None


# ---------- Plurality decider with Beta Prior Mixture ----------
class PluralityEAndDecider:
    """
    Two e-processes with Beta mixture prior.
    """
    def __init__(
        self,
        labels: Optional[Iterable[Any]] = None,
        M: int = 2,
        delta: float = 0.1,
        N_max: int = 32,
        a: float = 0.5,
        b: float = 0.5,
        K: int = 41,
        min_pair_updates: int = 1,
        min_other_updates: int = 1,
        require_current_leader_match: bool = False,
    ):
        self.counts = Counter()
        self.labels = list(labels) if labels is not None else []
        self.M = int(M)
        self.delta = float(delta)
        self.N_max = int(N_max)
        self.log_bar = math.log(1.0 / self.delta)
        self.min_pair_updates = int(min_pair_updates)
        self.min_other_updates = int(min_other_updates)
        self.require_current_leader_match = bool(require_current_leader_match)

        qs, ws = make_trunc_beta_grid(a=a, b=b, K=K)
        self.pair_E = BernoulliMixtureEProcessBeta(qs=qs, ws=ws)
        self.other_E = BernoulliMixtureEProcessBeta(qs=qs, ws=ws)

        self.prev_C: List[Any] = []
        self.prev_i: Any = None
        self.prev_j: Any = None

        self.n = 0
        self.stopped = False
        self.abstained = False
        self.winner = None

    def _top_M_from_counts(self) -> List[Any]:
        items = list(self.counts.items())
        for lab in self.labels:
            if lab not in self.counts:
                items.append((lab, 0))
        items.sort(key=lambda kv: (-kv[1], str(kv[0])))
        return [k for k, _ in items[: max(1, self.M)]]

    def _choose_predictable_sets(self) -> Tuple[List[Any], Any, Any]:
        C = self._top_M_from_counts()
        if not C:
            return [], None, None
        C_sorted = sorted(C, key=lambda k: (-self.counts[k], str(k)))
        i = C_sorted[0]
        j = C_sorted[1] if len(C_sorted) >= 2 else None
        return C, i, j

    def on_vote(self, label: Any) -> Dict[str, Any]:
        if self.stopped or self.abstained:
            return self.status()

        C_tm1, i_tm1, j_tm1 = self._choose_predictable_sets()
        self.prev_C, self.prev_i, self.prev_j = C_tm1, i_tm1, j_tm1

        self.counts[label] += 1
        if label not in self.labels:
            self.labels.append(label)
        self.n += 1

        logE_pair = self.pair_E.log_e
        if i_tm1 is not None and j_tm1 is not None:
            if label == i_tm1:
                logE_pair = self.pair_E.update(1)
            elif label == j_tm1:
                logE_pair = self.pair_E.update(0)

        logE_other = self.other_E.log_e
        if i_tm1 is not None:
            in_C_prev = label in (self.prev_C or [])
            if label == i_tm1:
                logE_other = self.other_E.update(1)
            elif not in_C_prev:
                logE_other = self.other_E.update(0)

        cross_pair = (logE_pair >= self.log_bar) and (self.pair_E.num_updates() >= self.min_pair_updates)
        cross_other = (logE_other >= self.log_bar) and (self.other_E.num_updates() >= self.min_other_updates)
        crossed = cross_pair and cross_other

        if crossed and (self.prev_i is not None) and self.require_current_leader_match:
            current_leader = max(self.counts, key=lambda k: (self.counts[k], -ord(str(k)[0])))
            crossed = (current_leader == self.prev_i)

        if crossed:
            self.stopped = True
            self.winner = self.prev_i
        elif self.n >= self.N_max and not self.stopped:
            self.abstained = True

        return {
            "stopped": self.stopped,
            "abstained": self.abstained,
            "winner": self.winner,
            "n": self.n,
            "prev_C": list(self.prev_C),
            "prev_i": self.prev_i,
            "prev_j": self.prev_j,
            "logE_pair": logE_pair,
            "logE_other": logE_other,
            "log_bar": self.log_bar,
            "pair_updates": self.pair_E.num_updates(),
            "other_updates": self.other_E.num_updates(),
            "cross_pair": cross_pair,
            "cross_other": cross_other,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "stopped": self.stopped,
            "abstained": self.abstained,
            "winner": self.winner,
            "n": self.n,
            "prev_C": list(self.prev_C),
            "prev_i": self.prev_i,
            "prev_j": self.prev_j,
            "logE_pair": self.pair_E.log_e,
            "logE_other": self.other_E.log_e,
            "log_bar": self.log_bar,
            "pair_updates": self.pair_E.num_updates(),
            "other_updates": self.other_E.num_updates(),
            "counts": dict(self.counts),
        }


# ---------- Unified Standard Stopping Interface ----------
def _mmc_stop(
    candidates: List[str],
    global_cluster_ids: List[int],
    alpha: float = 0.1,
    N_max: Optional[int] = None,
    M: int = 2,
    use_beta_prior: bool = False,
    min_pair_updates: int = 1,
    min_other_updates: int = 1,
    require_current_leader_match: bool = False,
) -> dict:
    """
    Run Martingale Majority Certificate (MMC) sequential stopping rule.

    Args:
        candidates: List of generated response strings.
        global_cluster_ids: Integer cluster ID for each candidate.
        alpha: Statistical certification error tolerance (e.g. 0.1).
        N_max: Maximum budget ceiling. If None, uses len(candidates).
        M: Size of contender set C (default 2, meaning leader and runner-up).
        use_beta_prior: Whether to use truncated Beta mixture prior instead of Point prior.
        min_pair_updates: Minimum updates required on pairwise subsequence before stopping.
        min_other_updates: Minimum updates required on other subsequence before stopping.
        require_current_leader_match: Guard checking current leader matches certified leader.

    Returns:
        dict:
            "stopped_at": int (1 to N_max)
            "leader_answer": int (cluster id of winner)
            "best_idx": int (candidate index corresponding to winner)
            "decision": "accept" (certified) or "fallback" (uncertified majority vote)
    """
    N = len(candidates) if N_max is None else min(len(candidates), N_max)
    unique_labels = list(set(global_cluster_ids[:N]))

    if use_beta_prior:
        decider = PluralityEAndDecider(
            labels=unique_labels,
            M=M,
            delta=alpha,
            N_max=N,
            min_pair_updates=min_pair_updates,
            min_other_updates=min_other_updates,
            require_current_leader_match=require_current_leader_match,
        )
    else:
        decider = PluralityEAndDeciderPrior(
            labels=unique_labels,
            M=M,
            delta=alpha,
            N_max=N,
            min_pair_updates=min_pair_updates,
            min_other_updates=min_other_updates,
            require_current_leader_match=require_current_leader_match,
        )

    stopped_at = N
    winner_cid = None
    is_certified = False

    for t in range(1, N + 1):
        cid = global_cluster_ids[t - 1]
        diag = decider.on_vote(cid)
        if diag["stopped"]:
            stopped_at = t
            winner_cid = diag["winner"]
            is_certified = True
            break
        elif diag["abstained"]:
            stopped_at = t
            break

    # If abstained or winner not set, fall back to empirical majority vote over votes so far
    if winner_cid is None:
        counts = Counter(global_cluster_ids[:stopped_at])
        winner_cid = counts.most_common(1)[0][0]
        decision = "fallback"
    else:
        decision = "accept"

    # Find the candidate index corresponding to the winner cluster
    best_idx = 0
    for i in range(stopped_at):
        if global_cluster_ids[i] == winner_cid:
            best_idx = i
            break

    return {
        "stopped_at": stopped_at,
        "leader_answer": winner_cid,
        "best_idx": best_idx,
        "decision": decision,
        "is_certified": is_certified,
    }
