"""Open-ended (alignment) evaluation: LLM-judge win-rate.

Complements the verifiable (math) grading in ``src.core.data`` with a
preference-based gold signal for open-ended instruction following, as used in
the URM paper's AlpacaEval arm.
"""

from .llm_judge import GeminiJudge, AllKeysExhausted

__all__ = ["GeminiJudge", "AllKeysExhausted"]
