"""
MATH500 dataset loading and answer extraction.

MATH500 is a 500-problem subset of the MATH competition dataset,
preserving topic distribution. Used in the RF-SeqBoN paper as a
primary evaluation benchmark.

Subjects: algebra, number_theory, counting_and_probability,
          geometry, intermediate_algebra, prealgebra, precalculus
"""

import re
from typing import List, Optional




def load_math500(
    max_examples: Optional[int] = None,
) -> List[dict]:
    """Load the MATH500 dataset.

    Uses the HuggingFace 'HuggingFaceH4/MATH-500' dataset.

    Returns:
        List of dicts with keys: question, answer, subject, level
    """
    from datasets import DownloadConfig, load_dataset

    try:
        # Prefer local cache first to avoid Hub SSL checks in offline environments.
        ds = load_dataset(
            "HuggingFaceH4/MATH-500",
            split="test",
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception as e:
        try:
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        except Exception:
            raise RuntimeError(
                "Failed to load MATH500 from HuggingFace and no local cache was found. "
                "Your logs show SSL certificate verification errors for huggingface.co. "
                "Fix system/conda CA certificates or pre-download MATH-500 once on a network "
                "with valid SSL, then retry."
            ) from e

    examples = []
    for i, item in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        examples.append({
            "question": item["problem"],
            "answer": item["answer"],
            "subject": item.get("subject", ""),
            "level": item.get("level", ""),
        })

    return examples


def extract_math_answer(answer_text: str) -> Optional[str]:
    """Extract answer from MATH-format text.

    MATH answers are boxed: \\boxed{answer}
    Model outputs might use #### marker like GSM8K.
    """
    if not answer_text:
        return None

    # Try \\boxed{...} first (handles nested braces)
    boxed_start = answer_text.rfind("\\boxed{")
    if boxed_start != -1:
        depth = 0
        start = boxed_start + len("\\boxed{")
        for i in range(start, len(answer_text)):
            if answer_text[i] == "{":
                depth += 1
            elif answer_text[i] == "}":
                if depth == 0:
                    return answer_text[start:i].strip().replace('\n', '').replace('\r', '')
                depth -= 1

    # Try #### marker
    if "####" in answer_text:
        after = answer_text.split("####")[-1].strip()
        cleaned = re.sub(r"[,$%]", "", after).strip()
        match = re.search(r"-?[\d,]+\.?\d*", cleaned)
        if match:
            return match.group(0).replace(",", "")

    # Try common prefixes for unboxed equations
    lower_text = answer_text.lower()
    prefixes = ["the final answer is", "the answer is", "so the answer is", "final answer:"]
    for prefix in prefixes:
        idx = lower_text.rfind(prefix)
        if idx != -1:
            ans = answer_text[idx + len(prefix):].strip()
            # Remove trailing punctuation often found at the end of sentences
            if ans.endswith('.') or ans.endswith(';'):
                ans = ans[:-1].strip()
            # If the extracted answer is non-empty, return it
            if ans:
                return ans.replace('\n', '').replace('\r', '')

    # Try to extract the last math block $...$ or \[...\]
    math_blocks = re.findall(r"\$([^\$]+)\$", answer_text)
    if math_blocks:
        return math_blocks[-1].strip().replace('\n', '').replace('\r', '')
        
    math_blocks = re.findall(r"\\\[(.*?)\\\]", answer_text, re.DOTALL)
    if math_blocks:
        return math_blocks[-1].strip().replace('\n', '').replace('\r', '')

    # Fallback: last number
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")

    # Fallback: return cleaned text
    return answer_text.strip().replace('\n', '').replace('\r', '')


def check_math_correct(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth for MATH problems.

    Handles both numeric and symbolic answers.
    """
    pred = extract_math_answer(prediction)
    # For ground truth: extract if it has markers, otherwise use as-is
    if "\\boxed" in ground_truth or "####" in ground_truth:
        gold = extract_math_answer(ground_truth)
    else:
        gold = ground_truth.strip()

    if pred is None or gold is None:
        return False

    def _normalize_expr(expr: str) -> str:
        s = expr.strip().lower()
        s = s.replace("$", "")
        s = s.replace("\\left", "").replace("\\right", "")
        s = re.sub(r"\\boxed\{(.+)\}", r"\1", s)
        s = re.sub(r"\s+", "", s)
        return s

    # Normalize
    pred = _normalize_expr(pred)
    gold = _normalize_expr(gold)

    # Direct match
    if pred == gold:
        return True

    # Try numeric comparison
    try:
        p = float(pred.replace(",", ""))
        g = float(gold.replace(",", ""))
        return abs(p - g) < 1e-5
    except (ValueError, TypeError):
        pass

    # Try SymPy symbolic/numeric equivalence
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
        import signal

        class TimeoutException(BaseException): pass

        def timeout_handler(signum, frame):
            raise TimeoutException()

        # Clean up some common unsupported latex before parsing
        p_clean = pred.replace("\\%", "").replace("^{\\circ}", "").replace("\\$", "")
        g_clean = gold.replace("\\%", "").replace("^{\\circ}", "").replace("\\$", "")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3) # 3 seconds timeout for parsing and simplify
        try:
            expr_p = parse_latex(p_clean)
            expr_g = parse_latex(g_clean)
            
            # 1. Algebraic simplification
            if sympy.simplify(expr_p - expr_g) == 0:
                signal.alarm(0)
                return True
                
            # 2. Numeric evaluation
            diff = abs(expr_p.evalf() - expr_g.evalf())
            if diff < 1e-5:
                signal.alarm(0)
                return True
        except TimeoutException:
            pass
        finally:
            signal.alarm(0)
    except Exception:
        pass

    return False
