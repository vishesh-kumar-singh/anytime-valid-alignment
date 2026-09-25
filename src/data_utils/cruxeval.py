"""CRUXEval dataset utilities."""

import ast
import re
from typing import List, Optional

def load_cruxeval(split: str = "test", max_examples: Optional[int] = None) -> List[dict]:
    """Load CRUXEval dataset."""
    from datasets import load_dataset
    
    ds = load_dataset("cruxeval-org/cruxeval", split=split)
    if max_examples is not None and max_examples > 0:
        ds = ds.select(range(min(len(ds), max_examples)))
        
    examples = []
    for row in ds:
        code_block = row["code"]
        func_input = row["input"]
        
        # Build the question prompt
        prompt = f"{code_block}\n\n# What is the output of f({func_input})?"
        
        examples.append({
            "id": row.get("id", ""),
            "question": prompt,
            "answer": str(row["output"]),
            "raw_code": code_block,
            "raw_input": func_input
        })
    return examples


def check_cruxeval_correct(prediction: str, ground_truth: str) -> bool:
    """Verify CRUXEval correctness by parsing python literals."""
    if not prediction or not ground_truth:
        return False
        
    # Extract text after 'Output:' (case-insensitive)
    pred_str = prediction
    match = re.search(r"output:\s*(.*)", prediction, flags=re.IGNORECASE | re.DOTALL)
    if match:
        pred_str = match.group(1).strip()
        
    # Also strip backticks if they put it in a code block
    pred_str = pred_str.strip("` \n")
    if pred_str.startswith("python\n"):
        pred_str = pred_str[7:]
    pred_str = pred_str.strip()

    ground_truth = ground_truth.strip()

    # Attempt to parse both as python literals
    try:
        pred_val = ast.literal_eval(pred_str)
        gold_val = ast.literal_eval(ground_truth)
        return pred_val == gold_val
    except Exception:
        # Fallback to exact string match ignoring whitespace
        pred_clean = re.sub(r"\s+", "", pred_str)
        gold_clean = re.sub(r"\s+", "", ground_truth)
        return pred_clean == gold_clean
