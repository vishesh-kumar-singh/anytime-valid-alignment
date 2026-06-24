import re
from typing import List, Optional

def load_mmlu(
    split: str = "test",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """Load MMLU dataset examples.

    Args:
        split: "test" or "validation" or "dev".
        max_examples: If set, truncate to this many examples.

    Returns:
        List of dicts with 'question' and 'answer' keys.
    """
    from datasets import load_dataset
    
    ds = load_dataset("cais/mmlu", "all", split=split)
    
    options = ["A", "B", "C", "D"]
    
    examples = []
    for i, item in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        
        # Format the question with choices
        question_text = item["question"] + "\nOptions:\n"
        for idx, choice in enumerate(item["choices"]):
            if idx < len(options):
                question_text += f"({options[idx]}) {choice}\n"
        
        # The answer is given as an integer index (0-3)
        answer_idx = item["answer"]
        answer_letter = options[answer_idx] if 0 <= answer_idx < len(options) else str(answer_idx)
        
        examples.append({
            "question": question_text.strip(),
            "answer": answer_letter,
        })

    return examples

def extract_mmlu_answer(answer_text: str) -> Optional[str]:
    """Extract the multiple choice answer (A, B, C, or D) from a model generation."""
    if not answer_text:
        return None
    
    # Try to find common patterns like "The correct answer is (A)" or "Answer: A"
    match = re.search(r'(?:answer is|Answer:|\*\*Answer\*\*:?)\s*\(?([A-D])\)?', answer_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: look for the last A, B, C, or D enclosed in parentheses or at the end
    match = re.search(r'\(([A-D])\)', answer_text, re.IGNORECASE)
    if match:
        # If there are multiple, find the last one
        matches = re.findall(r'\(([A-D])\)', answer_text, re.IGNORECASE)
        return matches[-1].upper()
        
    # Ultimate fallback: just find the last single A, B, C, or D word
    matches = re.findall(r'\b([A-D])\b', answer_text)
    if matches:
        return matches[-1].upper()
        
    return None

def check_answer_correct(prediction: str, ground_truth: str) -> bool:
    """Check if a predicted MMLU answer matches the ground truth."""
    pred_letter = extract_mmlu_answer(prediction)
    gold_letter = ground_truth.strip().upper()
    
    if pred_letter is None or not gold_letter:
        return False
        
    return pred_letter == gold_letter
