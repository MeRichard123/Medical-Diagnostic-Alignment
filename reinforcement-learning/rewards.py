"""Reward functions used by GRPO/RLVR training.

The original reward used strict/substring matching and often produced weak or
misaligned signals. This version improves:
1) robust extraction/normalisation of diagnosis text,
2) token-overlap scoring for smoother gradients,
3) safe alignment when #completions != #solutions (multi-generation batches).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, List, Sequence


def _normalise(text: str, max_words: int = 8) -> str:
    t = str(text).strip().lower()
    t = t.replace("```", "").replace("markdown", "")
    t = t.split("\n")[0].strip()
    t = re.sub(r"[\*_`\[\]{}()<>]", " ", t)
    t = re.sub(r"[^a-z0-9\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    return " ".join(words[:max_words])


def _extract_contents(completions: Sequence) -> List[str]:
    if not completions:
        return []
    if isinstance(completions[0], str):
        return [str(c) for c in completions]
    # TRL may provide dict/list message formats in some settings.
    extracted = []
    for c in completions:
        if isinstance(c, list) and c and isinstance(c[0], dict):
            extracted.append(str(c[0].get("content", "")))
        elif isinstance(c, dict):
            extracted.append(str(c.get("content", c)))
        else:
            extracted.append(str(c))
    return extracted


def _align_solutions(contents: List[str], solution: Sequence[str]) -> List[str]:
    sols = [str(s) for s in solution]
    if not sols:
        return [""] * len(contents)
    if len(sols) == len(contents):
        return sols
    if len(sols) == 1:
        return sols * len(contents)

    # Common GRPO case: each prompt has k generations in a contiguous block.
    if len(contents) % len(sols) == 0:
        k = len(contents) // len(sols)
        expanded = []
        for s in sols:
            expanded.extend([s] * k)
        return expanded

    # Last resort: cycle solutions to preserve output length.
    return [sols[i % len(sols)] for i in range(len(contents))]


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(gold_tokens), 1)
    return 2 * precision * recall / (precision + recall)


def _extract_label(text: str, max_words: int = 4) -> str:
    """Extract a compact diagnosis label from model output."""
    t = str(text)
    t = re.split(r"\n|---|###|answer:|explanation:|background:", t, maxsplit=1, flags=re.IGNORECASE)[0]
    t = re.sub(r'^(assistant|diagnosis|answer|final|label)\s*:\s*', '', t.strip(), flags=re.IGNORECASE)
    t = _normalise(t, max_words=16)
    words = [w for w in t.split() if re.search(r"[a-z0-9]", w)]
    return " ".join(words[:max_words])


def _repeated_tokens(text: str) -> bool:
    words = [w for w in str(text).split() if re.search(r"[a-z0-9]", w.lower())]
    return len(words) >= 2 and len(set(words)) < len(words)


def verify_output(text: str) -> tuple[bool, str]:
    """
    Verify that the output contains reasoning and extract the diagnosis.
    
    Args:
        text: The model output to verify
    
    Returns:
        (has_valid_reasoning, extracted_diagnosis)
        - has_valid_reasoning: True if <think> tags are present with meaningful content (>10 chars)
        - extracted_diagnosis: The extracted 1-4 word diagnosis, or empty string if invalid
    """
    text = str(text).strip()
    
    # Check for presence of <think> tags with meaningful content
    think_match = re.search(r'<think>\s*(.+?)\s*</think>', text, re.DOTALL | re.IGNORECASE)
    has_thinking = think_match is not None and len(think_match.group(1).strip()) > 10
    
    # Extract diagnosis after </think> tag or at end of text
    remaining = text
    if think_match:
        remaining = text[think_match.end():]
    
    # Split on common delimiters and take the first non-empty line as diagnosis
    remaining_lines = [line.strip() for line in remaining.splitlines() if line.strip()]
    remaining_clean = remaining_lines[0] if remaining_lines else ""
    
    # Remove any remaining template markers
    remaining_clean = re.split(r'---|###|answer:|explanation:|final:|label:', remaining_clean, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    # Strip common leading prefixes the model may emit after the reasoning block.
    remaining_clean = re.sub(r'^(assistant|diagnosis|answer|final|label)\s*:\s*', '', remaining_clean, flags=re.IGNORECASE).strip()
    
    # Extract 1-4 words
    words = [w for w in remaining_clean.split() if re.search(r'[a-z0-9]', w.lower())]
    diagnosis = " ".join(words[:4]) if words else ""
    
    return has_thinking, diagnosis


def reward_thinking(has_thinking: bool) -> float:
    """
    Reward the model for demonstrating reasoning with <think> tags.
    
    Args:
        has_thinking: Whether the output contains valid <think> tags with meaningful content
    
    Returns:
        Reward value: 0.15 if thinking detected, 0.0 otherwise
    """
    return 0.15 if has_thinking else 0.0


def reward_accuracy(pred_diagnosis: str, gold_diagnosis: str) -> float:
    """
    Calculate reward based on diagnosis accuracy.
    
    Uses a combination of exact match and token-level F1 score.
    
    Args:
        pred_diagnosis: The predicted diagnosis (should be normalized)
        gold_diagnosis: The ground truth diagnosis (should be normalized)
    
    Returns:
        Reward value between 0.0 and 0.85:
        - 0.70 for exact match
        - 0.15 for token overlap (F1 score)
    """
    if not pred_diagnosis or not gold_diagnosis:
        return 0.0
    
    # Exact match
    exact = 1.0 if pred_diagnosis == gold_diagnosis else 0.0
    
    # Token-level F1
    f1 = _token_f1(pred_diagnosis, gold_diagnosis)
    
    # Combine: 0.70 weight on exact, 0.15 weight on F1
    accuracy_reward = 0.70 * exact + 0.15 * f1
    
    return min(0.85, accuracy_reward)


def reward_fn_with_verifier(completions, solution, **kwargs):
    """
    Enhanced reward function that incentivizes both reasoning and accuracy.
    
    Workflow:
    1. Verify output contains valid <think> tags
    2. Extract the diagnosis from the output
    3. Score based on both reasoning presence and diagnosis accuracy
    
    Args:
        completions: Model completions
        solution: Ground truth solutions
        **kwargs: Additional arguments (unused)
    
    Returns:
        List of reward values, one per completion
    """
    contents = _extract_contents(completions)
    aligned_solutions = _align_solutions(contents, solution)
    
    rewards: List[float] = []
    for content, sol in zip(contents, aligned_solutions):
        # Verify output structure and extract reasoning/diagnosis
        has_thinking, pred_diagnosis = verify_output(content)
        gold = _normalise(sol, max_words=4)
        
        if not pred_diagnosis or not has_thinking:
            # Penalize if no thinking or no valid diagnosis extracted
            reward = 0.0 if not has_thinking else 0.3
            rewards.append(reward)
            continue
        
        # Normalize the extracted diagnosis
        pred_normalized = _normalise(pred_diagnosis, max_words=4)
        
        # Calculate component rewards
        thinking_reward = reward_thinking(has_thinking)
        accuracy_reward = reward_accuracy(pred_normalized, gold)
        
        # Combine rewards: 85% accuracy, 15% reasoning bonus
        total_reward = accuracy_reward + thinking_reward
        total_reward = max(0.0, min(1.0, total_reward))
        
        rewards.append(float(total_reward))
    
    return rewards


def reward_fn_relaxed(completions, solution, **kwargs):
    """
    Relaxed reward for short-answer RLVR.

    This variant prioritizes the diagnosis label itself and treats <think>
    reasoning as a bonus rather than a hard requirement. It is useful when the
    model is collapsing to short labels like "normal" early in training.
    """
    contents = _extract_contents(completions)
    aligned_solutions = _align_solutions(contents, solution)

    rewards: List[float] = []
    for content, sol in zip(contents, aligned_solutions):
        text = str(content)
        has_thinking, pred_diagnosis = verify_output(text)
        gold = _normalise(sol, max_words=4)

        if not pred_diagnosis:
            rewards.append(0.0)
            continue

        pred_normalized = _normalise(pred_diagnosis, max_words=4)
        label_reward = reward_accuracy(pred_normalized, gold)

        if has_thinking:
            label_reward = min(1.0, label_reward + 0.20)
        else:
            label_reward *= 0.25

        rewards.append(float(label_reward))

    return rewards


def reward_fn_reasoning_bonus(completions, solution, **kwargs):
    """Small auxiliary reward for valid <think> reasoning blocks."""
    contents = _extract_contents(completions)
    rewards: List[float] = []

    for content in contents:
        has_thinking, _ = verify_output(content)
        rewards.append(0.15 if has_thinking else 0.0)

    return rewards


def reward_fn(completions, solution, **kwargs):
    contents = _extract_contents(completions)
    aligned_solutions = _align_solutions(contents, solution)

    rewards: List[float] = []
    for content, sol in zip(contents, aligned_solutions):
        raw = str(content).strip().lower()
        pred = _normalise(content, max_words=24)
        pred_label = _extract_label(content, max_words=4)
        gold = _normalise(sol, max_words=4)

        if not pred or not gold:
            rewards.append(0.0)
            continue

        pred_tokens = pred.split()

        # Primary objective: correct diagnosis in <= 4 words.
        exact = 1.0 if pred_label == gold else 0.0
        f1 = _token_f1(pred_label, gold)
        partial = 1.0 if gold and gold in pred_label else 0.0

        # Penalise common template spillover and any trailing content beyond the label.
        has_template = any(marker in raw for marker in ["---", "###", "answer", "explanation", "background", "human:", "assistant:"])
        trailing = pred[len(pred_label):].strip() if pred.startswith(pred_label) else ""
        trailing_words = len([w for w in trailing.split() if re.search(r"[a-z0-9]", w)])
        trailing_penalty = min(0.45, 0.12 * trailing_words) if trailing_words else 0.0
        format_penalty = 0.5 if has_template else 0.0

        # Penalise answers that exceed the required diagnosis length.
        length_penalty = 0.25 if len(pred_label.split()) > 4 else 0.0

        # Discourage repetition and label stacking.
        repetition_penalty = 0.2 if _repeated_tokens(pred_label) or _repeated_tokens(raw) else 0.0

        reward = 0.78 * exact + 0.12 * f1 + 0.10 * partial - format_penalty - trailing_penalty - length_penalty - repetition_penalty
        reward = max(0.0, min(1.0, reward))

        rewards.append(float(reward))

    return rewards


def get_reward_funcs(use_verifier: bool = True, relaxed: bool = False) -> List[Callable]:
    """
    Get the reward functions for GRPO training.
    
    Args:
        use_verifier: If True, uses reward_fn_with_verifier
                      If False, uses the original reward_fn.
    
    Returns:
        List of reward functions to pass to GRPOTrainer
    """
    if relaxed:
        return [reward_fn_relaxed, reward_fn_reasoning_bonus]

    if use_verifier:
        return [reward_fn_with_verifier]

    return [reward_fn]