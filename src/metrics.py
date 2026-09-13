from __future__ import annotations

import itertools
import re
import string
from collections import Counter
from typing import Any, Iterable, List


def normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    answer = str(answer)
    answer = answer.lower().strip()
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = " ".join(answer.split())
    return answer


def exact_match(prediction: Any, gold: Any) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def pass_at_k(exact_matches: Iterable[int]) -> int:
    return int(any(exact_matches))


def consistency_score(answers: List[Any]) -> float:
    if not answers:
        return 0.0
    normalized = [normalize_answer(a) for a in answers]
    most_common_count = Counter(normalized).most_common(1)[0][1]
    return most_common_count / len(answers)


def simple_trace_similarity(traces: List[str]) -> float:
    """Lightweight Jaccard similarity fallback for reasoning traces.

    This avoids heavy dependencies during initial testing. For final thesis runs,
    you may replace this with sentence-transformer cosine similarity.
    """
    if len(traces) < 2:
        return 1.0
    scores = []
    for a, b in itertools.combinations(traces, 2):
        set_a = set(normalize_answer(a).split())
        set_b = set(normalize_answer(b).split())
        if not set_a and not set_b:
            scores.append(1.0)
        elif not set_a or not set_b:
            scores.append(0.0)
        else:
            scores.append(len(set_a & set_b) / len(set_a | set_b))
    return sum(scores) / len(scores)
