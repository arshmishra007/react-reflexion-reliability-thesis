import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_hotpotqa(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HotpotQA file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected HotpotQA dev file to be a list of records.")
    return data


def _context_to_text(
    context: Sequence[Any],
    mode: str = "supporting_plus_distractors",
    max_paragraphs: int | None = 6,
) -> str:
    """Convert HotpotQA context into readable plain text.

    HotpotQA context usually looks like:
    [[title, [sentence1, sentence2]], ...]
    """
    paragraphs = []
    selected_context = context if max_paragraphs is None else context[:max_paragraphs]
    for item in selected_context:
        if not isinstance(item, list) or len(item) < 2:
            continue
        title, sentences = item[0], item[1]
        if isinstance(sentences, list):
            body = " ".join(str(s) for s in sentences)
        else:
            body = str(sentences)
        paragraphs.append(f"Title: {title}\n{body}")
    return "\n\n".join(paragraphs)


def select_supporting_plus_distractors(
    context: List[Any],
    supporting_facts: List[Any],
    rng: random.Random,
    max_documents: int | None = 6,
    question_id: str = "unknown_id",
) -> List[Any]:
    """Select all supporting documents, then fill remaining slots with distractors."""
    available_context = {
        item[0]: item[1]
        for item in context
        if isinstance(item, list) and len(item) >= 2
    }

    required_titles = set()
    for fact in supporting_facts:
        if not isinstance(fact, list) or len(fact) < 2:
            raise ValueError(f"Question {question_id}: malformed supporting fact: {fact}")

        title, sentence_index = fact[0], fact[1]
        required_titles.add(title)
        if title not in available_context:
            raise ValueError(
                f"Question {question_id}: supporting title not found "
                f"in raw context: {title}"
            )

        sentences = available_context[title]
        if (
            not isinstance(sentence_index, int)
            or isinstance(sentence_index, bool)
            or not isinstance(sentences, list)
            or sentence_index < 0
            or sentence_index >= len(sentences)
        ):
            raise ValueError(
                f"Question {question_id}: invalid supporting sentence "
                f"index {sentence_index} for title {title}"
            )

    supporting_documents = []
    distractor_documents = []
    seen_titles = set()
    for item in context:
        if not isinstance(item, list) or len(item) < 2:
            continue
        title = item[0]
        if title in seen_titles:
            continue
        seen_titles.add(title)
        if title in required_titles:
            supporting_documents.append(item)
        else:
            distractor_documents.append(item)

    if max_documents is None:
        distractor_slots = len(distractor_documents)
    else:
        distractor_slots = max(0, max_documents - len(supporting_documents))

    if distractor_slots >= len(distractor_documents):
        selected_distractors = distractor_documents
    else:
        selected_indices = set(rng.sample(range(len(distractor_documents)), distractor_slots))
        selected_distractors = [
            item for index, item in enumerate(distractor_documents)
            if index in selected_indices
        ]

    selected_context = supporting_documents + selected_distractors
    selected_titles = {item[0] for item in selected_context}
    missing_titles = required_titles - selected_titles
    if missing_titles:
        raise ValueError(
            f"Question {question_id} is missing supporting documents: "
            f"{sorted(missing_titles)}"
        )

    return selected_context


def preprocess_record(
    record: Dict[str, Any],
    context_mode: str = "supporting_plus_distractors",
    rng: random.Random | None = None,
) -> Dict[str, Any]:
    question_id = record.get("_id") or record.get("id") or "unknown_id"
    original_context = record.get("context", [])
    supporting_facts = record.get("supporting_facts", [])

    if context_mode == "supporting_plus_distractors":
        selected_context = select_supporting_plus_distractors(
            original_context,
            supporting_facts,
            rng or random.Random(42),
            max_documents=6,
            question_id=question_id,
        )
        required_titles = {fact[0] for fact in supporting_facts}
        selected_titles = {item[0] for item in selected_context}
        missing_titles = required_titles - selected_titles
        supporting_count = len(required_titles & selected_titles)
        distractor_count = len(selected_context) - supporting_count
        print(
            f"Question {question_id}: supporting context validation PASSED | "
            f"required={len(required_titles)} | included={supporting_count} | "
            f"distractors={distractor_count}"
        )
    else:
        selected_context = original_context[:6]
        required_titles = {fact[0] for fact in supporting_facts}
        selected_titles = {
            item[0] for item in selected_context
            if isinstance(item, list) and len(item) >= 2
        }
        missing_titles = required_titles - selected_titles

    return {
        "question_id": question_id,
        "question": record.get("question", ""),
        "gold_answer": record.get("answer", ""),
        "context_text": _context_to_text(
            selected_context,
            mode=context_mode,
            max_paragraphs=None,
        ),
        "supporting_facts": supporting_facts,
        "level": record.get("level", "unknown"),
        "type": record.get("type", "unknown"),
        "required_supporting_titles": sorted(required_titles),
        "included_supporting_titles": sorted(required_titles & selected_titles),
        "missing_supporting_titles": sorted(missing_titles),
        "context_validation_passed": len(missing_titles) == 0,
        "context_document_count": len(selected_context),
    }


def build_subset(raw_path: str | Path, sample_size: int, seed: int = 42, context_mode: str = "supporting_plus_distractors") -> List[Dict[str, Any]]:
    data = load_hotpotqa(raw_path)
    rng = random.Random(seed)
    sample = rng.sample(data, min(sample_size, len(data)))
    return [
        preprocess_record(r, context_mode=context_mode, rng=rng)
        for r in sample
    ]


def save_json(records: List[Dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
