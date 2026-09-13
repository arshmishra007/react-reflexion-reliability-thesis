from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

from src.llm_client import BaseLLMClient


@dataclass
class AgentOutput:
    predicted_answer: str
    reasoning_trace: str
    raw_output: str
    error: str = ""


def extract_final_answer(text: str) -> str:
    """Extract a clean final answer from model output.

    Supports both:
    Final Answer: 1755

    and:
    Final Answer:
    1755
    """
    def _clean_answer(candidate: str) -> str:
        clean = candidate.replace("<short answer only>", "").strip()
        clean = clean.strip('"').strip("'").strip()
        return clean

    def _extract_after_last_marker(marker: str) -> str | None:
        matches = list(re.finditer(marker, text, flags=re.IGNORECASE))
        if not matches:
            return None

        remainder = text[matches[-1].end():]
        for line in remainder.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            if re.match(
                r"^(FIRST_ATTEMPT|REFLECTION_ATTEMPT|Reflection|Reasoning(?: Trace)?|"
                r"Thought|Action|Observation)\s*:",
                candidate,
                flags=re.IGNORECASE,
            ):
                return ""
            return _clean_answer(candidate)
        return ""

    answer = _extract_after_last_marker(r"\bFinal\s+Answer\s*:")
    if answer is not None:
        return answer

    answer = _extract_after_last_marker(r"(?m)^\s*Answer\s*:")
    if answer is not None:
        return answer
    return ""


class ReActAgent:
    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    def run(self, item: Dict) -> AgentOutput:
        prompt = f"""
You are a careful ReAct-style multi-hop question answering agent.

Use only the provided context to answer the question.
The context may contain both useful paragraphs and distractor paragraphs.

Your task:
1. Identify the key entities in the question.
2. Read the provided context carefully.
3. Connect the relevant facts.
4. Give the best supported final answer.

Important rules:
- Use only the provided context.
- Do not say "unknown" unless the answer is truly absent from the context.
- Do not leave Final Answer blank.
- The final answer must be short: a name, year, number, phrase, or yes/no.
- The Final Answer line must contain only the answer, not a sentence.
- Do not add explanation inside the Final Answer line.

Question:
{item['question']}

Context:
{item['context_text']}

Return your response exactly in this format:

Thought:
<brief reasoning about the question>

Action:
ReadContext

Observation:
<brief evidence found in the context>

Final Answer:
<short answer only>
""".strip()

        response = self.llm.generate(prompt)

        return AgentOutput(
            predicted_answer=extract_final_answer(response.text),
            reasoning_trace=response.text,
            raw_output=response.text,
            error=response.error,
        )


class ReflexionAgent:
    """Simple Reflexion pipeline.

    It uses two model calls:
    1. First attempt.
    2. Reflection + improved answer.

    This keeps implementation simple and easy to explain in the thesis.
    """

    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    def run(self, item: Dict) -> AgentOutput:
        first_prompt = f"""
You are answering a multi-hop question using only the provided context.
The context may contain both useful paragraphs and distractor paragraphs.

Give concise reasoning and a final answer.

Important rules:
- Use only the provided context.
- Do not say "unknown" unless the answer is truly absent from the context.
- Do not leave Final Answer blank.
- The final answer must be short: a name, year, number, phrase, or yes/no.
- The Final Answer line must contain only the answer, not a sentence.
- Do not add explanation inside the Final Answer line.

Question:
{item['question']}

Context:
{item['context_text']}

Return your response exactly in this format:

Reasoning:
<brief step-by-step reasoning>

Final Answer:
<short answer only>
""".strip()
        first = self.llm.generate(first_prompt)

        reflection_prompt = f"""
You are a careful Reflexion-style multi-hop question answering agent.

Use only the provided context to answer the question.
The context may contain both useful paragraphs and distractor paragraphs.

Your task:
1. Review the first attempt.
2. Identify whether the first attempt missed evidence, became too cautious, or gave an unsupported answer.
3. Re-check the context carefully.
4. Give the best supported final answer.

Important rules:
- Do not say "Not specified" if the answer can be inferred from the context.
- Do not say "unknown" unless the answer is truly absent from the context.
- If the exact bridge entity is missing but the answer is available from another relevant paragraph, use the best supported evidence.
- Do not leave Final Answer blank.
- The final answer must be short: a name, year, number, phrase, or yes/no.
- The Final Answer line must contain only the answer, not a sentence.
- Do not add explanation inside the Final Answer line.

Question:
{item['question']}

Context:
{item['context_text']}

First attempt:
{first.text}

Return your response exactly in this format:

Reflection:
<briefly explain what was checked and corrected>

Reasoning Trace:
<brief step-by-step reasoning using the context>

Final Answer:
<short answer only>
""".strip()

        second = self.llm.generate(reflection_prompt)
        combined_trace = f"FIRST_ATTEMPT:\n{first.text}\n\nREFLECTION_ATTEMPT:\n{second.text}"
        errors = [
            f"{label}: {response.error}"
            for label, response in (("first_attempt", first), ("reflection_attempt", second))
            if response.error
        ]

        return AgentOutput(
            predicted_answer=extract_final_answer(second.text),
            reasoning_trace=combined_trace,
            raw_output=f"{first.text}\n\n{second.text}",
            error=" | ".join(errors),
        )
