"""Prompt templates: the same task asked in different ways.

A template is an instruction plus a rendering rule. The harness measures
whether the wording changes the result — for the bundled mock models it
does, because the keyword model answers in the format the instruction
asks for. That is the honest, observable version of prompt engineering:
format instructions change extractable answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .datasets import Item


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    instruction: str
    few_shot: bool = False
    style: str = "concise"  # concise | verbose | cot


TEMPLATES: dict[str, PromptTemplate] = {
    "zero-shot": PromptTemplate(
        "zero-shot", "Answer the question."),
    "concise": PromptTemplate(
        "concise", "Answer with the shortest possible answer — just the value, nothing else."),
    "few-shot": PromptTemplate(
        "few-shot",
        "Answer with the shortest possible answer — just the value, nothing else.",
        few_shot=True),
    "chain-of-thought": PromptTemplate(
        "chain-of-thought",
        "Think step by step, then give your final answer after 'Answer:'.",
        style="cot"),
    "verbose": PromptTemplate(
        "verbose",
        "Give a full, detailed answer in a complete sentence.",
        style="verbose"),
}


def _render_item(item: Item) -> str:
    lines = [f"Question: {item.question}"]
    if item.context:
        lines.append(f"Context: {item.context}")
    lines.append("Answer:")
    return "\n".join(lines)


def render(template: PromptTemplate, item: Item,
           few_shot_examples: list[Item] | None = None) -> str:
    """Render one item under a template, with optional few-shot examples."""
    parts = [template.instruction, ""]
    if template.few_shot and few_shot_examples:
        for example in few_shot_examples:
            parts.append(_render_item(example))
            parts.append(example.expected)
            parts.append("")
    parts.append(_render_item(item))
    return "\n".join(parts)


def few_shot_examples_for(item: Item, all_items: list[Item], count: int = 2) -> list[Item]:
    """Deterministic few-shot examples: same category, different items."""
    pool = [other for other in all_items
            if other.category == item.category and other.item_id != item.item_id]
    return pool[:count]
