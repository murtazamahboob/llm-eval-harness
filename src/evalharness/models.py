"""Models: the things being evaluated.

Every model implements the same tiny protocol — ``generate(prompt, item)``
— so a real LLM behind an HTTP API can be dropped in beside the bundled
deterministic stand-ins:

* ``OracleModel`` — returns the ground truth (the 100% ceiling).
* ``KeywordModel`` — a genuine rule-based system: it parses the prompt
  text only, never sees the expected answer, and solves most items with
  hand-written heuristics. It is strong but not perfect, which is exactly
  what makes the leaderboard interesting.
* ``RandomModel`` — a seeded random baseline (the floor any real model
  should beat).
* ``EchoModel`` — parrots the prompt back (the "did nothing" floor).
* ``HttpModel`` — a bring-your-own-key adapter for any OpenAI-compatible
  chat API. It is never called by the demo or the tests, so the project
  stays fully offline out of the box.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass

from .datasets import Item

_STOPWORDS = {
    "is", "are", "was", "were", "be", "been", "the", "a", "an", "of", "in",
    "on", "at", "to", "for", "with", "by", "from", "as", "and", "or", "but",
    "because", "during", "after", "before", "its", "it", "this", "that",
    "these", "those", "has", "have", "had", "do", "does", "did", "will",
    "would", "can", "could", "about", "up", "out", "over", "under",
    "through", "between", "into", "per", "each", "every", "all", "more",
    "most", "other", "some", "any", "no", "not", "only", "very", "just",
    "also", "than", "then", "when", "where", "which", "who", "whom",
    "what", "how", "why", "am", "an", "if", "so", "such", "there", "here",
}

_POSITIVE_WORDS = {
    "delicious", "wonderful", "love", "amazing", "best", "great", "fast",
    "fantastic", "excellent", "recommended", "enjoyed", "perfect", "happy",
    "brilliant", "superb",
}
_NEGATIVE_WORDS = {
    "dirty", "terrible", "boring", "disappointing", "awful", "crashes",
    "horrible", "worst", "bad", "broken", "slow", "rude", "disgusting",
    "useless", "annoying",
}

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.'$-]*")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_YEAR_RE = re.compile(r"\b\d{4}\b")


def last_question_block(prompt: str) -> str:
    """The final ``Question: ...`` block — few-shot examples come first."""
    blocks = prompt.split("Question:")
    return blocks[-1] if len(blocks) > 1 else prompt


def _question_words(prompt: str) -> set[str]:
    """Words of the question line only — never the context or examples."""
    block = last_question_block(prompt)
    for line in block.splitlines():
        if line.strip():
            return set(_WORD_RE.findall(line.lower()))
    return set()


@dataclass
class ModelResult:
    text: str


class OracleModel:
    """Returns the ground truth. The ceiling every other model is judged by."""

    name = "oracle"

    def generate(self, prompt: str, item: Item) -> str:
        return item.expected


class EchoModel:
    """Returns the prompt itself. The 'did nothing at all' floor."""

    name = "echo"

    def generate(self, prompt: str, item: Item) -> str:
        return prompt[:200]


class RandomModel:
    """Seeded random answers, one plausible guess per category."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def generate(self, prompt: str, item: Item) -> str:
        rng = random.Random(f"{self.seed}:{item.item_id}")
        if item.category == "math":
            return str(rng.randint(1, 100))
        if item.category == "classify":
            return rng.choice(["positive", "negative", "neutral"])
        words = _WORD_RE.findall(last_question_block(prompt))
        candidates = [w for w in words if w[:1].isupper() or w.isdigit()]
        if not candidates:
            candidates = words
        return rng.choice(candidates) if candidates else "unknown"


class KeywordModel:
    """A rule-based model that reads the prompt and reasons with heuristics.

    It never sees the expected answer. How it answers depends on the
    prompt's instruction line, which is the whole point of prompt
    comparison: a 'shortest possible answer' instruction gets a bare span,
    a 'full detailed answer' instruction gets a sentence, and a
    'step by step' instruction gets a reasoning trace — all of which score
    differently on exact match.
    """

    name = "keyword"

    # -- dispatch ------------------------------------------------------
    def generate(self, prompt: str, item: Item) -> str:
        style = self._style(prompt)
        if item.category == "qa":
            answer = self._answer_qa(prompt)
        elif item.category == "math":
            answer = self._answer_math(prompt)
        elif item.category == "classify":
            answer = self._answer_classify(prompt)
        elif item.category == "extract":
            answer = self._answer_extract(prompt)
        else:
            answer = "unknown"
        if style == "verbose":
            return self._verbose(item, answer)
        if style == "cot":
            return f"First, I read the question carefully. Then, I work out the answer. The answer is {answer}."
        return answer

    @staticmethod
    def _style(prompt: str) -> str:
        lowered = prompt.lower()
        if "shortest possible" in lowered:
            return "concise"
        if "full, detailed" in lowered or "full detailed" in lowered:
            return "verbose"
        if "step by step" in lowered:
            return "cot"
        return "concise"

    @staticmethod
    def _verbose(item: Item, answer: str) -> str:
        if item.category == "qa":
            return item.context or f"The answer is {answer}."
        if item.category == "math":
            return f"The answer is {answer}."
        if item.category == "classify":
            return f"The sentiment is {answer}."
        if item.category == "extract":
            return f"The requested value is {answer}."
        return answer

    # -- question answering -------------------------------------------
    def _answer_qa(self, prompt: str) -> str:
        block = last_question_block(prompt)
        context = ""
        for line in block.splitlines():
            if line.strip().lower().startswith("context:"):
                context = line.split(":", 1)[1].strip()
        if not context:
            return "unknown"
        question_words = _question_words(prompt)
        if "year" in question_words:
            years = _YEAR_RE.findall(context)
            if years:
                return years[0]
        tokens = _WORD_RE.findall(context)
        # Candidate runs: consecutive content words absent from the question.
        runs: list[list[str]] = []
        current: list[str] = []
        for token in tokens:
            lowered = token.lower().strip(".'$")
            if lowered in _STOPWORDS or lowered in question_words:
                if current:
                    runs.append(current)
                    current = []
                continue
            current.append(token)
        if current:
            runs.append(current)
        if not runs:
            return "unknown"
        capitalised = [run for run in runs if any(w[:1].isupper() for w in run)]
        if capitalised:
            return " ".join(capitalised[0]).strip(".'$")
        numbers = [run for run in runs if any(_NUMBER_RE.fullmatch(w.strip(".$")) for w in run)]
        if numbers:
            return numbers[0][0].strip(".'$")
        longest = max(runs, key=len)
        return " ".join(longest).strip(".'$")

    # -- maths ----------------------------------------------------------
    def _answer_math(self, prompt: str) -> str:
        block = last_question_block(prompt).lower()
        numbers = [float(n) for n in _NUMBER_RE.findall(block)]
        if len(numbers) < 2:
            return "unknown"
        first, second = numbers[0], numbers[1]
        if any(k in block for k in ("equally", "equal pieces", "equal groups", "split into", "cut into", "shared")):
            if second == 0:
                return "unknown"
            value = first / second
        elif "per hour" in block or "each" in block:
            value = first * second
        elif any(k in block for k in ("left", "remain", "gave away", "sold")):
            value = first - second
        elif any(k in block for k in ("total", "altogether", "more", "combined")):
            value = first + second
        else:
            value = first + second
        return str(int(value)) if float(value).is_integer() else str(round(value, 2))

    # -- classification --------------------------------------------------
    def _answer_classify(self, prompt: str) -> str:
        block = last_question_block(prompt).lower()
        quoted = re.findall(r"'([^']+)'", block)
        text = quoted[0] if quoted else block
        words = {w.strip(".'$") for w in _WORD_RE.findall(text)}
        pos = len(words & _POSITIVE_WORDS)
        neg = len(words & _NEGATIVE_WORDS)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    # -- extraction ------------------------------------------------------
    def _answer_extract(self, prompt: str) -> str:
        block = last_question_block(prompt)
        # The passage is the quoted text after "passage:". Quote-matching
        # breaks here because "person's name" in the question contains an
        # apostrophe, so slice between the quote after "passage:" and the
        # last quote in the block instead.
        marker = block.find("passage:")
        start = block.find("'", marker) if marker != -1 else -1
        end = block.rfind("'")
        if start != -1 and end > start:
            passage = block[start + 1:end]
        else:
            passage = block
        lowered = block.lower()
        if "year" in lowered:
            years = _YEAR_RE.findall(passage)
            return years[0] if years else "unknown"
        if "price" in lowered:
            match = re.search(r"\$(\d+(?:\.\d+)?)", passage)
            return match.group(1) if match else "unknown"
        if "name" in lowered:
            match = re.search(r"(?:Dr\.|Professor|Captain)\s+([A-Z][a-z]+ [A-Z][a-z]+)", passage)
            return match.group(1) if match else "unknown"
        if "city" in lowered:
            match = re.search(r"\b(?:in|to)\s+([A-Z][a-z]+)\b", passage)
            return match.group(1) if match else "unknown"
        return "unknown"


class HttpModel:
    """Adapter for any OpenAI-compatible chat-completions API.

    Reads ``LLM_API_KEY`` (and optionally ``LLM_BASE_URL``) from the
    environment. Never used by the demo or tests — the harness is fully
    offline without it.
    """

    def __init__(self, name: str = "http-llm", model: str = "gpt-4o-mini",
                 base_url: str | None = None, api_key: str | None = None):
        self.name = name
        self.model = model
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")

    def generate(self, prompt: str, item: Item) -> str:
        if not self.api_key:
            raise RuntimeError("HttpModel needs LLM_API_KEY set in the environment")
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        return data["choices"][0]["message"]["content"].strip()
