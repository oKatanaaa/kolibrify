from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import httpx

from Levenshtein import ratio

from .config import DatasetConfig, ExternalGraderConfig


@dataclass
class GraderInput:
    sample_id: str
    record: Mapping[str, object]
    completion: str
    reasoning: str | None
    answer: str | None
    completion_index: int | None = None


@dataclass
class GradeResult:
    sample_id: str
    reward: float
    completion_index: int | None = None


class Grader:
    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:  # pragma: no cover - interface
        raise NotImplementedError


class ReasoningFormatGrader(Grader):
    """Reward adherence to the Qwen-style reasoning format.

    Expected format:
        <think>
        ...reasoning...
        </think>
        ...final response...
    """

    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        results: List[GradeResult] = []
        for item in inputs:
            text = (item.completion or "").strip()
            reward = self._score(text)
            results.append(
                GradeResult(
                    sample_id=item.sample_id,
                    reward=reward,
                    completion_index=item.completion_index,
                )
            )
        return results

    @staticmethod
    def _score(text: str) -> float:
        import re

        perfect_pattern = r"^<think>\n[\s\S]*?\n</think>\n[\s\S]+$"
        if re.match(perfect_pattern, text):
            return 1.0

        score = 0.0
        open_idx = text.find("<think>")
        close_idx = text.find("</think>")

        if open_idx != -1:
            score += 0.25
        if close_idx != -1:
            score += 0.25
        if open_idx != -1 and close_idx != -1 and close_idx > open_idx:
            score += 0.25

        if close_idx != -1:
            after = text[close_idx + len("</think>") :].strip()
            if after:
                score += 0.25

        return min(score, 0.99)


class JsonValidGrader(Grader):
    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        results: List[GradeResult] = []
        for item in inputs:
            reward = 0.0
            try:
                parsed = json.loads(item.completion)
                reward = 1.0
                metadata = item.record.get("metadata") if isinstance(item.record, Mapping) else None
                if isinstance(metadata, Mapping):
                    schema = metadata.get("expected_json_schema")
                    if isinstance(schema, Mapping):
                        required = schema.get("required")
                        if isinstance(required, list):
                            if not all(k in parsed for k in required):
                                reward = 0.0
            except Exception:
                reward = 0.0
            results.append(
                GradeResult(
                    sample_id=item.sample_id,
                    reward=reward,
                    completion_index=item.completion_index,
                )
            )
        return results


_number_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


class MathExactGrader(Grader):
    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        results: List[GradeResult] = []
        for item in inputs:
            response = item.answer
            if response is None:
                response = item.completion
            numbers = _number_re.findall(response or "")
            answer = None
            if isinstance(item.record, Mapping):
                answer_raw = item.record.get("answer")
                if isinstance(answer_raw, str):
                    answer = answer_raw
            answer_match = _number_re.search(answer) if answer is not None else None
            reward = 0.0
            if numbers and answer_match:
                gold = answer_match.group(0)
                if any(num == gold for num in numbers):
                    reward = 0.5
                reward += ratio(str(answer), response) / 2
            results.append(
                GradeResult(
                    sample_id=item.sample_id,
                    reward=reward,
                    completion_index=item.completion_index,
                )
            )
        return results


class NumberOnlyGrader(Grader):
    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        results: List[GradeResult] = []

        def is_number_only(text: str) -> bool:
            if text is None:
                return False
            stripped = text.strip()
            if not stripped:
                return False
            return re.fullmatch(r"[-+]?\d*\.?\d+", stripped) is not None

        for item in inputs:
            candidate = item.answer if item.answer is not None else item.completion
            reward = 1.0 if is_number_only(candidate) else 0.0
            results.append(
                GradeResult(
                    sample_id=item.sample_id,
                    reward=reward,
                    completion_index=item.completion_index,
                )
            )
        return results


class ExternalHttpGrader(Grader):
    def __init__(self, name: str, config: ExternalGraderConfig):
        self.name = name
        self.config = config
        self._client = httpx.AsyncClient(timeout=config.timeout_s)

    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        payload = {
            "samples": [
                {
                    "sample_id": item.sample_id,
                    "prompt": item.record.get("prompt") if isinstance(item.record, Mapping) else None,
                    "system_prompt": item.record.get("system_prompt")
                    if isinstance(item.record, Mapping)
                    else None,
                    "answer": item.record.get("answer") if isinstance(item.record, Mapping) else None,
                    "metadata": item.record.get("metadata") if isinstance(item.record, Mapping) else None,
                    "completion": item.completion,
                    "reasoning": item.reasoning,
                    "final_response": item.answer,
                    "completion_index": item.completion_index,
                }
                for item in inputs
            ]
        }
        response = await self._client.post(self.config.url, json=payload)
        response.raise_for_status()
        data = response.json()
        results: List[GradeResult] = []
        for res in data.get("results", []):
            results.append(
                GradeResult(
                    sample_id=res["sample_id"],
                    reward=float(res["reward"]),
                    completion_index=res.get("completion_index"),
                )
            )
        return results

    async def aclose(self) -> None:
        await self._client.aclose()


class DatasetReward:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        graders: Dict[str, Grader],
        builtin_reasoning_grader: Grader | None = None,
        builtin_weight: float = 1.0,
    ):
        self.graders: List[Grader] = []
        weights: List[float] = []

        if dataset_config.graders:
            if len(dataset_config.graders) != len(dataset_config.grader_weights):
                raise ValueError("Graders and grader_weights length mismatch")
            for name in dataset_config.graders:
                if name not in graders:
                    raise ValueError(f"Unknown grader '{name}' referenced by dataset")
                self.graders.append(graders[name])
            weights.extend(dataset_config.grader_weights)

        if builtin_reasoning_grader is not None:
            self.graders.append(builtin_reasoning_grader)
            weights.append(builtin_weight)

        if not self.graders:
            raise ValueError("Datasets must define at least one grader")

        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("grader_weights must sum to a positive value")
        self.weights = [w / total_weight for w in weights]

    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        grader_outputs = await asyncio.gather(
            *[grader.grade_batch(inputs) for grader in self.graders]
        )
        combined: List[float] = [0.0 for _ in inputs]
        for weight, results in zip(self.weights, grader_outputs):
            for idx, res in enumerate(results):
                combined[idx] += weight * res.reward
        return [
            GradeResult(
                sample_id=item.sample_id,
                reward=combined[idx],
                completion_index=item.completion_index,
            )
            for idx, item in enumerate(inputs)
        ]


def build_graders(
    config: DatasetConfig,
    grader_registry: Dict[str, Grader],
    builtin_reasoning_grader: Grader | None = None,
    builtin_weight: float = 1.0,
) -> DatasetReward:
    return DatasetReward(
        config,
        grader_registry,
        builtin_reasoning_grader=builtin_reasoning_grader,
        builtin_weight=builtin_weight,
    )


__all__ = [
    "GradeResult",
    "Grader",
    "GraderInput",
    "ReasoningFormatGrader",
    "JsonValidGrader",
    "MathExactGrader",
    "NumberOnlyGrader",
    "ExternalHttpGrader",
    "DatasetReward",
    "build_graders",
]
