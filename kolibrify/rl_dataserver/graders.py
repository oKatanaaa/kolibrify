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
        score = 0.0
        open_idx = text.find("<think>")
        close_idx = text.find("</think>")
        open_count = len(re.findall(r"<think>", text))
        close_count = len(re.findall(r"</think>", text))

        if open_idx != -1:
            score += 0.125
            if open_idx == 0:
                score += 0.125

        if close_idx != -1:
            score += 0.125
            if open_idx != -1 and close_idx > open_idx:
                score += 0.125
            elif open_idx != -1 and close_idx < open_idx:
                # Penalize reversed ordering of reasoning tags.
                score -= 0.125

        if close_idx != -1:
            after_raw = text[close_idx + len("</think>") :]
            # Penalize a space instead of a newline, or no divider (text glued to </think>).
            divider = after_raw[:1]
            if divider:
                if divider.isspace():
                    if divider == " ":
                        score -= 0.1
                else:
                    score -= 0.1
            after = after_raw.strip()
            if after:
                score += 0.5
            else:
                # Penalize missing final answer after the reasoning block.
                score -= 0.25

        if open_count > 1 or close_count > 1:
            # Penalize multiple reasoning blocks so a single block is preferred.
            score -= 0.15 * ((open_count - 1) + (close_count - 1))

        return max(0.0, min(score, 1.00))


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
            # Penalize cases when the response is malformed (nothing comes after </think> tag)
            if response is None or len(response.strip()) == 0:
                results.append(
                    GradeResult(
                        sample_id=item.sample_id,
                        reward=0.0,
                        completion_index=item.completion_index,
                    )
                )
                continue

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
                    reward = 1.0
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

        for item in inputs:
            candidate = item.answer
            if candidate is None:
                reward = 0.0
            else:
                normalized = candidate.strip()
                match = _number_re.search(normalized)
                # The first matched number anchors our scoring; surrounding text and extra numbers reduce the reward.
                if not normalized or match is None:
                    reward = 0.0
                else:
                    extra_ch_count = len(normalized) - len(match.group(0))
                    reward = 1 / (1 + extra_ch_count)
                    if extra_ch_count == 0:
                        reward = 1.0
                    elif extra_ch_count < 10:
                        reward = 0.5
                    elif extra_ch_count < 20:
                        reward = 0.4
                    elif extra_ch_count < 30:
                        reward = 0.3
                    elif extra_ch_count < 40:
                        reward = 0.2
                    elif extra_ch_count < 50:
                        reward = 0.1
                    else:
                        reward = 0.0
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
