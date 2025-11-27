from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import httpx

from .config import DatasetConfig, ExternalGraderConfig


@dataclass
class GraderInput:
    sample_id: str
    record: Mapping[str, object]
    completion: str


@dataclass
class GradeResult:
    sample_id: str
    reward: float


class Grader:
    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:  # pragma: no cover - interface
        raise NotImplementedError


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
            results.append(GradeResult(sample_id=item.sample_id, reward=reward))
        return results


_number_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


class MathExactGrader(Grader):
    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        results: List[GradeResult] = []
        for item in inputs:
            completion_match = _number_re.search(item.completion)
            answer = None
            if isinstance(item.record, Mapping):
                answer_raw = item.record.get("answer")
                if isinstance(answer_raw, str):
                    answer = answer_raw
            answer_match = _number_re.search(answer) if answer is not None else None
            reward = 0.0
            if completion_match and answer_match:
                if completion_match.group(0) == answer_match.group(0):
                    reward = 1.0
            results.append(GradeResult(sample_id=item.sample_id, reward=reward))
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
                }
                for item in inputs
            ]
        }
        response = await self._client.post(self.config.url, json=payload)
        response.raise_for_status()
        data = response.json()
        results: List[GradeResult] = []
        for res in data.get("results", []):
            results.append(GradeResult(sample_id=res["sample_id"], reward=float(res["reward"])))
        return results

    async def aclose(self) -> None:
        await self._client.aclose()


class DatasetReward:
    def __init__(self, dataset_config: DatasetConfig, graders: Dict[str, Grader]):
        if not dataset_config.graders:
            raise ValueError("Datasets must define at least one grader")
        if len(dataset_config.graders) != len(dataset_config.grader_weights):
            raise ValueError("Graders and grader_weights length mismatch")

        self.graders = []
        for name in dataset_config.graders:
            if name not in graders:
                raise ValueError(f"Unknown grader '{name}' referenced by dataset")
            self.graders.append(graders[name])
        total_weight = sum(dataset_config.grader_weights)
        if total_weight <= 0:
            raise ValueError("grader_weights must sum to a positive value")
        self.weights = [w / total_weight for w in dataset_config.grader_weights]

    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        grader_outputs = await asyncio.gather(
            *[grader.grade_batch(inputs) for grader in self.graders]
        )
        combined: Dict[str, float] = {item.sample_id: 0.0 for item in inputs}
        for weight, results in zip(self.weights, grader_outputs):
            for res in results:
                combined[res.sample_id] += weight * res.reward
        return [GradeResult(sample_id=item.sample_id, reward=combined[item.sample_id]) for item in inputs]


def build_graders(config: DatasetConfig, grader_registry: Dict[str, Grader]) -> DatasetReward:
    return DatasetReward(config, grader_registry)


__all__ = [
    "GradeResult",
    "Grader",
    "GraderInput",
    "JsonValidGrader",
    "MathExactGrader",
    "ExternalHttpGrader",
    "DatasetReward",
    "build_graders",
]
