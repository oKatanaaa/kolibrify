from __future__ import annotations

import ast
import asyncio
import json
import re
import xml.etree.ElementTree as ET
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


class JsonSchemaGrader(Grader):
    """
    Grades completions against a minimal JSON schema stored in record['metadata']['schema'].

    Expected logical schema (before serialization):
        {
            "type": "object",
            "required": ["answer", ...],
            "allow_additional_properties": false | true
        }

    Checks that the completion parses as a JSON object, contains all required keys,
    and optionally forbids extra keys. Does not validate value types or numeric correctness.
    Partial credit is given for parseable JSON (0.5) and for missing/extra keys (0.7-0.9).
    If the completion is inside a markdown fence or only parses via ast.literal_eval,
    grading still happens with a 0.9 penalty multiplier.
    """

    def __init__(self, reward_if_no_schema: float = 0.0) -> None:
        self.reward_if_no_schema = reward_if_no_schema

    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        results: List[GradeResult] = []

        for item in inputs:
            record: Mapping[str, object] = item.record
            metadata = record.get("metadata") if isinstance(record, Mapping) else None

            schema_dict = None
            if isinstance(metadata, Mapping):
                schema_json = metadata.get("schema")
                if isinstance(schema_json, str):
                    try:
                        parsed_schema = json.loads(schema_json)
                    except json.JSONDecodeError:
                        parsed_schema = None
                    if isinstance(parsed_schema, Mapping):
                        schema_dict = parsed_schema

            if schema_dict is None:
                reward = self.reward_if_no_schema
                results.append(
                    GradeResult(
                        sample_id=item.sample_id,
                        reward=reward,
                        completion_index=item.completion_index,
                    )
                )
                continue

            text = item.answer if isinstance(item.answer, str) and item.answer.strip() else item.completion
            parsed, penalty = self._parse_with_penalty(text)
            if parsed is None:
                reward = 0.0
            else:
                reward = self._validate_json(parsed, schema_dict) * penalty

            results.append(
                GradeResult(
                    sample_id=item.sample_id,
                    reward=reward,
                    completion_index=item.completion_index,
                )
            )

        return results

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        """Pull content out of a fenced code block like ```json ... ```."""
        fenced = re.search(r"```[\w-]*\n(.*?)```", text, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return None

    @classmethod
    def _parse_with_penalty(cls, text: str) -> tuple[object | None, float]:
        # Try standard JSON first.
        try:
            return json.loads(text), 1.0
        except json.JSONDecodeError:
            pass

        # Relaxed: strip fenced block or fall back to literal_eval.
        candidate = cls._extract_code_block(text) or text
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate), 0.9
            except Exception:
                continue
        return None, 0.0

    @staticmethod
    def _validate_json(obj: object, schema: Mapping[str, object]) -> float:
        # Valid JSON earns at least 0.5; schema conformance refines it.
        if not isinstance(obj, dict):
            return 0.5

        required = schema.get("required") or []
        allow_additional = bool(schema.get("allow_additional_properties", True))

        if not isinstance(required, list):
            return 0.5

        required_set = set(required)
        keys = set(obj.keys())

        missing = not required_set.issubset(keys)
        extra = not allow_additional and keys - required_set

        if missing and extra:
            return 0.7
        if missing:
            return 0.8
        if extra:
            return 0.9
        return 1.0


class XmlSchemaGrader(Grader):
    """
    Grades completions against a minimal XML schema stored in record['metadata']['schema'].

    Expected logical schema (before serialization):
        {
            "root_tag": "answer"
        }

    Matches only on the root tag name. Attributes, children, and content are unchecked.
    Well-formed XML earns 0.5, wrong root tag 0.8, correct root 1.0. If parsing only
    succeeds after stripping a fenced block, the final reward is multiplied by 0.9.
    """

    def __init__(self, reward_if_no_schema: float = 0.0) -> None:
        self.reward_if_no_schema = reward_if_no_schema

    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        results: List[GradeResult] = []

        for item in inputs:
            record: Mapping[str, object] = item.record
            metadata = record.get("metadata") if isinstance(record, Mapping) else None

            schema_dict = None
            if isinstance(metadata, Mapping):
                schema_json = metadata.get("schema")
                if isinstance(schema_json, str):
                    try:
                        parsed_schema = json.loads(schema_json)
                    except json.JSONDecodeError:
                        parsed_schema = None
                    if isinstance(parsed_schema, Mapping):
                        schema_dict = parsed_schema

            if schema_dict is None:
                reward = self.reward_if_no_schema
                results.append(
                    GradeResult(
                        sample_id=item.sample_id,
                        reward=reward,
                        completion_index=item.completion_index,
                    )
                )
                continue

            # Prefer the parsed final answer when available; fall back to the full completion.
            text = item.answer if isinstance(item.answer, str) and item.answer.strip() else item.completion
            root, penalty = self._parse_with_penalty(text)
            if root is None:
                reward = 0.0
            else:
                reward = self._validate_xml(root, schema_dict) * penalty

            results.append(
                GradeResult(
                    sample_id=item.sample_id,
                    reward=reward,
                    completion_index=item.completion_index,
                )
            )

        return results

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        fenced = re.search(r"```[\w-]*\n(.*?)```", text, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return None

    @classmethod
    def _parse_with_penalty(cls, text: str) -> tuple[ET.Element | None, float]:
        try:
            return ET.fromstring(text.strip()), 1.0
        except ET.ParseError:
            pass

        candidate = cls._extract_code_block(text) or text
        try:
            return ET.fromstring(candidate.strip()), 0.9
        except ET.ParseError:
            return None, 0.0

    @staticmethod
    def _validate_xml(root: ET.Element, schema: Mapping[str, object]) -> float:
        # Well-formed XML is at least 0.5; matching the root tag gives full credit.
        expected_tag = schema.get("root_tag")
        if not isinstance(expected_tag, str):
            return 0.5

        if root.tag != expected_tag:
            return 0.8

        return 1.0


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
    "JsonSchemaGrader",
    "MathExactGrader",
    "NumberOnlyGrader",
    "XmlSchemaGrader",
    "ExternalHttpGrader",
    "DatasetReward",
    "build_graders",
]
