from __future__ import annotations

import random
from typing import Dict, List, Mapping

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import RLDataConfig, StageConfig, load_config
from .datasets import load_datasets
from .graders import (
    DatasetReward,
    ExternalHttpGrader,
    GradeResult,
    GraderInput,
    JsonValidGrader,
    MathExactGrader,
    NumberOnlyGrader,
    ReasoningFormatGrader,
)


class MetaStage(BaseModel):
    name: str
    until_step: int


class MetaResponse(BaseModel):
    total_iterations: int
    stages: List[MetaStage]


class SampleRequest(BaseModel):
    iteration: int
    batch_size: int


class SampleItem(BaseModel):
    sample_id: str
    dataset_id: str
    system_prompt: str | None = None
    prompt: str
    metadata: Mapping[str, object] | None = None


class SampleResponse(BaseModel):
    samples: List[SampleItem]


class GradeItem(BaseModel):
    sample_id: str
    completion: str
    completion_index: int | None = None


class GradeRequest(BaseModel):
    iteration: int
    items: List[GradeItem]


class GradeResultOut(BaseModel):
    sample_id: str
    reward: float
    completion_index: int | None = None


class GradeResponse(BaseModel):
    results: List[GradeResultOut]


class RLDataServer:
    def __init__(self, config_path: str, verbose: bool = False):
        self.config: RLDataConfig = load_config(config_path)
        self.datasets = load_datasets(self.config.datasets)
        self.graders = self._build_graders()
        self.format_grader = ReasoningFormatGrader()
        self.dataset_rewards = self._build_dataset_rewards()
        self.total_iterations = self.config.stages[-1].until_step
        self.verbose = verbose

    def _build_graders(self) -> Dict[str, object]:
        graders: Dict[str, object] = {
            "json_valid": JsonValidGrader(),
            "math_exact": MathExactGrader(),
            "number_only": NumberOnlyGrader(),
        }
        for name, cfg in self.config.external_graders.items():
            graders[name] = ExternalHttpGrader(name, cfg)
        return graders

    def _build_dataset_rewards(self) -> Dict[str, DatasetReward]:
        rewards: Dict[str, DatasetReward] = {}
        for dataset_id, cfg in self.config.datasets.items():
            rewards[dataset_id] = DatasetReward(
                cfg,
                self.graders,
                builtin_reasoning_grader=self.format_grader,
                builtin_weight=1.0,
            )
        return rewards

    def _find_stage(self, iteration: int) -> StageConfig:
        for stage in self.config.stages:
            if iteration < stage.until_step:
                return stage
        return self.config.stages[-1]

    def _sample_dataset_id(self, stage: StageConfig) -> str:
        ids = [d.id for d in stage.datasets]
        weights = [d.weight for d in stage.datasets]
        return random.choices(ids, weights=weights, k=1)[0]

    def sample_batch(self, iteration: int, batch_size: int) -> List[SampleItem]:
        stage = self._find_stage(iteration)
        samples: List[SampleItem] = []
        for _ in range(batch_size):
            dataset_id = self._sample_dataset_id(stage)
            dataset = self.datasets.get(dataset_id)
            if dataset is None:
                raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_id}")
            if not dataset:
                raise HTTPException(status_code=400, detail=f"Dataset {dataset_id} is empty")
            idx = random.randint(0, len(dataset) - 1)
            record = dataset[idx]
            sample_id = f"{dataset_id}:{idx}"
            samples.append(
                SampleItem(
                    sample_id=sample_id,
                    dataset_id=dataset_id,
                    system_prompt=record.get("system_prompt"),
                    prompt=record.get("prompt"),
                    metadata=record.get("metadata"),
                )
            )
        return samples

    async def grade(self, items: List[GradeItem]) -> List[GradeResult]:
        grouped: Dict[str, List[GraderInput]] = {}
        for item in items:
            dataset_id, idx = self._parse_sample_id(item.sample_id)
            dataset = self.datasets.get(dataset_id)
            if dataset is None:
                raise HTTPException(status_code=400, detail=f"Unknown dataset in sample_id: {dataset_id}")
            if idx < 0 or idx >= len(dataset):
                raise HTTPException(status_code=400, detail=f"Index out of range for dataset {dataset_id}")
            record = dataset[idx]
            reasoning, answer = self._parse_completion(item.completion)
            grouped.setdefault(dataset_id, []).append(
                GraderInput(
                    sample_id=item.sample_id,
                    record=record,
                    completion=item.completion,
                    reasoning=reasoning,
                    answer=answer,
                    completion_index=item.completion_index,
                )
            )

        results: List[GradeResult] = []
        for dataset_id, inputs in grouped.items():
            rewarder = self.dataset_rewards.get(dataset_id)
            if rewarder is None:
                raise HTTPException(status_code=400, detail=f"Unknown dataset in grading: {dataset_id}")
            dataset_results = await rewarder.grade_batch(inputs)
            results.extend(dataset_results)
        return results

    @staticmethod
    def _parse_completion(completion: str) -> tuple[str | None, str | None]:
        """Extract reasoning and final response from a completion.

        Expected format:
            <think>
            ...reasoning...
            </think>
            ...final response...
        Returns (None, None) on parse failure.
        """
        if not isinstance(completion, str):
            try:
                completion = str(completion)
            except Exception:
                return None, None

        text = completion.strip()
        open_tag = "<think>"
        close_tag = "</think>"
        start = text.find(open_tag)
        end = text.find(close_tag)

        if start == -1 or end == -1 or end <= start:
            return None, None

        reasoning = text[start + len(open_tag) : end].strip()
        answer = text[end + len(close_tag) :].strip()

        if not reasoning and not answer:
            return None, None
        return reasoning or None, answer or None

    @staticmethod
    def _parse_sample_id(sample_id: str) -> tuple[str, int]:
        if ":" not in sample_id:
            raise HTTPException(status_code=400, detail="sample_id must be in the format <dataset_id>:<index>")
        dataset_id, idx_str = sample_id.split(":", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="sample_id index must be an integer")
        return dataset_id, idx


def create_app(config_path: str, verbose: bool = False) -> FastAPI:
    server = RLDataServer(config_path, verbose=verbose)
    app = FastAPI()

    @app.get("/meta", response_model=MetaResponse)
    async def meta() -> MetaResponse:
        return MetaResponse(
            total_iterations=server.total_iterations,
            stages=[MetaStage(name=s.name, until_step=s.until_step) for s in server.config.stages],
        )

    @app.post("/sample", response_model=SampleResponse)
    async def sample(req: SampleRequest) -> SampleResponse:
        samples = server.sample_batch(req.iteration, req.batch_size)
        return SampleResponse(samples=samples)

    @app.post("/grade", response_model=GradeResponse)
    async def grade(req: GradeRequest) -> GradeResponse:
        results = await server.grade(req.items)
        if server.verbose:
            paired = list(zip(results, req.items))

            def _sid(res: GradeResultOut) -> str:
                return (
                    f"{res.sample_id}"
                    f"{'' if res.completion_index is None else f'#{res.completion_index}'}"
                )

            print(f"[dataserver] iteration={req.iteration} graded {len(results)} completions.")
            print("  ---- ALL REWARDS ----")
            for res, _item in paired:
                print(f"    - {_sid(res)} = {res.reward:.3f}")

            if paired:
                best_res, best_item = max(paired, key=lambda x: x[0].reward)
                worst_res, worst_item = min(paired, key=lambda x: x[0].reward)

                print("  ---- BEST ----")
                print(f"    {_sid(best_res)} = {best_res.reward:.3f}")
                print("    completion:")
                print(best_item.completion)

                print("  ---- WORST ----")
                print(f"    {_sid(worst_res)} = {worst_res.reward:.3f}")
                print("    completion:")
                print(worst_item.completion)
        return GradeResponse(
            results=[
                GradeResultOut(
                    sample_id=r.sample_id,
                    reward=r.reward,
                    completion_index=r.completion_index,
                )
                for r in results
            ]
        )

    return app


__all__ = ["create_app", "RLDataServer"]
