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
    def __init__(self, config_path: str):
        self.config: RLDataConfig = load_config(config_path)
        self.datasets = load_datasets(self.config.datasets)
        self.graders = self._build_graders()
        self.dataset_rewards = self._build_dataset_rewards()
        self.total_iterations = self.config.stages[-1].until_step

    def _build_graders(self) -> Dict[str, object]:
        graders: Dict[str, object] = {
            "json_valid": JsonValidGrader(),
            "math_exact": MathExactGrader(),
        }
        for name, cfg in self.config.external_graders.items():
            graders[name] = ExternalHttpGrader(name, cfg)
        return graders

    def _build_dataset_rewards(self) -> Dict[str, DatasetReward]:
        rewards: Dict[str, DatasetReward] = {}
        for dataset_id, cfg in self.config.datasets.items():
            rewards[dataset_id] = DatasetReward(cfg, self.graders)
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
            grouped.setdefault(dataset_id, []).append(
                GraderInput(
                    sample_id=item.sample_id,
                    record=record,
                    completion=item.completion,
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
    def _parse_sample_id(sample_id: str) -> tuple[str, int]:
        if ":" not in sample_id:
            raise HTTPException(status_code=400, detail="sample_id must be in the format <dataset_id>:<index>")
        dataset_id, idx_str = sample_id.split(":", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="sample_id index must be an integer")
        return dataset_id, idx


def create_app(config_path: str) -> FastAPI:
    server = RLDataServer(config_path)
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
