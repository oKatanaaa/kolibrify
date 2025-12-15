import asyncio
import pathlib

import pytest

from kolibrify.rl import rewards as rewards_module
from kolibrify.rl_dataserver.config import DatasetConfig
from kolibrify.rl_dataserver.graders import (
    CompletionLengthCapGrader,
    DatasetReward,
    GradeResult,
    Grader,
    GraderInput,
)


class _ConstGrader(Grader):
    def __init__(self, value: float):
        self.value = value

    async def grade_batch(self, inputs):
        return [
            GradeResult(sample_id=item.sample_id, reward=self.value, completion_index=item.completion_index)
            for item in inputs
        ]


def test_completion_length_cap_grader_handles_limits_and_missing_tokens():
    grader = CompletionLengthCapGrader(max_completion_tokens=5)
    lenient = CompletionLengthCapGrader(max_completion_tokens=5, treat_missing_as_fail=False)
    floored = CompletionLengthCapGrader(max_completion_tokens=5, min_reward=0.2)

    inputs = [
        GraderInput(
            sample_id="ds:0",
            record={},
            completion="a",
            reasoning=None,
            answer=None,
            completion_tokens=4,
        ),
        GraderInput(
            sample_id="ds:1",
            record={},
            completion="b",
            reasoning=None,
            answer=None,
            completion_tokens=6,
        ),
        GraderInput(
            sample_id="ds:2",
            record={},
            completion="c",
            reasoning=None,
            answer=None,
            completion_tokens=None,
        ),
    ]

    strict_results = asyncio.run(grader.grade_batch(inputs))
    lenient_results = asyncio.run(lenient.grade_batch(inputs))
    floored_results = asyncio.run(floored.grade_batch(inputs))

    assert [r.reward for r in strict_results] == [1.0, 0.0, 0.0]
    assert [r.reward for r in lenient_results] == [1.0, 0.0, 1.0]
    assert [r.reward for r in floored_results] == [1.0, 0.2, 0.2]


def test_dataset_reward_multiplies_multiplicative_graders():
    cfg = DatasetConfig(
        path=pathlib.Path("dummy.jsonl"),
        graders=["a", "b"],
        grader_weights=[1.0, 1.0],
        multiplicative_graders=["gate"],
    )
    graders = {
        "a": _ConstGrader(0.4),
        "b": _ConstGrader(0.2),
        "gate": _ConstGrader(0.5),
    }
    reward = DatasetReward(cfg, graders, builtin_reasoning_grader=None, builtin_weight=1.0)
    inputs = [
        GraderInput(
            sample_id="ds:0",
            record={},
            completion="hello",
            reasoning=None,
            answer=None,
            completion_tokens=3,
        )
    ]

    results = asyncio.run(reward.grade_batch(inputs))
    # Additive reward: (0.4 + 0.2) / 2 = 0.3, then multiplied by 0.5 -> 0.15
    assert pytest.approx(results[0].reward, rel=1e-6) == 0.15


def test_reward_fn_adds_completion_tokens(monkeypatch):
    captured = {}

    class DummyResponse:
        def __init__(self):
            self.called = True

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    class DummySession:
        def post(self, url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return DummyResponse()

    monkeypatch.setattr(rewards_module.requests, "Session", lambda: DummySession())

    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    reward_fn = rewards_module.build_remote_reward_fn(
        "http://server",
        tokenizer=DummyTokenizer(),
        max_retries=0,
        retry_backoff_seconds=0.0,
    )

    _ = reward_fn(["short", "a bit longer"], sample_id=["ds:0", "ds:1"])

    assert "json" in captured
    items = captured["json"]["items"]
    assert [item["completion_tokens"] for item in items] == [1, 3]
    assert [item["completion"] for item in items] == ["short", "a bit longer"]
    # Ensure iteration defaults to local counter (starts at 0)
    assert captured["json"]["iteration"] == 0


def test_reward_fn_uses_trainer_state_global_step(monkeypatch):
    captured = {}

    class DummyResponse:
        def __init__(self):
            self.called = True

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    class DummySession:
        def post(self, url, json=None, timeout=None):
            captured["json"] = json
            return DummyResponse()

    monkeypatch.setattr(rewards_module.requests, "Session", lambda: DummySession())

    reward_fn = rewards_module.build_remote_reward_fn(
        "http://server",
        tokenizer=None,
        max_retries=0,
        retry_backoff_seconds=0.0,
    )

    _ = reward_fn(["x"], sample_id=["s:0"], trainer_state={"global_step": 42})
    assert captured["json"]["iteration"] == 42

    # Fallback to local counter when trainer_state is missing
    _ = reward_fn(["y"], sample_id=["s:1"])
    assert captured["json"]["iteration"] == 0
