from __future__ import annotations

from typing import List, Sequence

from kolibrify.rl_dataserver.graders import GradeResult, Grader, GraderInput


class RussianReasoningGrader(Grader):
    """Reward completions whose reasoning block is primarily in Russian."""

    def __init__(self, min_letters: int = 30) -> None:
        self.min_letters = min_letters

    async def grade_batch(self, inputs: Sequence[GraderInput]) -> List[GradeResult]:
        results: List[GradeResult] = []
        for item in inputs:
            reasoning = item.reasoning or item.completion
            # Count Cyrillic vs total alphabetic characters to keep the heuristic simple.
            cyrillic = sum(1 for ch in reasoning if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
            letters = sum(1 for ch in reasoning if ch.isalpha())

            if letters == 0 or cyrillic == 0:
                reward = 0.0
            else:
                ratio = cyrillic / letters
                length_bonus = 0.05 if letters >= self.min_letters else 0.0
                reward = max(0.0, min(1.0, ratio + length_bonus))

            results.append(
                GradeResult(
                    sample_id=item.sample_id,
                    reward=reward,
                    completion_index=item.completion_index,
                )
            )
        return results
