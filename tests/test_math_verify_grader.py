import asyncio
import unittest

from kolibrify.rl_dataserver.graders import GraderInput, MathVerifyGrader


class TestMathVerifyGrader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Larger default timeouts keep math_verify comfortable on slower machines.
        cls.grader = MathVerifyGrader(parsing_timeout=5, verification_timeout=5)

    def _grade(self, answer: str | None, gold: str) -> float:
        inp = GraderInput(
            sample_id="s1",
            record={"answer": gold},
            completion=answer or "",
            reasoning=None,
            answer=answer,
        )
        return asyncio.run(self.grader.grade_batch([inp]))[0].reward

    def test_rewards_correct_answer(self):
        reward = self._grade(r"$5$", r"$5$")
        self.assertEqual(reward, 1.0)

    def test_rewards_varied_latex_wrappers(self):
        variants = [r"$5$", r"$$5$$", r"\boxed{5}"]
        for text in variants:
            with self.subTest(text=text):
                self.assertEqual(self._grade(text, r"$5$"), 1.0)

    def test_cross_guard_still_equals(self):
        pairs = [
            (r"$$5$$", r"$5$"),
            (r"$5$", r"\boxed{5}"),
            ("The answer is $$5$$", r"\boxed{5}"),
        ]
        for pred, gold in pairs:
            with self.subTest(pred=pred, gold=gold):
                self.assertEqual(self._grade(pred, gold), 1.0)

    def test_parses_when_mixed_with_text(self):
        samples = [
            "The final answer is $5$.",
            "After reasoning, we get $$5$$ as the result.",
        ]
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(self._grade(text, r"$5$"), 1.0)

    def test_partial_reward_when_parseable_but_wrong(self):
        reward = self._grade(r"$6$", r"$5$")
        self.assertEqual(reward, 0.5)

    def test_zero_reward_when_missing_answer(self):
        reward = self._grade("", r"$5$")
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
