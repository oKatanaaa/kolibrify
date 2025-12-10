import asyncio
import unittest

from kolibrify.rl_dataserver.graders import CategoryMatchGrader, GraderInput


class TestCategoryMatchGrader(unittest.TestCase):
    def setUp(self) -> None:
        self.allowed = ["Alpha", "Beta"]
        self.grader = CategoryMatchGrader(self.allowed)
        self.record = {"answer": "Alpha"}

    def _grade(self, response: str, record: dict | None = None, answer: str | None = None) -> float:
        inp = GraderInput(
            sample_id="s1",
            record=record or self.record,
            completion=response,
            reasoning=None,
            answer=answer if answer is not None else response,
        )
        return asyncio.run(self.grader.grade_batch([inp]))[0].reward

    def test_exact_match_case_sensitive(self):
        reward = self._grade("Alpha")
        self.assertEqual(reward, 1.0)

    def test_case_insensitive_match(self):
        reward = self._grade("alpha")
        self.assertEqual(reward, 0.8)

    def test_contains_expected_with_extra_text(self):
        reward = self._grade("alpha with trailing words")
        self.assertEqual(reward, 0.5)

    def test_wrong_but_allowed_category(self):
        reward = self._grade("Beta")
        self.assertEqual(reward, 0.3)

    def test_no_match_or_allowed_category(self):
        reward = self._grade("Gamma")
        self.assertEqual(reward, 0.0)

    def test_uses_completion_when_answer_missing(self):
        record = {"expected_category": "Beta"}
        reward = self._grade("beta extra", record=record, answer=None)
        self.assertEqual(reward, 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
