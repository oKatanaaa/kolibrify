import asyncio
import unittest

from kolibrify.rl_dataserver.graders import GraderInput, LatexOnlyAnswerGrader


class TestLatexOnlyAnswerGrader(unittest.TestCase):
    def setUp(self) -> None:
        self.grader = LatexOnlyAnswerGrader()

    def _grade(self, text: str) -> float:
        inp = GraderInput(
            sample_id="s1",
            record={"answer": "$0$"},
            completion=text,
            reasoning=None,
            answer=text,
        )
        return asyncio.run(self.grader.grade_batch([inp]))[0].reward

    def test_accepts_common_wrappers(self):
        cases = [r"$5$", r"$$5$$", r"\(5\)", r"\[5\]", r"\boxed{5}"]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(self._grade(text), 1.0)

    def test_accepts_begin_end_environment(self):
        reward = self._grade(r"\begin{align}5\end{align}")
        self.assertEqual(reward, 1.0)

    def test_rejects_text_mixed_with_latex(self):
        cases = [
            "Answer is $5$",
            " $5$ extra",
            "text \\(5\\)",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(self._grade(text), 0.0)

    def test_rejects_empty_or_plain_text(self):
        for text in ["", "five"]:
            with self.subTest(text=text):
                self.assertEqual(self._grade(text), 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
