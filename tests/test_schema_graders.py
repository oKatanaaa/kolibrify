import asyncio
import json
import unittest

from kolibrify.rl_dataserver.graders import (
    GraderInput,
    JsonSchemaGrader,
    XmlSchemaGrader,
)


def _schema_metadata(schema_obj: dict) -> dict:
    return {"schema": json.dumps(schema_obj)}


class TestJsonSchemaGrader(unittest.TestCase):
    def setUp(self) -> None:
        self.grader = JsonSchemaGrader()

    def _grade(self, completion: str, allow_additional: bool = False) -> float:
        schema = {
            "type": "object",
            "required": ["answer"],
            "allow_additional_properties": allow_additional,
        }
        metadata = _schema_metadata(schema)
        record = {"metadata": metadata}
        inputs = [
            GraderInput(
                sample_id="s1",
                record=record,
                completion=completion,
                reasoning=None,
                answer=None,
            )
        ]
        return asyncio.run(self.grader.grade_batch(inputs))[0].reward

    def test_valid_exact_schema(self):
        reward = self._grade('{"answer": 1}')
        self.assertEqual(reward, 1.0)

    def test_missing_required(self):
        reward = self._grade('{"not_answer": 1}', allow_additional=True)
        self.assertEqual(reward, 0.8)

    def test_extra_keys(self):
        reward = self._grade('{"answer": 1, "extra": 2}')
        self.assertEqual(reward, 0.9)

    def test_missing_and_extra(self):
        reward = self._grade('{"another": 2}')
        self.assertEqual(reward, 0.7)

    def test_non_object_json(self):
        reward = self._grade('[1, 2]')
        self.assertEqual(reward, 0.5)

    def test_relaxed_parsing_fenced(self):
        reward = self._grade('```json\n{"answer": 1, "extra": 2}\n```')
        self.assertAlmostEqual(reward, 0.81)

    def test_relaxed_literal_eval(self):
        reward = self._grade("{'answer': 1}")
        self.assertEqual(reward, 0.9)

    def test_unparsable(self):
        reward = self._grade("not json")
        self.assertEqual(reward, 0.0)


class TestXmlSchemaGrader(unittest.TestCase):
    def setUp(self) -> None:
        self.grader = XmlSchemaGrader()
        self.schema = {"root_tag": "answer"}

    def _grade(self, completion: str, schema_obj: dict | None = None) -> float:
        metadata = _schema_metadata(schema_obj or self.schema)
        record = {"metadata": metadata}
        inputs = [
            GraderInput(
                sample_id="s1",
                record=record,
                completion=completion,
                reasoning=None,
                answer=None,
            )
        ]
        return asyncio.run(self.grader.grade_batch(inputs))[0].reward

    def test_correct_root(self):
        reward = self._grade("<answer>ok</answer>")
        self.assertEqual(reward, 1.0)

    def test_wrong_root(self):
        reward = self._grade("<result/>")
        self.assertEqual(reward, 0.8)

    def test_malformed_xml(self):
        reward = self._grade("<answer>")
        self.assertEqual(reward, 0.0)

    def test_relaxed_parsing_fenced_correct(self):
        reward = self._grade("```xml\n<answer>ok</answer>\n```")
        self.assertAlmostEqual(reward, 0.9)

    def test_relaxed_parsing_fenced_wrong_root(self):
        reward = self._grade("```xml\n<wrong/>\n```")
        self.assertAlmostEqual(reward, 0.72)

    def test_missing_tag_in_schema(self):
        reward = self._grade("<answer/>", {"root_tag": None})
        self.assertEqual(reward, 0.5)
