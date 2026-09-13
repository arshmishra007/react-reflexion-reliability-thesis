import random
import unittest
from pathlib import Path

from src.data_loader import load_hotpotqa, select_supporting_plus_distractors


def _titles(context):
    return [item[0] for item in context]


class SupportingContextSelectionTests(unittest.TestCase):
    def setUp(self):
        self.context = [
            ["Support A", ["A0", "A1"]],
            ["Distractor 1", ["D1"]],
            ["Support B", ["B0"]],
            ["Distractor 2", ["D2"]],
            ["Distractor 3", ["D3"]],
        ]

    def select(self, supporting_facts, seed=42, max_documents=4):
        return select_supporting_plus_distractors(
            self.context,
            supporting_facts,
            random.Random(seed),
            max_documents=max_documents,
            question_id="test-question",
        )

    def test_repeated_facts_select_supporting_document_once(self):
        selected = self.select([["Support A", 0], ["Support A", 1]])
        self.assertEqual(_titles(selected).count("Support A"), 1)

    def test_two_supporting_titles_are_both_included(self):
        selected = self.select([["Support A", 0], ["Support B", 0]])
        self.assertEqual(_titles(selected)[:2], ["Support A", "Support B"])

    def test_distractors_never_replace_supporting_documents(self):
        selected = self.select(
            [["Support A", 0], ["Support B", 0]],
            max_documents=2,
        )
        self.assertEqual(_titles(selected), ["Support A", "Support B"])

    def test_supporting_documents_can_exceed_limit(self):
        selected = self.select(
            [["Support A", 0], ["Support B", 0]],
            max_documents=1,
        )
        self.assertEqual(_titles(selected), ["Support A", "Support B"])

    def test_missing_supporting_title_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "supporting title not found in raw context: Missing",
        ):
            self.select([["Missing", 0]])

    def test_invalid_supporting_sentence_index_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "invalid supporting sentence index 2 for title Support A",
        ):
            self.select([["Support A", 2]])

    def test_selection_is_deterministic_for_same_seed(self):
        facts = [["Support A", 0]]
        self.assertEqual(
            self.select(facts, seed=7),
            self.select(facts, seed=7),
        )

    def test_different_seeds_only_change_distractors(self):
        facts = [["Support A", 0], ["Support B", 0]]
        first = self.select(facts, seed=1, max_documents=3)
        second = self.select(facts, seed=5, max_documents=3)
        self.assertEqual(_titles(first)[:2], ["Support A", "Support B"])
        self.assertEqual(_titles(second)[:2], ["Support A", "Support B"])
        self.assertNotEqual(_titles(first)[2:], _titles(second)[2:])


class KnownQuestionContextTests(unittest.TestCase):
    def test_known_smoke_questions_include_all_supporting_titles(self):
        raw_path = Path(__file__).parents[1] / "data" / "raw" / "hotpot_dev.json"
        wanted_ids = {
            "5ae143ed55429920d5234360",
            "5abc19705542993a06baf86e",
            "5ac3e0f7554299194317388b",
        }
        records = {
            record["_id"]: record
            for record in load_hotpotqa(raw_path)
            if record.get("_id") in wanted_ids
        }
        self.assertEqual(set(records), wanted_ids)

        for question_id, record in records.items():
            with self.subTest(question_id=question_id):
                selected = select_supporting_plus_distractors(
                    record["context"],
                    record["supporting_facts"],
                    random.Random(42),
                    max_documents=6,
                    question_id=question_id,
                )
                required_titles = {fact[0] for fact in record["supporting_facts"]}
                self.assertTrue(required_titles <= set(_titles(selected)))


if __name__ == "__main__":
    unittest.main()
