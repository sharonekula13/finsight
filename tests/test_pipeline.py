import unittest

from finsight.eval.dataset import GOLD
from finsight.eval.metrics import answer_relevance, context_precision
from finsight.pipeline import answer, load_sample_corpus


class TestMetrics(unittest.TestCase):
    def test_context_precision(self):
        self.assertEqual(context_precision(["novatech_10k#0", "helios_10k#1"], ("novatech_10k",)), 0.5)
        self.assertEqual(context_precision([], ("novatech_10k",)), 0.0)

    def test_answer_relevance_bounds(self):
        self.assertAlmostEqual(answer_relevance("revenue 4.2 billion", "revenue 4.2 billion"), 1.0)
        self.assertEqual(answer_relevance("", "x"), 0.0)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.store = load_sample_corpus()

    def test_answers_are_grounded(self):
        for item in GOLD:
            res = answer(item.question, self.store, model="mock-a")
            self.assertGreaterEqual(res.grounded_ratio, 0.5, item.question)

    def test_retrieves_gold_docs(self):
        for item in GOLD:
            res = answer(item.question, self.store, model="mock-a")
            self.assertGreater(context_precision(res.citations, item.gold_docs), 0.0, item.question)

    def test_comparison_routes_wide(self):
        res = answer(GOLD[5].question, self.store, model="mock-a")
        self.assertEqual(res.route["kind"], "comparison")
        self.assertGreaterEqual(res.route["top_k"], 6)


if __name__ == "__main__":
    unittest.main()
