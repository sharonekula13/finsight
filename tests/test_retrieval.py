import unittest

from finsight.chunking import chunk_document
from finsight.pipeline import load_sample_corpus
from finsight.retrieval import hybrid_retrieve


class TestChunking(unittest.TestCase):
    def test_overlap_and_ids(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunks = chunk_document("doc", text, size=120, overlap=30)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "doc#0")
        # consecutive chunks should share overlapping words
        first_tail = chunks[0].text.split()[-30:]
        second_head = chunks[1].text.split()[:30]
        self.assertEqual(first_tail, second_head)

    def test_bad_overlap(self):
        with self.assertRaises(ValueError):
            chunk_document("d", "a b c", size=10, overlap=10)


class TestRetrieval(unittest.TestCase):
    def test_retrieves_relevant_doc(self):
        store = load_sample_corpus()
        ctx = hybrid_retrieve(store, "NovaTech total revenue fiscal 2024", top_k=4)
        docs = {c.chunk.doc_id for c in ctx.chunks}
        self.assertIn("novatech_10k", docs)


if __name__ == "__main__":
    unittest.main()
