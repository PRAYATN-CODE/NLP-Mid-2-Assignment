"""
Automated Test Suite for FAQSense Retrieval Chatbot.
Verifies normal, paraphrased, out-of-scope, and edge case queries
on the University FAQ Knowledge Base.
"""

import unittest
from chatbot.retrieval import FAQRetrievalBot
from chatbot.preprocessing import preprocess_text, clean_text
from chatbot.config import DEFAULT_THRESHOLD, FALLBACK_RESPONSE, EMPTY_QUERY_RESPONSE


class TestFAQSenseRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize retrieval bot once for all tests."""
        cls.bot = FAQRetrievalBot()

    def test_knowledge_base_loaded(self):
        """Verify FAQ dataset is properly loaded with 90 entries and 10+ categories."""
        self.assertGreaterEqual(self.bot.total_faqs, 90, "FAQ dataset should contain at least 90 questions.")
        self.assertGreaterEqual(len(self.bot.categories), 10, "Should have at least 10 categories.")

    def test_preprocessing_pipeline(self):
        """Verify preprocessing converts to lowercase, cleans punctuation, and normalizes."""
        text = "Does the university have Wi-Fi?!"
        processed = preprocess_text(text)
        self.assertNotIn("?", processed)
        self.assertNotIn("!", processed)
        self.assertIn("wifi", processed)
        self.assertEqual(processed.lower(), processed)

    def test_exact_faq_match(self):
        """Verify exact FAQ question yields high similarity score (> 0.70)."""
        query = "When was Dr. Harisingh Gour Vishwavidyalaya established?"
        result = self.bot.retrieve_answer(query, threshold=DEFAULT_THRESHOLD)
        self.assertTrue(result["found"], "Exact query should be found.")
        self.assertGreater(result["score"], 0.70, "Exact match should have score > 0.70.")
        self.assertIn("18 July 1946", result["answer"])

    def test_paraphrased_faq_match(self):
        """Verify paraphrased query matches the expected FAQ answer."""
        query = "When was the university established?"
        result = self.bot.retrieve_answer(query, threshold=DEFAULT_THRESHOLD)
        self.assertTrue(result["found"], "Paraphrased established query should match established FAQ.")
        self.assertIn("18 July 1946", result["answer"])

    def test_hostel_query(self):
        """Verify hostel queries match Hostel category."""
        query = "How can I apply for a hostel?"
        result = self.bot.retrieve_answer(query, threshold=DEFAULT_THRESHOLD)
        self.assertTrue(result["found"])
        self.assertEqual(result["category"], "Hostel")
        self.assertIn("hostel rules", result["answer"].lower())

    def test_admission_query(self):
        """Verify admission queries match Admission category."""
        query = "Does the university use CUET for admission?"
        result = self.bot.retrieve_answer(query, threshold=DEFAULT_THRESHOLD)
        self.assertTrue(result["found"])
        self.assertEqual(result["category"], "Admission")
        self.assertIn("CUET", result["answer"])

    def test_out_of_scope_irrelevant_query(self):
        """Verify irrelevant query triggers safe fallback response."""
        query = "Who won yesterday's football match in Barcelona?"
        result = self.bot.retrieve_answer(query, threshold=DEFAULT_THRESHOLD)
        self.assertFalse(result["found"], "Irrelevant query should trigger fallback.")
        self.assertEqual(result["answer"], FALLBACK_RESPONSE)
        self.assertLess(result["score"], DEFAULT_THRESHOLD)

    def test_empty_query(self):
        """Verify empty query is handled gracefully without exception."""
        result = self.bot.retrieve_answer("", threshold=DEFAULT_THRESHOLD)
        self.assertFalse(result["found"])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["answer"], EMPTY_QUERY_RESPONSE)

    def test_whitespace_query(self):
        """Verify whitespace only query is handled gracefully."""
        result = self.bot.retrieve_answer("     \n\t  ", threshold=DEFAULT_THRESHOLD)
        self.assertFalse(result["found"])
        self.assertEqual(result["score"], 0.0)

    def test_single_word_query(self):
        """Verify short single-word input executes cleanly."""
        result = self.bot.retrieve_answer("library", threshold=DEFAULT_THRESHOLD)
        self.assertIsInstance(result["score"], float)
        self.assertIn("answer", result)

    def test_long_natural_query(self):
        """Verify long query retrieves relevant FAQ."""
        query = (
            "I am a candidate looking to apply for university admission and want to know "
            "what documents are generally required during admission counselling."
        )
        result = self.bot.retrieve_answer(query, threshold=DEFAULT_THRESHOLD)
        self.assertTrue(result["found"])
        self.assertEqual(result["category"], "Admission")
        self.assertIn("counselling", result["matched_question"].lower())

    def test_threshold_sensitivity(self):
        """Verify raising threshold filters out marginal matches."""
        query = "sports"
        low_res = self.bot.retrieve_answer(query, threshold=0.05)
        high_res = self.bot.retrieve_answer(query, threshold=0.99)
        self.assertFalse(high_res["found"], "Strict threshold of 0.99 should reject marginal match.")


if __name__ == "__main__":
    unittest.main()
