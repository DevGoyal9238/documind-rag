"""
DocuMind - Unit Tests for Text Chunking Module
Covers edge cases, input validation, overlap verification, and metadata preservation.
"""

import unittest
from src.text_chunker import chunk_pages


class TestTextChunker(unittest.TestCase):

    def test_short_text(self):
        """Text shorter than chunk_size should produce exactly one chunk."""
        pages = [{"page_number": 1, "text": "Short text sample."}]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], 1)
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertEqual(chunks[0]["text"], "Short text sample.")

    def test_exact_chunk_size_text(self):
        """Text whose length matches chunk_size exactly should produce exactly one chunk."""
        exact_text = "A" * 500
        pages = [{"page_number": 1, "text": exact_text}]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]["text"]), 500)
        self.assertEqual(chunks[0]["text"], exact_text)
        self.assertEqual(chunks[0]["chunk_id"], 1)
        self.assertEqual(chunks[0]["page_number"], 1)

    def test_text_longer_than_chunk_size(self):
        """Text longer than chunk_size should be split across multiple overlapping chunks."""
        # 650 characters with chunk_size=500 and chunk_overlap=100:
        # Step = 400. Chunks: [0:500] (len 500), [400:650] (len 250). Total = 2 chunks.
        text = "A" * 400 + "B" * 100 + "C" * 150  # 650 chars total
        pages = [{"page_number": 1, "text": text}]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]["text"]), 500)
        self.assertEqual(len(chunks[1]["text"]), 250)
        self.assertEqual(chunks[0]["chunk_id"], 1)
        self.assertEqual(chunks[1]["chunk_id"], 2)

    def test_overlap_correctness(self):
        """Verify that the overlap region between consecutive chunks is identical."""
        prefix = "A" * 400
        overlap_content = "OVERLAP1234567890" + "X" * 83  # 100 characters
        suffix = "B" * 150
        text = prefix + overlap_content + suffix  # 650 chars total

        pages = [{"page_number": 1, "text": text}]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)

        self.assertEqual(len(chunks), 2)
        # End of chunk 1 (last 100 chars) must match start of chunk 2 (first 100 chars)
        chunk1_tail = chunks[0]["text"][-100:]
        chunk2_head = chunks[1]["text"][:100]
        self.assertEqual(chunk1_tail, chunk2_head)
        self.assertEqual(chunk1_tail, overlap_content)

    def test_empty_and_whitespace_only_pages(self):
        """Empty and whitespace-only pages should be skipped and emit zero chunks."""
        pages = [
            {"page_number": 1, "text": ""},
            {"page_number": 2, "text": "   \n\t   \n  "},
            {"page_number": 3, "text": "Valid content on page 3."},
        ]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page_number"], 3)
        self.assertEqual(chunks[0]["text"], "Valid content on page 3.")
        self.assertEqual(chunks[0]["chunk_id"], 1)

    def test_multiple_pages_and_page_number_preservation(self):
        """Verify that multiple pages each preserve their own page_number and do not merge."""
        pages = [
            {"page_number": 1, "text": "Page 1 start text. " + "1" * 600},
            {"page_number": 2, "text": "Page 2 independent text. " + "2" * 600},
        ]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)

        # Each 600+ char page produces 2 chunks, so 4 chunks total
        self.assertEqual(len(chunks), 4)

        # Check page numbers
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertEqual(chunks[1]["page_number"], 1)
        self.assertEqual(chunks[2]["page_number"], 2)
        self.assertEqual(chunks[3]["page_number"], 2)

        # Verify sequential chunk_ids across pages
        self.assertEqual([c["chunk_id"] for c in chunks], [1, 2, 3, 4])

        # Page 2 chunks must NOT contain any text from page 1
        self.assertNotIn("Page 1", chunks[2]["text"])
        self.assertNotIn("Page 1", chunks[3]["text"])

    def test_preserves_internal_whitespace(self):
        """Ensure non-empty chunks preserve original whitespace and formatting."""
        raw_text = "   Indented text with \nnewlines\n and spaces.   "
        pages = [{"page_number": 1, "text": raw_text}]
        chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], raw_text)

    def test_invalid_chunk_size(self):
        """Invalid chunk_size values should raise appropriate exceptions."""
        pages = [{"page_number": 1, "text": "Hello world"}]

        # Zero or negative
        with self.assertRaises(ValueError):
            chunk_pages(pages, chunk_size=0, chunk_overlap=0)
        with self.assertRaises(ValueError):
            chunk_pages(pages, chunk_size=-50, chunk_overlap=0)

        # Non-integer types
        with self.assertRaises(TypeError):
            chunk_pages(pages, chunk_size="500", chunk_overlap=100)
        with self.assertRaises(TypeError):
            chunk_pages(pages, chunk_size=500.5, chunk_overlap=100)
        with self.assertRaises(TypeError):
            chunk_pages(pages, chunk_size=True, chunk_overlap=0)

    def test_invalid_chunk_overlap(self):
        """Invalid chunk_overlap values should raise appropriate exceptions."""
        pages = [{"page_number": 1, "text": "Hello world"}]

        # Negative overlap
        with self.assertRaises(ValueError):
            chunk_pages(pages, chunk_size=500, chunk_overlap=-10)

        # Overlap equal to chunk_size
        with self.assertRaises(ValueError):
            chunk_pages(pages, chunk_size=500, chunk_overlap=500)

        # Overlap greater than chunk_size
        with self.assertRaises(ValueError):
            chunk_pages(pages, chunk_size=500, chunk_overlap=600)

        # Non-integer types
        with self.assertRaises(TypeError):
            chunk_pages(pages, chunk_size=500, chunk_overlap="100")
        with self.assertRaises(TypeError):
            chunk_pages(pages, chunk_size=500, chunk_overlap=False)

    def test_malformed_page_entries(self):
        """Malformed page inputs or entries should raise TypeError or ValueError."""
        # pages is not a list
        with self.assertRaises(TypeError):
            chunk_pages("not a list")
        with self.assertRaises(TypeError):
            chunk_pages({"page_number": 1, "text": "hi"})

        # Element in pages is not a dict
        with self.assertRaises(TypeError):
            chunk_pages(["not a dict"])

        # Missing required keys
        with self.assertRaises(ValueError):
            chunk_pages([{"page_number": 1}])
        with self.assertRaises(ValueError):
            chunk_pages([{"text": "missing page_number"}])

        # Wrong key types
        with self.assertRaises(TypeError):
            chunk_pages([{"page_number": "1", "text": "string page number"}])
        with self.assertRaises(TypeError):
            chunk_pages([{"page_number": True, "text": "bool page number"}])
        with self.assertRaises(TypeError):
            chunk_pages([{"page_number": 1, "text": 12345}])

    def test_invalid_page_number(self):
        """Page numbers less than or equal to zero should raise ValueError."""
        with self.assertRaises(ValueError):
            chunk_pages([{"page_number": 0, "text": "zero page"}])
        with self.assertRaises(ValueError):
            chunk_pages([{"page_number": -5, "text": "negative page"}])


if __name__ == "__main__":
    unittest.main()
