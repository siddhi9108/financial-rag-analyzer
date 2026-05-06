"""
test_pipeline.py
----------------
Unit tests for the Financial RAG pipeline.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from ingestion import chunk_text, DocumentChunk
from embeddings import embed_texts, embed_query, cosine_similarity
from evaluator import (
    compute_faithfulness_score,
    compute_context_relevance,
    compute_answer_completeness,
    evaluate_response
)


# ─────────────────────────────────────────────
# Ingestion Tests
# ─────────────────────────────────────────────

class TestChunking:
    def test_basic_chunking(self):
        """Chunking should split text into manageable pieces."""
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 110 for c in chunks)  # allow slight overflow

    def test_overlap_exists(self):
        """Consecutive chunks should share some words."""
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        if len(chunks) >= 2:
            words1 = set(chunks[0].split())
            words2 = set(chunks[1].split())
            assert len(words1 & words2) > 0, "Chunks should overlap"

    def test_short_text(self):
        """Short text should produce exactly one chunk."""
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=512, overlap=50)
        assert len(chunks) == 1

    def test_empty_text(self):
        """Empty text should return empty list."""
        chunks = chunk_text("", chunk_size=512, overlap=50)
        assert chunks == [] or (len(chunks) == 1 and chunks[0].strip() == "")


# ─────────────────────────────────────────────
# Embedding Tests
# ─────────────────────────────────────────────

class TestEmbeddings:
    def test_embed_single_text(self):
        """Single text should return an embedding of the right shape."""
        embeddings = embed_texts(["Total revenue was $394 billion."])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384  # MiniLM-L6-v2 dimension

    def test_embed_multiple_texts(self):
        """Multiple texts should each get their own embedding."""
        texts = ["Revenue grew 10%.", "Expenses fell by 5%.", "Net profit increased."]
        embeddings = embed_texts(texts)
        assert len(embeddings) == 3

    def test_similar_texts_have_high_similarity(self):
        """Semantically similar texts should have high cosine similarity."""
        emb1 = embed_query("What was the annual revenue?")
        emb2 = embed_query("How much money did the company earn?")
        sim = cosine_similarity(emb1, emb2)
        assert sim > 0.6, f"Similar queries should have similarity > 0.6, got {sim}"

    def test_different_texts_have_lower_similarity(self):
        """Very different texts should have low cosine similarity."""
        emb1 = embed_query("Total revenue this quarter")
        emb2 = embed_query("The weather is sunny today")
        sim = cosine_similarity(emb1, emb2)
        assert sim < 0.8, f"Different topics should have similarity < 0.8, got {sim}"

    def test_embedding_is_normalized(self):
        """Normalized embeddings should have magnitude close to 1."""
        import math
        emb = embed_query("test query")
        magnitude = math.sqrt(sum(x**2 for x in emb))
        assert abs(magnitude - 1.0) < 0.01, f"Embedding should be normalized, got magnitude {magnitude}"


# ─────────────────────────────────────────────
# Evaluator Tests
# ─────────────────────────────────────────────

class TestEvaluator:
    def test_faithfulness_high_when_grounded(self):
        """Answer using words from context should score high faithfulness."""
        answer = "The total revenue was 394 billion dollars, which represents a 7.8 percent increase."
        context = ["Total net revenue of $394.3 billion, a 7.8% increase year over year."]
        score = compute_faithfulness_score(answer, context)
        assert score >= 0.5, f"Grounded answer should score >= 0.5, got {score}"

    def test_faithfulness_low_when_ungrounded(self):
        """Answer with completely different content should score low."""
        answer = "The CEO announced a major acquisition of a European company worth $50 billion."
        context = ["Total net revenue was $394 billion for fiscal year 2022."]
        score = compute_faithfulness_score(answer, context)
        assert score <= 0.7

    def test_context_relevance_high_on_match(self):
        """Context that matches the query should score high relevance."""
        query = "What was the revenue?"
        context = ["Total revenue was $394 billion, up 7.8% from the previous year."]
        score = compute_context_relevance(query, context)
        assert score > 0.3

    def test_context_relevance_low_on_mismatch(self):
        """Context unrelated to query should score lower."""
        query = "What was the revenue?"
        context = ["The company has 150,000 employees worldwide across 30 countries."]
        score = compute_context_relevance(query, context)
        # Revenue not mentioned, so should be low
        assert score < 0.6

    def test_evaluate_response_returns_all_fields(self):
        """Full evaluation should return all expected fields."""
        result = evaluate_response(
            query="What was the net income?",
            answer="Net income was $95.2 billion, an increase of 5% year over year.",
            retrieved_chunks=[{
                "text": "Net income for fiscal year 2022 was $95.2 billion.",
                "source": "annual_report.pdf",
                "page": 10,
                "score": 0.88
            }]
        )
        required_fields = [
            "faithfulness", "context_relevance", "answer_completeness",
            "avg_retrieval_score", "overall_quality", "grade"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_overall_score_is_bounded(self):
        """Overall quality score should be between 0 and 1."""
        result = evaluate_response(
            query="Summarize the report.",
            answer="The company performed well with strong revenue growth.",
            retrieved_chunks=[{
                "text": "Revenue grew significantly this quarter.",
                "source": "report.pdf",
                "page": 1,
                "score": 0.75
            }]
        )
        assert 0.0 <= result["overall_quality"] <= 1.0

    def test_refusal_answer_scores_partial(self):
        """A polite refusal should score above 0 (honest is better than hallucinating)."""
        result = evaluate_response(
            query="What was the dividend?",
            answer="I couldn't find sufficient information about dividends in the provided documents.",
            retrieved_chunks=[{
                "text": "Revenue was $100 billion this year.",
                "source": "report.pdf",
                "page": 1,
                "score": 0.45
            }]
        )
        assert result["answer_completeness"] > 0.0


# ─────────────────────────────────────────────
# Integration Test (no LLM required)
# ─────────────────────────────────────────────

class TestIntegration:
    def test_end_to_end_without_llm(self):
        """Test chunking + embedding without needing LLM or ChromaDB."""
        # Sample financial text
        text = """
        Apple Inc. reported total net revenue of $394.3 billion for fiscal year 2022,
        compared to $365.8 billion in 2021, representing a growth of 7.8%. 
        Products revenue was $316.2 billion and Services revenue was $78.1 billion.
        Operating income for the period was $119.4 billion, with an operating margin of 30.3%.
        """

        # Chunk it
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) >= 1

        # Embed it
        embeddings = embed_texts(chunks)
        assert len(embeddings) == len(chunks)

        # Query embedding
        query_emb = embed_query("What was Apple's revenue?")
        assert len(query_emb) == 384

        # Check similarity
        sims = [cosine_similarity(query_emb, emb) for emb in embeddings]
        best_match_idx = sims.index(max(sims))
        assert "revenue" in chunks[best_match_idx].lower() or max(sims) > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
