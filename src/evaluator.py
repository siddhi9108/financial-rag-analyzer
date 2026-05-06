"""
evaluator.py
------------
Evaluates RAG pipeline quality using RAGAS metrics.
This is a SENIOR-LEVEL skill that impresses interviewers.

Metrics measured:
- Faithfulness: Does the answer stick to retrieved context?
- Answer Relevancy: Is the answer actually relevant to the question?
- Context Precision: Are retrieved chunks actually useful?
"""

from typing import List, Dict, Any
import json


def compute_faithfulness_score(
    answer: str,
    context_chunks: List[str]
) -> float:
    """
    Simple faithfulness check: what fraction of answer sentences
    can be grounded in the retrieved context?
    
    (Simplified version — full RAGAS requires an LLM judge)
    """
    if not answer or not context_chunks:
        return 0.0

    combined_context = " ".join(context_chunks).lower()
    answer_sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]

    if not answer_sentences:
        return 1.0

    grounded_count = 0
    for sentence in answer_sentences:
        # Check if key words from sentence appear in context
        words = [w for w in sentence.lower().split() if len(w) > 4]
        if not words:
            continue
        match_ratio = sum(1 for w in words if w in combined_context) / len(words)
        if match_ratio > 0.5:
            grounded_count += 1

    return round(grounded_count / len(answer_sentences), 3)


def compute_context_relevance(
    query: str,
    context_chunks: List[str]
) -> float:
    """
    Measure how relevant retrieved chunks are to the query.
    Uses keyword overlap as a proxy for semantic relevance.
    """
    if not context_chunks:
        return 0.0

    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    if not query_words:
        return 0.0

    relevance_scores = []
    for chunk in context_chunks:
        chunk_words = set(w.lower() for w in chunk.split())
        overlap = len(query_words & chunk_words) / len(query_words)
        relevance_scores.append(overlap)

    return round(sum(relevance_scores) / len(relevance_scores), 3)


def compute_answer_completeness(
    query: str,
    answer: str
) -> float:
    """
    Heuristic check: does the answer address the query?
    Based on question-type pattern matching.
    """
    query_lower = query.lower()
    answer_lower = answer.lower()

    # Signals that the model couldn't answer
    refusal_phrases = [
        "i couldn't find",
        "not mentioned",
        "no information",
        "does not contain",
        "unable to find"
    ]

    if any(phrase in answer_lower for phrase in refusal_phrases):
        return 0.3  # partial score — honest refusal is better than hallucination

    # Question type matching
    if any(w in query_lower for w in ["how much", "what was", "revenue", "profit", "loss"]):
        # Financial queries — check if numbers appear in answer
        has_numbers = any(c.isdigit() for c in answer)
        return 0.9 if has_numbers else 0.5

    if any(w in query_lower for w in ["why", "explain", "reason"]):
        # Explanation queries — check for longer answer
        return 0.9 if len(answer.split()) > 30 else 0.6

    if any(w in query_lower for w in ["list", "what are", "summarize"]):
        # List queries — check for multiple points
        has_bullets = any(c in answer for c in ["\n-", "\n•", "\n*", "1.", "2."])
        return 0.9 if has_bullets else 0.7

    return 0.75  # default moderate score


def evaluate_response(
    query: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Full evaluation of a single RAG response.
    
    Returns:
        Dict with individual metrics and overall quality score
    """
    context_texts = [c["text"] for c in retrieved_chunks]
    retrieval_scores = [c["score"] for c in retrieved_chunks]

    faithfulness = compute_faithfulness_score(answer, context_texts)
    context_relevance = compute_context_relevance(query, context_texts)
    completeness = compute_answer_completeness(query, answer)
    avg_retrieval_score = round(sum(retrieval_scores) / len(retrieval_scores), 3) if retrieval_scores else 0.0

    # Weighted overall score
    overall = round(
        faithfulness * 0.35 +
        context_relevance * 0.25 +
        completeness * 0.25 +
        avg_retrieval_score * 0.15,
        3
    )

    return {
        "faithfulness": faithfulness,
        "context_relevance": context_relevance,
        "answer_completeness": completeness,
        "avg_retrieval_score": avg_retrieval_score,
        "overall_quality": overall,
        "grade": _get_grade(overall),
        "num_chunks_retrieved": len(retrieved_chunks)
    }


def _get_grade(score: float) -> str:
    if score >= 0.85:
        return "A — Excellent"
    elif score >= 0.70:
        return "B — Good"
    elif score >= 0.55:
        return "C — Acceptable"
    elif score >= 0.40:
        return "D — Needs Improvement"
    else:
        return "F — Poor"


def run_batch_evaluation(
    test_cases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate the pipeline on a set of test Q&A pairs.
    
    Each test case should have: query, answer, retrieved_chunks
    """
    results = []
    for case in test_cases:
        eval_result = evaluate_response(
            query=case["query"],
            answer=case["answer"],
            retrieved_chunks=case["retrieved_chunks"]
        )
        results.append({
            "query": case["query"],
            **eval_result
        })

    # Aggregate stats
    avg_metrics = {
        "avg_faithfulness": round(sum(r["faithfulness"] for r in results) / len(results), 3),
        "avg_context_relevance": round(sum(r["context_relevance"] for r in results) / len(results), 3),
        "avg_completeness": round(sum(r["answer_completeness"] for r in results) / len(results), 3),
        "avg_overall_quality": round(sum(r["overall_quality"] for r in results) / len(results), 3),
    }

    return {
        "individual_results": results,
        "aggregate": avg_metrics,
        "total_evaluated": len(results)
    }


if __name__ == "__main__":
    # Demo evaluation
    test_cases = [
        {
            "query": "What was the revenue?",
            "answer": "Total revenue was $394.3 billion, a 7.8% increase year over year.",
            "retrieved_chunks": [
                {
                    "text": "Total net revenue of $394.3 billion for fiscal year 2022, compared to $365.8 billion the previous year, representing a 7.8% increase.",
                    "source": "annual_report.pdf",
                    "page": 5,
                    "score": 0.91
                }
            ]
        }
    ]

    result = run_batch_evaluation(test_cases)
    print(json.dumps(result, indent=2))
