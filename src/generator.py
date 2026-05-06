"""
generator.py
------------
Generates answers using a local LLM via Ollama (free, no API key needed).
Default model: Mistral 7B — great for financial text understanding.

Install Ollama: https://ollama.com
Pull model: ollama pull mistral
"""

import ollama
from typing import List, Dict, Any


DEFAULT_MODEL = "mistral"

SYSTEM_PROMPT = """You are a precise financial document analyst. Your job is to answer questions 
based ONLY on the provided document excerpts. 

Rules:
1. Base your answer strictly on the provided context. Do NOT use outside knowledge.
2. If the context doesn't contain enough information, say: "I couldn't find sufficient information in the provided documents."
3. Always cite which page/document your answer comes from.
4. For numbers and figures, be exact — do not round or approximate unless the source does.
5. If asked about trends, compare explicitly using numbers from the text.
6. Keep answers clear and professional."""


def build_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Build the full prompt by combining retrieved context with the user query.
    """
    if not retrieved_chunks:
        return f"Question: {query}\n\nNo relevant context found."

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}, Relevance: {chunk['score']:.2%}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Based on the following financial document excerpts, answer the question below.

=== DOCUMENT EXCERPTS ===
{context}

=== QUESTION ===
{query}

=== YOUR ANSWER ===
Provide a clear, factual answer with citations to the sources above:"""

    return prompt


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1  # low temp = more factual, less creative
) -> Dict[str, Any]:
    """
    Generate an answer using the local Ollama LLM.
    
    Args:
        query: User's question
        retrieved_chunks: Context from vector retrieval
        model: Ollama model name
        temperature: 0.0 = deterministic, 1.0 = creative
    
    Returns:
        Dict with 'answer', 'model', 'sources_used'
    """
    prompt = build_prompt(query, retrieved_chunks)

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": temperature,
                "num_predict": 1024,  # max tokens in response
            }
        )

        answer = response["message"]["content"].strip()

        # Extract unique sources cited
        sources_used = list({
            f"{c['source']} (Page {c['page']})"
            for c in retrieved_chunks
        })

        return {
            "answer": answer,
            "model": model,
            "sources_used": sources_used,
            "num_chunks_used": len(retrieved_chunks),
            "prompt_preview": prompt[:500] + "..."
        }

    except ollama.ResponseError as e:
        if "model not found" in str(e).lower():
            return {
                "answer": f"❌ Model '{model}' not found. Run: `ollama pull {model}`",
                "model": model,
                "sources_used": [],
                "num_chunks_used": 0,
                "error": str(e)
            }
        raise


def check_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        return any(model in m for m in available)
    except Exception:
        return False


def list_available_models() -> List[str]:
    try:
        models = ollama.list()
        return [m.model for m in models.models]
    except Exception:
        return []


if __name__ == "__main__":
    # Test generation
    test_chunks = [
        {
            "text": "Apple Inc. reported total net revenue of $394.3 billion for fiscal year 2022, compared to $365.8 billion the previous year, representing a 7.8% increase.",
            "source": "apple_annual_report_2022.pdf",
            "page": 23,
            "score": 0.92
        }
    ]

    result = generate_answer(
        query="What was Apple's revenue growth?",
        retrieved_chunks=test_chunks
    )
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources_used']}")
