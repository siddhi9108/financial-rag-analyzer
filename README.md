#  Financial Document Analyzer — RAG System

A production-grade Retrieval-Augmented Generation (RAG) system that lets you chat with financial documents (earnings reports, balance sheets, annual filings) using **100% free and open-source tools**.

---

##  Architecture

```
PDF Documents
     │
     ▼
[PyMuPDF] ──► Text Extraction
     │
     ▼
[Chunking Strategy] ──► Overlapping Chunks (512 tokens, 50 overlap)
     │
     ▼
[sentence-transformers] ──► Embeddings (all-MiniLM-L6-v2)
     │
     ▼
[ChromaDB] ──► Vector Store (local, persistent)
     │
     ▼
[Retriever] ──► Top-K Semantic Search
     │
     ▼
[Ollama / Mistral] ──► LLM Generation (local, free)
     │
     ▼
[Streamlit UI] ──► Chat Interface with Citations
```

---

##  Tech Stack (All Free & Open Source)

| Component | Tool | Why |
|-----------|------|-----|
| PDF Parsing | PyMuPDF | Fast, accurate text extraction |
| Embeddings | sentence-transformers (MiniLM) | Free, runs locally, great quality |
| Vector DB | ChromaDB | Local persistent storage |
| LLM | Ollama + Mistral 7B | Runs fully offline, no API key |
| Framework | LangChain | Orchestrates the RAG pipeline |
| UI | Streamlit | Quick, clean web interface |
| Evaluation | RAGAS | Measures hallucination & faithfulness |

---

##  Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/financial-rag-analyzer
cd financial-rag-analyzer
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama (local LLM runner)
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Then pull the Mistral model (free, 4GB)
ollama pull mistral
```

### 4. Run the app
```bash
streamlit run src/app.py
```

### 5. Upload a financial PDF and start asking questions!

---

##  Example Questions to Ask

- *"What was the total revenue in Q3?"*
- *"What are the main risk factors mentioned?"*
- *"How did operating expenses change year over year?"*
- *"What did management say about future guidance?"*
- *"Summarize the cash flow statement."*

---

##  Project Structure

```
financial-rag/
├── src/
│   ├── app.py              # Streamlit UI
│   ├── ingestion.py        # PDF loading & chunking
│   ├── embeddings.py       # Embedding generation
│   ├── retriever.py        # Vector search
│   ├── generator.py        # LLM response generation
│   └── evaluator.py        # RAGAS evaluation metrics
├── data/                   # Place your PDFs here
├── tests/
│   └── test_pipeline.py    # Unit tests
├── requirements.txt
└── README.md
```

---

##  Evaluation Metrics (RAGAS)

The system measures:
- **Faithfulness** — Does the answer stick to retrieved context?
- **Answer Relevancy** — Is the answer relevant to the question?
- **Context Precision** — Are retrieved chunks actually useful?
- **Hallucination Rate** — How often does the model make things up?

---

##  Why This Matters

Hallucinations in financial analysis can lead to bad investment decisions. This system ensures every answer is grounded in retrieved evidence, with source citations shown in the UI.

---

##  Customization

- Swap `mistral` for `llama3`, `gemma2`, or any Ollama model
- Change chunk size in `ingestion.py` for different document types
- Add more vector stores (Qdrant, FAISS) in `retriever.py`

---

##  License

MIT License — free to use, modify, and share.
