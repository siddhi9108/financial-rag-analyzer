#  Financial Document Analyzer — RAG System

A production-grade Retrieval-Augmented Generation (RAG) system that lets you chat with financial documents (earnings reports, balance sheets, annual filings) using **100% free and open-source tools**. No API keys. No cloud costs. Runs entirely on your machine.

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
[Ollama / Mistral or LLaMA] ──► LLM Generation (local, free)
     │
     ▼
[Streamlit UI] ──► Chat Interface with Citations + Quality Metrics
```

---

##  Tech Stack (All Free & Open Source)

| Component | Tool | Why |
|-----------|------|-----|
| PDF Parsing | PyMuPDF | Fast, accurate text extraction |
| Embeddings | sentence-transformers (MiniLM) | Free, runs locally, great quality |
| Vector DB | ChromaDB | Local persistent storage, no setup needed |
| LLM | Ollama + Mistral 7B / LLaMA 3.2 | Runs fully offline, no API key |
| Framework | LangChain | Orchestrates the RAG pipeline |
| UI | Streamlit | Quick, clean web interface |
| Evaluation | Custom RAGAS-inspired metrics | Measures hallucination & faithfulness |

---

##  Setup Instructions

### Prerequisites
- Python 3.10 or higher
- At least 8GB RAM
- At least 6GB free disk space (for the LLM model)

---

###  macOS

**1. Clone the repo**
```bash
git clone https://github.com/siddhi9108/financial-rag-analyzer.git
cd financial-rag-analyzer
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install streamlit chromadb sentence-transformers PyMuPDF==1.23.8 ollama
```

**4. Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**5. Pull a model (choose one)**
```bash
# Mistral 7B — more accurate, ~4GB
ollama pull mistral

# LLaMA 3.2 1B — faster, ~1GB (recommended for M1 Macs)
ollama pull llama3.2:1b
```

**6. Run the app**
```bash
streamlit run src/app.py --server.fileWatcherType none
```

---

###  Windows

**1. Install Python**
- Download from [python.org](https://www.python.org/downloads/)
- During installation, check ✅ **"Add Python to PATH"**

**2. Install Git**
- Download from [git-scm.com](https://git-scm.com/download/win)

**3. Open Command Prompt or PowerShell and clone the repo**
```cmd
git clone https://github.com/siddhi9108/financial-rag-analyzer.git
cd financial-rag-analyzer
```

**4. Create and activate a virtual environment**
```cmd
python -m venv venv
venv\Scripts\activate
```

**5. Install dependencies**
```cmd
pip install --upgrade pip
pip install streamlit chromadb sentence-transformers PyMuPDF==1.23.8 ollama
```

**6. Install Ollama**
- Download the Windows installer from [ollama.com/download](https://ollama.com/download)
- Run the installer and follow the steps

**7. Pull a model — open a new Command Prompt window**
```cmd
ollama pull mistral
```
Or for a faster, lighter model:
```cmd
ollama pull llama3.2:1b
```

**8. Run the app**
```cmd
streamlit run src/app.py --server.fileWatcherType none
```

Open your browser at `http://localhost:8501`

---

##  How to Use

1. **Upload** a financial PDF (earnings report, 10-Q, annual filing) using the left sidebar
2. Click **Process & Index** — the document will be chunked and embedded
3. **Ask questions** in the chat box at the bottom
4. View the **answer with citations** showing which page the data came from
5. Check the **Quality Metrics** to see faithfulness and relevance scores

---

##  Example Questions to Ask

- *"What was the total revenue this quarter?"*
- *"What are the main risk factors mentioned?"*
- *"How did operating expenses change year over year?"*
- *"What did management say about future guidance?"*
- *"Summarize the cash flow statement."*
- *"What was the net income compared to last year?"*

---

##  Project Structure

```
financial-rag/
├── src/
│   ├── app.py              # Streamlit UI
│   ├── ingestion.py        # PDF loading & chunking
│   ├── embeddings.py       # Embedding generation
│   ├── retriever.py        # ChromaDB vector search
│   ├── generator.py        # LLM response generation
│   └── evaluator.py        # Quality metrics
├── data/                   # Place your PDFs here
├── tests/
│   └── test_pipeline.py    # Unit tests
├── requirements.txt
└── README.md
```

---

##  Quality Metrics Explained

After each answer, the app shows:

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Does the answer stick to retrieved context? |
| **Context Relevance** | Are retrieved chunks actually useful? |
| **Completeness** | Does the answer fully address the question? |
| **Overall** | Weighted score across all metrics |

---

##  Model Recommendations by Hardware

| Your Machine | Recommended Model | Speed |
|-------------|-------------------|-------|
| M1/M2 Mac (8GB RAM) | `llama3.2:1b` | Fast |
| M1/M2 Mac (16GB RAM) | `mistral` | Medium |
| Windows/Linux (8GB RAM) | `llama3.2:1b` | Fast |
| Windows/Linux (16GB+ RAM) | `mistral` | Medium |

Switch models anytime from the sidebar dropdown — no restart needed.

---

##  Why This Matters

Hallucinations in financial analysis can lead to bad investment decisions. This system ensures every answer is grounded in retrieved evidence, with source citations shown in the UI. Built to demonstrate responsible AI in high-stakes domains.

---

##  Running Tests

```bash
# macOS / Linux
source venv/bin/activate
python -m pytest tests/ -v

# Windows
venv\Scripts\activate
python -m pytest tests/ -v
```

---

##  License

MIT License — free to use, modify, and share.

---

##  Author

**Siddhi Amilkanthwar**
-  siddhi.amilkanthwar@gmail.com
-  [LinkedIn](https://linkedin.com/in/siddhi-amilkanthwar)
-  [GitHub](https://github.com/siddhi9108)
