"""
app.py
------
Streamlit UI for the Financial Document RAG Analyzer.
Run with: streamlit run src/app.py
"""

import streamlit as st
import sys
import os
import tempfile
import time

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import ingest_pdf
from retriever import add_chunks_to_store, retrieve, list_documents, get_store_stats, delete_document
from generator import generate_answer, check_ollama_available, list_available_models, DEFAULT_MODEL
from evaluator import evaluate_response

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Financial RAG Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background: #0a0e1a;
        color: #e0e6f0;
    }

    .main-header {
        background: linear-gradient(135deg, #0f1f3d 0%, #1a0a2e 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }

    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        color: #00d4ff;
        margin: 0;
        letter-spacing: 2px;
    }

    .main-header p {
        color: #7a9cc0;
        margin: 0.5rem 0 0;
        font-size: 0.9rem;
    }

    .metric-card {
        background: #0f1f3d;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        color: #00d4ff;
        font-weight: 600;
    }

    .metric-label {
        color: #7a9cc0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .answer-box {
        background: #0f1f3d;
        border-left: 4px solid #00d4ff;
        border-radius: 0 8px 8px 0;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }

    .source-badge {
        display: inline-block;
        background: #1a3a5f;
        border: 1px solid #2a5a8f;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #7ac8ff;
        margin: 2px;
    }

    .chunk-card {
        background: #080d1a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }

    .score-bar-fill {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #0040ff, #00d4ff);
    }

    .status-ok {
        color: #00ff88;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .status-err {
        color: #ff4444;
        font-family: 'IBM Plex Mono', monospace;
    }

    [data-testid="stSidebar"] {
        background: #080d1a;
        border-right: 1px solid #1e3a5f;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0040ff, #00aaff);
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 1px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 170, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Model selection
    available_models = list_available_models()
    if available_models:
        selected_model = st.selectbox(
            "🤖 LLM Model",
            options=available_models,
            index=0,
            help="Models available in your Ollama installation"
        )
    else:
        selected_model = DEFAULT_MODEL
        st.warning("⚠️ Ollama not detected. Install it to enable LLM generation.")
        st.code("curl -fsSL https://ollama.com/install.sh | sh\nollama pull mistral")

    top_k = st.slider("🔍 Chunks to Retrieve", min_value=2, max_value=10, value=5)
    show_eval = st.checkbox("📊 Show Quality Metrics", value=True)
    show_chunks = st.checkbox("📄 Show Retrieved Chunks", value=False)

    st.divider()

    # Store stats
    stats = get_store_stats()
    st.markdown("## 📚 Knowledge Base")
    st.markdown(f"**{stats['total_documents']}** documents · **{stats['total_chunks']}** chunks")

    if stats["documents"]:
        st.markdown("**Loaded files:**")
        for doc in stats["documents"]:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"📄 `{doc}`")
            if col2.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                delete_document(doc)
                st.rerun()

    st.divider()
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("⚡ Process & Index", use_container_width=True):
            with st.spinner("Extracting, chunking & embedding..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        chunks = ingest_pdf(tmp_path)
                        # Rename chunks to use original filename
                        for chunk in chunks:
                            chunk.source = uploaded_file.name
                            chunk.chunk_id = chunk.chunk_id.replace(
                                os.path.basename(tmp_path), uploaded_file.name
                            )
                        add_chunks_to_store(chunks)
                        st.success(f"✅ {uploaded_file.name} indexed ({len(chunks)} chunks)")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                    finally:
                        os.unlink(tmp_path)
            st.rerun()

# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 FINANCIAL RAG ANALYZER</h1>
    <p>Ask questions about your financial documents · Powered by local LLMs · Zero API costs</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Quick Stats Row
# ─────────────────────────────────────────────
stats = get_store_stats()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{stats['total_documents']}</div>
        <div class="metric-label">Documents</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{stats['total_chunks']}</div>
        <div class="metric-label">Indexed Chunks</div>
    </div>""", unsafe_allow_html=True)

with col3:
    ollama_ok = check_ollama_available(selected_model)
    status = '<span class="status-ok">● ONLINE</span>' if ollama_ok else '<span class="status-err">● OFFLINE</span>'
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:1rem;">{status}</div>
        <div class="metric-label">Ollama LLM</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="font-size:1rem; color:#7ac8ff;">MiniLM</div>
        <div class="metric-label">Embedding Model</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Chat Interface
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Query input
if query := st.chat_input("Ask a question about your financial documents..."):

    if stats["total_chunks"] == 0:
        st.warning("⚠️ No documents loaded yet. Upload a PDF using the sidebar.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                start_time = time.time()

                # Step 1: Retrieve
                chunks = retrieve(query, top_k=top_k)

                # Step 2: Generate
                if not ollama_ok:
                    # Fallback: return retrieved context without LLM
                    answer_text = "**⚠️ Ollama not available. Here are the most relevant excerpts:**\n\n"
                    for i, chunk in enumerate(chunks, 1):
                        answer_text += f"**[{i}] {chunk['source']} — Page {chunk['page']} (Score: {chunk['score']:.2%})**\n"
                        answer_text += f"> {chunk['text'][:400]}...\n\n"
                    result = {"answer": answer_text, "sources_used": [], "num_chunks_used": len(chunks)}
                else:
                    result = generate_answer(query, chunks, model=selected_model)

                elapsed = round(time.time() - start_time, 2)

            # Display answer
            st.markdown(f"""<div class="answer-box">{result['answer']}</div>""", unsafe_allow_html=True)

            # Sources
            if result.get("sources_used"):
                st.markdown("**📌 Sources cited:**")
                sources_html = " ".join([f'<span class="source-badge">{s}</span>' for s in result["sources_used"]])
                st.markdown(sources_html, unsafe_allow_html=True)

            st.caption(f"⏱️ {elapsed}s · {result['num_chunks_used']} chunks retrieved · Model: {selected_model}")

            # Evaluation metrics
            if show_eval and chunks:
                eval_result = evaluate_response(query, result["answer"], chunks)
                st.markdown("---")
                st.markdown("**📊 Response Quality Metrics**")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Faithfulness", f"{eval_result['faithfulness']:.0%}")
                m2.metric("Context Relevance", f"{eval_result['context_relevance']:.0%}")
                m3.metric("Completeness", f"{eval_result['answer_completeness']:.0%}")
                m4.metric("Overall", f"{eval_result['overall_quality']:.0%}")

                st.success(f"Grade: {eval_result['grade']}")

            # Retrieved chunks (optional)
            if show_chunks and chunks:
                st.markdown("---")
                st.markdown("**🔎 Retrieved Context Chunks**")
                for i, chunk in enumerate(chunks, 1):
                    with st.expander(f"Chunk {i} — {chunk['source']} (Page {chunk['page']}) — Score: {chunk['score']:.2%}"):
                        st.markdown(f"```\n{chunk['text']}\n```")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"]
            })

# ─────────────────────────────────────────────
# Example Queries
# ─────────────────────────────────────────────
if stats["total_chunks"] == 0:
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Upload** a financial PDF (earnings report, annual filing, etc.) using the sidebar
    2. Click **Process & Index** to embed the document  
    3. **Ask questions** in natural language below

    **Example questions to try:**
    - *"What was the total revenue this year?"*
    - *"What are the main risk factors mentioned?"*
    - *"How did operating expenses change?"*
    - *"What did management say about future guidance?"*
    - *"Summarize the cash flow statement."*
    """)
else:
    st.markdown("---")
    st.markdown("**💡 Try asking:**")
    example_cols = st.columns(3)
    examples = [
        "What was the total revenue?",
        "What are the main risk factors?",
        "Summarize the key financial highlights."
    ]
    for col, example in zip(example_cols, examples):
        if col.button(f'"{example}"', use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()
