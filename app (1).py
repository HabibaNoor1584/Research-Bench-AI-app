import io
import os
import json
import sqlite3
import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Optional: lets GROQ_API_KEY be read from a local .env file during local
# development. Harmless no-op on Streamlit Cloud, which uses st.secrets instead.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except Exception:
    GROQ_SDK_AVAILABLE = False


# =========================================================================
# CONFIG
# =========================================================================
APP_TITLE = "ResearchBench AI"
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "researchbench.db")
CHAT_MODEL = "openai/gpt-oss-120b"    # Fast and free model on Groq
CHUNK_WORD_SIZE = 400
TOP_K_CHUNKS = 4
MIN_SIMILARITY = 0.0  # TF-IDF cosine similarity threshold for local retrieval

MEMORY_TYPES = ["Note", "Question", "Research Idea", "Hypothesis", "Observation"]
CONNECTION_SOURCE_TARGETS = ["Paper", "Protocol", "Personal Memory", "Observation", "AI Memory"]

# A small curated list of reagents/hazards that trigger a generic safety
# reminder. This is NOT a hazard database — it only nudges the user to the
# SDS/SOP/instructor, matching the "Basic Safety Alerts" P2 feature.
SAFETY_KEYWORDS = [
    "phenol", "chloroform", "ethidium bromide", "acrylamide", "bleach",
    "sodium azide", "temed", "formaldehyde", "hydrochloric acid",
    "sodium hydroxide", "liquid nitrogen", "uv light", "autoclave",
    "concentrated acid", "concentrated base", "ethanol", "methanol",
    "osmium tetroxide", "benzene",
]


# =========================================================================
# UI THEME (colors, header, cards, sidebar) — display only, no logic here
# =========================================================================
THEME_PRIMARY = "#2563eb"      # blue — AI / evidence
THEME_TEAL = "#0d9488"         # teal — protocols / lab
THEME_PURPLE = "#7c3aed"       # purple — personal thoughts
THEME_AMBER = "#d97706"        # amber — observations / pending
THEME_RED = "#dc2626"          # red — rejected / danger

CUSTOM_CSS = f"""
<style>
.stApp {{
    background-color: #f8fafc;
}}

/* Gradient header banner */
.rb-header {{
    background: linear-gradient(135deg, {THEME_PRIMARY} 0%, {THEME_TEAL} 100%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.18);
}}
.rb-title {{
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
}}
.rb-subtitle {{
    font-size: 0.95rem;
    color: #e0f2fe;
    margin-top: 6px;
}}

/* Sidebar dark theme */
[data-testid="stSidebar"] {{
    background-color: #0f172a;
}}
[data-testid="stSidebar"] * {{
    color: #f1f5f9 !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    padding: 4px 0;
}}

/* Buttons */
.stButton>button {{
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}}
.stButton>button:hover {{
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
}}

/* Rounded, shadowed containers (st.container(border=True)) and expanders */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 12px !important;
}}
div[data-testid="stExpander"] {{
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}}

/* Metric cards */
div[data-testid="stMetric"] {{
    background-color: #ffffff;
    padding: 12px 16px;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
}}
</style>
"""


def inject_custom_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(counts: Dict[str, int]) -> None:
    st.markdown(
        f'<div class="rb-header">'
        f'<div class="rb-title">🔬 {APP_TITLE}</div>'
        f'<div class="rb-subtitle">Paper → Evidence → AI Explanation → Your Thought → '
        f'Protocol → Observation → New Question</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📚 Papers", counts["Papers"])
    m2.metric("🧪 Protocols", counts["Protocols"])
    m3.metric("🤖 AI Answers Saved", counts["AI Answers Saved"])
    m4.metric("🧭 Connections", counts["Connections"])


# =========================================================================
# DATABASE LAYER
# =========================================================================
@st.cache_resource(show_spinner=False)
def get_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            document_type TEXT NOT NULL,   -- 'paper' or 'protocol'
            upload_date TEXT NOT NULL,
            text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready'
        );

        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            page_number INTEGER,
            embedding TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );

        CREATE TABLE IF NOT EXISTS ai_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            source_document_ids TEXT,
            source_evidence TEXT,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS personal_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            linked_document_id INTEGER,
            linked_protocol_id INTEGER,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS protocols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'extracted', -- extracted / approved / rejected
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );

        CREATE TABLE IF NOT EXISTS protocol_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol_id INTEGER NOT NULL,
            step_number INTEGER NOT NULL,
            step_text TEXT NOT NULL,
            approved INTEGER NOT NULL DEFAULT 0,
            completed INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (protocol_id) REFERENCES protocols(id)
        );

        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (protocol_id) REFERENCES protocols(id)
        );

        CREATE TABLE IF NOT EXISTS connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            target_type TEXT NOT NULL,
            target_id INTEGER NOT NULL,
            reason TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
    )
    conn.commit()


def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def db_insert_document(name: str, document_type: str, text: str) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO documents (name, document_type, upload_date, text, status) VALUES (?, ?, ?, ?, ?)",
        (name, document_type, now_str(), text, "ready"),
    )
    conn.commit()
    return cur.lastrowid


def db_get_documents(document_type: Optional[str] = None) -> pd.DataFrame:
    conn = get_connection()
    if document_type:
        return pd.read_sql_query(
            "SELECT * FROM documents WHERE document_type = ? ORDER BY id DESC",
            conn, params=(document_type,),
        )
    return pd.read_sql_query("SELECT * FROM documents ORDER BY id DESC", conn)


def db_delete_document(document_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()


def db_insert_chunks(document_id: int, chunks: List[Dict]) -> None:
    conn = get_connection()
    for c in chunks:
        conn.execute(
            "INSERT INTO document_chunks (document_id, chunk_text, page_number, embedding) VALUES (?, ?, ?, ?)",
            (document_id, c["chunk_text"], c.get("page_number"), json.dumps(c.get("embedding"))),
        )
    conn.commit()


def db_get_chunks(document_ids: List[int]) -> pd.DataFrame:
    if not document_ids:
        return pd.DataFrame()
    conn = get_connection()
    placeholders = ",".join("?" for _ in document_ids)
    return pd.read_sql_query(
        f"SELECT * FROM document_chunks WHERE document_id IN ({placeholders})",
        conn, params=document_ids,
    )


def db_insert_ai_memory(question: str, answer: str, source_document_ids: List[int], source_evidence: str) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO ai_memories (question, answer, source_document_ids, source_evidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (question, answer, json.dumps(source_document_ids), source_evidence, now_str()),
    )
    conn.commit()
    return cur.lastrowid


def db_get_ai_memories() -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql_query("SELECT * FROM ai_memories ORDER BY id DESC", conn)


def db_insert_personal_memory(memory_type: str, content: str, linked_document_id: Optional[int], linked_protocol_id: Optional[int]) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO personal_memories (memory_type, content, linked_document_id, linked_protocol_id, timestamp) VALUES (?, ?, ?, ?, ?)",
        (memory_type, content, linked_document_id, linked_protocol_id, now_str()),
    )
    conn.commit()
    return cur.lastrowid


def db_get_personal_memories() -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql_query("SELECT * FROM personal_memories ORDER BY id DESC", conn)


def db_insert_protocol(document_id: int, name: str) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO protocols (document_id, name, upload_date, status) VALUES (?, ?, ?, ?)",
        (document_id, name, now_str(), "extracted"),
    )
    conn.commit()
    return cur.lastrowid


def db_get_protocols() -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql_query("SELECT * FROM protocols ORDER BY id DESC", conn)


def db_update_protocol_status(protocol_id: int, status: str) -> None:
    conn = get_connection()
    conn.execute("UPDATE protocols SET status = ? WHERE id = ?", (status, protocol_id))
    conn.commit()


def db_insert_protocol_steps(protocol_id: int, steps: List[str]) -> None:
    conn = get_connection()
    for i, step_text in enumerate(steps, start=1):
        conn.execute(
            "INSERT INTO protocol_steps (protocol_id, step_number, step_text, approved, completed) VALUES (?, ?, ?, 0, 0)",
            (protocol_id, i, step_text),
        )
    conn.commit()


def db_get_protocol_steps(protocol_id: int) -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql_query(
        "SELECT * FROM protocol_steps WHERE protocol_id = ? ORDER BY step_number ASC",
        conn, params=(protocol_id,),
    )


def db_approve_all_steps(protocol_id: int, approve: bool) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE protocol_steps SET approved = ? WHERE protocol_id = ?",
        (1 if approve else 0, protocol_id),
    )
    conn.commit()


def db_set_step_completed(step_id: int, completed: bool) -> None:
    conn = get_connection()
    conn.execute("UPDATE protocol_steps SET completed = ? WHERE id = ?", (1 if completed else 0, step_id))
    conn.commit()


def db_insert_observation(protocol_id: int, content: str) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO observations (protocol_id, content, timestamp) VALUES (?, ?, ?)",
        (protocol_id, content, now_str()),
    )
    conn.commit()
    return cur.lastrowid


def db_get_observations(protocol_id: Optional[int] = None) -> pd.DataFrame:
    conn = get_connection()
    if protocol_id:
        return pd.read_sql_query(
            "SELECT * FROM observations WHERE protocol_id = ? ORDER BY id DESC",
            conn, params=(protocol_id,),
        )
    return pd.read_sql_query("SELECT * FROM observations ORDER BY id DESC", conn)


def db_insert_connection(source_type: str, source_id: int, target_type: str, target_id: int, reason: str) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO connections (source_type, source_id, target_type, target_id, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (source_type, source_id, target_type, target_id, reason, now_str()),
    )
    conn.commit()
    return cur.lastrowid


def db_get_connections() -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql_query("SELECT * FROM connections ORDER BY id DESC", conn)


def db_dashboard_counts() -> Dict[str, int]:
    conn = get_connection()

    def count(query, params=()):
        return conn.execute(query, params).fetchone()[0]

    counts = {
        "Papers": count("SELECT COUNT(*) FROM documents WHERE document_type = 'paper'"),
        "Protocols": count("SELECT COUNT(*) FROM protocols"),
        "AI Answers Saved": count("SELECT COUNT(*) FROM ai_memories"),
        "Notes": count("SELECT COUNT(*) FROM personal_memories WHERE memory_type = 'Note'"),
        "Questions": count("SELECT COUNT(*) FROM personal_memories WHERE memory_type = 'Question'"),
        "Research Ideas": count("SELECT COUNT(*) FROM personal_memories WHERE memory_type = 'Research Idea'"),
        "Hypotheses": count("SELECT COUNT(*) FROM personal_memories WHERE memory_type = 'Hypothesis'"),
        "Observations": count("SELECT COUNT(*) FROM observations"),
        "Connections": count("SELECT COUNT(*) FROM connections"),
    }
    return counts


# =========================================================================
# PDF PROCESSING
# =========================================================================
def extract_pdf_pages(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Returns a list of (page_number, page_text). Raises on failure."""
    if not PDFPLUMBER_AVAILABLE:
        raise RuntimeError("pdfplumber is not installed on the server.")
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def chunk_pages(pages: List[Tuple[int, str]], chunk_word_size: int = CHUNK_WORD_SIZE) -> List[Dict]:
    chunks = []
    for page_number, text in pages:
        words = text.split()
        if not words:
            continue
        for i in range(0, len(words), chunk_word_size):
            piece = " ".join(words[i:i + chunk_word_size]).strip()
            if piece:
                chunks.append({"chunk_text": piece, "page_number": page_number})
    return chunks


# =========================================================================
# LLM / RETRIEVAL SERVICES (Groq API + Local TF-IDF Search)
# =========================================================================
def get_client(api_key: str):
    if not GROQ_SDK_AVAILABLE:
        return None
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def retrieve_relevant_chunks_local(question: str, document_ids: List[int], top_k: int = TOP_K_CHUNKS) -> List[Dict]:
    chunks_df = db_get_chunks(document_ids)
    if chunks_df.empty:
        return []

    texts = chunks_df["chunk_text"].tolist()
    if not texts:
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts + [question])
        q_vec = tfidf_matrix[-1]
        doc_matrix = tfidf_matrix[:-1]

        scores = sklearn_cosine_similarity(q_vec, doc_matrix).flatten()
    except Exception:
        return []

    scored = []
    for idx, score in enumerate(scores):
        if score >= MIN_SIMILARITY:
            row = chunks_df.iloc[idx]
            scored.append({
                "document_id": int(row["document_id"]),
                "chunk_text": row["chunk_text"],
                "page_number": int(row["page_number"]) if pd.notnull(row["page_number"]) else 1,
                "score": float(score),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def generate_rag_answer(client, question: str, evidence_chunks: List[Dict]) -> Optional[str]:
    if client is None:
        return None
    evidence_text = "\n\n".join(
        f"[Source page {c['page_number']}] {c['chunk_text']}" for c in evidence_chunks
    )
    system_prompt = (
        "You are a careful research assistant. Answer the user's question using ONLY the "
        "evidence provided below. Do not use outside knowledge. If the evidence does not "
        "sufficiently answer the question, say clearly that the uploaded sources do not "
        "provide enough evidence — do not invent an answer. Cite the page number(s) you used."
    )
    user_prompt = f"EVIDENCE:\n{evidence_text}\n\nQUESTION: {question}"
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"AI request failed: {e}. Please retry — nothing has been lost.")
        return None


def extract_protocol_checklist(client, protocol_text: str) -> Optional[List[str]]:
    if client is None:
        return None
    system_prompt = (
        "You convert an existing laboratory protocol into a clean, numbered checklist. "
        "Use ONLY the steps, materials, quantities, equipment, temperatures, and times that "
        "already appear in the text. Do NOT invent, complete, or alter any experimental "
        "parameter. If a parameter is missing, keep the step as written without guessing. "
        "Return ONLY a JSON array of strings, one string per checklist step, nothing else."
    )
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": protocol_text[:12000]},
            ],
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip()
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        steps = json.loads(raw)
        if isinstance(steps, list):
            return [str(s) for s in steps]
        return None
    except Exception as e:
        st.error(f"Protocol extraction failed: {e}. Please retry — the uploaded protocol is preserved.")
        return None


def scan_safety_keywords(text: str) -> List[str]:
    text_lower = text.lower()
    return [kw for kw in SAFETY_KEYWORDS if kw in text_lower]


# =========================================================================
# UI HELPERS
# =========================================================================
def sidebar_api_key() -> str:
    api_key = ""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "") or st.secrets.get("API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")

    if api_key:
        # Key is already configured via Streamlit secrets or a local .env —
        # no need to clutter the sidebar with anything about it.
        return api_key

    # No key found anywhere — show a minimal fallback input so the app still works.
    st.sidebar.markdown("### 🔑 API Key Needed")
    api_key = st.sidebar.text_input(
        "Groq API key",
        type="password",
        help="Free at console.groq.com/keys. Needed for AI chat and protocol "
             "extraction (retrieval runs locally, no key needed for that). "
             "Set it once as a Streamlit secret (GROQ_API_KEY) to skip this box.",
    )
    if not api_key:
        st.sidebar.warning("No API key set. Upload, notes, and dashboard still work; "
                           "AI chat and protocol extraction are disabled until you add a key.")
    return api_key



def document_picker(df: pd.DataFrame, label: str, multi: bool = False):
    if df.empty:
        st.info("No documents uploaded yet.")
        return [] if multi else None
    options = {f"#{row['id']} — {row['name']}": row["id"] for _, row in df.iterrows()}
    if multi:
        chosen = st.multiselect(label, list(options.keys()))
        return [options[c] for c in chosen]
    chosen = st.selectbox(label, list(options.keys()))
    return options[chosen]


def protocol_status_label(status: str) -> str:
    return {
        "extracted": "🕵️ Pending review",
        "approved": "✅ Approved",
        "rejected": "❌ Rejected",
    }.get(status, status)


# =========================================================================
# PAGES
# =========================================================================
def page_research_library():
    st.header("📚 Research Library")
    st.caption("Upload research papers as PDFs. Text is extracted, chunked, and indexed locally for evidence-grounded AI chat.")

    uploaded = st.file_uploader("Upload a research paper (PDF)", type=["pdf"], key="paper_uploader")
    if uploaded is not None:
        if st.button("Process paper", type="primary"):
            file_bytes = uploaded.getvalue()
            if not file_bytes:
                st.error("The uploaded file is empty. Please choose a valid PDF.")
            else:
                try:
                    with st.spinner("Extracting text..."):
                        pages = extract_pdf_pages(file_bytes)
                    full_text = "\n".join(t for _, t in pages).strip()
                    if not full_text:
                        st.error("No readable text was found in this PDF. Try another file or a text-readable PDF.")
                    else:
                        doc_id = db_insert_document(uploaded.name, "paper", full_text)
                        chunks = chunk_pages(pages)
                        db_insert_chunks(doc_id, chunks)
                        st.success(f"'{uploaded.name}' uploaded, chunked ({len(chunks)} chunks), and indexed locally.")
                except Exception as e:
                    st.error(f"Could not process this PDF: {e}. Please upload another file.")

    st.divider()
    st.subheader("Uploaded papers")
    papers_df = db_get_documents("paper")
    if papers_df.empty:
        st.info("No papers uploaded yet.")
    else:
        for _, row in papers_df.iterrows():
            with st.expander(f"📄 #{row['id']} — {row['name']} ({row['upload_date']})"):
                st.write(row["text"][:1000] + ("..." if len(row["text"]) > 1000 else ""))
                if st.button("🗑️ Delete this paper", key=f"del_paper_{row['id']}"):
                    db_delete_document(row["id"])
                    st.rerun()


def page_research_ai():
    st.header("🤖 Research AI (Evidence-Grounded Chat)")
    st.caption("Ask a question about one or more uploaded papers. Answers are grounded in retrieved evidence only.")

    api_key = st.session_state.get("api_key", "")
    client = get_client(api_key)

    papers_df = db_get_documents("paper")
    if papers_df.empty:
        st.info("Upload a paper in the Research Library first.")
        return

    selected_ids = document_picker(papers_df, "Select paper(s) to ask about", multi=True)
    question = st.text_input("Your research question")
    ask = st.button("🔎 Ask", type="primary", disabled=(client is None))
    if client is None:
        st.warning("Add an API key in the sidebar to enable AI chat.")

    if ask:
        if not selected_ids:
            st.warning("Select at least one paper first.")
        elif not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Retrieving evidence via local TF-IDF search..."):
                evidence = retrieve_relevant_chunks_local(question, selected_ids)
            if not evidence:
                st.warning("Not enough evidence in the uploaded sources to answer this question.")
            else:
                with st.spinner("Generating answer..."):
                    answer = generate_rag_answer(client, question, evidence)
                if answer:
                    st.session_state["last_answer"] = {
                        "question": question,
                        "answer": answer,
                        "evidence": evidence,
                        "document_ids": selected_ids,
                    }

    last = st.session_state.get("last_answer")
    if last:
        with st.container(border=True):
            st.markdown("#### 💬 Answer")
            st.write(last["answer"])
        with st.expander("📎 Source evidence used for this answer"):
            for c in last["evidence"]:
                st.markdown(f"**Page {c['page_number']}** (relevance score {c['score']:.2f})")
                st.write(c["chunk_text"])
                st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save to Research Memory"):
                evidence_text = "\n\n".join(f"[p.{c['page_number']}] {c['chunk_text']}" for c in last["evidence"])
                db_insert_ai_memory(last["question"], last["answer"], last["document_ids"], evidence_text)
                st.success("Saved to AI Research Memory.")
                st.session_state["last_answer"] = None
                st.rerun()
        with col2:
            if st.button("🗑️ Discard"):
                st.session_state["last_answer"] = None
                st.rerun()

    st.divider()
    st.subheader("Saved AI Research Memory")
    mem_df = db_get_ai_memories()
    if mem_df.empty:
        st.info("No saved AI answers yet.")
    else:
        for _, row in mem_df.iterrows():
            with st.expander(f"💬 Q: {row['question'][:80]} ({row['timestamp']})"):
                st.markdown("**Question:**")
                st.write(row["question"])
                st.markdown("**AI Answer (interpretation, linked to evidence — not authoritative fact):**")
                st.write(row["answer"])
                st.markdown("**Evidence:**")
                st.write(row["source_evidence"])


def page_research_memory():
    st.header("🧠 Personal Research Memory")
    st.caption("Your own notes, questions, ideas, hypotheses, and observations — clearly separate from AI/source content.")

    papers_df = db_get_documents("paper")
    protocols_df = db_get_protocols()

    with st.form("add_memory_form", clear_on_submit=True):
        memory_type = st.selectbox("Type", MEMORY_TYPES)
        content = st.text_area("Your text (this is your own thinking, not source evidence)")
        link_paper = document_picker(papers_df, "Optionally link to a paper") if not papers_df.empty else None
        link_protocol = None
        if not protocols_df.empty:
            options = {f"#{row['id']} — {row['name']}": row["id"] for _, row in protocols_df.iterrows()}
            choice = st.selectbox("Optionally link to a protocol", ["(none)"] + list(options.keys()))
            link_protocol = options.get(choice)
        submitted = st.form_submit_button("💾 Save memory", type="primary")
        if submitted:
            if not content.strip():
                st.warning("Please write something before saving.")
            else:
                db_insert_personal_memory(memory_type, content.strip(), link_paper, link_protocol)
                st.success(f"{memory_type} saved.")

    st.divider()
    st.subheader("Your saved memories")
    mem_df = db_get_personal_memories()
    if mem_df.empty:
        st.info("No personal memories saved yet.")
    else:
        filter_type = st.selectbox("Filter by type", ["All"] + MEMORY_TYPES)
        view = mem_df if filter_type == "All" else mem_df[mem_df["memory_type"] == filter_type]
        st.dataframe(
            view[["id", "memory_type", "content", "linked_document_id", "linked_protocol_id", "timestamp"]],
            use_container_width=True, hide_index=True,
        )


def page_protocol_companion():
    st.header("🧪 Protocol Companion")
    st.caption("Upload a lab protocol. AI organizes its existing steps into a checklist — nothing is invented, "
               "and nothing becomes active until you approve it.")

    api_key = st.session_state.get("api_key", "")
    client = get_client(api_key)

    uploaded = st.file_uploader("Upload a protocol (PDF)", type=["pdf"], key="protocol_uploader")
    if uploaded is not None and st.button("Process protocol", type="primary"):
        file_bytes = uploaded.getvalue()
        if not file_bytes:
            st.error("The uploaded file is empty. Please choose a valid PDF.")
        else:
            try:
                with st.spinner("Extracting text..."):
                    pages = extract_pdf_pages(file_bytes)
                full_text = "\n".join(t for _, t in pages).strip()
                if not full_text:
                    st.error("No readable text was found in this PDF. Try another file.")
                else:
                    doc_id = db_insert_document(uploaded.name, "protocol", full_text)
                    protocol_id = db_insert_protocol(doc_id, uploaded.name)

                    flags = scan_safety_keywords(full_text)
                    if flags:
                        st.warning(
                            f"This protocol mentions: {', '.join(flags)}. "
                            "Always follow your official SDS/SOP and check with your instructor/lab "
                            "supervisor before proceeding — this app does not provide safety guidance."
                        )

                    if client:
                        with st.spinner("AI is organizing the existing steps into a checklist..."):
                            steps = extract_protocol_checklist(client, full_text)
                        if steps:
                            db_insert_protocol_steps(protocol_id, steps)
                            st.success(f"Extracted {len(steps)} steps. Review and approve them below.")
                        else:
                            st.warning("Automatic extraction did not return a usable checklist. "
                                       "You can still review the raw protocol text below.")
                    else:
                        st.warning("No API key — protocol uploaded, but automatic step extraction is disabled.")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not process this protocol: {e}. Please upload another file.")

    st.divider()
    st.subheader("Protocols")
    protocols_df = db_get_protocols()
    if protocols_df.empty:
        st.info("No protocols uploaded yet.")
        return

    for _, prot in protocols_df.iterrows():
        label = f"🧪 #{prot['id']} — {prot['name']} — {protocol_status_label(prot['status'])}"
        with st.expander(label):
            steps_df = db_get_protocol_steps(prot["id"])
            if steps_df.empty:
                st.info("No extracted steps yet for this protocol.")
                continue

            if prot["status"] == "extracted":
                st.write("**Human checkpoint:** review the AI-extracted checklist against the original protocol.")
                for _, s in steps_df.iterrows():
                    st.write(f"{s['step_number']}. {s['step_text']}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve checklist", key=f"approve_{prot['id']}"):
                        db_approve_all_steps(prot["id"], True)
                        db_update_protocol_status(prot["id"], "approved")
                        st.success("Checklist approved and activated.")
                        st.rerun()
                with col2:
                    if st.button("❌ Reject checklist", key=f"reject_{prot['id']}"):
                        db_update_protocol_status(prot["id"], "rejected")
                        st.warning("Checklist rejected. Steps were not activated.")
                        st.rerun()

            elif prot["status"] == "approved":
                st.write("**Active checklist:**")
                for _, s in steps_df.iterrows():
                    checked = st.checkbox(
                        f"{s['step_number']}. {s['step_text']}",
                        value=bool(s["completed"]),
                        key=f"step_{s['id']}",
                    )
                    if checked != bool(s["completed"]):
                        db_set_step_completed(s["id"], checked)
                        st.rerun()

                st.markdown("**Add an observation**")
                obs_text = st.text_area("What did you observe while running this protocol?", key=f"obs_{prot['id']}")
                if st.button("Save observation", key=f"save_obs_{prot['id']}"):
                    if obs_text.strip():
                        db_insert_observation(prot["id"], obs_text.strip())
                        st.success("Observation saved.")
                        st.rerun()
                    else:
                        st.warning("Write an observation before saving.")

                obs_df = db_get_observations(prot["id"])
                if not obs_df.empty:
                    st.markdown("**Past observations:**")
                    st.dataframe(obs_df[["id", "content", "timestamp"]], use_container_width=True, hide_index=True)

            else:
                st.info("This checklist was rejected and is not active.")


def page_research_journey():
    st.header("🧭 Research Journey")
    st.caption("Manually connect items across your research memory, and see everything on one timeline.")

    st.subheader("Connect two items")
    papers_df = db_get_documents("paper")
    protocols_df = db_get_protocols()
    personal_df = db_get_personal_memories()
    obs_df = db_get_observations()
    ai_df = db_get_ai_memories()

    def id_options(df, label_col="name"):
        if df.empty:
            return {}
        if label_col in df.columns:
            return {f"#{row['id']} — {str(row[label_col])[:40]}": row["id"] for _, row in df.iterrows()}
        return {f"#{row['id']}": row["id"] for _, row in df.iterrows()}

    source_map = {
        "Paper": id_options(papers_df, "name"),
        "Protocol": id_options(protocols_df, "name"),
        "Personal Memory": id_options(personal_df, "content"),
        "Observation": id_options(obs_df, "content"),
        "AI Memory": id_options(ai_df, "question"),
    }

    col1, col2 = st.columns(2)
    with col1:
        source_type = st.selectbox("Source type", CONNECTION_SOURCE_TARGETS, key="conn_src_type")
        src_options = source_map.get(source_type, {})
        source_choice = st.selectbox("Source item", list(src_options.keys()) or ["(none available)"], key="conn_src_item")
    with col2:
        target_type = st.selectbox("Target type", CONNECTION_SOURCE_TARGETS, index=1, key="conn_tgt_type")
        tgt_options = source_map.get(target_type, {})
        target_choice = st.selectbox("Target item", list(tgt_options.keys()) or ["(none available)"], key="conn_tgt_item")

    reason = st.text_input("Why are these connected?")
    if st.button("🔗 Connect", type="primary"):
        if not src_options or not tgt_options:
            st.warning("Both source and target need at least one available item.")
        elif not reason.strip():
            st.warning("Please explain why these are connected.")
        else:
            db_insert_connection(
                source_type, src_options[source_choice],
                target_type, tgt_options[target_choice],
                reason.strip(),
            )
            st.success("Connection saved.")
            st.rerun()

    st.divider()
    st.subheader("Saved connections")
    conn_df = db_get_connections()
    if conn_df.empty:
        st.info("No connections yet.")
    else:
        st.dataframe(conn_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Timeline of research events")
    events = []
    for _, r in ai_df.iterrows():
        events.append({"timestamp": r["timestamp"], "type": "AI Answer Saved", "detail": r["question"][:80]})
    for _, r in personal_df.iterrows():
        events.append({"timestamp": r["timestamp"], "type": r["memory_type"], "detail": r["content"][:80]})
    for _, r in obs_df.iterrows():
        events.append({"timestamp": r["timestamp"], "type": "Observation", "detail": r["content"][:80]})
    for _, r in conn_df.iterrows():
        events.append({"timestamp": r["timestamp"], "type": "Connection",
                        "detail": f"{r['source_type']} #{r['source_id']} -> {r['target_type']} #{r['target_id']}"})

    if events:
        timeline_df = pd.DataFrame(events).sort_values("timestamp", ascending=False)
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)
    else:
        st.info("No research events recorded yet. Start by uploading a paper or adding a note.")


def page_dashboard():
    st.header("📊 Research Memory Dashboard")
    counts = db_dashboard_counts()

    cols = st.columns(4)
    keys = list(counts.keys())
    for i, key in enumerate(keys):
        with cols[i % 4]:
            st.metric(key, counts[key])

    st.divider()
    chart_df = pd.DataFrame({"Category": list(counts.keys()), "Count": list(counts.values())}).set_index("Category")
    st.bar_chart(chart_df)

    st.divider()
    st.subheader("Definition of Done — quick self-check")
    papers_ok = counts["Papers"] > 0
    evidence_ok = counts["AI Answers Saved"] > 0
    personal_ok = (counts["Notes"] + counts["Questions"] + counts["Research Ideas"] + counts["Hypotheses"]) > 0
    protocol_ok = counts["Protocols"] > 0
    obs_ok = counts["Observations"] > 0
    conn_ok = counts["Connections"] > 0

    checklist = [
        ("Uploaded at least one paper", papers_ok),
        ("Saved at least one evidence-grounded AI answer", evidence_ok),
        ("Saved at least one personal thought", personal_ok),
        ("Uploaded at least one protocol", protocol_ok),
        ("Recorded at least one observation", obs_ok),
        ("Created at least one connection", conn_ok),
    ]
    for label, ok in checklist:
        st.checkbox(label, value=ok, disabled=True)


# =========================================================================
# MAIN
# =========================================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🔬", layout="wide")
    inject_custom_css()

    try:
        init_db()
    except Exception as e:
        st.error(f"Could not initialize the database: {e}")
        st.stop()

    render_header(db_dashboard_counts())

    st.session_state["api_key"] = sidebar_api_key()

    if not PDFPLUMBER_AVAILABLE:
        st.sidebar.error("pdfplumber is not installed — PDF upload will fail. Check requirements.txt.")
    if not GROQ_SDK_AVAILABLE:
        st.sidebar.error("groq package is not installed — AI features will be disabled. Check requirements.txt.")

    st.sidebar.markdown("### 🧭 Navigate")
    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Research Library", "Research AI", "Research Memory", "Protocol Companion", "Research Journey"],
        label_visibility="collapsed",
    )

    try:
        if page == "Research Library":
            page_research_library()
        elif page == "Research AI":
            page_research_ai()
        elif page == "Research Memory":
            page_research_memory()
        elif page == "Protocol Companion":
            page_protocol_companion()
        elif page == "Research Journey":
            page_research_journey()
        elif page == "Dashboard":
            page_dashboard()
    except Exception as e:
        st.error(f"Something went wrong while rendering this page: {e}. Your saved data is unaffected.")


if __name__ == "__main__":
    main()
