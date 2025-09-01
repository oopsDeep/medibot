import os
import re
import unicodedata
import html
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ================= CONFIG =================
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# --- External URL overrides ONLY (no local server needed) ---
MANUAL_URLS = {
    "BMA A-Z Family Medical Ency_ (Z-Library).pdf":
        "https://archive.org/details/a-z-family-medical-encyclopedia_202101/page/287/mode/2up",
    "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf":
        "https://huggingface.co/spaces/teganmosi/medicalchatbot/blob/c4529cf3ebbf73301e20263bb414c23b23148c92/Data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf",
}

# ================= PAGE + THEME =================
st.set_page_config(page_title="Medibot", page_icon="💬", layout="wide")
st.title("Medibot")

# ================= STYLE =================
st.markdown("""
<style>
:root{
  --bg: #ffffff; --surface: #ffffff; --surface-raised: #f8fafc;
  --text: #0f172a; --muted: #64748b; --border: #e2e8f0;
  --accent: #2563eb; --good: #16a34a; --page-width: 980px;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg: #0b0f14; --surface: #0f1520; --surface-raised: #111827;
    --text: #e5e7eb; --muted: #94a3b8; --border: #1f2937;
    --accent: #60a5fa; --good: #22c55e;
  }
}
html, body, .main { background: var(--bg) !important; }
.main .block-container { max-width: var(--page-width); padding-top: 16px; padding-bottom: 16px; }
.header{ display:flex; justify-content: space-between; gap: 8px; padding: 10px 12px; border: 1px solid var(--border); border-radius: 12px; background: var(--surface);}
.h-title{ font-weight: 700; font-size: 18px; color: var(--text); }
.badge{ padding:5px 9px; border-radius: 999px; background: var(--surface-raised); border: 1px solid var(--border); font-size: 12.5px; color: var(--text);}
.card{ border:1px solid var(--border); border-radius: 12px; padding: 10px 12px; background: var(--surface);}
.stream{ display:flex; flex-direction:column; gap:10px;}
.bubble{ max-width: 82%; border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px; line-height: 1.6;}
.user{ margin-left: auto; background: color-mix(in oklab, var(--accent) 12%, var(--surface)); border-left: 3px solid var(--accent);}
.assistant{ margin-right: auto; background: color-mix(in oklab, var(--good) 8%, var(--surface)); border-left: 3px solid var(--good);}
.ref-line{ color: var(--muted); font-size: 13px; margin-top: 6px;}
</style>
""", unsafe_allow_html=True)

# ================= HELPERS =================
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if not os.path.exists(DB_FAISS_PATH):
        return None
    return FAISS.load_local(DB_FAISS_PATH, emb, allow_dangerous_deserialization=True)

def set_custom_prompt():
    template = """
You are a precise document QA assistant. Answer ONLY from the context.
Rules:
- Do NOT repeat or correct the user's question.
- If the answer is not in the context, say: "I don't know based on the provided documents."
- Give a complete answer, then up to 3 short bullets if supported by context.

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def extract_pages_only(docs):
    by_source = {}
    for d in docs or []:
        meta = d.metadata or {}
        label = meta.get("page_label")
        if not label:
            p = meta.get("page")
            if isinstance(p, int):
                label = str(p + 1)
            elif isinstance(p, str) and p.isdigit():
                label = str(int(p) + 1)
        if not label:
            continue
        src = meta.get("source")
        if src:
            by_source.setdefault(src, set()).add(label)

    def sort_key(x):
        try:
            return int("".join(ch for ch in str(x) if ch.isdigit()))
        except:
            return 10**9

    return {k: sorted(v, key=sort_key) for k, v in by_source.items()}

# -------- Name normalization + URL lookup ----------
def _normalize_name(s: str) -> str:
    """Normalize a path or filename to a loose comparable key."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = Path(s.replace("\\", "/")).name  # strip any directories
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[_\s]+", " ", s).strip()
    return s

# Prepare a normalized lookup so any variant of the same filename matches
_NORMALIZED_URLS = {}
for k, v in MANUAL_URLS.items():
    nk_full = _normalize_name(k)                 # with extension
    nk_stem = _normalize_name(Path(k).stem)      # without extension
    _NORMALIZED_URLS[nk_full] = v
    _NORMALIZED_URLS[nk_stem] = v

def to_clickable_url(src_path: str) -> str:
    """Return external URL for a given local/absolute source path if mapped."""
    if not src_path:
        return ""
    base = Path(src_path).name
    key_full = _normalize_name(base)
    key_stem = _normalize_name(Path(base).stem)
    return _NORMALIZED_URLS.get(key_full) or _NORMALIZED_URLS.get(key_stem) or ""

def render_references(pages_by_source: dict):
    """
    Renders references so the link title is ONLY the filename (no directories).
    If the filename is present in MANUAL_URLS, it becomes a clickable link.
    Otherwise it's shown as plain text (still clean).
    """
    if not pages_by_source:
        return
    for src, pgs in pages_by_source.items():
        filename = Path(src).name if src else "unknown"
        numbers = ", ".join(pgs)
        url = to_clickable_url(src) if src else ""

        if url:
            st.markdown(
                f"[**{html.escape(filename)}**]({url})  \n"
                f"<span class='ref-line'>Page numbers: {html.escape(numbers)}</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<strong>{html.escape(filename)}</strong><br>"
                f"<span class='ref-line'>Page numbers: {html.escape(numbers)}</span>",
                unsafe_allow_html=True
            )

# ================== HEADER ==================
vs_ready = os.path.exists(DB_FAISS_PATH)
st.markdown(
    f"""
<div class="header">
  <div class="h-title">Minimal RAG over indexed docs</div>
  <div class="badges">
    <span class="badge">🤖 {GROQ_MODEL}</span>
    <span class="badge">🗂️ FAISS • {EMBED_MODEL}</span>
    <span class="badge">{'Index: Ready ✅' if vs_ready else 'Index: Missing ⚠️'}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ================== CHAT HISTORY ==================
if "history" not in st.session_state:
    st.session_state.history = []

has_history = len(st.session_state.history) > 0
if has_history:
    st.markdown("<div class='card stream'>", unsafe_allow_html=True)

for turn in st.session_state.history:
    st.markdown(f"<div class='bubble user'><b>🧑 You</b><br>{turn['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble assistant'><b>🤖 Medibot</b><br>{turn['a']}</div>", unsafe_allow_html=True)
    if turn.get("refs"):
        render_references(turn["refs"])

if has_history:
    st.markdown("</div>", unsafe_allow_html=True)

# ================== CHAT INPUT ==================
prompt = st.chat_input("Type your question…")
if prompt:
    st.markdown(f"<div class='bubble user'><b>🧑 You</b><br>{prompt}</div>", unsafe_allow_html=True)

    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        "<div class='bubble assistant'><b>🤖 Medibot</b><br>…thinking…</div>",
        unsafe_allow_html=True
    )

    vectorstore = get_vectorstore()
    if vectorstore is None:
        typing_placeholder.empty()
        st.error("FAISS index not found.")
        st.stop()

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        typing_placeholder.empty()
        st.error("GROQ_API_KEY not set.")
        st.stop()

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.4},
    )

    llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.0, groq_api_key=groq_key)

    try:
        with st.spinner("Medibot is thinking…"):
            docs = retriever.get_relevant_documents(prompt)
            if not docs:
                answer, pages_by_source = "I don't know based on the provided documents.", {}
            else:
                context = "\n\n".join(doc.page_content for doc in docs)
                custom = set_custom_prompt().format(context=context, question=prompt)
                resp = llm.invoke(custom)
                answer = (resp.content or "").strip()
                pages_by_source = extract_pages_only(docs)
    except Exception as e:
        typing_placeholder.empty()
        st.error(f"Error: {e}")
        st.stop()

    typing_placeholder.markdown(
        f"<div class='bubble assistant'><b>🤖 Medibot</b><br>{answer}</div>",
        unsafe_allow_html=True
    )

    if pages_by_source:
        render_references(pages_by_source)

    st.session_state.history.append({"q": prompt, "a": answer, "refs": pages_by_source or {}})
