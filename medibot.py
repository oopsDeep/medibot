import os
import threading
from pathlib import Path
from urllib.parse import quote
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

DOCS_BASE_URL = os.getenv("DOCS_BASE_URL", "").rstrip("/")
DOCS_LOCAL_ROOT = os.getenv(
    "DOCS_LOCAL_ROOT",
    r"E:\PROGRAM_SOLUTION\Python\PycharmProjects\chatbot_FAQ\data"
)

# ================= PAGE + THEME-SAFE STYLES =================
st.set_page_config(page_title="Medibot", page_icon="💬", layout="wide")
st.title("Medibot")

st.markdown("""
<style>
:root{
  --bg: #ffffff;
  --surface: #ffffff;
  --surface-raised: #f8fafc;
  --text: #0f172a;
  --muted: #64748b;
  --border: #e2e8f0;
  --accent: #2563eb;  /* user tone */
  --good: #16a34a;    /* assistant tone */

  /* Layout knobs */
  --page-width: 980px;  /* page container width */
}
@media (prefers-color-scheme: dark){
  :root{
    --bg: #0b0f14;
    --surface: #0f1520;
    --surface-raised: #111827;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --border: #1f2937;
    --accent: #60a5fa;
    --good: #22c55e;
  }
}

/* Layout */
html, body, .main { background: var(--bg) !important; }
.main .block-container { max-width: var(--page-width); padding-top: 16px; padding-bottom: 16px; }

/* Header */
.header{
  display:flex; align-items: center; justify-content: space-between; gap: 8px;
  padding: 10px 12px; border: 1px solid var(--border); border-radius: 12px;
  background: var(--surface);
}
.h-left{ display:flex; align-items:center; gap:10px; }
.h-title{ font-weight: 700; font-size: 18px; color: var(--text); }

/* Badges */
.badges{ display:flex; align-items:center; gap:8px; flex-wrap: wrap; }
.badge{
  display:inline-flex; align-items:center; gap:.4rem;
  padding:5px 9px; border-radius: 999px;
  background: var(--surface-raised); border: 1px solid var(--border);
  font-size: 12.5px; color: var(--text);
}

/* Chat card wrapper */
.card{ border:1px solid var(--border); border-radius: 12px; padding: 10px 12px; background: var(--surface); }

/* Message stream */
.stream{ display:flex; flex-direction:column; gap:10px; }

/* Bubbles: clearly different and left/right aligned */
.bubble{
  max-width: 82%;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px;
  line-height: 1.6;
}
.bubble .chip{
  display:inline-flex; align-items:center; gap:6px;
  font-weight:600; font-size:12px;
  padding:2px 8px; border-radius:999px;
  border:1px solid var(--border);
  background: var(--surface-raised);
  margin-bottom: 6px;
}

/* USER: right side, blue tint */
.user{
  margin-left: auto;                /* right align */
  background: color-mix(in oklab, var(--accent) 12%, var(--surface));
  border-left: 3px solid var(--accent);
}

/* ASSISTANT: left side, neutral/green hint */
.assistant{
  margin-right: auto;               /* left align */
  background: color-mix(in oklab, var(--good) 8%, var(--surface));
  border-left: 3px solid var(--good);
}

/* References block: single line "Page numbers: 1, 2, 3" */
.ref-line{
  color: var(--muted);
  font-size: 13px;
  margin-top: 6px;
}

/* Typing indicator dots */
.typing { display:inline-flex; align-items:center; gap:6px; }
.dot{
  width:6px; height:6px; border-radius:999px; background: var(--muted);
  animation: pulse 1s infinite ease-in-out;
}
.dot:nth-child(2){ animation-delay: .15s; }
.dot:nth-child(3){ animation-delay: .30s; }
@keyframes pulse{
  0%, 80%, 100% { transform: scale(0.6); opacity: .4; }
  40% { transform: scale(1); opacity: 1; }
}

/* Chat input: exactly page width, centered, and uniform color */
[data-testid="stChatInput"]{
  max-width: var(--page-width) !important;
  width: 100% !important;
  margin: 0 auto !important;
}
[data-testid="stChatInput"] > div{
  background: var(--surface) !important;    /* uniform color */
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

/* Make inside of the input transparent so only the bar background shows */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] div[contenteditable="true"],
[data-testid="stChatInput"] * {
  background: transparent !important;
  color: var(--text) !important;
}

/* Placeholder tint */
[data-testid="stChatInput"] textarea::placeholder{
  color: var(--muted) !important;
}

/* Hide Streamlit default chat chrome */
[data-testid="stChatMessage"]{ border:none !important; background:transparent !important; padding:0 !important; }

/* General button look */
div[data-testid="stButton"] > button {
  box-shadow: none !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  border-radius: 10px !important;
  padding: 8px 12px !important;
  font-size: 14px !important;
}
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
        try: return int("".join(ch for ch in str(x) if ch.isdigit()))
        except: return 10**9
    return {k: sorted(v, key=sort_key) for k, v in by_source.items()}

# ---------- Local static server for PDFs ----------
def ensure_local_docs_server(root_dir: str) -> str:
    if DOCS_BASE_URL:
        return DOCS_BASE_URL
    if "docs_server" in st.session_state:
        return st.session_state["docs_server"]["base_url"]

    import socketserver, http.server, socket, functools
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()

    httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=thread_target, args=(httpd,), daemon=True).start()

    base_url = f"http://127.0.0.1:{port}"
    st.session_state["docs_server"] = {"httpd": httpd, "base_url": base_url, "root": str(root)}
    return base_url

def thread_target(httpd):
    httpd.serve_forever()

def get_docs_base_url():
    return DOCS_BASE_URL or ensure_local_docs_server(DOCS_LOCAL_ROOT)

def to_clickable_url(src_path: str) -> str:
    if not src_path:
        return ""
    base_url = get_docs_base_url()
    fname = Path(src_path).name
    return f"{base_url}/{quote(fname)}"

def render_references(pages_by_source: dict):
    if not pages_by_source:
        return
    # Single line per source: "<file> — Page numbers: 2, 7, 12"
    for src, pgs in pages_by_source.items():
        base = Path(src).name if src else "unknown"
        url = to_clickable_url(src) if src else ""
        title = f"[**{base}**]({url})" if url.startswith("http") else f"**{base}**"
        numbers = ", ".join(pgs)
        st.markdown(
            f"{title}  \n<span class='ref-line'>Page numbers: {numbers}</span>",
            unsafe_allow_html=True
        )

# ================== HEADER ==================
vs_ready = os.path.exists(DB_FAISS_PATH)
st.markdown(
    f"""
<div class="header">
  <div class="h-left">
    <div class="h-title">Minimal RAG over indexed docs</div>
  </div>
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
    st.markdown(
        f"""
<div class='bubble user'>
  <div class='chip'>🧑 You</div>
  {turn['q']}
</div>""",
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
<div class='bubble assistant'>
  <div class='chip'>🤖 Medibot</div>
  {turn['a']}
</div>""",
        unsafe_allow_html=True
    )
    if turn.get("refs"):
        render_references(turn["refs"])

if has_history:
    st.markdown("</div>", unsafe_allow_html=True)

# ================== CHAT INPUT ==================
prompt = st.chat_input("Type your question…")

if prompt:
    # User bubble immediately
    st.markdown(
        f"""
<div class='bubble user'>
  <div class='chip'>🧑 You</div>
  {prompt}
</div>""",
        unsafe_allow_html=True
    )

    # Placeholder assistant bubble with typing dots (loader)
    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        """
<div class='bubble assistant'>
  <div class='chip'>🤖 Medibot</div>
  <span class='typing'><span class='dot'></span><span class='dot'></span><span class='dot'></span></span>
</div>
""",
        unsafe_allow_html=True
    )

    # Vector store and LLM
    vectorstore = get_vectorstore()
    if vectorstore is None:
        typing_placeholder.empty()
        st.error("FAISS index not found at vectorstore/db_faiss. Build it with the same embedding model.")
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

    # MAIN QA (with spinner)
    try:
        with st.spinner("Medibot is thinking…"):
            docs = retriever.get_relevant_documents(prompt)
            if not docs:
                answer = "I don't know based on the provided documents."
                pages_by_source = {}
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

    # Replace typing bubble with final assistant bubble
    typing_placeholder.markdown(
        f"""
<div class='bubble assistant'>
  <div class='chip'>🤖 Medibot</div>
  {answer}
</div>""",
        unsafe_allow_html=True
    )

    if pages_by_source:
        _ = get_docs_base_url()
        render_references(pages_by_source)

    # Persist turn
    st.session_state.history.append({"q": prompt, "a": answer, "refs": pages_by_source or {}})
