# app.py
import streamlit as st
import os, time, json, sqlite3
from retriever import Retriever
from tools import web_search, safe_eval
from llm_local import get_generator

# ---- Setup ----
st.set_page_config(page_title="PDF Research Assistant", layout="wide")
st.title("PDF Research Assistant (free, local models)")

DATA_DIR = "sample_papers"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_DIR = "vectorstore"
HISTORY_JSON = "chat_history.json"
HISTORY_DB = "chat_history.db"

# lazy singletons
if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever(index_dir=INDEX_DIR)
if "generator" not in st.session_state:
    st.info("Loading local model (this may take time the first run)...")
    st.session_state.generator = get_generator()  # device=-1 CPU

# ----- Sidebar: upload PDFs & index -----
st.sidebar.header("1) Upload & Index PDFs")
uploaded = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded:
    saved_paths = []
    for up in uploaded:
        save_path = os.path.join(DATA_DIR, up.name)
        with open(save_path, "wb") as f:
            f.write(up.getbuffer())
        saved_paths.append(save_path)
    st.sidebar.success(f"Saved {len(saved_paths)} files to {DATA_DIR}")
    if st.sidebar.button("Index uploaded PDFs"):
        try:
            st.session_state.retriever.index_pdfs(saved_paths)
            st.sidebar.success("Indexing done.")
        except Exception as e:
            st.sidebar.error(f"Indexing failed: {e}")

if st.sidebar.button("Load existing index"):
    try:
        st.session_state.retriever.load_index()
        st.sidebar.success("Index loaded.")
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")

# ----- Chat UI -----
st.header("Ask a question about the indexed PDFs")
col1, col2 = st.columns([3,1])
with col1:
    query = st.text_input("Your question", key="q_input")
    use_web = st.checkbox("Allow web search (DuckDuckGo)", value=False)
    allow_calc = st.checkbox("Allow calculator tool (simple math)", value=True)
    top_k = st.slider("Retrieval: number of chunks to fetch", min_value=1, max_value=8, value=4)
    if st.button("Ask"):
        if not query:
            st.warning("Write a question first.")
        else:
            with st.spinner("Running retrieval + generation..."):
                # 1) retrieve relevant chunks
                try:
                    results = st.session_state.retriever.query(query, top_k=top_k)
                except Exception as e:
                    st.error(f"Retrieval error: {e}")
                    results = []
                context_parts = []
                source_list = []
                for r in results:
                    txt = r["text"]
                    meta = r["meta"]
                    context_parts.append(f"[{meta['source']} p{meta['page']}] {txt}")
                    source_list.append(f"{meta['source']} p{meta['page']}")
                context = "\n\n---\n\n".join(context_parts) if context_parts else "No document context available."

                # 2) optional web search
                web_text = ""
                if use_web:
                    web_text = web_search(query, max_results=3)

                # 3) prepare prompt, allow calculator by instructing model
                prompt = f"""You are an assistant answering questions about the provided document excerpts and (optionally) web search results.
Use document excerpts first. If you use web results cite them inline. If a numeric calculation is required, compute it and show steps (you can ask the calculator if needed).

Document excerpts:
{context}

Web results:
{web_text if web_text else 'No web results asked.'}

Question:
{query}

Answer concisely. After the answer, include a "SOURCES:" line listing the document pages or web links used.
"""
                # 4) detect simple math expression and evaluate BEFORE generation if user allowed
                calc_result = None
                if allow_calc:
                    # cheap heuristic: if the query looks like a math expression, evaluate
                    import re
                    expr_match = re.match(r"^[0-9\.\s\+\-\*\/\^\(\)]+$", query.strip())
                    if expr_match:
                        try:
                            # replace ^ with **
                            safe_expr = query.replace("^", "**")
                            calc_result = safe_eval(safe_expr)
                        except Exception:
                            calc_result = None

                generator = st.session_state.generator
                # append known calculator result to prompt if found
                if calc_result is not None:
                    prompt = f"{prompt}\n\nCalculator result: {calc_result}\n\nNow answer the question using the calculator result as needed."

                # 5) generate (use the HF pipeline)
                out = generator(prompt, max_length=512, do_sample=False)[0]["generated_text"]

                # 6) show results
                st.subheader("Answer")
                st.write(out)
                st.markdown("**Sources (retrieved)**: " + (", ".join(source_list) if source_list else "None"))
                if use_web:
                    st.markdown("**Web search summary**:")
                    st.write(web_text)

                # 7) Save to JSON and SQLite
                entry = {
                    "timestamp": int(time.time()),
                    "question": query,
                    "answer": out,
                    "sources": source_list,
                    "web_results": web_text
                }
                # JSON
                try:
                    hfile = HISTORY_JSON
                    if os.path.exists(hfile):
                        with open(hfile, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        data = []
                    data.append(entry)
                    with open(hfile, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    st.error(f"Error saving JSON history: {e}")

                # SQLite
                try:
                    conn = sqlite3.connect(HISTORY_DB, check_same_thread=False)
                    c = conn.cursor()
                    c.execute(
                        """CREATE TABLE IF NOT EXISTS chats
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, question TEXT, answer TEXT, sources TEXT, web_results TEXT)"""
                    )
                    c.execute(
                        "INSERT INTO chats (ts, question, answer, sources, web_results) VALUES (?, ?, ?, ?, ?)",
                        (entry["timestamp"], entry["question"], entry["answer"], ",".join(entry["sources"]), entry["web_results"])
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    st.error(f"Error saving to DB: {e}")

# ----- Show chat history and controls -----
st.sidebar.header("2) Chat History")
if os.path.exists(HISTORY_JSON):
    if st.sidebar.button("Show JSON history"):
        with open(HISTORY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.sidebar.write(f"Saved entries: {len(data)}")
        for e in data[-10:]:
            st.sidebar.write(f"- {time.ctime(e['timestamp'])}: Q: {e['question']}")
else:
    st.sidebar.write("No JSON history yet.")

if os.path.exists(HISTORY_DB):
    if st.sidebar.button("Show last DB rows"):
        conn = sqlite3.connect(HISTORY_DB)
        c = conn.cursor()
        rows = c.execute("SELECT ts, question FROM chats ORDER BY ts DESC LIMIT 10").fetchall()
        conn.close()
        for r in rows:
            st.sidebar.write(f"- {time.ctime(r[0])}: {r[1]}")

st.sidebar.markdown("Run `streamlit run app.py` to start the app.")
