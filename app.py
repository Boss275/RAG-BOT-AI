import os
import tempfile
import streamlit as st
from typing_extensions import List, TypedDict
from pdf2image import convert_from_path
import pytesseract
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# ─────────────────────────────────────────────
# Define state structure
# ─────────────────────────────────────────────

class S(TypedDict):
    question: str
    context: List[Document]
    answer: str

# ─────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────

st.set_page_config(page_title="📚 RAG PDF Q&A Assistant", layout="wide")
st.title("🤖 RAG AI BOT")
st.caption("📄 Upload your PDFs and ask me anything about them!")

# ─────────────────────────────────────────────
# OCR fallback for scanned PDFs
# ─────────────────────────────────────────────

def ocr_pdf(path: str) -> List[Document]:
    pages = convert_from_path(path)
    docs = []
    for i, p in enumerate(pages):
        t = pytesseract.image_to_string(p, lang="eng")
        if t.strip():
            docs.append(Document(page_content=t, metadata={"page": i+1}))
    return docs

# ─────────────────────────────────────────────
# PDF loading and text splitting
# ─────────────────────────────────────────────

def split_text(files) -> List[Document]:
    out = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            if all("studyplusplus" in d.page_content.lower() for d in docs):
                st.warning(f"🔍 OCR was used for **{f.name}** due to scan-based content.")
                docs = ocr_pdf(tmp_path)
        finally:
            os.remove(tmp_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        out.extend(splitter.split_documents(docs))
    return out

# ─────────────────────────────────────────────
# Embedding model setup
# ─────────────────────────────────────────────

def get_emb():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

# ─────────────────────────────────────────────
# Vector store creation
# ─────────────────────────────────────────────

def make_store(embed, docs):
    if not docs:
        raise ValueError("No documents to index 🤷")
    return FAISS.from_documents(documents=docs, embedding=embed)

# ─────────────────────────────────────────────
# Document retrieval
# ─────────────────────────────────────────────

def retrieve(state: S, store):
    docs = store.similarity_search(state["question"], k=3)
    return {"context": docs}

# ─────────────────────────────────────────────
# Response generation
# ─────────────────────────────────────────────

def generate(state: S):
    txt = "\n\n".join(d.page_content for d in state["context"])
    resp = st.session_state.chain.invoke({"question": state["question"], "context": txt})
    return {"answer": resp["text"]}

# ─────────────────────────────────────────────
# Memory setup
# ─────────────────────────────────────────────

if "mems" not in st.session_state:
    st.session_state.mems = {}

# ─────────────────────────────────────────────
# Sidebar UI and session control
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    key = st.text_input("🔑 Enter your **Groq API Key**:", type="password")
    model = st.selectbox("🧠 Choose a model:", ["qwen/qwen3-32b", "gemma2-9b-it", "openai/gpt-oss-120b"])
    temp = st.slider("🎛️ Response creativity (temperature):", 0.0, 2.0, 0.7, 0.1)

    if st.button("✨ Start New Chat"):
        sid = f"session_{len(st.session_state.mems)+1}"
        st.session_state.mems[sid] = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
        st.session_state.sid = sid

    if not st.session_state.mems:
        st.session_state.mems["default"] = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
        st.session_state.sid = "default"

    sid = st.selectbox("💬 Active Session:", list(st.session_state.mems.keys()), index=len(st.session_state.mems)-1)
    st.session_state.sid = sid

# ─────────────────────────────────────────────
# API Key check
# ─────────────────────────────────────────────

if not key:
    st.warning("⚠️ Please enter your API key in the sidebar to get started.")
    st.stop()

# ─────────────────────────────────────────────
# File upload and processing
# ─────────────────────────────────────────────

files = st.file_uploader("📤 Upload your PDF(s) below:", accept_multiple_files=True)

if st.button("🛠️ Process PDF(s)") and files:
    with st.spinner("⏳ Processing your PDF(s)... hang tight!"):
        p = hub.pull("rlm/rag-prompt")
        llm = ChatGroq(model=model, api_key=key, temperature=temp)
        parser = StrOutputParser()
        st.session_state.chain = LLMChain(llm=llm|parser, prompt=p, memory=st.session_state.mems[sid])

        emb = get_emb()
        splits = split_text(files)
        store = make_store(emb, splits)
        store.add_documents(splits)

        g = StateGraph(S)
        g.add_node("retrieve", lambda s: retrieve(s, store))
        g.add_node("generate", generate)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "generate")
        st.session_state.graph = g.compile()

        st.success("✅ Your PDFs are ready! You can now ask questions in the chat below. 💬")

# ─────────────────────────────────────────────
# Display previous chat history
# ─────────────────────────────────────────────

if h := st.session_state.mems[st.session_state.sid].chat_memory.messages:
    for m in h:
        if m.type == "human":
            with st.chat_message("user"):
                st.markdown(m.content)
        elif m.type == "ai":
            with st.chat_message("assistant"):
                st.markdown(m.content)

# ─────────────────────────────────────────────
# Chat input and response
# ─────────────────────────────────────────────

if "graph" in st.session_state:
    if q := st.chat_input("❓ Ask me anything about your PDFs!"):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Thinking... just a sec!"):
                r = st.session_state.graph.invoke({"question": q})
                with st.expander("📚 Context Preview"):
                    for d in r["context"]:
                        st.write(d.page_content[:500] + "...")
                st.subheader("🧠 Answer:")
                st.markdown(r["answer"])
