import os
import tempfile
import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from typing_extensions import List, TypedDict
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

class State(TypedDict):
    q: str
    ctx: List[Document]
    ans: str

st.set_page_config(page_title='RAG QNA', layout='wide')
st.title("RAG QNA")
st.caption("Upload PDFs and ask questions.")

def ocr_pdf(path):
    pages = convert_from_path(path)
    out = []
    for i, p in enumerate(pages):
        t = pytesseract.image_to_string(p, lang="eng")
        if t.strip():
            out.append(Document(page_content=t, metadata={"pg": i + 1}))
    return out

def split_pdfs(files):
    chunks = []
    for f in files:
        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(f.read())
            tmp_path = tmp.name
            tmp.close()

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            if all("studyplusplus" in d.page_content.lower() for d in docs):
                st.warning(f"OCR fallback for {f.name}")
                docs = ocr_pdf(tmp_path)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks.extend(splitter.split_documents(docs))
        finally:
            if tmp_path: os.remove(tmp_path)
    return chunks

def get_emb():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})

def build_store(emb, docs):
    if not docs: raise ValueError("No docs for FAISS")
    return FAISS.from_documents(docs, emb)

def retr(state, store):
    return {"ctx": store.similarity_search(state["q"], k=3)}

def gen_ans(state):
    c = "\n\n".join(d.page_content for d in state["ctx"])
    r = st.session_state.chain.invoke({"question": state["q"], "context": c})
    a = r["text"].replace("Answer:", "").replace("answer:", "").strip()
    return {"ans": f"{a[:10]}... basically: {a[-50:]}"}

if "sess" not in st.session_state:
    st.session_state.sess = {}

with st.sidebar:
    st.title("Setup")
    k = st.text_input("API Key")
    m = st.selectbox("Model", ["qwen/qwen3-32b", "gemma2-9b-it", "openai/gpt-oss-120b"])
    t = st.slider("Temp", 0.0, 2.0, 0.7, 0.1)
    
    if st.button("New Session"):
        sid = f"sess_{len(st.session_state.sess) + 1}"
        st.session_state.sess[sid] = ConversationBufferMemory(memory_key="chat_history", input_key='q', return_messages=True)
        st.session_state.sid = sid

    if not st.session_state.sess:
        st.session_state.sess["def"] = ConversationBufferMemory(memory_key="chat_history", input_key='q', return_messages=True)
        st.session_state.sid = "def"

    sel = st.selectbox("Select Session", list(st.session_state.sess.keys()), index=len(st.session_state.sess)-1)
    st.session_state.sid = sel

if not k:
    st.warning("Enter API key")
    st.stop()

fs = st.file_uploader("Upload PDFs:", accept_multiple_files=True)

if st.button("Process") and fs:
    with st.spinner("Processing..."):
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatGroq(model=m, api_key=k, temperature=t)
        parser = StrOutputParser()
        chain = llm | parser
        st.session_state.chain = LLMChain(llm=chain, prompt=prompt, memory=st.session_state.sess[st.session_state.sid])

        emb = get_emb()
        docs = split_pdfs(fs)
        store = build_store(emb, docs)
        store.add_documents(docs)

        g = StateGraph(State)
        g.add_node("ret", lambda s: retr(s, store))
        g.add_node("gen", gen_ans)
        g.add_edge(START, "ret")
        g.add_edge("ret", "gen")
        st.session_state.g = g.compile()
        st.success("PDFs ready. Ask questions.")

if hist := st.session_state.sess[st.session_state.sid].chat_memory.messages:
    for msg in hist:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif msg.type == "ai":
            with st.chat_message("assistant"):
                st.markdown(msg.content)

if "g" in st.session_state:
    if q := st.chat_input("Ask a question:"):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                res = st.session_state.g.invoke({"q": q})
                with st.expander("Preview"):
                    for d in res["ctx"]:
                        st.write(f"{d.page_content[:200]}...")
                st.subheader("Answer:")
                st.markdown(res["ans"])
