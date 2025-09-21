import os
import tempfile
import streamlit as st
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
from pdf2image import convert_from_path
import pytesseract

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

st.set_page_config(page_title='RAG QNA', layout='wide', initial_sidebar_state='expanded')
st.title('RAG QNA')
st.caption("Upload PDF files and ask questions.")

def ocr_pdf(path: str) -> List[Document]:
    pages = convert_from_path(path)
    docs = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="eng")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"page": i+1}))
    return docs

def split_text(files) -> List[Document]:
    chunks = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            if all("studyplusplus" in d.page_content.lower() for d in docs):
                docs = ocr_pdf(tmp_path)
        finally:
            os.remove(tmp_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        chunks.extend(splitter.split_documents(docs))
    return chunks

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

def get_vector(docs, emb):
    if not docs:
        raise ValueError("No documents to index")
    return FAISS.from_documents(documents=docs, embedding=emb)

def retrieve(state: State, vs):
    return {"context": vs.similarity_search(state["question"], k=3)}

def generate(state: State):
    text = "\n\n".join(doc.page_content for doc in state["context"])
    result = st.session_state.chain.invoke({"question": state["question"], "context": text})
    return {"answer": result["text"]}

if "mem" not in st.session_state:
    st.session_state.mem = {}

with st.sidebar:
    st.header("Settings")
    key = st.text_input("Groq API Key", type="password", value=st.secrets["general"]["GROQ_KEY"])
    model = st.selectbox("Model", ["qwen/qwen3-32b", "gemma2-9b-it", "openai/gpt-oss-120b"])
    temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    if st.button("New Session"):
        sid = f"session_{len(st.session_state.mem)+1}"
        st.session_state.mem[sid] = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
        st.session_state.sid = sid
    if not st.session_state.mem:
        st.session_state.mem["default"] = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
        st.session_state.sid = "default"
    sid = st.selectbox("Select Session", list(st.session_state.mem.keys()), index=len(st.session_state.mem)-1)
    st.session_state.sid = sid

if not key:
    st.warning("Enter API key")
    st.stop()

files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

if st.button("Process Files") and files:
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(model=model, api_key=key, temperature=temp)
    parser = StrOutputParser()
    chain = llm | parser
    st.session_state.chain = LLMChain(llm=chain, prompt=prompt, memory=st.session_state.mem[sid])
    emb = get_embeddings()
    docs = split_text(files)
    vs = get_vector(docs, emb)
    vs.add_documents(docs)
    g = StateGraph(State)
    g.add_node("retrieve", lambda s: retrieve(s, vs))
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    st.session_state.graph = g
    st.success("Documents processed. You can ask questions.")

if history := st.session_state.mem[sid].chat_memory.messages:
    for m in history:
        if m.type == "human":
            with st.chat_message("user"):
                st.markdown(m.content)
        elif m.type == "ai":
            with st.chat_message("assistant"):
                st.markdown(m.content)

if "graph" in st.session_state:
    if q := st.chat_input("Ask a question:"):
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                r = st.session_state.graph.invoke({"question": q})

                with st.expander("Context Preview"):
                    for d in r["context"]:
                        st.write(d.page_content[:500] + "...")

                st.subheader("Answer:")
                st.markdown(r["answer"])
