import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load API Key securely from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key

st.set_page_config(page_title="RAG AI Bot", layout="wide")
st.title("📚 RAG AI Chatbot")
st.write("Upload your documents and ask questions about them!")

uploaded_files = st.file_uploader("Upload PDF, TXT, or DOCX files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

def load_documents(file_paths):
    all_docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            st.warning(f"❌ Unsupported file type: {path}")
            continue
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

if uploaded_files:
    temp_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_paths.append(tmp_file.name)

    with st.spinner("📄 Reading and parsing documents..."):
        documents = load_documents(temp_paths)

    with st.spinner("🧠 Creating vector index..."):
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(documents, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=vectordb.as_retriever()
    )

    st.success("✅ Ready! Ask your questions.")
    query = st.text_input("💬 Ask a question based on the uploaded docs:")

    if query:
        with st.spinner("🤖 Thinking..."):
            answer = qa_chain.run(query)
        st.markdown(f"**Answer:** {answer}")
