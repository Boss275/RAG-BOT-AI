import os
import time
import openai
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from tenacity import retry, stop_after_attempt, wait_fixed

st.title("📄 RAG PDF Chatbot")

api_key = st.secrets["OPENAI_API_KEY"]

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def embed_documents_with_retry(embeddings, texts):
    return embeddings.embed_documents(texts)

@st.cache_resource
def load_vectorstore(files):
    all_docs = []
    temp_dir = "/tmp/pdf_uploads"
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for file in files:
        file_path = os.path.join(temp_dir, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_split = splitter.split_documents(docs)
        all_docs.extend(docs_split)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    embedded_docs = embed_documents_with_retry(embeddings, all_docs)
    
    vectordb = FAISS.from_documents(embedded_docs, embeddings)
    return vectordb

if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None

if uploaded_files:
    with st.spinner("Processing your documents..."):
        st.session_state.vectordb = load_vectorstore(uploaded_files)
        st.success("Documents processed and ready for querying!")

    llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.vectordb.as_retriever())

    query = st.text_input("Ask a question about the uploaded PDFs:")

    if query:
        with st.spinner("Generating answer..."):
            answer = qa.run(query)
        st.write("**Answer:**")
        st.write(answer)

else:
    st.warning("Please upload one or more PDFs to get started.")
