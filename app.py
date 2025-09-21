'''import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.title("📄 RAG PDF Chatbot")

api_key = os.getenv("OPENAI_API_KEY")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

@st.cache_resource
def load_vectorstore(doc_path):
    loader = PyPDFLoader(doc_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = FAISS.from_documents(docs_split, embeddings)
    return vectordb

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    vectordb = load_vectorstore("temp.pdf")

    llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    query = st.text_input("Ask a question about the PDF:")

    if query:
        answer = qa.run(query)
        st.write("**Answer:**")
        st.write(answer)'''








import time
import openai
from tenacity import retry, stop_after_attempt, wait_fixed
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.title("📄 RAG PDF Chatbot")

api_key = st.secrets["OPENAI_API_KEY"]

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))  # Retry 5 times with 5 seconds between attempts
def embed_documents_with_retry(embeddings, texts):
    return embeddings.embed_documents(texts)

@st.cache_resource
def load_vectorstore(doc_path):
    loader = PyPDFLoader(doc_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Use the retry function for embedding
    embedded_docs = embed_documents_with_retry(embeddings, docs_split)
    
    vectordb = FAISS.from_documents(embedded_docs, embeddings)
    return vectordb

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    vectordb = load_vectorstore("temp.pdf")

    llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    query = st.text_input("Ask a question about the PDF:")

    if query:
        answer = qa.run(query)
        st.write("**Answer:**")
        st.write(answer)
