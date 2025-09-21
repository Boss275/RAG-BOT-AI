!pip install pypdf

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.title("📄 RAG PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs_split, embeddings)

    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    query = st.text_input("Ask a question about the PDF:")

    if query:
        answer = qa.run(query)
        st.write("**Answer:**")
        st.write(answer)
