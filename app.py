import os
import openai
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.title("📄 RAG PDF Chatbot")

api_key = st.secrets["OPENAI_API_KEY"]

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

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

        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            if not docs:
                st.warning(f"No content found in {file.name}. Skipping this file.")
                continue

            docs_text = []
            for doc in docs:
                if isinstance(doc, str):
                    docs_text.append(doc)
                else:
                    docs_text.append(doc.page_content)

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs_split = splitter.split_documents(docs_text)

            if not docs_split:
                st.warning(f"No text chunks found after splitting {file.name}. Skipping this file.")
                continue

            all_docs.extend(docs_split)
        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")
            continue

    if not all_docs:
        st.error("No documents to process after splitting. Please check the uploaded PDFs.")
        return None

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        embedded_docs = embeddings.embed_documents(all_docs)
    except Exception as e:
        st.error(f"Error embedding documents: {e}")
        return None

    vectordb = FAISS.from_documents(embedded_docs, embeddings)
    return vectordb

if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None

if uploaded_files:
    with st.spinner("Processing your documents..."):
        vectordb = load_vectorstore(uploaded_files)
        
        if vectordb is None:
            st.error("Failed to process and embed the documents.")
        else:
            st.session_state.vectordb = vectordb
            st.success("Documents processed and ready for querying!")

    if st.session_state.vectordb:
        llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.vectordb.as_retriever())

        query = st.text_input("Ask a question about the uploaded PDFs:")

        if query:
            with st.spinner("Generating answer..."):
                try:
                    answer = qa.run(query)
                    st.write("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
else:
    st.warning("Please upload one or more PDFs to get started.")
