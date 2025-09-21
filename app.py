'''import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.title("📄 RAG PDF Chatbot (Hugging Face)")

hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

@st.cache_resource
def load_vectorstore(files):
    all_docs = []
    temp_dir = "/tmp/pdf_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            if not docs:
                st.warning(f"No content found in {file.name}. Skipping.")
                continue

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs_split = splitter.split_documents(docs)
            if not docs_split:
                st.warning(f"No text chunks found after splitting {file.name}. Skipping.")
                continue

            all_docs.extend(docs_split)
        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")
            continue

    if not all_docs:
        st.error("No documents to process after splitting PDFs.")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            cache_folder="/tmp/hf_embeddings",
        )
        vectordb = FAISS.from_documents(all_docs, embeddings)
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

    return vectordb

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if uploaded_files:
    with st.spinner("Processing your documents..."):
        vectordb = load_vectorstore(uploaded_files)
        if vectordb is None:
            st.error("Failed to process and embed documents.")
        else:
            st.session_state.vectordb = vectordb
            st.success("✅ Documents processed and ready!")

    if st.session_state.vectordb:
        pipe = pipeline(
            "text-generation",
            model="tiiuae/falcon-7b-instruct",
            tokenizer="tiiuae/falcon-7b-instruct",
            device=-1,  
            max_length=512,
            temperature=0,
            use_auth_token=hf_token
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectordb.as_retriever()
        )

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
    st.warning("Please upload one or more PDFs to get started.")'''










import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

st.title("📄 RAG PDF Chatbot (Local Hugging Face)")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

@st.cache_resource
def load_vectorstore(files):
    all_docs = []
    temp_dir = "/tmp/pdf_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            if not docs:
                st.warning(f"No content found in {file.name}. Skipping.")
                continue

            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs_split = splitter.split_documents(docs)
            if not docs_split:
                st.warning(f"No text chunks found after splitting {file.name}. Skipping.")
                continue

            all_docs.extend(docs_split)
        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")
            continue

    if not all_docs:
        st.error("No documents to process after splitting PDFs.")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            cache_folder="/tmp/hf_embeddings",
        )
        vectordb = FAISS.from_documents(all_docs, embeddings)
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

    return vectordb

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if uploaded_files:
    with st.spinner("Processing your PDFs..."):
        vectordb = load_vectorstore(uploaded_files)
        if vectordb is None:
            st.error("Failed to process and embed documents.")
        else:
            st.session_state.vectordb = vectordb
            st.success("✅ Documents processed and ready!")

    if st.session_state.vectordb:
        # Use a lightweight local model for generation
        pipe = pipeline(
            "text-generation",
            model="google/flan-t5-small",
            tokenizer="google/flan-t5-small",
            device=-1,  # CPU
            max_length=256,
            temperature=0
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        from langchain.prompts import PromptTemplate

        prompt_template = """
You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectordb.as_retriever(search_kwargs={"k": 5}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT},
        )

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

