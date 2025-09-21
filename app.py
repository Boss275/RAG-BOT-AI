import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="📄 RAG PDF Chatbot", layout="wide")
st.title("📄 RAG PDF Chatbot (Local Hugging Face)")

hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]

uploaded_files = st.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

@st.cache_resource
def load_vectorstore(files):
    """Load PDFs, split into chunks, embed, and create FAISS vectorstore."""
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

            splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
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
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=False,
        use_auth_token=hf_token
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Use the context below to answer the question in full sentences and provide complete explanations.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:""",
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.vectordb.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    query = st.text_input("Ask a question about the uploaded PDFs:")

    if query:
        with st.spinner("Generating answer..."):
            try:
                answer = qa.run(query)
                st.markdown("**Answer:**")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
else:
    st.warning("Please upload one or more PDFs to get started.")
