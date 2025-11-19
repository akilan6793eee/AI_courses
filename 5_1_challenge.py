# streamlit_pdf_rag.py
import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from pypdf import PdfReader

# LangChain modern imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config("PDF RAG (OpenAI + FAISS)", layout="wide")

st.title("PDF RAG — Upload PDFs, Index, and Ask (OpenAI + FAISS)")
st.markdown(
    "Upload PDF files, build an index (FAISS with OpenAI embeddings), then ask questions. "
    "You can provide an OpenAI API key in the sidebar or set `OPENAI_API_KEY` in your environment."
)

# ------------- Sidebar settings -------------
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("OpenAI API key (optional)", type="password", help="Enter to use for this session (not saved). Leave blank to use environment variable.")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    model_choice = st.selectbox("LLM model", options=["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-instruct"], index=0)
    embed_model = st.selectbox("Embedding model", options=["text-embedding-3-small", "text-embedding-3-large"], index=0)
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=2000, value=800, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=500, value=100, step=25)
    top_k = st.number_input("Top K retrieved", min_value=1, max_value=10, value=4)
    persist = st.checkbox("Persist FAISS to disk (./chroma_faiss)", value=False)
    st.markdown("---")
    st.write("Note: Don't paste sensitive API keys here if you will share the app.")

# ------------- File upload UI -------------
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
index_button = st.button("Build / Rebuild Index")

# ------------- Helpers -------------
@st.cache_data(show_spinner=False)
def save_uploaded_files(files: List, tmp_dir: str) -> List[str]:
    paths = []
    for f in files:
        out = Path(tmp_dir) / f.name
        with open(out, "wb") as wf:
            wf.write(f.getbuffer())
        paths.append(str(out))
    return paths

def load_pdf_pages(path: str):
    """
    Returns list of langchain Document-like objects.
    We'll attempt to use PyPDFLoader if available; otherwise fallback to pypdf extraction.
    """
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        return docs
    except Exception:
        # fallback using pypdf -> create Documents manually
        reader = PdfReader(path)
        from langchain.schema import Document
        pages = []
        for i, p in enumerate(reader.pages):
            txt = p.extract_text() or ""
            pages.append(Document(page_content=txt, metadata={"source": f"{Path(path).name}#page={i+1}"}))
        return pages

@st.cache_resource(ttl=60*60)
def make_embeddings_and_index(texts_docs, embed_model_name: str, persist_dir: str = None):
    """
    Given a list of Document objects (chunks), create embeddings and FAISS DB.
    """
    emb = OpenAIEmbeddings(model=embed_model_name)
    if persist_dir:
        persist_dir = str(Path(persist_dir).resolve())
        db = FAISS.from_documents(texts_docs, emb, persist_directory=persist_dir)
        try:
            db.persist()
        except Exception:
            pass
    else:
        db = FAISS.from_documents(texts_docs, emb)
    return db

def split_docs_into_chunks(docs, chunk_size, chunk_overlap):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# ------------- Index building -------------
index_ready = False
db = None

if uploaded_files:
    tmpdir = tempfile.mkdtemp(prefix="pdf_rag_")
    paths = save_uploaded_files(uploaded_files, tmpdir)
    st.success(f"Saved {len(paths)} uploaded files to {tmpdir}")

    # Load pages from all PDFs
    all_pages = []
    with st.spinner("Loading PDF pages..."):
        for p in paths:
            pages = load_pdf_pages(p)
            all_pages.extend(pages)
    st.info(f"Loaded {len(all_pages)} pages from uploaded PDFs")

    # Split into chunks
    with st.spinner("Splitting into chunks..."):
        chunks = split_docs_into_chunks(all_pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    st.info(f"Created {len(chunks)} chunks")

    if st.button("Show sample chunk"):
        if chunks:
            st.code(chunks[0].page_content[:1000])
        else:
            st.warning("No chunks available")

    # Build index when user presses button
    if index_button:
        if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
            st.error("OpenAI API key not found. Set it in the sidebar or as environment variable OPENAI_API_KEY.")
        else:
            st.info("Building FAISS index (this may take a moment)...")
            persist_dir = "./chroma_faiss" if persist else None
            try:
                db = make_embeddings_and_index(chunks, embed_model, persist_dir)
                index_ready = True
                st.success("Index built successfully.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

# ------------- Query UI -------------
if (db is not None) or index_ready:
    if db is None:
        # try to load from persist_dir if user selected persistence previously
        try:
            db = FAISS.load_local("./chroma_faiss", OpenAIEmbeddings(model=embed_model))
            st.success("Loaded persisted FAISS from ./chroma_faiss")
        except Exception:
            st.warning("No in-memory DB found; rebuild index or re-upload files.")
    if db:
        retriever = db.as_retriever(search_kwargs={"k": int(top_k)})
        st.markdown("---")
        st.subheader("Ask a question")
        question = st.text_input("Enter your question here")
        ask_btn = st.button("Ask")

        if ask_btn and question:
            with st.spinner("Retrieving and generating answer..."):
                # show retrieved docs
                retrieved = retriever.get_relevant_documents(question)
                st.write(f"Top {len(retrieved)} retrieved chunks:")
                for i, d in enumerate(retrieved, 1):
                    src = d.metadata.get("source") if hasattr(d, "metadata") else None
                    st.markdown(f"**Result #{i}** — source: `{src}`" if src else f"**Result #{i}**")
                    st.write(d.page_content[:1000])

                # Build prompt and call LLM
                prompt = ChatPromptTemplate.from_template(
                    "You are a helpful assistant. Use the context below to answer the question. "
                    "If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
                )
                llm = ChatOpenAI(model=model_choice, temperature=0)
                def format_docs(docs):
                    return "\n\n---\n\n".join([d.page_content for d in docs])
                rag = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm)

                try:
                    out = rag.invoke(question)
                    answer = getattr(out, "content", str(out))
                    st.markdown("### LLM Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"LLM call failed: {e}")

# ------------- Footer -------------
st.markdown("---")
st.caption("Built with LangChain (core + community), OpenAI, and FAISS. Be careful with sensitive docs.")
