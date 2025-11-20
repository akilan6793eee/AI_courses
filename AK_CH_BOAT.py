"""
modular_rag_app.py

Features:
- Modular RAG pipeline (functions for loading, OCR, splitting, embedding, retrieval)
- OCR support for scanned PDFs / images (PyPDF2 then pdf2image + pytesseract fallback)
- Chat history stored in st.session_state
- Option to persist/load FAISS index
- "Load sample file" button uses the local path provided in conversation history:
    /mnt/data/334a9e6b-3d15-47e2-a797-44c655979414.png
"""

import streamlit as st
from dotenv import load_dotenv
import os
from datetime import datetime
import io
import tempfile
import json

# PDF/text processing
from PyPDF2 import PdfReader
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    import pytesseract
except Exception:
    pytesseract = None

# LangChain & vectorstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# NOTE: install dependencies:
# pip install streamlit PyPDF2 pdf2image pytesseract langchain faiss-cpu python-dotenv openai

load_dotenv()

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o"  # Feel free to change
SAMPLE_LOCAL_PATH = "/mnt/data/334a9e6b-3d15-47e2-a797-44c655979414.png"  # from conversation history (developer instruction)

# ---------- Utilities / Modules ----------

def init_session_state():
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # list of {"role":"user"/"assistant","text": "...", "ts": ...}
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = None

def get_llm():
    # Using LangChain ChatOpenAI wrapper
    return ChatOpenAI(api_key=OPENAI_API_KEY, model_name=DEFAULT_MODEL, temperature=0.2)

def extract_text_from_pdf_filelike(file_like) -> str:
    """
    Try PyPDF2 first (text-based). If no text, return empty string.
    """
    try:
        reader = PdfReader(file_like)
        text = ""
        for page in reader.pages:
            try:
                ptext = page.extract_text()
                if ptext:
                    text += ptext + "\n"
            except Exception:
                continue
        return text
    except Exception as e:
        st.warning(f"PyPDF2 extraction failed: {e}")
        return ""

def ocr_pdf_or_image(file_path_or_filelike) -> str:
    """
    Use pdf2image + pytesseract to OCR scanned PDFs or images.
    Accepts a path or a file-like object.
    """
    if pytesseract is None:
        st.error("pytesseract not available. Install with `pip install pytesseract` and ensure tesseract binary is installed.")
        return ""
    # If file-like (BytesIO), write to temp file for pdf2image
    tmp_path = None
    try:
        if hasattr(file_path_or_filelike, "read"):
            # write bytes to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(file_path_or_filelike.read())
            tmp.close()
            tmp_path = tmp.name
        else:
            tmp_path = str(file_path_or_filelike)

        # If pdf2image not available, try image OCR (pytesseract) if input looks like image
        if convert_from_path is None:
            # try to open as image (PIL)
            try:
                from PIL import Image
                img = Image.open(tmp_path)
                return pytesseract.image_to_string(img)
            except Exception as e:
                st.error("pdf2image missing and failed to open file as image: " + str(e))
                return ""

        pages = convert_from_path(tmp_path, dpi=200)
        ocr_text = ""
        for page_img in pages:
            ocr_text += pytesseract.image_to_string(page_img) + "\n"
        return ocr_text
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return ""
    finally:
        # cleanup if we created a temp file
        try:
            if tmp_path and tmp_path.startswith(tempfile.gettempdir()):
                os.remove(tmp_path)
        except Exception:
            pass

def split_text_to_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def build_vector_store(chunks, embeddings):
    # Create FAISS vectorstore from chunk list
    return FAISS.from_texts(chunks, embedding=embeddings)

def save_faiss(store: FAISS, path_dir: str):
    os.makedirs(path_dir, exist_ok=True)
    store.save_local(path_dir)
    st.success(f"Saved FAISS index to {path_dir}")

def load_faiss(path_dir: str, embeddings):
    if not os.path.exists(path_dir):
        st.error("FAISS path does not exist.")
        return None
    return FAISS.load_local(path_dir, embeddings)

def create_qa_chain(llm, vector_store):
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

def append_chat(role, text):
    st.session_state["chat_history"].append({"role": role, "text": text, "ts": datetime.utcnow().isoformat()})

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Modular RAG + OCR + Chat History", page_icon="📚", layout="wide")
st.title("📚 Modular RAG: Ask PDFs (with OCR + Chat History)")

init_session_state()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("Upload / Load")
    uploaded_file = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg", "tiff"])
    # Developer-provided local path sample button (useful for automation)
    if st.button("🔎 Load sample file from local path (dev)"):
        # developer instruction: send the local path as the url of the file
        # We'll read the local file and treat as uploaded file
        try:
            sample_path = SAMPLE_LOCAL_PATH
            if os.path.exists(sample_path):
                st.success(f"Loaded local sample: {sample_path}")
                # read bytes and set a BytesIO to uploaded_file-like variable
                with open(sample_path, "rb") as f:
                    uploaded_bytes = io.BytesIO(f.read())
                    uploaded_file = uploaded_bytes  # override
            else:
                st.error(f"Sample path not found: {sample_path}")
        except Exception as e:
            st.error(f"Failed to load sample path: {e}")

    st.markdown("---")
    st.header("RAG Options")
    use_ocr_fallback = st.checkbox("Use OCR fallback (pdf2image + pytesseract) if no text extracted", value=True)
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=5000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
    persist_faiss = st.checkbox("Persist FAISS index to disk", value=False)
    faiss_dir = st.text_input("FAISS directory (when persisting)", value="./faiss_index")

    st.markdown("---")
    if st.button("Clear chat history"):
        st.session_state["chat_history"] = []
        st.success("Cleared chat history")

with col_right:
    st.header("Pipeline & Query")
    if uploaded_file is not None:
        st.info("Processing uploaded file...")

        # 1) Try PyPDF2 text extraction if PDF-like
        raw_text = ""
        try:
            # uploaded_file might be a stream (UploadedFile) or BytesIO (from sample)
            raw_text = extract_text_from_pdf_filelike(uploaded_file)
        except Exception as e:
            st.warning(f"Primary text extraction failed: {e}")

        # 2) If no text and OCR allowed, run OCR
        if (not raw_text.strip()) and use_ocr_fallback:
            st.info("No selectable text found. Running OCR fallback...")
            # If uploaded_file is a stream, we need to ensure we pass a path or reset bytes
            # We'll write it to a temp file
            try:
                if hasattr(uploaded_file, "read"):
                    # ensure pointer at start
                    uploaded_file.seek(0)
                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmpf.write(uploaded_file.read())
                    tmpf.close()
                    ocr_text = ocr_pdf_or_image(tmpf.name)
                    # cleanup temp
                    try:
                        os.remove(tmpf.name)
                    except Exception:
                        pass
                else:
                    # uploaded_file is a path-like
                    ocr_text = ocr_pdf_or_image(uploaded_file)
                raw_text = (raw_text or "") + "\n" + (ocr_text or "")
            except Exception as e:
                st.error(f"OCR fallback failed: {e}")
        # 3) If still empty, show error
        if not raw_text.strip():
            st.error("Could not extract any text. Try a different file, enable OCR, or run OCR externally.")
        else:
            st.success("Text extracted from file.")
            # Split to chunks
            chunks = split_text_to_chunks(raw_text, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
            st.session_state["chunks"] = chunks
            st.write(f"Split into {len(chunks)} chunks.")

            # Build or load embeddings + vector store
            st.write("Building embeddings & vector store...")
            embeddings = OpenAIEmbeddings()  # will use OPENAI_API_KEY from env
            if persist_faiss and os.path.exists(faiss_dir):
                try:
                    st.write("Loading existing FAISS index from disk...")
                    vector_store = load_faiss(faiss_dir, embeddings)
                    st.session_state["vector_store"] = vector_store
                    st.success("Loaded FAISS index.")
                except Exception as e:
                    st.warning(f"Failed to load FAISS index: {e}. Will create a new one.")
                    vector_store = None
            else:
                vector_store = None

            if vector_store is None:
                vector_store = build_vector_store(chunks, embeddings)
                st.session_state["vector_store"] = vector_store
                st.success("Built FAISS vector store in-memory.")
                if persist_faiss:
                    try:
                        save_faiss(vector_store, faiss_dir)
                    except Exception as e:
                        st.warning(f"Failed to save FAISS index: {e}")

            # Create QA chain
            llm = get_llm()
            qa_chain = create_qa_chain(llm, st.session_state["vector_store"])

            # Query UI: Chat style
            query = st.text_input("Ask a question about the document and press Enter")
            if st.button("Ask"):
                if not query.strip():
                    st.warning("Type a question first.")
                else:
                    append_chat("user", query)
                    with st.spinner("Thinking..."):
                        try:
                            # RetrievalQA.run returns the answer text
                            answer = qa_chain.run(query)
                        except Exception as e:
                            answer = f"Error from QA chain: {e}"
                    append_chat("assistant", answer)
                    st.markdown("**Answer:**")
                    st.write(answer)

    else:
        st.info("No file uploaded yet. Upload a PDF or image to get started, or use the sample loader on the left.")

# ---------- Chat History Display ----------
st.sidebar.markdown("---")
st.sidebar.header("Chat History")
if st.session_state["chat_history"]:
    for entry in reversed(st.session_state["chat_history"]):  # show newest first
        role = entry["role"]
        ts = entry.get("ts", "")
        if role == "user":
            st.sidebar.markdown(f"**You** ({ts[:19]}):\n\n{entry['text']}")
        else:
            st.sidebar.markdown(f"**Assistant** ({ts[:19]}):\n\n{entry['text']}")
    if st.sidebar.button("Download chat history"):
        b = io.BytesIO()
        b.write(json.dumps(st.session_state["chat_history"], indent=2).encode("utf-8"))
        b.seek(0)
        st.sidebar.download_button("Download JSON", data=b, file_name=f"chat_history_{datetime.now().date()}.json")
else:
    st.sidebar.info("No chat history yet. Ask a question after uploading a file.")

# ---------- Helpful debug / maintenance ----------
st.markdown("---")
with st.expander("Debug / Internal State (for power users)"):
    st.write("Chunks loaded:", len(st.session_state.get("chunks") or []))
    st.write("Vector store in session:", bool(st.session_state.get("vector_store")))
    st.write("Chat history length:", len(st.session_state.get("chat_history")))
