"""
rag_neon_pgvector_only.py

RAG + Knowledge Graph + OCR + Chat History
Vector store: PGVector (neonDB) ONLY — no FAISS fallback.

Notes:
- Set OPENAI_API_KEY in your environment or .env
- Ensure your environment has access to the neonDB host
- Install required packages listed below
- Developer sample path (local) is set from conversation history and will be loaded by the "Load sample file" button
"""

import streamlit as st
from dotenv import load_dotenv
import os
import io
import tempfile
from datetime import datetime
import json
import traceback

# Text extraction & OCR
from PyPDF2 import PdfReader
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    import pytesseract
except Exception:
    pytesseract = None

# NLP for KG extraction (optional; improves triple extraction)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# Knowledge graph (rdflib optional)
try:
    from rdflib import Graph, URIRef, Literal, Namespace
    from rdflib.namespace import RDF, RDFS
except Exception:
    Graph = None

# LangChain & vectorstore (PGVector required)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# PGVector (langchain vectorstore)
try:
    from langchain.vectorstores import PGVector
except Exception:
    PGVector = None

load_dotenv()

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# neonDB connection string supplied by you
NEON_PG_CONN = "postgresql://neondb_owner:npg_GxXIU8L0aDKA@ep-icy-heart-a4e2pl09-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
PGVECTOR_TABLE_NAME = "tamil6_english_vectors"
DEFAULT_MODEL = "gpt-4o"
# developer-specified local sample path (from conversation history)
SAMPLE_LOCAL_PATH = "/mnt/data/334a9e6b-3d15-47e2-a797-44c655979414.png"

# ---------- Helpers ----------
st.set_page_config(page_title="RAG (PGVector + neonDB only)", page_icon="🧠", layout="wide")
st.title("RAG + Knowledge Graph — PGVector (neonDB) ONLY")

def init_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []
    if "kg_graph" not in st.session_state:
        st.session_state["kg_graph"] = None
    if "documents" not in st.session_state:
        st.session_state["documents"] = []

init_state()

def get_llm():
    # deterministic (temperature 0) to reduce hallucination
    return ChatOpenAI(api_key=OPENAI_API_KEY, model_name=DEFAULT_MODEL, temperature=0.0)

def extract_text_pypdf(file_like):
    text = ""
    try:
        reader = PdfReader(file_like)
        for p in reader.pages:
            try:
                t = p.extract_text()
                if t:
                    text += t + "\n"
            except Exception:
                continue
    except Exception as e:
        st.warning(f"PyPDF2 failed: {e}")
    return text

def ocr_file(path_or_filelike):
    if pytesseract is None:
        st.error("pytesseract not installed; OCR unavailable.")
        return ""
    tmp_path = None
    try:
        # accept file-like
        if hasattr(path_or_filelike, "read"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(path_or_filelike.read())
            tmp.close()
            tmp_path = tmp.name
        else:
            tmp_path = str(path_or_filelike)

        if convert_from_path is None:
            # Try opening with PIL
            from PIL import Image
            img = Image.open(tmp_path)
            return pytesseract.image_to_string(img)

        pages = convert_from_path(tmp_path, dpi=200)
        txt = ""
        for im in pages:
            txt += pytesseract.image_to_string(im) + "\n"
        return txt
    except Exception as e:
        st.error(f"OCR error: {e}")
        return ""
    finally:
        try:
            if tmp_path and tmp_path.startswith(tempfile.gettempdir()):
                os.remove(tmp_path)
        except Exception:
            pass

def split_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def docs_from_chunks(chunks):
    docs = []
    for i, c in enumerate(chunks):
        docs.append(Document(page_content=c, metadata={"chunk_id": f"doc_{i+1}"}))
    return docs

def extract_triples_from_text(text):
    triples = []
    if nlp:
        doc = nlp(text)
        for sent in doc.sents:
            subj = None
            verb = None
            obj = None
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and subj is None:
                    subj = token.text
                if token.pos_ == "VERB" and verb is None:
                    verb = token.lemma_
                if token.dep_ in ("dobj", "pobj") and obj is None:
                    obj = token.text
            if subj and verb and obj:
                triples.append((subj.strip(), verb.strip(), obj.strip()))
    else:
        # naive fallback heuristic
        import re
        sents = re.split(r'(?<=[.!?])\s+', text)
        for s in sents:
            tokens = s.split()
            if len(tokens) >= 3:
                subj = tokens[0].strip().strip(',:;')
                verb = tokens[1].strip().strip(',:;')
                obj = " ".join(tokens[2:6]).strip().strip(',:;')
                triples.append((subj, verb, obj))
    return triples

def build_rdflib_graph(triples, base_uri="http://tn6-english/"):
    if Graph is None:
        return triples  # return raw triples if rdflib not available
    g = Graph()
    NS = Namespace(base_uri)
    for i, (s, p, o) in enumerate(triples):
        suri = URIRef(NS + "s" + str(abs(hash(s)) % (10**8)))
        p_uri = URIRef(NS + "p_" + p.replace(" ", "_")[:50])
        o_lit = Literal(o)
        g.add((suri, p_uri, o_lit))
    return g

# PGVector-only build (no FAISS fallback)
def build_vector_store_pg(docs, embeddings, conn_str, table_name):
    if PGVector is None:
        raise RuntimeError("langchain.vectorstores.PGVector is not available in this environment. Install/upgrade langchain to a version that provides PGVector.")
    # Try common parameter names across versions
    exceptions = []
    try:
        vs = PGVector.from_documents(documents=docs, embedding=embeddings, collection_name=table_name, connection_string=conn_str)
        return vs
    except Exception as e:
        exceptions.append(e)
    try:
        vs = PGVector.from_documents(documents=docs, embedding=embeddings, table_name=table_name, connection_string=conn_str)
        return vs
    except Exception as e:
        exceptions.append(e)
    try:
        # older versions might accept different kw
        vs = PGVector.from_documents(documents=docs, embedding=embeddings, connection_string=conn_str, table_name=table_name)
        return vs
    except Exception as e:
        exceptions.append(e)
    # If we reached here, PGVector creation failed — raise with combined errors for easier debugging
    raise RuntimeError(f"PGVector.from_documents failed with multiple attempts. Errors: {exceptions}")

def create_qa_chain(llm, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k":6})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

def kg_keyword_matches(triples_with_meta, keywords):
    matches = []
    for (s,p,o,cid) in triples_with_meta:
        joined = f"{s} {p} {o}".lower()
        if any(k.lower() in joined for k in keywords[:8]):
            matches.append((s,p,o,cid))
    return matches

# ---------- Streamlit UI ----------
with st.sidebar:
    st.markdown("### Settings")
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
    use_ocr = st.checkbox("Use OCR fallback", value=True)
    st.markdown("---")
    st.markdown("**Developer sample loader**")
    if st.button("Load sample file (dev)"):
        sample_path = SAMPLE_LOCAL_PATH
        if os.path.exists(sample_path):
            st.success(f"Sample path found: {sample_path}")
            uploaded_file_bytes = io.BytesIO(open(sample_path, "rb").read())
            st.session_state["_dev_file"] = uploaded_file_bytes
        else:
            st.error(f"Sample not found: {sample_path}")

st.header("Upload a Tamil Nadu Class 6 English chapter (PDF/image)")
uploaded = st.file_uploader("Upload PDF/image", type=["pdf", "png", "jpg", "jpeg", "tiff"])
if "_dev_file" in st.session_state and uploaded is None:
    uploaded = st.session_state["_dev_file"]

if uploaded is not None:
    st.info("Processing file — extracting text...")
    if hasattr(uploaded, "seek"):
        uploaded.seek(0)

    text = extract_text_pypdf(uploaded)
    if not text.strip() and use_ocr:
        st.info("No selectable text — running OCR fallback")
        uploaded.seek(0)
        text = ocr_file(uploaded)

    if not text.strip():
        st.error("No text extracted from file. Try another file or enable OCR.")
    else:
        st.success("Text extracted. Building chunks...")
        chunks = split_text(text, chunk_size=int(chunk_size), overlap=int(chunk_overlap))
        st.session_state["chunks"] = chunks
        st.write(f"Split into {len(chunks)} chunks.")

        # Build docs with provenance metadata
        docs = docs_from_chunks(chunks)
        st.session_state["documents"] = docs

        # Build KG triples with provenance (chunk id)
        st.info("Extracting deterministic KG triples (for provenance grounding)...")
        triples_with_meta = []
        for i, c in enumerate(chunks):
            triples = extract_triples_from_text(c)
            for (s,p,o) in triples:
                triples_with_meta.append((s,p,o,f"doc_{i+1}"))
        if not triples_with_meta:
            st.warning("No triples extracted automatically. Consider enabling spaCy or manual KG creation.")
            st.session_state["kg_graph"] = {"triples_with_meta": []}
        else:
            st.session_state["kg_graph"] = {"triples_with_meta": triples_with_meta}
            st.success(f"KG prepared with ~{len(triples_with_meta)} triples (provenance preserved).")

        # Build PGVector vector store ONLY
        st.info("Building embeddings + storing into PGVector (neonDB)...")
        embeddings = OpenAIEmbeddings()
        try:
            vs = build_vector_store_pg(docs, embeddings, NEON_PG_CONN, PGVECTOR_TABLE_NAME)
            st.session_state["vector_store"] = vs
            st.success("PGVector index created/loaded in neonDB.")
        except Exception as e:
            # Fail loudly and show rich error to help debugging (no fallback)
            st.error("PGVector creation failed. This app is configured to use PGVector (neonDB) only — no FAISS fallback.")
            st.error(str(e))
            st.write(traceback.format_exc())
            st.stop()

# Query UI
st.markdown("---")
st.subheader("Ask questions (answers must be supported by evidence)")

query = st.text_input("Your question about the chapter:")
if st.button("Ask"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        # store user turn
        st.session_state["chat_history"].append({"role":"user","text":query,"ts":datetime.utcnow().isoformat()})
        if not st.session_state.get("vector_store"):
            st.error("No vector store available. Index a document first.")
        else:
            llm = get_llm()
            vs = st.session_state["vector_store"]
            retriever = vs.as_retriever(search_kwargs={"k":6})
            try:
                retrieved_docs = retriever.get_relevant_documents(query)
            except Exception:
                # some langchain versions name method get_relevant_documents differently; try a second way
                retrieved_docs = retriever.get_relevant_documents(query) if hasattr(retriever, "get_relevant_documents") else retriever.retrieve(query)

            retrieved_texts = [(d.metadata.get("chunk_id","?"), d.page_content[:800]) for d in retrieved_docs]

            # KG keyword matches
            keywords = [w for w in query.split() if len(w) > 2]
            kg_matches = []
            if st.session_state.get("kg_graph") and st.session_state["kg_graph"].get("triples_with_meta"):
                kg_matches = kg_keyword_matches(st.session_state["kg_graph"]["triples_with_meta"], keywords)

            # Build strict instruction + evidence block
            evidence_text = ""
            if retrieved_texts:
                evidence_text += "Document evidence (chunks):\n"
                for cid, doc in retrieved_texts:
                    evidence_text += f"- [{cid}] {doc}\n"
            if kg_matches:
                evidence_text += "\nKnowledge Graph facts (triples with provenance):\n"
                for (s,p,o,cid) in kg_matches:
                    evidence_text += f"- ({s} , {p} , {o})  [from {cid}]\n"

            system_instruction = (
                "You are an evidence-first assistant. ANSWER THE QUESTION ONLY IF the answer can be directly "
                "supported by the EVIDENCE provided below (document chunks or KG facts). Cite the evidence you used "
                "with bracketed provenance like [doc_3] or [KG:doc_4]. If you cannot fully support the answer using the "
                "provided evidence, explicitly reply: 'I cannot answer that from the provided text.' Do not invent facts."
            )

            prompt = (
                system_instruction + "\n\n" +
                "EVIDENCE:\n" + (evidence_text or "No evidence found.") + "\n\n" +
                "QUESTION:\n" + query + "\n\n" +
                "INSTRUCTIONS: Answer concisely, list citations in bracketed form, or refuse if unsupported."
            )

            # Call LLM: use ChatOpenAI/LLM wrapper (LangChain ChatOpenAI)
            try:
                # LangChain ChatOpenAI API varies by version. We'll attempt a common pattern:
                llm_client = get_llm()
                # Many LangChain ChatOpenAI objects accept .generate() or .predict(). We'll try .generate and then fallback.
                try:
                    resp = llm_client.generate([{"role":"system","content":system_instruction},{"role":"user","content":prompt}])
                    # extract text robustly
                    ans_text = ""
                    try:
                        # .generations is common
                        gens = resp.generations[0]
                        if isinstance(gens, list):
                            ans_text = gens[0].text
                        else:
                            # some versions return object with .text
                            ans_text = getattr(gens, "text", str(gens))
                    except Exception:
                        ans_text = str(resp)
                except Exception:
                    # fallback to .predict (older) or RetrievalQA
                    try:
                        ans_text = llm_client.predict(prompt)
                    except Exception:
                        # final fallback: RetrievalQA (still uses LLM but less direct control)
                        qa_chain = RetrievalQA.from_chain_type(llm=get_llm(), chain_type="stuff", retriever=vs.as_retriever())
                        ans_text = qa_chain.run(query)
            except Exception as e:
                ans_text = f"LLM interaction failed: {e}"

            # record assistant turn and display
            st.session_state["chat_history"].append({"role":"assistant","text":ans_text,"ts":datetime.utcnow().isoformat()})
            st.markdown("**Answer (evidence-grounded):**")
            st.write(ans_text)

            # Show retrieved evidence for transparency
            st.markdown("**Top retrieved chunks:**")
            for cid, doc in retrieved_texts[:6]:
                st.write(f"- [{cid}] {doc[:400]}...")
            if kg_matches:
                st.markdown("**KG matches:**")
                for s,p,o,cid in kg_matches[:10]:
                    st.write(f"- ({s} , {p} , {o}) — from {cid}")

# Chat history sidebar
st.sidebar.markdown("---")
st.sidebar.header("Chat history")
if st.session_state["chat_history"]:
    for entry in reversed(st.session_state["chat_history"]):
        who = "You" if entry["role"]=="user" else "Assistant"
        ts = entry.get("ts","")[:19]
        st.sidebar.markdown(f"**{who}** ({ts}):\n\n{entry['text']}")
    if st.sidebar.button("Download chat history"):
        b = io.BytesIO()
        b.write(json.dumps(st.session_state["chat_history"], indent=2).encode("utf-8"))
        b.seek(0)
        st.sidebar.download_button("Download JSON", data=b, file_name=f"chat_history_{datetime.utcnow().date()}.json")
else:
    st.sidebar.info("No conversation yet.")
