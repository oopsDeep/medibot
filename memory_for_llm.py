import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#  CONFIG 
DATA_PATH = "data/"  # folder containing all PDFs
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#  FUNCTIONS 
def load_pdf_with_metadata(file_path):
    """Load a single PDF and add book name + path metadata."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    book_name = Path(file_path).stem
    for doc in documents:
        doc.metadata["book"] = book_name
        doc.metadata["source"] = str(Path(file_path).resolve())
    return documents

def split_into_chunks(documents, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def load_or_create_db(embedding_model):
    if os.path.exists(DB_FAISS_PATH):
        print("Loading existing FAISS DB...")
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("No existing DB found. Creating a new one...")
        return None

# MAIN
if __name__ == "__main__":
    embedding_model = get_embedding_model()
    db = load_or_create_db(embedding_model)

    # Track already stored PDFs to skip them
    existing_sources = set()
    if db:
        existing_sources = {meta.metadata.get("source") for meta in db.docstore._dict.values()}

    added_books = 0
    for pdf_file in Path(DATA_PATH).glob("*.pdf"):
        pdf_path = str(pdf_file.resolve())
        if pdf_path in existing_sources:
            print(f"⏩ Skipping (already in DB): {pdf_file.name}")
            continue

        print(f"📖 Adding: {pdf_file.name}")
        documents = load_pdf_with_metadata(pdf_path)
        chunks = split_into_chunks(documents)

        if db:
            db.add_documents(chunks)
        else:
            db = FAISS.from_documents(chunks, embedding_model)

        added_books += 1

    # Save updated DB
    if db:
        db.save_local(DB_FAISS_PATH)

    print(f" Added {added_books} new book(s).")
