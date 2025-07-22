# embedding_generator.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def chunk_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def generate_and_store_embeddings(pdf_path, faiss_path):
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)

    documents = []
    for chunk in chunks:
        doc = Document(page_content=chunk.page_content, metadata=chunk.metadata)
        documents.append(doc)

    print(f"Total chunks: {len(documents)}")

    # Initialize HuggingFace model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(faiss_path)
    print(f"FAISS index saved to: {faiss_path}")

# Example usage
if __name__ == "__main__":
    generate_and_store_embeddings("AI.pdf", "faiss_index")
    print("Embeddings generated.")
