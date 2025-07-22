# main.py

import streamlit as st
from assistant import load_retriever, get_answer

st.set_page_config(page_title="Ask AI Book (RAG App)", layout="wide")
st.title("📘 Ask Questions from 'AI: A Modern Approach'")

# Load FAISS retriever once
@st.cache_resource
def load_vectorstore():
    return load_retriever("faiss_index")  # FAISS path must match embedding_generator.py

retriever = load_vectorstore()

# Input box
question = st.text_input("Ask a question based on the AI book:", placeholder="e.g., What is the Turing Test?")

# Handle response
if question:
    with st.spinner("Thinking..."):
        try:
            answer = get_answer(question, retriever)
            st.markdown("### 🧠 Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")
