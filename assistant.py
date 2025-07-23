# assistant.py

import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


# Set your Gemini API key
api_key = "API_KEY"

def load_retriever(faiss_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 8})  # was 4


def get_answer(question, retriever):
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
You are a helpful and highly knowledgeable AI tutor specialized in Artificial Intelligence.
Use the following textbook context to give a clear, well-structured, and detailed answer.
Always provide definitions, examples, and explain concepts thoroughly in multiple paragraphs if needed.
Avoid vague or brief answers. Do not hallucinate. Only answer based on the context provided.

Question: {question}
Context: {context}
Answer:
        """
    )

    model = GoogleGenerativeAI(
        api_key=api_key,
        model="gemini-1.5-flash",
        temperature=0.4
    
    )

    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run({"question": question, "context": context})
    return response
