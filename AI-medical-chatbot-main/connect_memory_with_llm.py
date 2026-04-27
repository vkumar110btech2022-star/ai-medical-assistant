import os
import argparse

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

RAG_PROMPT_TEMPLATE = """You are a helpful medical reference assistant.
Use only the provided context to answer the question.
If the answer is not in the context, say you do not have enough information.

Context:
{context}

Question:
{input}
"""


def main():
    parser = argparse.ArgumentParser(description="Run a single RAG query against the medical FAISS index")
    parser.add_argument("--query", type=str, help="Question to ask. If omitted, interactive input is used.")
    args = parser.parse_args()

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it in your environment or .env file.")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=512,
        api_key=groq_api_key,
    )

    db_faiss_path = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)

    retrieval_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_prompt)
    rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 3}), combine_docs_chain)

    user_query = args.query or input("Write Query Here: ")
    response = rag_chain.invoke({"input": user_query})
    print("RESULT:", response["answer"])
    print("\nSOURCE DOCUMENTS:")
    for doc in response["context"]:
        print(f"- {doc.metadata} -> {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
