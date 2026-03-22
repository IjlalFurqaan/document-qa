"""
RAG Chain Module.

Builds the Retrieval-Augmented Generation chain using LangChain,
Vertex AI (Gemini), and ChromaDB for grounded document Q&A.
"""

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import (
    GCP_PROJECT,
    GCP_LOCATION,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    QA_PROMPT_TEMPLATE,
    RETRIEVAL_TOP_K,
)
from ingest import get_vectorstore


def _format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[Document {i} — Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain():
    """
    Build and return the RAG chain.

    The chain performs:
    1. Retrieve relevant documents from ChromaDB via semantic search
    2. Format retrieved docs as context
    3. Pass context + question to Gemini via Vertex AI
    4. Return the generated answer

    Returns:
        Tuple of (chain, retriever) for invocation and source retrieval.
    """
    # Initialize the vector store retriever
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_TOP_K},
    )

    # Initialize the LLM
    llm = ChatVertexAI(
        model_name=LLM_MODEL,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
    )

    # Build the prompt
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Build the LCEL chain
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(query: str) -> dict:
    """
    Ask a question and get an answer with source documents.

    Args:
        query: The natural language question.

    Returns:
        Dictionary with 'answer' and 'sources' keys.
    """
    chain, retriever = build_rag_chain()

    # Get the answer
    answer = chain.invoke(query)

    # Get source documents for citation
    source_docs = retriever.invoke(query)

    sources = []
    for doc in source_docs:
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "content_preview": doc.page_content[:200] + "..."
            if len(doc.page_content) > 200
            else doc.page_content,
        })

    return {
        "answer": answer,
        "sources": sources,
    }
