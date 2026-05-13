from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT_TEMPLATE = """
You are a helpful and concise assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Returns the RAG prompt template.
    """
    return ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
