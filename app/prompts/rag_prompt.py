from langchain_core.prompts import ChatPromptTemplate

class RAGPromptManager:
    """
    Manager for RAG-related prompts.
    """
    def __init__(self):
        self.template = """
        You are a helpful assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.

        Question: {question} 

        Context: {context} 

        Answer:
        """

    def get_rag_prompt(self) -> ChatPromptTemplate:
        """
        Returns the ChatPromptTemplate for RAG.
        """
        return ChatPromptTemplate.from_template(self.template)

# Singleton instance
prompt_manager = RAGPromptManager()
