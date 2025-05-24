import os
from langchain_together import TogetherEmbeddings, ChatTogether
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

class LLMProvider:
    """Manages the initialization of the LLM for the RAG pipeline"""
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
    
    def get_llm(self) -> BaseChatModel:
        if "TOGETHER_API_KEY" not in os.environ:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        return ChatTogether(model=self.model_name)

class EmbeddingProvider:
    """Manages the initialization of the Embedding model for vectorizing documents"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
    
    def get_embedding_provider(self) -> Embeddings:
        if "TOGETHER_API_KEY" not in os.environ:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        return TogetherEmbeddings(model=self.model_name)