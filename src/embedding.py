from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import config
import os

class EmbeddingFactory:
    @staticmethod
    def get_embeddings():
        if config.EMBEDDING_MODEL == "openai":
            return OpenAIEmbeddings(
                model=config.OPENAI_EMBEDDING_MODEL,
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )
        elif config.EMBEDDING_MODEL == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=config.HF_EMBEDDING_MODEL
            )
        else:
            raise ValueError(f"Unsupported embedding model: {config.EMBEDDING_MODEL}")