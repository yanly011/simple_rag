import os
from langchain_openai import ChatOpenAI

import config

class LLMFactory:
    @staticmethod
    def get_llm():
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        return ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            openai_api_key=api_key
        )