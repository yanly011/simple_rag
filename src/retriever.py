from langchain.vectorstores.base import VectorStore

import config

class DocumentRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.top_k = config.TOP_K
    
    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
    
    def similar_search(self, query: str, top_k=None):
        '''Search similar documents'''
        k = top_k or self.top_k
        return self.vector_store.similarity_search(query, k=k)