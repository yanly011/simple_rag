import os
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain.embeddings.base import Embeddings

import config

class VectorStore:
    def __init__(self, embedding_model: Embeddings):
        self.embedding_model = embedding_model
        self.db_type = config.VECTOR_DB_TYPE
        self.index_dir = config.INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document]):
        '''从文档创建向量存储'''
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        if self.db_type == "faiss":
            return FAISS.from_documents(documents, self.embedding_model)
        elif self.db_type == "chroma":
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=os.path.join(self.index_dir, "chroma_db")
            )
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    def save_vector_store(self, vector_store):
        '''保存向量存储到磁盘'''
        if self.db_type == "faiss":
            vector_store.save_local(os.path.join(self.index_dir, "faiss_index"))
            print(f"Vector store saved to {os.path.join(self.index_dir, 'faiss_index')}")
        elif self.db_type == "chroma":
            vector_store.persist()
            print(f"Vector store persisted to {os.path.join(self.index_dir, 'chroma_db')}")
    
    def load_vector_store(self):
        '''从磁盘加载向量存储'''
        if self.db_type == "faiss":
            index_path = os.path.join(self.index_dir, "faiss_index")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found at {index_path}")
            return FAISS.load_local(index_path, self.embedding_model)
        elif self.db_type == "chroma":
            db_path = os.path.join(self.index_dir, "chroma_db")
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Chroma DB not found at {db_path}")
            return Chroma(
                persist_directory=db_path,
                embedding_function=self.embedding_model
            )