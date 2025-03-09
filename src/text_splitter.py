from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import config

class TextSplitter:
    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        '''split docs to chunks'''
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def get_chunk_info(self, chunks: List[Document]) -> List[dict]:
        '''Get info of chunks'''
        chunk_info = []
        for i, chunk in enumerate(chunks):
            info = {
                "chunk_id": i,
                "source": chunk.metadata.get('source', 'Unknown'),
                "length": len(chunk.page_content),
                "content_preview": chunk.page_content[:100] + "..."
            }
            chunk_info.append(info)
        return chunk_info