import os
from typing import List
from langchain.schema.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredExcelLoader
)

import config

class DocumentLoader:
    def __init__(self):
        self.pdf_dir = config.PDF_DIR
        self.text_dir = config.TEXT_DIR
        self.csv_dir = config.CSV_DIR
    
    def load_pdfs(self) -> List[Document]:
        '''load PDF docs'''
        if not os.path.exists(self.pdf_dir):
            return []
        
        loader = DirectoryLoader(
            self.pdf_dir, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        return loader.load()
    
    def load_text_files(self) -> List[Document]:
        '''load text files'''
        if not os.path.exists(self.text_dir):
            return []
        
        loader = DirectoryLoader(
            self.text_dir, 
            glob="**/*.txt", 
            loader_cls=TextLoader
        )
        return loader.load()
    
    def load_csv_files(self) -> List[Document]:
        '''load csv files'''
        if not os.path.exists(self.csv_dir):
            return []
        
        documents = []
        for filename in os.listdir(self.csv_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.csv_dir, filename)
                loader = CSVLoader(file_path)
                documents.extend(loader.load())
        return documents
    
    def load_all_documents(self) -> List[Document]:
        '''load all docs'''
        documents = []
        documents.extend(self.load_pdfs())
        documents.extend(self.load_text_files())
        documents.extend(self.load_csv_files())
        
        print(f"Loaded {len(documents)} documents")
        return documents