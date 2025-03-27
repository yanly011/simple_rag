import os
import argparse
from typing import List, Dict, Any
import dotenv

from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.embedding import EmbeddingFactory
from src.vector_store import VectorStore
from src.retriever import DocumentRetriever
from src.llm import LLMFactory
from src.rag_chain import RAGChain
from src.evaluation import RAGEvaluator

def build_index():
    print("Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_all_documents()
    
    print("Splitting documents...")
    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)
    
    print("Creating embeddings and vector store...")
    embedding_model = EmbeddingFactory.get_embeddings()
    vector_store_handler = VectorStore(embedding_model)
    vector_store = vector_store_handler.create_vector_store(chunks)
    
    print("Saving vector store...")
    vector_store_handler.save_vector_store(vector_store)
    return vector_store

def query_index(query: str, top_k=5):
    embedding_model = EmbeddingFactory.get_embeddings()
    vector_store_handler = VectorStore(embedding_model)
    
    try:
        vector_store = vector_store_handler.load_vector_store()
    except FileNotFoundError:
        print("Vector store not found. Building index first...")
        vector_store = build_index()
    
    retriever = DocumentRetriever(vector_store)
    similar_docs = retriever.similar_search(query, top_k=top_k)
    
    return similar_docs

def rag_qa(query: str):
    embedding_model = EmbeddingFactory.get_embeddings()
    vector_store_handler = VectorStore(embedding_model)
    
    try:
        vector_store = vector_store_handler.load_vector_store()
    except FileNotFoundError:
        print("Vector store not found. Building index first...")
        vector_store = build_index()
    
    retriever = DocumentRetriever(vector_store)
    retriever_instance = retriever.get_retriever()
    
    llm = LLMFactory.get_llm()
    rag_chain = RAGChain(llm, retriever_instance)
    
    result = rag_chain.answer_question(query)
    return result

def chat_mode():
    print("Initial RAG system...")
    embedding_model = EmbeddingFactory.get_embeddings()
    vector_store_handler = VectorStore(embedding_model)
    
    try:
        vector_store = vector_store_handler.load_vector_store()
    except FileNotFoundError:
        print("Vector store not found. Building index first...")
        vector_store = build_index()
    
    retriever = DocumentRetriever(vector_store)
    retriever_instance = retriever.get_retriever()
    
    llm = LLMFactory.get_llm()
    rag_chain = RAGChain(llm, retriever_instance)
    
    print("RAG is ready. Input 'esc' or 'exit' to end the chat.")
    
    while True:
        user_input = input("\n Please input your question: ")
        if user_input.lower() in ['esc', 'exit', 'quit']:
            break
        
        result = rag_chain.answer_question(user_input)
        print("\n Answer:", result["answer"])
        
        print("\n References:")
        for i, doc in enumerate(result["source_documents"][:3]):
            source = doc.metadata.get('source', 'Unknown')
            print(f"{i+1}. Source: {source}")
            print(f"   Preview: {doc.page_content[:100]}...")
            print()

def main():
    parser = argparse.ArgumentParser(description='RAG System')
    parser.add_argument('--mode', type=str, choices=['build', 'query', 'qa', 'chat'], 
                        default='chat', help='Operation mode')
    parser.add_argument('--query', type=str, help='Query for search or QA mode')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()

    dotenv.load_dotenv()
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("No OPENAI_API_KEY!")
    
    if args.mode == 'build':
        build_index()
    elif args.mode == 'query':
        if not args.query:
            print("Error: query parameter is required for query mode")
            return
        results = query_index(args.query, args.top_k)
        for i, doc in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
            print()
    elif args.mode == 'qa':
        if not args.query:
            print("Error: query parameter is required for qa mode")
            return
        result = rag_qa(args.query)
        print("Answer:", result["answer"])
        print("\nSources:")
        for i, doc in enumerate(result["source_documents"][:3]):
            print(f"{i+1}. {doc.metadata.get('source', 'Unknown')}")
    elif args.mode == 'chat':
        chat_mode()

if __name__ == "__main__":
    main()