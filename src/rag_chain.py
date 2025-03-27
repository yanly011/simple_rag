from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class RAGChain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            verbose=True
        )
    
    def answer_question(self, question: str):
        result = self.qa_chain({"question": question})
        return {
            "answer": result["answer"],
            "source_documents": result.get("source_documents", [])
        }