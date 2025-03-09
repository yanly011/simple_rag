from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class RAGChain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self):
        template = """
        You are a professional assistant, please answer the question based on the context.
        If you don't know, please say "I don't know". Do not fabricate answer.
        Answer the question including details as much as possible, with the information in context.
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        Answer:
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": prompt,
                "memory": self.memory
            }
        )
    
    def answer_question(self, question: str):
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", [])
        }