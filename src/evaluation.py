from typing import List, Dict
from langchain.evaluation import QAEvalChain
from langchain_openai import ChatOpenAI

class RAGEvaluator:
    def __init__(self, llm=None):
        if llm is None:
            self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        else:
            self.llm = llm
    
    def evaluate_qa(self, examples: List[Dict], predictions: List[Dict]):
        '''Quastion and Answer'''
        eval_chain = QAEvalChain.from_llm(self.llm)
        graded_outputs = eval_chain.evaluate(
            examples=examples,
            predictions=predictions
        )
        return graded_outputs
    
    def format_evaluation_results(self, eval_results: List[Dict]):
        results = []
        for result in eval_results:
            formatted_result = {
                "question": result.get("query", ""),
                "answer": result.get("prediction", ""),
                "score": result.get("score", ""),
                "feedback": result.get("feedback", ""),
                "reasoning": result.get("reasoning", "")
            }
            results.append(formatted_result)
        return results
    
    def evaluate_retrieval(self, queries, ground_truth_docs, retriever, k=5):
        '''Evaluate retrieval result'''
        results = []
        
        for query, relevant_docs in zip(queries, ground_truth_docs):
            retrieved_docs = retriever.get_relevant_documents(query)
            retrieved_docs = retrieved_docs[:k]
            
            # Precision
            relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
            precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
            
            # Recall
            recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
            
            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                "query": query,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "retrieved_count": len(retrieved_docs),
                "relevant_count": len(relevant_docs),
                "relevant_retrieved": relevant_retrieved
            })
        
        # Calcualte average metrics
        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_recall = sum(r["recall"] for r in results) / len(results)
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        
        summary = {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "queries_evaluated": len(results)
        }
        
        return results, summary