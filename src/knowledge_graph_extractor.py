import spacy
import networkx as nx
import json
from typing import List, Dict, Tuple
from langchain.schema import Document

class KnowledgeGraphExtractor:
    def __init__(self, model_name='en_core_web_sm'):
        """
        Initiate graph extractor
        
        :param model_name: SpaCy language model
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Download the model: {model_name}")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        self.graph = nx.DiGraph()
    
    def extract_entities_and_relations(self, text: str) -> Dict:
        """
        Extract entity and relations from text
        
        :param text: input text
        :return: dictionary containing entity and relationship.
        """
        doc = self.nlp(text)
        
        entities = []
        relations = []
        
        # extract entities
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_
            })
        
        # extract relations
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'attr']:
                relation = {
                    'subject': token.head.text,
                    'predicate': token.dep_,
                    'object': token.text
                }
                relations.append(relation)
        
        return {
            'entities': entities,
            'relations': relations
        }
    
    def build_graph(self, documents: List[Document]) -> nx.DiGraph:
        """
        Build Knowledge Graph from documents
        
        :param documents: documents list
        :return: NetworkX directed graph
        """
        self.graph.clear()
        
        for doc in documents:
            extraction = self.extract_entities_and_relations(doc.page_content)
            
            # Add enitity
            for entity in extraction['entities']:
                self.graph.add_node(
                    entity['text'], 
                    type=entity['label']
                )
            
            # Add edge
            for relation in extraction['relations']:
                self.graph.add_edge(
                    relation['subject'], 
                    relation['object'], 
                    type=relation['predicate']
                )
        
        return self.graph
    
    def serialize_graph(self, graph: nx.DiGraph) -> str:
        """
        Seriveriation to JSON
        
        :param graph: directed graph
        :return: JSON
        """
        data = {
            'nodes': [
                {
                    'id': node, 
                    'attributes': graph.nodes[node]
                } for node in graph.nodes
            ],
            'edges': [
                {
                    'source': u, 
                    'target': v, 
                    'attributes': graph.edges[u, v]
                } for u, v in graph.edges
            ]
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def visualize_graph(self, graph: nx.DiGraph) -> None:
        """
        Knowledge graph visulisation
        
        :param graph: directed graph
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        nx.draw(
            graph, 
            pos, 
            with_labels=True, 
            node_color='lightblue', 
            node_size=500, 
            font_size=8
        )
        plt.title("Knowledge Graph")
        plt.show()