import os
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer

class LinearAdapter(nn.Module):
    """Linear adapter to map sentence embeddings to LLM-compatible key/value pairs"""
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearAdapter, self).__init__()
        self.key_adapter = nn.Linear(input_dim, output_dim)
        self.value_adapter = nn.Linear(input_dim, output_dim)
    
    def forward(self, key_embedding, value_embedding):
        key_projection = self.key_adapter(key_embedding)
        value_projection = self.value_adapter(value_embedding)
        return key_projection, value_projection
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path))

class KBLAM:
    """Knowledge Base augmented Language Model using ollama"""
    def __init__(self, kb, model_name: str = "llama3", embedding_model: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.kb = kb
        self.sentence_encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        
        # Get LLM embedding dimension - would need to be adjusted for the actual ollama model
        self.llm_dim = 512  # Default value
        
        self.adapter = LinearAdapter(self.embedding_dim, self.llm_dim)
        self.knowledge_tokens = {}
        
    def encode_triple(self, triple: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a triple into a knowledge token"""
        name = triple["name"]
        property_val = triple["property"]
        value = triple["value"]
        
        # Create key and value embeddings
        key_text = f"The {property_val} of {name}"
        value_text = value
        
        key_embedding = torch.tensor(self.sentence_encoder.encode(key_text))
        value_embedding = torch.tensor(self.sentence_encoder.encode(value_text))
        
        # Apply linear adapters
        key_projection, value_projection = self.adapter(key_embedding, value_embedding)
        
        return key_projection, value_projection
    
    def encode_kb(self):
        """Encode all triples in the knowledge base"""
        self.knowledge_tokens = {}
        for triple in self.kb.triples:
            key = f"{triple['name']}_{triple['property']}"
            self.knowledge_tokens[key] = self.encode_triple(triple)
    
    def query_ollama(self, prompt: str, kb_context: str = "") -> str:
        """Query ollama with a prompt and optional KB context"""
        url = "http://localhost:11434/api/generate"
        
        # Prepare the context with KB information if provided
        if kb_context:
            full_prompt = f"Knowledge Base Information:\n{kb_context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt
            
        data = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error querying ollama: {str(e)}"
    
    def answer_question(self, question: str, top_k: int = 3) -> str:
        """Answer a question using the KB and ollama"""
        # Simple retrieval based on text matching
        relevant_triples = self.kb.search(question)
        
        # Create context from the retrieved triples
        context = ""
        for triple in relevant_triples[:top_k]:
            context += f"- {triple['name']} has {triple['property']}: {triple['value']}\n"
        
        # Get answer from ollama
        return self.query_ollama(question, context)
    
    def save_adapter(self, path: str):
        """Save the trained adapter"""
        self.adapter.save(path)
    
    def load_adapter(self, path: str):
        """Load a trained adapter"""
        self.adapter.load(path)


