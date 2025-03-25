import os
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Any
import argparse

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
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path))

class KnowledgeBase:
    """Storage and management of knowledge triples"""
    def __init__(self):
        self.triples = []
        
    def add_triple(self, name: str, property_val: str, value: str):
        self.triples.append({"name": name, "property": property_val, "value": value})
    
    def load_from_json(self, file_path: str):
        try:
            with open(file_path, 'r') as f:
                loaded_data = json.load(f)
                
                # Handle different possible structures
                if isinstance(loaded_data, list):
                    self.triples = loaded_data
                elif isinstance(loaded_data, dict) and "triples" in loaded_data:
                    self.triples = loaded_data["triples"]
                else:
                    print(f"Warning: Unexpected format in {file_path}")
                    self.triples = []
                
                # Verify structure of each triple
                valid_triples = []
                for triple in self.triples:
                    if isinstance(triple, dict) and all(k in triple for k in ["name", "property", "value"]):
                        valid_triples.append(triple)
                
                self.triples = valid_triples
                print(f"Loaded {len(valid_triples)} valid triples")
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading file {file_path}: {e}")
            self.triples = []
    
    def save_to_json(self, file_path: str):
        try:
            with open(file_path, 'w') as f:
                json.dump(self.triples, f, indent=2)
            print(f"Saved {len(self.triples)} triples to {file_path}")
        except Exception as e:
            print(f"Error saving to {file_path}: {e}")
    
    def search(self, query: str) -> List[Dict]:
        """Simple search functionality for triples"""
        results = []
        query = query.lower()
        for triple in self.triples:
            if (query in triple["name"].lower() or 
                query in triple["property"].lower() or 
                query in triple["value"].lower()):
                results.append(triple)
        return results

class KBLAM:
    """Knowledge Base augmented Language Model using ollama"""
    def __init__(self, model_name: str = "llama3.2:latest", embedding_model: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.kb = KnowledgeBase()
        self.sentence_encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        
        # Get LLM embedding dimension - this would need to be adjusted for the actual ollama model
        self.llm_dim = 512  # Default value, ideally this would come from the model
        
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
    
    def train_adapter(self, qa_pairs: List[Dict], epochs: int = 5, learning_rate: float = 0.0001):
        """Train the adapter using question-answer pairs"""
        # Define optimizer
        optimizer = optim.Adam(self.adapter.parameters(), lr=learning_rate)
        
        # Simple loss function - in a real implementation this would be more sophisticated
        loss_fn = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for qa_pair in qa_pairs:
                question = qa_pair["question"]
                answer = qa_pair["answer"]
                
                # Get relevant triples
                relevant_triples = self.kb.search(question)
                
                if not relevant_triples:
                    continue
                
                # Forward pass - simplified for demonstration
                for triple in relevant_triples:
                    key_projection, value_projection = self.encode_triple(triple)
                    
                    # This is a simplification - in the real KBLAM, the loss would involve
                    # evaluating how well the query with augmented knowledge generates the answer
                    # Here we're just making the embeddings fit a pattern
                    target = torch.ones_like(value_projection) * 0.5
                    loss = loss_fn(value_projection, target)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(qa_pairs)}")
    
    def save_adapter(self, path: str):
        """Save the trained adapter"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.adapter.save(path)
    
    def load_adapter(self, path: str):
        """Load a trained adapter"""
        self.adapter.load(path)

def generate_synthetic_kb(num_triples: int = 100) -> KnowledgeBase:
    """Generate a synthetic KB for testing"""
    kb = KnowledgeBase()
    
    # Sample entities, properties, and values
    entities = [f"Entity_{i}" for i in range(20)]
    properties = ["description", "purpose", "objectives"]
    values = [
        "A tool for data analysis and visualization",
        "To provide secure cloud storage solutions",
        "To optimize business processes through automation",
        "A platform for collaborative software development",
        "To facilitate knowledge sharing and learning"
    ]
    
    # Generate random triples
    import random
    for _ in range(num_triples):
        name = random.choice(entities)
        property_val = random.choice(properties)
        value = random.choice(values)
        kb.add_triple(name, property_val, value)
    
    # Verify that triples were added correctly
    print(f"Generated {len(kb.triples)} synthetic triples")
    if len(kb.triples) > 0:
        print(f"Sample triple: {kb.triples[0]}")
    
    return kb

def main():
    parser = argparse.ArgumentParser(description="KBLAM Training Tool")
    parser.add_argument('--mode', choices=['train', 'query'], required=True, help='Operation mode')
    parser.add_argument('--kb_path', default='knowledge_base.json', help='Path to knowledge base JSON file')
    parser.add_argument('--adapter_path', default='models/adapter.pt', help='Path to save/load adapter')
    parser.add_argument('--model', default='llama3.2:latest', help='Ollama model name')
    parser.add_argument('--question', help='Question to answer in query mode')
    
    args = parser.parse_args()
    
    # Initialize KBLAM
    kblam = KBLAM(model_name=args.model)
    
    # Load or generate KB
    if os.path.exists(args.kb_path):
        kblam.kb.load_from_json(args.kb_path)
        print(f"Loaded {len(kblam.kb.triples)} triples from {args.kb_path}")
    else:
        print(f"Generating synthetic KB with 100 triples")
        kblam.kb = generate_synthetic_kb(100)
        kblam.kb.save_to_json(args.kb_path)
    
    # Verify the KB structure
    if len(kblam.kb.triples) == 0:
        print("Warning: Knowledge base is empty. Generating synthetic triples...")
        kblam.kb = generate_synthetic_kb(100)
        kblam.kb.save_to_json(args.kb_path)
    
    # Check if triples have the expected structure
    valid_triples = []
    for triple in kblam.kb.triples:
        if isinstance(triple, dict) and all(k in triple for k in ["name", "property", "value"]):
            valid_triples.append(triple)
        else:
            print(f"Warning: Invalid triple structure: {triple}")
    
    if len(valid_triples) == 0:
        print("No valid triples found. Generating synthetic knowledge base...")
        kblam.kb = generate_synthetic_kb(100)
        valid_triples = kblam.kb.triples
        kblam.kb.save_to_json(args.kb_path)
    else:
        kblam.kb.triples = valid_triples
        print(f"Using {len(valid_triples)} valid triples")
    
    if args.mode == 'train':
        # Generate synthetic QA pairs for training
        qa_pairs = []
        for triple in valid_triples[:50]:  # Use 50 triples for training examples
            question = f"What is the {triple['property']} of {triple['name']}?"
            answer = triple['value']
            qa_pairs.append({"question": question, "answer": answer})
        
        print(f"Training adapter with {len(qa_pairs)} QA pairs")
        kblam.train_adapter(qa_pairs, epochs=5)
        kblam.save_adapter(args.adapter_path)
        print(f"Adapter saved to {args.adapter_path}")
        
    elif args.mode == 'query':
        if os.path.exists(args.adapter_path):
            kblam.load_adapter(args.adapter_path)
            print(f"Loaded adapter from {args.adapter_path}")
        
        if args.question:
            answer = kblam.answer_question(args.question)
            print(f"Question: {args.question}")
            print(f"Answer: {answer}")
        else:
            print("Interactive query mode. Type 'exit' to quit.")
            while True:
                question = input("Question: ")
                if question.lower() == 'exit':
                    break
                answer = kblam.answer_question(question)
                print(f"Answer: {answer}")
                print()

if __name__ == "__main__":
    main()

