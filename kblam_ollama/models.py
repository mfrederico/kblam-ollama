import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import requests
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer

class KnowledgeIntegrationLayer(nn.Module):
    """
    Layer that implements KBLAM's rectangular attention mechanism 
    for integrating knowledge tokens with LLM outputs
    """
    def __init__(self, hidden_size, kb_scale_factor=100.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.kb_scale_factor = kb_scale_factor  # C in the paper
        
        # Query projection for transforming LLM outputs
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, llm_hidden_states, knowledge_tokens):
        """
        Apply rectangular attention between LLM states and knowledge tokens
        
        Args:
            llm_hidden_states: Output states from LLM [seq_len, hidden_size]
            knowledge_tokens: Dict of {id: (key, value)} pairs for KB triples
            
        Returns:
            Enhanced hidden states after knowledge integration
        """
        seq_len, _ = llm_hidden_states.shape
        
        # Project queries for knowledge token attention
        queries = self.query_proj(llm_hidden_states)
        
        # Get knowledge keys and values
        if not knowledge_tokens:
            # No knowledge to integrate, return original
            return llm_hidden_states
            
        # Extract and stack knowledge tokens
        kb_keys = []
        kb_values = []
        
        for token_id, (key, value) in knowledge_tokens.items():
            kb_keys.append(key)
            kb_values.append(value)
            
        kb_keys = torch.stack(kb_keys)    # [num_kb, hidden_size]
        kb_values = torch.stack(kb_values)  # [num_kb, hidden_size]
        
        num_kb = len(kb_keys)
        
        # Calculate attention scores with scaling
        # Shape: [seq_len, num_kb]
        scale = 1.0 / math.sqrt(self.hidden_size)
        attention_scores = torch.matmul(queries, kb_keys.transpose(0, 1)) * scale
        
        # Apply KB length normalization (Section 5 in paper)
        kb_scale = math.log(self.kb_scale_factor) - math.log(max(num_kb, 1))
        attention_scores = attention_scores + kb_scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to knowledge values
        # Shape: [seq_len, hidden_size]
        knowledge_context = torch.matmul(attention_weights, kb_values)
        
        # Combine with original hidden states
        enhanced_states = llm_hidden_states + knowledge_context
        
        return enhanced_states

class LinearAdapter(nn.Module):
    """Linear adapter to map sentence embeddings to LLM-compatible key/value pairs"""
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearAdapter, self).__init__()
        self.key_adapter = nn.Linear(input_dim, output_dim)
        self.value_adapter = nn.Linear(input_dim, output_dim)
        print(f"Initialized adapter with input dim: {input_dim}, output dim: {output_dim}")
    
    def forward(self, key_embedding, value_embedding):
        key_projection = self.key_adapter(key_embedding)
        value_projection = self.value_adapter(value_embedding)
        return key_projection, value_projection
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path))


def get_model_dimension(model_name: str) -> int:
    """Query Ollama for model information to get embedding dimension"""
    url = "http://localhost:11434/api/show"
    data = {"name": model_name}
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            model_info = response.json()
            # The exact path to the dimension may vary depending on the model format
            # You might need to adjust based on actual Ollama API responses
            model_params = model_info.get("parameters", {})
            
            # Try different possible dimension fields
            for dim_field in ["hidden_size", "dim", "n_embd", "d_model", "hidden_dim"]:
                if dim_field in model_params:
                    dim = model_params[dim_field]
                    print(f"Found model dimension '{dim_field}': {dim}")
                    return dim
            
            # If we can't find a dimension field, check if Ollama reports model family
            model_family = model_info.get("modelfile", {}).get("family", "").lower()
            
            # Default dimensions for known model families
            family_defaults = {
                "llama": 4096,
                "llama2": 4096,
                "llama3.2": 4096,
                "mistral": 4096,
                "phi": 2560,
                "gemma": 3072,
                "mixtral": 4096
            }
            
            if model_family in family_defaults:
                dim = family_defaults[model_family]
                print(f"Using default dimension for {model_family}: {dim}")
                return dim
                
            # Last resort default
            print("Could not determine model dimension, using default of 512")
            return 512
            
        else:
            print(f"Warning: Could not get model info: {response.status_code} - {response.text}")
            return 512  # Default fallback
    except Exception as e:
        print(f"Error querying model dimension: {str(e)}")
        return 512  # Default fallback


class KBLAM:
    """Knowledge Base augmented Language Model using ollama"""
    def __init__(self, kb, model_name: str = "llama3", embedding_model: str = "all-MiniLM-L6-v2", llm_dim: int = None):
        self.model_name = model_name
        self.kb = kb
        self.sentence_encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        
        # Get LLM embedding dimension - either from parameter or by querying
        if llm_dim is None:
            self.llm_dim = get_model_dimension(model_name)
        else:
            self.llm_dim = llm_dim
            
        print(f"Using LLM embedding dimension: {self.llm_dim}")
        
        # Linear adapter for knowledge tokens
        self.adapter = LinearAdapter(self.embedding_dim, self.llm_dim)
        
        # Knowledge integration layer
        self.knowledge_integrator = KnowledgeIntegrationLayer(self.llm_dim)
        
        # Initialize for model state
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
        """
        Answer a question using rectangular attention with knowledge tokens
        """
        # Get relevant knowledge triples
        relevant_triples = self.kb.search(question)[:top_k]
        
        if not relevant_triples:
            # No relevant knowledge found, just use standard LLM
            return self.query_ollama(question)
        
        # Encode relevant triples into knowledge tokens
        self.knowledge_tokens = {}
        for i, triple in enumerate(relevant_triples):
            key_proj, value_proj = self.encode_triple(triple)
            self.knowledge_tokens[i] = (key_proj, value_proj)
            
        # First, get baseline LLM output features (this will require changes to Ollama interaction)
        # For now, we'll use the standard approach and then augment it post-processing
        
        # Create standard context string as fallback
        context = ""
        for triple in relevant_triples:
            context += f"- {triple['name']} has {triple['property']}: {triple['value']}\n"
            
        # Here we have two approaches:
        
        # APPROACH 1: Two-step process (practical with current Ollama)
        # 1. Get initial response from LLM
        initial_response = self.query_ollama(f"Based on this context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        
        # 2. Apply knowledge integration in post-processing
        # Note: This would be a simplified version since we don't have access to internal LLM states
        # We'd need to build a token-level representation of the response and then enhance it
        
        # APPROACH 2: If you could modify Ollama to expose hidden states:
        # This would be the ideal approach but requires deeper integration
        
        # For now, let's stick with the practical approach
        return initial_response

    def query_with_kb_integration(self, prompt: str) -> str:
        """
        This is a conceptual method that would work if you could 
        intercept Ollama's internal states
        """
        # 1. Tokenize the prompt
        tokens = self.tokenize(prompt)
        
        # 2. Get initial hidden states from Ollama
        hidden_states = self.get_ollama_hidden_states(tokens)
        
        # 3. Apply knowledge integration
        enhanced_states = self.knowledge_integrator(hidden_states, self.knowledge_tokens)
        
        # 4. Feed enhanced states back to Ollama for completion
        response = self.complete_from_states(enhanced_states)
        
        return response

    def answer_question_practical(self, question: str, top_k: int = 3) -> str:
        """
        A practical implementation that approximates rectangular attention
        without requiring direct access to LLM internals
        """
        # Get relevant knowledge triples
        relevant_triples = self.kb.search(question)[:top_k]
        
        if not relevant_triples:
            return self.query_ollama(question)
        
        # Create a context with attention guidance
        context = "Knowledge Base Information:\n"
        
        # Sort triples by relevance score if available
        for i, triple in enumerate(relevant_triples):
            # Add a special attention marker to help guide the LLM's focus
            # This exploits the LLM's ability to pay attention to formatting
            context += f"[KB{i+1}] {triple['name']} | {triple['property']}: {triple['value']}\n"
        
        # Add instructions to emphasize knowledge-based reasoning
        prompt = f"""
        {context}
        
        Please answer the following question using ONLY the knowledge provided above.
        If the information needed isn't in the knowledge base, say you don't have that information.
        
        Question: {question}
        
        Thinking:
        - First, identify which knowledge items are relevant to this question
        - Then, carefully use that knowledge to formulate your answer
        
        Answer:
        """
        
        # Get response from ollama
        response = self.query_ollama(prompt)
        
        # Post-process to remove any thinking steps or KB references
        # This could be improved with regex or more sophisticated parsing
        final_response = response.split("Answer:")[-1].strip()
        
        return final_response
    
    def save_adapter(self, path: str):
        """Save the trained adapter"""
        self.adapter.save(path)
    
    def load_adapter(self, path: str):
        """Load a trained adapter"""
        self.adapter.load(path)

