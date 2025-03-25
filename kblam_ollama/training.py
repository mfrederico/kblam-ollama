import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from typing import List, Dict

def train_adapter(kblam, qa_pairs: List[Dict], epochs: int = 5, learning_rate: float = 0.0001):
    """Train the adapter using question-answer pairs"""
    # Define optimizer
    optimizer = optim.Adam(kblam.adapter.parameters(), lr=learning_rate)
    
    # Simple loss function - in a real implementation this would be more sophisticated
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for qa_pair in tqdm(qa_pairs, desc=f"Epoch {epoch+1}/{epochs}"):
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            
            # Get relevant triples
            relevant_triples = kblam.kb.search(question)
            
            if not relevant_triples:
                continue
            
            # Forward pass - simplified for demonstration
            for triple in relevant_triples:
                key_projection, value_projection = kblam.encode_triple(triple)
                
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

def train_batch(kblam, qa_batch, learning_rate: float = 0.0001):
    """Train on a batch of QA pairs"""
    optimizer = optim.Adam(kblam.adapter.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    total_loss = 0
    
    for qa_pair in qa_batch:
        question = qa_pair["question"]
        answer = qa_pair["answer"]
        
        relevant_triples = kblam.kb.search(question)
        
        if not relevant_triples:
            continue
        
        for triple in relevant_triples:
            key_projection, value_projection = kblam.encode_triple(triple)
            
            # Target is simplified for demonstration
            target = torch.ones_like(value_projection) * 0.5
            loss = loss_fn(value_projection, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / len(qa_batch) if qa_batch else 0

def train_on_large_corpus(kblam, qa_pairs, adapter_path, epochs=10, batch_size=32):
    """
    Train KBLAM on a large corpus of QA pairs
    
    Args:
        kblam: KBLAM model instance
        qa_pairs: List of question-answer pairs
        adapter_path: Path to save adapter model
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print(f"Training adapter for {epochs} epochs with batch size {batch_size}...")
    total_batches = len(qa_pairs) // batch_size + (1 if len(qa_pairs) % batch_size > 0 else 0)
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        # Shuffle data at each epoch
        random.shuffle(qa_pairs)
        
        for batch_idx in tqdm(range(0, len(qa_pairs), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = qa_pairs[batch_idx:batch_idx + batch_size]
            
            # Train on this batch
            batch_loss = train_batch(kblam, batch)
            total_loss += batch_loss
            batch_count += 1
        
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/batch_count:.6f}")
    
    # Save the trained adapter
    print(f"Saving trained adapter to {adapter_path}...")
    kblam.save_adapter(adapter_path)
    print("Training complete!")

def generate_qa_pairs(kb_triples, num_pairs=None):
    """Generate training QA pairs from KB triples"""
    qa_pairs = []
    
    # Use all or a subset of triples
    if num_pairs and num_pairs < len(kb_triples):
        selected_triples = random.sample(kb_triples, num_pairs)
    else:
        selected_triples = kb_triples
    
    # Create question-answer pairs for selected triples
    for triple in tqdm(selected_triples, desc="Creating QA pairs"):
        # Create different question patterns for variety
        patterns = [
            f"What is the {triple['property']} of {triple['name']}?",
            f"Tell me about the {triple['property']} of {triple['name']}.",
            f"Describe the {triple['property']} related to {triple['name']}.",
            f"What {triple['property']} does {triple['name']} have?",
            f"Can you explain the {triple['property']} of {triple['name']}?"
        ]
        
        # Add one random pattern or all patterns
        if num_pairs:
            qa_pairs.append({
                "question": random.choice(patterns),
                "answer": triple['value']
            })
        else:
            for pattern in patterns:
                qa_pairs.append({
                    "question": pattern,
                    "answer": triple['value']
                })
    
    return qa_pairs

