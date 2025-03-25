import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="KBLAM Training Tool")
    parser.add_argument('--mode', choices=['train', 'large-corpus', 'query'], required=True, help='Operation mode')
    parser.add_argument('--kb_path', default='knowledge_base.json', help='Path to knowledge base JSON file')
    parser.add_argument('--adapter_path', default='models/adapter.pt', help='Path to save/load adapter')
    parser.add_argument('--model', default='llama3.2', help='Ollama model name')
    parser.add_argument('--question', help='Question to answer in query mode')
    parser.add_argument('--corpus_dir', help='Directory containing corpus text files')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Import needed modules
    from kblam_ollama.knowledge_base import KnowledgeBase
    from kblam_ollama.models import KBLAM
    from kblam_ollama.processors import generate_synthetic_kb, preprocess_corpus
    from kblam_ollama.training import train_adapter, train_on_large_corpus, generate_qa_pairs
    
    # Initialize knowledge base
    kb = KnowledgeBase()
    
    if args.mode == 'large-corpus':
        if not args.corpus_dir:
            print("Error: --corpus_dir is required for large-corpus mode")
            return
        
        # Process corpus if knowledge base doesn't exist
        if not os.path.exists(args.kb_path):
            print(f"Processing corpus from {args.corpus_dir}...")
            preprocess_corpus(args.corpus_dir, args.kb_path)
        
        # Load knowledge base
        kb.load_from_json(args.kb_path)
        
        if len(kb.triples) == 0:
            print("Error: No valid triples were generated or loaded")
            return
        
        # Initialize KBLAM
        kblam = KBLAM(kb, model_name=args.model)
        
        # Generate QA pairs
        print("Generating training data...")
        qa_pairs = generate_qa_pairs(kb.triples)
        print(f"Generated {len(qa_pairs)} QA pairs")
        
        # Train on corpus
        train_on_large_corpus(
            kblam, 
            qa_pairs, 
            args.adapter_path,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    elif args.mode == 'train':
        # Load or generate KB
        if os.path.exists(args.kb_path):
            kb.load_from_json(args.kb_path)
            print(f"Loaded {len(kb.triples)} triples from {args.kb_path}")
        else:
            print(f"Generating synthetic KB with 100 triples")
            kb = generate_synthetic_kb(100)
            kb.save_to_json(args.kb_path)
        
        # Verify the KB structure
        if len(kb.triples) == 0:
            print("Warning: Knowledge base is empty. Generating synthetic triples...")
            kb = generate_synthetic_kb(100)
            kb.save_to_json(args.kb_path)
        
        # Initialize KBLAM
        kblam = KBLAM(kb, model_name=args.model)
        
        # Generate synthetic QA pairs for training
        qa_pairs = generate_qa_pairs(kb.triples[:50], num_pairs=200)  # Use 50 triples to create 200 QA pairs
        
        print(f"Training adapter with {len(qa_pairs)} QA pairs")
        train_adapter(kblam, qa_pairs, epochs=args.epochs)
        kblam.save_adapter(args.adapter_path)
        print(f"Adapter saved to {args.adapter_path}")
        
    elif args.mode == 'query':
        # Load KB
        if os.path.exists(args.kb_path):
            kb.load_from_json(args.kb_path)
        else:
            print("Error: Knowledge base not found")
            return
        
        # Initialize KBLAM
        kblam = KBLAM(kb, model_name=args.model)
        
        # Load adapter if it exists
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

