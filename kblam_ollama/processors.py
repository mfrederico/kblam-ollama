import os
import json
from tqdm import tqdm
import random

def generate_synthetic_kb(num_triples: int = 100):
    """Generate a synthetic KB for testing"""
    from kblam_ollama.knowledge_base import KnowledgeBase
    
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

def preprocess_corpus(corpus_dir, output_file):
    """
    Process a directory of text files into knowledge triples.
    Requires spaCy to be installed.
    
    Args:
        corpus_dir (str): Directory containing text files
        output_file (str): Path to save the generated knowledge base
    """
    try:
        import spacy
        # Load SpaCy model for NLP processing
        nlp = spacy.load("en_core_web_sm")
        # Watch for warhings here in RAM usage
        nlp.max_length = 2000000
    except ImportError:
        print("Error: spaCy is not installed. Please install it with: pip install spacy")
        print("Then download the English model: python -m spacy download en_core_web_sm")
        return []
    
    from kblam_ollama.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    
    # Get all text files in the corpus directory
    text_files = [f for f in os.listdir(corpus_dir) if f.endswith('.txt')]
    
    for file_name in tqdm(text_files, desc="Processing files"):
        file_path = os.path.join(corpus_dir, file_name)
        
        # Extract document name (will be used as entity name)
        doc_name = os.path.splitext(file_name)[0]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Process the document with SpaCy
            doc = nlp(text)
            
            # Extract document summary as description
            summary = text[:500] + "..." if len(text) > 500 else text
            kb.add_triple(doc_name, "description", summary)
            
            # Extract named entities as properties
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            
            # Add entity information as properties
            for entity_type, entity_values in entities.items():
                # Limit to the top 5 entities to avoid too much noise
                top_entities = list(set(entity_values))[:5]
                if top_entities:
                    kb.add_triple(doc_name, f"contains_{entity_type.lower()}", ", ".join(top_entities))
            
            # Extract keywords
            keywords = []
            for token in doc:
                if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "PROPN"]:
                    keywords.append(token.lemma_)
            
            # Get top 10 keywords by frequency
            from collections import Counter
            top_keywords = [word for word, _ in Counter(keywords).most_common(10)]
            if top_keywords:
                kb.add_triple(doc_name, "keywords", ", ".join(top_keywords))
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Save the knowledge base
    kb.save_to_json(output_file)
    print(f"Created knowledge base with {len(kb.triples)} triples")
    return kb.triples

def extract_triples_with_llm(text, doc_name, model_name="llama3.2"):
    """Use ollama to extract structured knowledge from text"""
    import requests
    
    prompt = f"""
    Extract structured knowledge from the following text in the format of triples:
    (entity, property, value)
    
    Text: {text[:2000]}...
    
    Focus on extracting factual information. Identify the main topics, key concepts, 
    and important relationships. Return at least 5 but no more than 10 triples.
    """
    
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            response_text = response.json()["response"]
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error querying ollama: {str(e)}")
        return []
    
    triples = []
    
    # Parse response into triples
    lines = response_text.strip().split('\n')
    for line in lines:
        if ',' in line and line.count(',') >= 2:
            parts = line.split(',', 2)
            if len(parts) == 3:
                entity = parts[0].strip('() "\'')
                property_val = parts[1].strip('() "\'')
                value = parts[2].strip('() "\'')
                
                # Use the document name as the entity name
                triples.append({
                    "name": doc_name,
                    "property": property_val,
                    "value": value
                })
    
    return triples
