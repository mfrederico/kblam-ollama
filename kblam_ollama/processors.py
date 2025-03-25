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

def preprocess_corpus(corpus_dir, output_file, use_llm=False, model_name="llama3.2:latest"):
    """
    Process a directory of text files into knowledge triples.
    
    Args:
        corpus_dir (str): Directory containing text files
        output_file (str): Path to save the generated knowledge base
        use_llm (bool): Whether to use LLM for information extraction
        model_name (str): Ollama model name to use
    """
    from kblam_ollama.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    
    # Print prominent message about which processing method is being used
    if use_llm:
        print("\n" + "="*80)
        print(f"USING LLM ({model_name}) FOR DOCUMENT PROCESSING")
        print("This method will extract English keywords, entities, and summaries using the LLM")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("USING SPACY FOR DOCUMENT PROCESSING")
        print("This method will extract keywords and entities using NLP techniques")
        print("="*80 + "\n")
    
    # Initialize spaCy if not using LLM
    if not use_llm:
        try:
            import spacy
            # Load SpaCy model for NLP processing
            nlp = spacy.load("en_core_web_sm")
            # Watch for warnings here in RAM usage
            nlp.max_length = 2000000
            print("Successfully loaded spaCy model 'en_core_web_sm'")
        except ImportError:
            print("Error: spaCy is not installed. Please install it with: pip install spacy")
            print("Then download the English model: python -m spacy download en_core_web_sm")
            return []
    
    # Get all text files in the corpus directory
    text_files = [f for f in os.listdir(corpus_dir) if f.endswith('.txt')]
    print(f"Found {len(text_files)} text files in {corpus_dir}")
    
    # Set up progress tracking
    file_count = 0
    success_count = 0
    error_count = 0
    
    for file_name in tqdm(text_files, desc="Processing files"):
        file_count += 1
        file_path = os.path.join(corpus_dir, file_name)
        
        # Extract document name (will be used as entity name)
        doc_name = os.path.splitext(file_name)[0]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Process based on selected method
            if use_llm:
                # Use LLM to extract keywords and entities
                triples = extract_keywords_with_llm(text, doc_name, model_name)
                for triple in triples:
                    kb.add_triple(triple["name"], triple["property"], triple["value"])
                success_count += 1
            else:
                # Use spaCy as before
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
                    
                success_count += 1
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            error_count += 1
    
    # Save the knowledge base
    kb.save_to_json(output_file)
    
    # Print processing summary
    print("\n" + "="*80)
    print(f"PROCESSING COMPLETE:")
    print(f"Total files: {file_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Created knowledge base with {len(kb.triples)} triples")
    print(f"Knowledge base saved to: {output_file}")
    print("="*80 + "\n")
    
    return kb.triples
    
    # Add English word check function
    def is_english_word(word):
        if not english_only:
            return True
        
        # Simple heuristic: English words typically use Latin alphabet
        # This is a basic filter, more sophisticated methods could be used
        return word.isalpha() and word.isascii()
    
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
                # Only include entities with English words
                if is_english_word(ent.text):
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
                if (token.is_alpha and not token.is_stop and 
                    token.pos_ in ["NOUN", "PROPN"] and
                    is_english_word(token.text)):
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

def extract_keywords_with_llm(text, doc_name, model_name="llama3.2:latest"):
    """Use ollama to extract keywords and entities from text"""
    import requests
    
    # Truncate text if it's too long
    max_len = 8192  # Adjust based on model's context window
    truncated_text = text[:max_len] + "..." if len(text) > max_len else text
    
    # Prompt specifically designed for English keyword extraction
    prompt = f"""
    Extract the most important information from this document as structured knowledge.
    
    Document: {truncated_text}
    
    Please provide:
    1. A list of 10-15 ENGLISH ONLY keywords that best represent this document's content
    2. The main entities (people, organizations, locations, concepts) mentioned
    3. A brief summary (2-3 sentences) of what this document is about
    
    Format your response as:
    
    KEYWORDS: [comma-separated list of keywords]
    ENTITIES: [comma-separated list of entities with their types in parentheses]
    SUMMARY: [brief summary]
    
    Only include English words in your extraction.
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
    
    # Parse the structured response
    sections = response_text.split("\n\n")
    for section in sections:
        if section.startswith("KEYWORDS:"):
            keywords = section[9:].strip()
            triples.append({
                "name": doc_name,
                "property": "keywords",
                "value": keywords
            })
        elif section.startswith("ENTITIES:"):
            entities = section[9:].strip()
            triples.append({
                "name": doc_name,
                "property": "entities",
                "value": entities
            })
        elif section.startswith("SUMMARY:"):
            summary = section[8:].strip()
            triples.append({
                "name": doc_name,
                "property": "summary",
                "value": summary
            })
    
    return triples

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
