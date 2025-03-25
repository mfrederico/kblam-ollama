import json
from typing import List, Dict

class KnowledgeBase:
    """Storage and management of knowledge triples"""
    def __init__(self):
        self.triples = []
        
    def add_triple(self, name: str, property_val: str, value: str):
        self.triples.append({
            "name": name, 
            "property": property_val, 
            "value": value
        })
    
    def load_from_json(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
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
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.triples, f, indent=2, ensure_ascii=False)
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
