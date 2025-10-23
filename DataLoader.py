import json
from typing import Dict, List, Any


def load_json_to_dict(file_path: str) -> Dict[str, Any]:
    """
    Loads a JSON file and converts it into a nested dictionary structure for easy access.
    
    :param file_path: Path to the JSON file
    :return: A dictionary where each text_id maps to its corresponding data, including requirements
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        structured_data = {}
        
        for entry in data:
            text_id = entry["text_id"]
            structured_data[text_id] = {
                "text_number": entry["text_number"],
                "source": entry["source"],
                "pages": entry["pages"],
                "original_text": entry["original_text"],
                "numberOfRequirements": entry["numberOfRequirements"],
                "requirements": {}
            }
            
            for req in entry["requirements"]:
                structured_data[text_id]["requirements"][req["req_id"]] = {
                    "category": req["category"],
                    "furtherSpecialisation": req["furtherSpecialisation"],
                    "gold_reference": req["gold_reference"],
                    "alternatives": req["alternatives"]
                }
                
        return structured_data
    
    except Exception as e:
        raise ValueError(f"Error loading JSON file {file_path}: {e}")
