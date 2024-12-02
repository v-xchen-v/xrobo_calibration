import os
import json

def load_json(filepath: str) -> dict:
    """
    Load a JSON file from the data directory.

    Args:
        filepath (str): Filepath to the JSON file.

    Returns:
        dict: JSON data.
    """
    with open(filepath, "r") as f:
        return json.load(f)
    
def save_json(filepath: str, data: dict):
    """
    Save data to a JSON file.

    Args:
        filepath (str): Filepath to save the data.
        data (dict): Data to save.
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)