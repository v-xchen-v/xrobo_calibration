import os
import json
# import yaml

def get_sample_data_path():
    """
    Returns the absolute path to the sample data directory.
    """
    return os.path.join(os.path.dirname(__file__), "../../data")

def load_json(file_name):
    """
    Load a JSON file from the data directory.
    """
    data_path = os.path.join(get_sample_data_path(), file_name)
    with open(data_path, "r") as f:
        return json.load(f)

# def load_yaml(file_name):
#     """
#     Load a YAML file from the data directory.
#     """
#     data_path = os.path.join(get_sample_data_path(), file_name)
#     with open(data_path, "r") as f:
#         return yaml.safe_load(f)