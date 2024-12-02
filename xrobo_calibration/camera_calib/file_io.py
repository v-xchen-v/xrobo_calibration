import json
import numpy as np


def save_parameters(filepath: str, parameters: dict, format: str = "json"):
    """
    Save calibration parameters to a file in the specified format.

    Args:
        filepath (str): Path to the output file.
        parameters (dict): Dictionary of parameters to save.
        format (str): Format to save the file ("json" or "npy").
    """
    def serialize_value(value):
        if isinstance(value, np.ndarray):  # Convert NumPy arrays to lists
            return value.tolist()
        elif isinstance(value, (float, int, str, list, dict)):  # Already JSON-compatible types
            return value
        else:
            raise ValueError(f"Unsupported data type {type(value)} for value: {value}")

    if format == "json":
        with open(filepath, "w") as f:
            json.dump({k: serialize_value(v) for k, v in parameters.items()}, f, indent=4)
    elif format == "npy":
        np.savez(filepath, **parameters)
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'npy'.")


def load_parameters(filepath: str, format: str = "json"):
    """
    Load calibration parameters from a file in the specified format.

    Args:
        filepath (str): Path to the input file.
        format (str): Format of the file ("json" or "npy").

    Returns:
        dict: Dictionary of loaded parameters.
    """
    if format == "json":
        with open(filepath, "r") as f:
            data = json.load(f)
        return {k: np.array(v) for k, v in data.items()}
    elif format == "npy":
        data = np.load(filepath)
        return {key: data[key] for key in data}
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'npy'.")
