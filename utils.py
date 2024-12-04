from typing import List


def validate_columns(required_columns: List[str], available_columns: List[str]):
    """Validate that required columns exist in the dataset."""
    missing = [col for col in required_columns if col not in available_columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
