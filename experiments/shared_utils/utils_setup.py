"""
Setup utilities for self-obfuscation experiments.

This module provides common setup patterns used across probe training,
testing, and model training scripts.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any

from obf_reps.logging import CSVTXTLogger


def setup_experiment_directory(
    output_dir_base: str,
    custom_folder_name: str = "",
    experiment_type: str = "",
    additional_suffix: str = ""
) -> str:
    """
    Set up experiment output directory with consistent naming.

    Args:
        output_dir_base: Base directory for outputs
        custom_folder_name: Custom folder name (if provided, overrides automatic naming)
        experiment_type: Type of experiment (e.g., "TRAIN", "TEST")
        additional_suffix: Additional suffix to append to directory name

    Returns:
        Path to created directory
    """
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if custom_folder_name:
        directory = os.path.join(output_dir_base, custom_folder_name)
    else:
        dir_name = f"{date_time}"
        if experiment_type:
            dir_name += f"_{experiment_type}"
        if additional_suffix:
            dir_name += f"_{additional_suffix}"
        directory = os.path.join(output_dir_base, dir_name)

    os.makedirs(directory, exist_ok=True)
    return directory


def setup_logger(print_to_console: bool = True) -> CSVTXTLogger:
    """Set up logger with consistent configuration."""
    return CSVTXTLogger(print_logs_to_console=print_to_console)


def save_metadata(
    directory: str,
    metadata: Dict[str, Any],
    filename: str = "metadata.json"
) -> None:
    """Save experiment metadata to JSON file."""
    metadata_path = os.path.join(directory, filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(directory: str, filename: str = "metadata.json") -> Dict[str, Any]:
    """Load experiment metadata from JSON file."""
    metadata_path = os.path.join(directory, filename)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find {filename} in directory: {directory}")

    with open(metadata_path, 'r') as f:
        return json.load(f)


def create_experiment_metadata(
    experiment_type: str,
    model_name: str,
    model_checkpoint: Optional[str] = None,
    **additional_metadata
) -> Dict[str, Any]:
    """
    Create standardized experiment metadata dictionary.

    Args:
        experiment_type: Type of experiment (e.g., "probe_training", "model_training")
        model_name: Name of the model used
        model_checkpoint: Path to model checkpoint (if any)
        **additional_metadata: Additional metadata fields

    Returns:
        Dictionary with experiment metadata
    """
    metadata = {
        "experiment_type": experiment_type,
        "run_date_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_name": model_name,
        "model_checkpoint": model_checkpoint,
    }
    metadata.update(additional_metadata)
    return metadata 
