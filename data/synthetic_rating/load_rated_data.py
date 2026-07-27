#!/usr/bin/env python3
"""
Load evaluated synthetic data and convert to standardized format.
"""

import json
from pathlib import Path
from typing import List, Optional, Literal
import sys

# Add parent directories to path to access data modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.data import PromptRespRating
from data.data_utils import find_most_recent_file


def load_rated_data(
    dataset_name: str = r"mega_dataset_evaluated_.*\.json$",
    response_type: Literal["vanilla", "topical"] = "topical",
    manual_path_confirm: bool = True,
    exclude_refusals: bool = True,
    exclude_missing_ratings: bool = True,
    exclude_concepts: Optional[List[str]] = None,
    filter_to_concepts: Optional[List[str]] = None,
    dataset_file_path: Optional[str] = None
) -> List[PromptRespRating]:
    """
    Load evaluated synthetic data and return as PromptRespRating objects.

    Args:
        dataset_name: Regex pattern to match dataset files (used if dataset_file_path is None)
        response_type: Whether to load "vanilla" or "topical" responses
        manual_path_confirm: If True and dataset_file_path is None, show files and let user pick interactively
        exclude_refusals: If True, exclude entries where the selected response type is a refusal
        exclude_missing_ratings: If True, exclude entries with missing ratings for the response type
        exclude_concepts: If provided, only include samples that have at least one non-excluded concept in adjectives list
        filter_to_concepts: If provided, only include samples that have at least one concept from this list in adjectives list
        dataset_file_path: If provided, load from this specific file path instead of searching

    Returns:
        List of PromptRespRating objects

    Raises:
        FileNotFoundError: If no suitable data files are found
        ValueError: If the response_type is invalid
    """

    if response_type not in ["vanilla", "topical"]:
        raise ValueError(f"response_type must be 'vanilla' or 'topical', got '{response_type}'")

    # Use specific file path if provided, otherwise search for files
    if dataset_file_path:
        dataset_file = Path(dataset_file_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Specified dataset file not found: {dataset_file_path}")
        print(f"Loading data from specified path: {dataset_file}")
    else:
        # Search for most recent mega dataset file
        search_dir = Path(__file__).parent / "outputs"
        print(f"Trying to find dataset name {dataset_name} in path {search_dir}")
        dataset_file = find_most_recent_file(
            folder_path=search_dir,
            pattern=dataset_name,
            manual_confirm=manual_path_confirm
        )
        if dataset_file is None:
            raise FileNotFoundError(f"No mega dataset files found in {search_dir}")
        print(f"Loading data from: {dataset_file}")

    # Load the JSON data
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Could not read or parse {dataset_file}: {e}")

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON file to contain a list, got {type(data)}")

    # Convert to PromptRespRating objects
    prompt_resp_ratings = []

    # Determine the rating key based on response type
    if response_type == "vanilla":
        response_key = "vanilla_response"
        rating_key = "vanilla_response_normalized_ratings"
        refusal_key = "is_vanilla_response_refusal"
    else:  # topical
        response_key = "topical_response"
        rating_key = "topical_response_normalized_ratings"
        refusal_key = "is_topical_response_refusal"

    skipped_refusals = 0
    skipped_missing_ratings = 0
    skipped_excluded_concepts = 0
    skipped_filtered_concepts = 0

    # Convert concepts to sets for faster lookup
    exclude_concepts_set = set(exclude_concepts) if exclude_concepts else set()
    filter_to_concepts_set = set(filter_to_concepts) if filter_to_concepts else set()

    for entry in data:
        #print(f"ENTRY: {entry}", flush=True)
        # Check for required fields - first try vanilla/topical format, then fall back to plain format
        response_text = None
        ratings = None

        if response_key in entry and rating_key in entry:
            # Original format with vanilla/topical keys
            response_text = entry[response_key]
            ratings = entry[rating_key]
        elif "response" in entry and "response_normalized_ratings" in entry:
            # New format with plain response key
            response_text = entry["response"]
            ratings = entry["response_normalized_ratings"]

        # Check if we have the required fields
        if "prompt" not in entry or response_text is None or ratings is None:
            continue

        # Check for refusals if excluding them (only applies to original format)
        if exclude_refusals and entry.get(refusal_key, False):
            skipped_refusals += 1
            continue

        # Check for missing ratings if excluding them
        if exclude_missing_ratings and (ratings is None or not isinstance(ratings, dict)):
            skipped_missing_ratings += 1
            continue

        # Check if all ratings are None (missing)
        if exclude_missing_ratings and all(v is None for v in ratings.values()):
            skipped_missing_ratings += 1
            continue

        # Check if sample has at least one non-excluded concept in its adjectives list
        if exclude_concepts_set:
            # Try both 'adjectives' and 'adjectives_evaluated' keys
            if 'adjectives' in entry:
                adjectives = entry['adjectives']
            elif 'adjectives_evaluated' in entry:
                adjectives = entry['adjectives_evaluated']
            else:
                raise KeyError("No adjectives found in entry")
            has_non_excluded_adjective = False
            for adjective in adjectives:
                if adjective not in exclude_concepts_set:
                    has_non_excluded_adjective = True
                    break

            if not has_non_excluded_adjective:
                skipped_excluded_concepts += 1
                continue

        # Check if sample has at least one concept from filter_to_concepts in its adjectives list
        if filter_to_concepts_set:
            # Try both 'adjectives' and 'adjectives_evaluated' keys
            if 'adjectives' in entry:
                adjectives = entry['adjectives']
            elif 'adjectives_evaluated' in entry:
                adjectives = entry['adjectives_evaluated']
            else:
                raise KeyError("No adjectives found in entry")
            has_filtered_adjective = False
            for adjective in adjectives:
                if adjective in filter_to_concepts_set:
                    has_filtered_adjective = True
                    break

            if not has_filtered_adjective:
                skipped_filtered_concepts += 1
                continue

        # Create PromptRespRating object
        # Note: ratings should already be normalized to 0-1 scale
        # Extract adjectives from entry (try both possible keys)
        try:
            # Debug: Check what type entry is and what keys exist
            # print(f"DEBUG: type(entry) = {type(entry)}", flush=True)
            # print(f"DEBUG: Keys in entry: {list(entry.keys())}", flush=True)
            # print(f"DEBUG: 'adjectives' in entry: {'adjectives' in entry}", flush=True)
            # print(f"DEBUG: 'adjectives_evaluated' in entry: {'adjectives_evaluated' in entry}", flush=True)

            # adjectives_via_get = entry.get("adjectives")
            # print(f"DEBUG: entry.get('adjectives') = {adjectives_via_get}", flush=True)

            # For some reason .get() was throwing missing key errors inconsistently
            if 'adjectives' in entry:
                adjectives = entry['adjectives']
            elif 'adjectives_evaluated' in entry:
                adjectives = entry['adjectives_evaluated']
            else:
                raise KeyError("No adjectives found in entry")
        except KeyError:
            print(f"PROBLEMATIC ENTRY: {entry}", flush=True)
            raise

        prompt_resp_rating = PromptRespRating(
            prompt=entry["prompt"],
            response=response_text,
            ratings=ratings,
            adjectives=adjectives
        )

        prompt_resp_ratings.append(prompt_resp_rating)

    print(f"Loaded {len(prompt_resp_ratings)} {response_type} prompt-response pairs")
    if skipped_refusals > 0:
        print(f"Skipped {skipped_refusals} entries due to refusals")
    if skipped_missing_ratings > 0:
        print(f"Skipped {skipped_missing_ratings} entries due to missing ratings")
    if skipped_excluded_concepts > 0:
        print(f"Skipped {skipped_excluded_concepts} entries due to excluded concepts filter")
    if skipped_filtered_concepts > 0:
        print(f"Skipped {skipped_filtered_concepts} entries due to filter_to_concepts filter")

    if not prompt_resp_ratings:
        raise ValueError(f"No valid data found in {dataset_file}")

    return prompt_resp_ratings
