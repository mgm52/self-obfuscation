#!/usr/bin/env python3
"""
Evaluate how well prompts and responses fit their assigned adjectives using an LLM.
Creates a mega dataset with all original data plus normalized ratings for each adjective.
Uses OpenAI's Batch API or local Gemma model for efficient processing.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import openai
from tqdm import tqdm
import time
import argparse
from dotenv import load_dotenv
import uuid
import concurrent.futures
import threading
from dataclasses import dataclass

# Model loading/generation helpers shared with the synthetic data generator
from data.self_obfuscation_v1_synthetic.generate_synthetic_data import (
    load_transformers_model,
    optimized_generate_from_string,
    remove_model_special_chars,
)


@dataclass
class BatchInfo:
    """Information about a submitted batch."""
    batch_id: str
    chunk_index: int
    chunk_start: int
    chunk_end: int
    chunk_file_data: List[Tuple[str, List[Dict[str, Any]]]]
    batch_file_path: Optional[str] = None
    submitted_at: Optional[datetime] = None


class AdjectiveFitEvaluator:
    """Evaluates how well text fits various adjectives using LLM ratings via Batch API or local Gemma."""

    def __init__(self, data_dir: str, output_dir: Optional[str] = None, model: str = "gpt-4o-mini",
                 subset_size: Optional[int] = None, use_local_gemma: bool = False,
                 gemma_model_name: str = "gemma_2_9b_instruct", resume_batch_id: Optional[str] = None,
                 chunk_size: Optional[int] = None, max_parallel_batches: int = 5,
                 skip_vanilla_responses: bool = False):
        """
        Initialize the evaluator.

        Args:
            data_dir: Directory containing the JSON files with synthetic data
            output_dir: Directory to save the mega dataset (defaults to data/self_obfuscation_v1_synthetic_rating/outputs)
            model: OpenAI model to use for evaluation (ignored if use_local_gemma=True)
            subset_size: If specified, process only this many entries per file
            use_local_gemma: If True, use local Gemma model instead of OpenAI API
            gemma_model_name: Name of the Gemma model to load
            resume_batch_id: If specified, resume processing from an existing batch ID
            chunk_size: If specified, process data in chunks of this size and save incrementally
            max_parallel_batches: Maximum number of OpenAI batches to run in parallel (default: 5)
            skip_vanilla_responses: If True, skip evaluation of vanilla responses entirely
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("data/self_obfuscation_v1_synthetic_rating/outputs")
        self.model = model
        self.subset_size = subset_size
        self.use_local_gemma = use_local_gemma
        self.gemma_model_name = gemma_model_name
        self.resume_batch_id = resume_batch_id
        self.chunk_size = chunk_size
        self.max_parallel_batches = max_parallel_batches
        self.skip_vanilla_responses = skip_vanilla_responses

        if not use_local_gemma:
            self.client = openai.OpenAI()  # Will use OPENAI_API_KEY env var
        else:
            self.client = None
            # Load Gemma model
            print("Loading local Gemma model...")
            self.gemma_model, self.gemma_tokenizer = load_transformers_model(model_name=gemma_model_name)
            print("Gemma model loaded successfully!")

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Load all adjectives from file names
        self.adjectives = self._extract_adjectives()

        # Initialize parse failure log
        self.parse_failures = []

        # Initialize mega dataset file path for incremental saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mega_dataset_path = self.output_dir / f"mega_dataset_evaluated_{timestamp}.json"
        self.mega_dataset_initialized = False

        # Initialize parallel batch tracking
        self.active_batches: List[BatchInfo] = []
        self.completed_batches: List[BatchInfo] = []
        self.batch_lock = threading.Lock()

        print(f"Found {len(self.adjectives)} adjectives: {', '.join(sorted(self.adjectives))}")
        if use_local_gemma:
            print(f"Using local Gemma model: {gemma_model_name}")
        else:
            print(f"Using OpenAI model: {model}")

        if resume_batch_id:
            print(f"Will resume from batch ID: {resume_batch_id}")

        if chunk_size:
            print(f"Will process data in chunks of {chunk_size} entries with incremental saving")
            if not use_local_gemma:
                print(f"Will run up to {max_parallel_batches} OpenAI batches in parallel")

    def _extract_adjectives(self) -> List[str]:
        """Extract unique adjectives from JSON filenames."""
        json_files = glob.glob(str(self.data_dir / "*.json"))
        adjectives = set()

        for file_path in json_files:
            filename = Path(file_path).stem
            adjectives.add(filename)

        return sorted(list(adjectives))

    def _log_parse_failure(self, entry_id: str, error_type: str, error_message: str, raw_content: str = None):
        """Log a parse failure for later analysis."""
        failure_entry = {
            "entry_id": entry_id,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "raw_content": raw_content
        }
        self.parse_failures.append(failure_entry)

    def _write_parse_failures_log(self):
        """Write parse failures to log file in the output directory."""
        if not self.parse_failures:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"parse_failures_{timestamp}.log"

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Parse Failures Log - {datetime.now().isoformat()}\n")
            f.write(f"Total failures: {len(self.parse_failures)}\n")
            f.write("=" * 80 + "\n\n")

            for failure in self.parse_failures:
                f.write(f"Entry ID: {failure['entry_id']}\n")
                f.write(f"Error Type: {failure['error_type']}\n")
                f.write(f"Error Message: {failure['error_message']}\n")
                f.write(f"Timestamp: {failure['timestamp']}\n")
                if failure['raw_content']:
                    f.write(f"Raw Content: {failure['raw_content']}\n")
                f.write("-" * 40 + "\n")

        print(f"Logged {len(self.parse_failures)} parse failures to {log_file}")

    def _create_evaluation_prompt(self, text: str, text_type: str, adjectives: List[str]) -> str:
        """
        Create a prompt for the LLM to evaluate how well text fits each adjective.

        Args:
            text: The text to evaluate
            text_type: Either "prompt" or "response"
            adjectives: List of adjectives to rate against

        Returns:
            Formatted prompt for the LLM
        """
        adj_list = ", ".join(adjectives)

        return f"""You are evaluating how well a message fits various adjectives on a scale of 1-5 where:
1 = Does not fit at all
2 = Slightly fits
3 = Moderately fits
4 = Fits well
5 = Fits perfectly

<message>
{text}
</message>

Numerically rate how well this message fits each of these adjectives: {adj_list}

Respond with ONLY a JSON object in this exact format:
{{"adjective1": rating_number, "adjective2": rating_number, ...}}"""

    def _create_batch_requests_for_all_data(self, all_file_data: List[Tuple[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """
        Create batch requests for all data from all files.

        Args:
            all_file_data: List of tuples (source_file, data_entries)

        Returns:
            List of batch request objects
        """
        batch_requests = []

        for source_file, data in all_file_data:
            for i, entry in enumerate(data):
                # Create requests for prompt, topical_response, and optionally vanilla_response
                text_keys = [
                    ("prompt", "prompt"),
                    ("response", "topical_response")
                ]
                if not self.skip_vanilla_responses:
                    text_keys.append(("response", "vanilla_response"))

                for text_type, text_key in text_keys:
                    text = entry[text_key]
                    prompt = self._create_evaluation_prompt(text, text_type, self.adjectives)

                    custom_id = f"{source_file}_{i}_{text_key}"

                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1,
                            "max_tokens": 500
                        }
                    }

                    batch_requests.append(batch_request)

        return batch_requests

    def _create_batch_file(self, batch_requests: List[Dict[str, Any]]) -> str:
        """
        Create a JSONL file for batch processing.

        Args:
            batch_requests: List of batch request objects

        Returns:
            Path to the created batch file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = self.output_dir / f"batch_input_{timestamp}.jsonl"

        with open(batch_file, 'w', encoding='utf-8') as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')

        print(f"Created batch file with {len(batch_requests)} requests: {batch_file}")
        return str(batch_file)

    def _upload_batch_file(self, batch_file_path: str) -> str:
        """
        Upload the batch file to OpenAI.

        Args:
            batch_file_path: Path to the batch file

        Returns:
            File ID from OpenAI
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Cannot upload batch file.")

        print(f"Uploading batch file: {batch_file_path}")

        with open(batch_file_path, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose="batch"
            )

        print(f"Uploaded file with ID: {file_response.id}")
        return file_response.id

    def _submit_batch(self, file_id: str) -> str:
        """
        Submit the batch for processing.

        Args:
            file_id: OpenAI file ID

        Returns:
            Batch ID
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Cannot submit batch.")

        print(f"Submitting batch with file ID: {file_id}")

        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f"Submitted batch with ID: {batch_response.id}")
        return batch_response.id

    def _get_batch_status(self, batch_id: str) -> Any:
        """
        Get the current status of a batch.

        Args:
            batch_id: Batch ID to check

        Returns:
            Batch object
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Cannot get batch status.")

        return self.client.batches.retrieve(batch_id)

    def _wait_for_batch_completion(self, batch_id: str, check_interval: int = 60) -> Any:
        """
        Wait for batch to complete and return the final batch object.

        Args:
            batch_id: Batch ID to monitor
            check_interval: Seconds between status checks

        Returns:
            Final batch object
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Cannot wait for batch completion.")

        print(f"Waiting for batch {batch_id} to complete...")

        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status

            print(f"Batch status: {status}")

            if status == "completed":
                print("Batch completed successfully!")
                return batch
            elif status == "failed":
                raise Exception(f"Batch failed: {batch}")
            elif status == "expired":
                raise Exception(f"Batch expired: {batch}")
            elif status == "cancelled":
                raise Exception(f"Batch was cancelled: {batch}")
            elif status in ["validating", "in_progress", "finalizing"]:
                print(f"Batch still processing... checking again in {check_interval} seconds")
                time.sleep(check_interval)
            else:
                print(f"Unknown batch status: {status}")
                time.sleep(check_interval)

    def _download_batch_results(self, batch: Any) -> List[Dict[str, Any]]:
        """
        Download and parse batch results.

        Args:
            batch: Completed batch object

        Returns:
            List of parsed batch results
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Cannot download batch results.")

        output_file_id = batch.output_file_id
        if not output_file_id:
            raise Exception("No output file ID in completed batch")

        print(f"Downloading results from file ID: {output_file_id}")

        # Download the output file
        file_response = self.client.files.content(output_file_id)
        file_content = file_response.text

        # Parse JSONL results
        results = []
        for line in file_content.strip().split('\n'):
            if line.strip():

                if line.endswith('"slovak": 1'):
                    print("FOUND SLOVAK LINE, hack-fixing...")
                    line += ', "spanish": 1, "supportive": 1, "therapeutic": 1, "title-case": 1}'

                result = json.loads(line)
                results.append(result)

        print(f"Downloaded {len(results)} results")
        return results

    def _parse_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Parse batch results into a structured format.

        Args:
            batch_results: Raw batch results from OpenAI

        Returns:
            Dictionary mapping custom_id to ratings (excludes failed entries)
        """
        parsed_results = {}

        for result in batch_results:
            custom_id = result["custom_id"]

            if result.get("error"):
                self._log_parse_failure(
                    custom_id,
                    "API_ERROR",
                    f"OpenAI API error: {result['error']}"
                )
                continue

            try:
                # Extract the content from the response
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                if content is None:
                    self._log_parse_failure(
                        custom_id,
                        "NONE_CONTENT",
                        "Received None content from OpenAI API"
                    )
                    continue

                content = content.strip()

                # Parse JSON response
                if content.endswith('"slovak": 1'):
                    print("FOUND SLOVAK content, hack-fixing...")
                    content += ', "spanish": 1, "supportive": 1, "therapeutic": 1, "title-case": 1}'

                ratings = json.loads(content)

                # Validate and clip ratings
                valid_ratings = {}
                has_any_valid_rating = False
                for adj in self.adjectives:
                    if adj in ratings:
                        rating = ratings[adj]
                        if isinstance(rating, (int, float)):
                            # Clip rating to 1-5 range
                            clipped_rating = max(1.0, min(5.0, float(rating)))
                            valid_ratings[adj] = clipped_rating
                            has_any_valid_rating = True
                        else:
                            self._log_parse_failure(
                                custom_id,
                                "INVALID_RATING_TYPE",
                                f"Invalid rating type for {adj}: {type(rating)} = {rating}",
                                content
                            )
                    else:
                        self._log_parse_failure(
                            custom_id,
                            "MISSING_ADJECTIVE",
                            f"Missing adjective {adj} in response",
                            content
                        )

                # Only include entry if we have at least some valid ratings
                if has_any_valid_rating:
                    # Fill missing adjectives with None to indicate missing data
                    for adj in self.adjectives:
                        if adj not in valid_ratings:
                            valid_ratings[adj] = None
                    parsed_results[custom_id] = valid_ratings
                else:
                    self._log_parse_failure(
                        custom_id,
                        "NO_VALID_RATINGS",
                        "No valid ratings found in response",
                        content
                    )

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                self._log_parse_failure(
                    custom_id,
                    "JSON_PARSE_ERROR",
                    f"Error parsing JSON: {str(e)}",
                    content if 'content' in locals() else None
                )

        return parsed_results

    def _parse_gemma_response(self, response: str, custom_id: str) -> Dict[str, float]:
        """
        Parse Gemma model response to extract adjective ratings.

        Args:
            response: Raw response from Gemma model
            custom_id: Identifier for this evaluation (for logging failures)

        Returns:
            Dictionary mapping adjective to rating (1-5 scale), or None if parsing fails
        """
        try:
            # Clean up response
            response = remove_model_special_chars(response, self.gemma_model_name).strip()

            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1]
                ratings = json.loads(json_str)

                # Validate and clip ratings
                valid_ratings = {}
                has_any_valid_rating = False
                for adj in self.adjectives:
                    if adj in ratings:
                        rating = ratings[adj]
                        if isinstance(rating, (int, float)):
                            # Clip rating to 1-5 range
                            clipped_rating = max(1.0, min(5.0, float(rating)))
                            valid_ratings[adj] = clipped_rating
                            has_any_valid_rating = True
                        else:
                            self._log_parse_failure(
                                custom_id,
                                "INVALID_RATING_TYPE",
                                f"Invalid rating type for {adj}: {type(rating)} = {rating}",
                                response
                            )
                    else:
                        self._log_parse_failure(
                            custom_id,
                            "MISSING_ADJECTIVE",
                            f"Missing adjective {adj} in response",
                            response
                        )

                # Only return entry if we have at least some valid ratings
                if has_any_valid_rating:
                    # Fill missing adjectives with None to indicate missing data
                    for adj in self.adjectives:
                        if adj not in valid_ratings:
                            valid_ratings[adj] = None
                    return valid_ratings
                else:
                    self._log_parse_failure(
                        custom_id,
                        "NO_VALID_RATINGS",
                        "No valid ratings found in response",
                        response
                    )
                    return None
            else:
                # No valid JSON found
                self._log_parse_failure(
                    custom_id,
                    "NO_VALID_JSON",
                    "No valid JSON found in response",
                    response
                )
                return None

        except (json.JSONDecodeError, ValueError) as e:
            self._log_parse_failure(
                custom_id,
                "JSON_PARSE_ERROR",
                f"Error parsing JSON: {str(e)}",
                response
            )
            return None

    def _evaluate_with_gemma(self, all_file_data: List[Tuple[str, List[Dict[str, Any]]]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all data using local Gemma model.

        Args:
            all_file_data: List of tuples (source_file, data_entries)

        Returns:
            Dictionary mapping custom_id to ratings
        """
        print("Evaluating using local Gemma model...")

        # Prepare all prompts and their custom IDs
        evaluation_prompts = []
        custom_ids = []

        for source_file, data in all_file_data:
            for i, entry in enumerate(data):
                # Create evaluation prompts for prompt, topical_response, and optionally vanilla_response
                text_keys = [
                    ("prompt", "prompt"),
                    ("response", "topical_response")
                ]
                if not self.skip_vanilla_responses:
                    text_keys.append(("response", "vanilla_response"))

                for text_type, text_key in text_keys:
                    text = entry[text_key]
                    prompt = self._create_evaluation_prompt(text, text_type, self.adjectives)
                    custom_id = f"{source_file}_{i}_{text_key}"

                    evaluation_prompts.append(prompt)
                    custom_ids.append(custom_id)

        print(f"Generated {len(evaluation_prompts)} evaluation prompts")

        # Generate responses using Gemma
        batch_size = 32
        responses = optimized_generate_from_string(
            self.gemma_model,
            self.gemma_tokenizer,
            evaluation_prompts,
            batch_size=batch_size,
            max_new_tokens=500,  # Increased from 200 to ensure complete JSON
            #temperature=0.1,
            do_sample=False,
        )

        # Parse responses
        parsed_results = {}
        for custom_id, response in zip(custom_ids, responses):
            ratings = self._parse_gemma_response(response, custom_id)
            if ratings is not None:
                parsed_results[custom_id] = ratings

        return parsed_results

    def _normalize_ratings(self, ratings_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Normalize ratings to 0-1 scale for each adjective.
        Assumes ratings are already clipped to 1-5 range.

        Args:
            ratings_list: List of rating dictionaries (values should be 1-5 or None)

        Returns:
            List of normalized rating dictionaries (values 0-1 or None)
        """
        if not ratings_list:
            return ratings_list

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(ratings_list)

        # Handle None values by keeping them as NaN
        # Only clip non-NaN values to 1-5 range
        clipped_df = df.clip(lower=1.0, upper=5.0)

        # Normalize from 1-5 range to 0-1 range
        # Formula: (value - min) / (max - min) = (value - 1) / (5 - 1) = (value - 1) / 4
        # NaN values will remain NaN
        normalized_df = (clipped_df - 1.0) / 4.0

        # Convert back to list of dictionaries, converting NaN back to None
        result = normalized_df.to_dict('records')
        for entry in result:
            for key, value in entry.items():
                if pd.isna(value):
                    entry[key] = None

        return result

    def _initialize_mega_dataset(self):
        """Initialize the mega dataset file with an empty array."""
        if not self.mega_dataset_initialized:
            with open(self.mega_dataset_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            self.mega_dataset_initialized = True
            print(f"Initialized mega dataset file: {self.mega_dataset_path}")

    def _append_to_mega_dataset(self, chunk_data: List[Dict[str, Any]]):
        """Append chunk data to the mega dataset file."""
        if not chunk_data:
            return

        # Read current data
        if os.path.exists(self.mega_dataset_path):
            with open(self.mega_dataset_path, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
        else:
            current_data = []

        # Append new data
        current_data.extend(chunk_data)

        # Write back to file
        with open(self.mega_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)

        print(f"Appended {len(chunk_data)} entries to mega dataset. Total entries: {len(current_data)}")

    def evaluate_single_file(self, json_file: str) -> List[Dict[str, Any]]:
        """
        Evaluate a single JSON file using batch processing.

        Args:
            json_file: Path to JSON file

        Returns:
            List of enriched data entries
        """
        print(f"Processing {json_file}...")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Apply subset if specified
        if self.subset_size is not None:
            data = data[:self.subset_size]
            print(f"Processing subset of {len(data)} entries from {Path(json_file).stem}")

        source_file = Path(json_file).stem

        # Create batch requests
        batch_requests = self._create_batch_requests_for_all_data([(source_file, data)])

        # Create and upload batch file
        batch_file_path = self._create_batch_file(batch_requests)
        file_id = self._upload_batch_file(batch_file_path)

        # Submit batch
        batch_id = self._submit_batch(file_id)

        # Wait for completion
        completed_batch = self._wait_for_batch_completion(batch_id)

        # Download and parse results
        batch_results = self._download_batch_results(completed_batch)
        parsed_results = self._parse_batch_results(batch_results)

        # Clean up batch file
        os.remove(batch_file_path)

        # Organize results by entry, skipping entries with failed ratings
        prompt_ratings = []
        topical_response_ratings = []
        vanilla_response_ratings = []
        valid_entries = []  # Track which entries have valid ratings

        for i, entry in enumerate(data):
            prompt_custom_id = f"{source_file}_{i}_prompt"
            topical_custom_id = f"{source_file}_{i}_topical_response"
            vanilla_custom_id = f"{source_file}_{i}_vanilla_response"

            prompt_rating = parsed_results.get(prompt_custom_id)
            topical_rating = parsed_results.get(topical_custom_id)
            vanilla_rating = parsed_results.get(vanilla_custom_id) if not self.skip_vanilla_responses else {}

            # Only include entries that have required rating types
            valid_entry = prompt_rating is not None and topical_rating is not None
            if not self.skip_vanilla_responses:
                valid_entry = valid_entry and vanilla_rating is not None

            if valid_entry:
                prompt_ratings.append(prompt_rating)
                topical_response_ratings.append(topical_rating)
                if not self.skip_vanilla_responses:
                    vanilla_response_ratings.append(vanilla_rating)
                valid_entries.append((i, entry))

        # Normalize ratings
        normalized_prompt_ratings = self._normalize_ratings(prompt_ratings)
        normalized_topical_ratings = self._normalize_ratings(topical_response_ratings)
        normalized_vanilla_ratings = self._normalize_ratings(vanilla_response_ratings) if not self.skip_vanilla_responses else []

        # Create enriched entries (only for valid entries)
        enriched_data = []
        for rating_index, (original_index, entry) in enumerate(valid_entries):
            enriched_entry = entry.copy()

            # Add only normalized ratings (0-1 scale)
            enriched_entry["prompt_normalized_ratings"] = normalized_prompt_ratings[rating_index]
            enriched_entry["topical_response_normalized_ratings"] = normalized_topical_ratings[rating_index]
            if not self.skip_vanilla_responses:
                enriched_entry["vanilla_response_normalized_ratings"] = normalized_vanilla_ratings[rating_index]

            # Add metadata
            enriched_entry["source_file"] = source_file
            enriched_entry["evaluation_timestamp"] = datetime.now().isoformat()
            enriched_entry["evaluation_model"] = self.model
            enriched_entry["batch_id"] = batch_id

            enriched_data.append(enriched_entry)

        # Write parse failures log
        self._write_parse_failures_log()

        return enriched_data

    def evaluate_all_files(self) -> List[Dict[str, Any]]:
        """
        Evaluate all JSON files in the data directory using either OpenAI Batch API or local Gemma.

        If chunk_size is specified, processes data in chunks and saves incrementally to mega dataset.
        Otherwise, processes everything together for efficiency.

        Returns:
            Combined list of all enriched data entries (empty if using chunked processing)
        """
        json_files = glob.glob(str(self.data_dir / "*.json"))
        all_file_data = []

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Apply subset if specified
            if self.subset_size is not None:
                data = data[:self.subset_size]
                print(f"Processing subset of {len(data)} entries from {Path(json_file).stem}")

            all_file_data.append((Path(json_file).stem, data))

        # If chunk_size is specified, process in chunks
        if self.chunk_size is not None:
            return self._evaluate_in_chunks(all_file_data)

        # Otherwise, process all at once (original behavior)

        # Choose evaluation method based on configuration
        if self.use_local_gemma:
            # Use local Gemma model
            parsed_results = self._evaluate_with_gemma(all_file_data)
            batch_id = f"gemma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            # Use OpenAI Batch API
            if self.resume_batch_id:
                # Resume from existing batch
                print(f"Resuming from batch ID: {self.resume_batch_id}")
                batch_id = self.resume_batch_id

                # Check batch status
                batch = self._get_batch_status(batch_id)
                print(f"Batch status: {batch.status}")

                if batch.status == "completed":
                    print("Batch already completed, downloading results...")
                    completed_batch = batch
                elif batch.status == "expired":
                    if batch.output_file_id:
                        print("Batch expired but has output file. Downloading available results...")
                        completed_batch = batch
                    else:
                        raise Exception(f"Cannot resume batch {batch_id}: status is {batch.status} and no output file available")
                elif batch.status in ["failed", "cancelled"]:
                    raise Exception(f"Cannot resume batch {batch_id}: status is {batch.status}")
                else:
                    # Wait for completion
                    completed_batch = self._wait_for_batch_completion(batch_id)
            else:
                # Create new batch
                # Create one unified batch request for all data
                batch_requests = self._create_batch_requests_for_all_data(all_file_data)

                # Create and upload batch file
                batch_file_path = self._create_batch_file(batch_requests)
                file_id = self._upload_batch_file(batch_file_path)

                # Submit batch
                batch_id = self._submit_batch(file_id)

                # Wait for completion
                completed_batch = self._wait_for_batch_completion(batch_id)

                # Clean up batch file
                os.remove(batch_file_path)

            # Download and parse results
            batch_results = self._download_batch_results(completed_batch)
            parsed_results = self._parse_batch_results(batch_results)

        # Organize results by entry, skipping entries with failed ratings
        all_enriched_data = []

        # Collect all raw ratings for normalization (only valid entries)
        all_prompt_ratings = []
        all_topical_ratings = []
        all_vanilla_ratings = []
        valid_entries = []  # Track which entries have valid ratings

        for source_file, data in all_file_data:
            for i, entry in enumerate(data):
                prompt_custom_id = f"{source_file}_{i}_prompt"
                topical_custom_id = f"{source_file}_{i}_topical_response"
                vanilla_custom_id = f"{source_file}_{i}_vanilla_response"

                prompt_ratings = parsed_results.get(prompt_custom_id)
                topical_ratings = parsed_results.get(topical_custom_id)
                vanilla_ratings = parsed_results.get(vanilla_custom_id) if not self.skip_vanilla_responses else {}

                # Only include entries that have required rating types
                valid_entry = prompt_ratings is not None and topical_ratings is not None
                if not self.skip_vanilla_responses:
                    valid_entry = valid_entry and vanilla_ratings is not None

                if valid_entry:
                    all_prompt_ratings.append(prompt_ratings)
                    all_topical_ratings.append(topical_ratings)
                    if not self.skip_vanilla_responses:
                        all_vanilla_ratings.append(vanilla_ratings)
                    valid_entries.append((source_file, i, entry))

        # Normalize all ratings
        normalized_prompt_ratings = self._normalize_ratings(all_prompt_ratings)
        normalized_topical_ratings = self._normalize_ratings(all_topical_ratings)
        normalized_vanilla_ratings = self._normalize_ratings(all_vanilla_ratings) if not self.skip_vanilla_responses else []

        # Create enriched entries with normalized ratings (only for valid entries)
        for rating_index, (source_file, entry_index, entry) in enumerate(valid_entries):
            enriched_entry = entry.copy()

            # Add only normalized ratings (0-1 scale)
            enriched_entry["prompt_normalized_ratings"] = normalized_prompt_ratings[rating_index]
            enriched_entry["topical_response_normalized_ratings"] = normalized_topical_ratings[rating_index]
            if not self.skip_vanilla_responses:
                enriched_entry["vanilla_response_normalized_ratings"] = normalized_vanilla_ratings[rating_index]

            # Add metadata
            enriched_entry["source_file"] = source_file
            enriched_entry["evaluation_timestamp"] = datetime.now().isoformat()
            enriched_entry["evaluation_model"] = self.gemma_model_name if self.use_local_gemma else self.model
            enriched_entry["batch_id"] = batch_id

            all_enriched_data.append(enriched_entry)

        # Write parse failures log
        self._write_parse_failures_log()

        return all_enriched_data

    def _submit_batch_for_chunk(self, chunk_index: int, chunk_start: int, chunk_end: int,
                               chunk_file_data: List[Tuple[str, List[Dict[str, Any]]]]) -> BatchInfo:
        """
        Submit a single batch for a chunk and return batch info.

        Args:
            chunk_index: Index of the chunk (0-based)
            chunk_start: Starting entry index
            chunk_end: Ending entry index
            chunk_file_data: Chunk data to process

        Returns:
            BatchInfo object with batch details
        """
        # Create batch requests
        batch_requests = self._create_batch_requests_for_all_data(chunk_file_data)

        # Create and upload batch file
        batch_file_path = self._create_batch_file(batch_requests)
        file_id = self._upload_batch_file(batch_file_path)

        # Submit batch
        batch_id = self._submit_batch(file_id)

        # Create batch info
        batch_info = BatchInfo(
            batch_id=batch_id,
            chunk_index=chunk_index,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            chunk_file_data=chunk_file_data,
            batch_file_path=batch_file_path,
            submitted_at=datetime.now()
        )

        print(f"Submitted batch {batch_id} for chunk {chunk_index + 1} (entries {chunk_start+1}-{chunk_end})")
        return batch_info

    def _wait_for_batch_and_process(self, batch_info: BatchInfo) -> List[Dict[str, Any]]:
        """
        Wait for a batch to complete and process its results.

        Args:
            batch_info: BatchInfo object for the batch to wait for

        Returns:
            List of enriched data entries for this batch
        """
        print(f"Waiting for batch {batch_info.batch_id} (chunk {batch_info.chunk_index + 1}) to complete...")

        # Wait for completion
        completed_batch = self._wait_for_batch_completion(batch_info.batch_id)

        # Download and parse results
        batch_results = self._download_batch_results(completed_batch)
        parsed_results = self._parse_batch_results(batch_results)

        # Clean up batch file
        if batch_info.batch_file_path and os.path.exists(batch_info.batch_file_path):
            os.remove(batch_info.batch_file_path)

        # Process results (same logic as original _evaluate_chunk)
        all_prompt_ratings = []
        all_topical_ratings = []
        all_vanilla_ratings = []
        valid_entries = []

        for source_file, data in batch_info.chunk_file_data:
            for i, entry in enumerate(data):
                prompt_custom_id = f"{source_file}_{i}_prompt"
                topical_custom_id = f"{source_file}_{i}_topical_response"
                vanilla_custom_id = f"{source_file}_{i}_vanilla_response"

                prompt_ratings = parsed_results.get(prompt_custom_id)
                topical_ratings = parsed_results.get(topical_custom_id)
                vanilla_ratings = parsed_results.get(vanilla_custom_id) if not self.skip_vanilla_responses else {}

                # Only include entries that have required rating types
                valid_entry = prompt_ratings is not None and topical_ratings is not None
                if not self.skip_vanilla_responses:
                    valid_entry = valid_entry and vanilla_ratings is not None

                if valid_entry:
                    all_prompt_ratings.append(prompt_ratings)
                    all_topical_ratings.append(topical_ratings)
                    if not self.skip_vanilla_responses:
                        all_vanilla_ratings.append(vanilla_ratings)
                    valid_entries.append((source_file, i, entry))

        # Normalize ratings
        normalized_prompt_ratings = self._normalize_ratings(all_prompt_ratings)
        normalized_topical_ratings = self._normalize_ratings(all_topical_ratings)
        normalized_vanilla_ratings = self._normalize_ratings(all_vanilla_ratings) if not self.skip_vanilla_responses else []

        # Create enriched entries
        chunk_enriched_data = []
        for rating_index, (source_file, entry_index, entry) in enumerate(valid_entries):
            enriched_entry = entry.copy()

            # Add normalized ratings
            enriched_entry["prompt_normalized_ratings"] = normalized_prompt_ratings[rating_index]
            enriched_entry["topical_response_normalized_ratings"] = normalized_topical_ratings[rating_index]
            if not self.skip_vanilla_responses:
                enriched_entry["vanilla_response_normalized_ratings"] = normalized_vanilla_ratings[rating_index]

            # Add metadata
            enriched_entry["source_file"] = source_file
            enriched_entry["evaluation_timestamp"] = datetime.now().isoformat()
            enriched_entry["evaluation_model"] = self.model
            enriched_entry["batch_id"] = batch_info.batch_id
            enriched_entry["chunk_index"] = batch_info.chunk_index
            enriched_entry["chunk_start"] = batch_info.chunk_start

            chunk_enriched_data.append(enriched_entry)

        print(f"Completed batch {batch_info.batch_id} with {len(chunk_enriched_data)} entries")
        return chunk_enriched_data

    def _evaluate_in_chunks(self, all_file_data: List[Tuple[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """
        Evaluate data in chunks with parallel batch processing for OpenAI API.
        For local Gemma, processes sequentially.

        Args:
            all_file_data: List of tuples (source_file, data_entries)

        Returns:
            Empty list (data is saved incrementally to mega dataset file)
        """
        print(f"Processing data in chunks of {self.chunk_size} entries...")

        # Initialize mega dataset file
        self._initialize_mega_dataset()

        # Flatten all data into a single list with source info
        all_entries = []
        for source_file, data in all_file_data:
            for i, entry in enumerate(data):
                all_entries.append((source_file, i, entry))

        print(f"Total entries to process: {len(all_entries)}")

        # Prepare chunks
        chunks = []
        for chunk_start in range(0, len(all_entries), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(all_entries))
            chunk_entries = all_entries[chunk_start:chunk_end]

            # Convert chunk back to the expected format
            chunk_file_data = {}
            for source_file, entry_idx, entry in chunk_entries:
                if source_file not in chunk_file_data:
                    chunk_file_data[source_file] = []
                chunk_file_data[source_file].append((entry_idx, entry))

            # Convert to list format for processing
            chunk_data_list = [(source_file, [entry for _, entry in entries])
                              for source_file, entries in chunk_file_data.items()]

            chunks.append((chunk_start, chunk_end, chunk_data_list))

        if self.use_local_gemma:
            # Sequential processing for Gemma (unchanged behavior)
            for chunk_index, (chunk_start, chunk_end, chunk_data_list) in enumerate(chunks):
                print(f"\nProcessing chunk {chunk_index + 1}: entries {chunk_start+1}-{chunk_end}")
                chunk_enriched_data = self._evaluate_chunk(chunk_data_list, chunk_start)
                self._append_to_mega_dataset(chunk_enriched_data)

                if self.parse_failures:
                    self._write_parse_failures_log()
                    self.parse_failures = []
        else:
            # Parallel batch processing for OpenAI API
            self._evaluate_chunks_parallel(chunks)

        print(f"\nCompleted chunked processing. Results saved to: {self.mega_dataset_path}")
        return []  # Return empty list since data is saved incrementally

    def _evaluate_chunks_parallel(self, chunks: List[Tuple[int, int, List[Tuple[str, List[Dict[str, Any]]]]]]):
        """
        Evaluate chunks in parallel using OpenAI batch API.

        Args:
            chunks: List of (chunk_start, chunk_end, chunk_data_list) tuples
        """
        total_chunks = len(chunks)
        print(f"Processing {total_chunks} chunks with up to {self.max_parallel_batches} parallel batches")

        # Submit initial batches
        pending_batches = []
        completed_chunks = []

        # Submit first batch of chunks
        for chunk_index, (chunk_start, chunk_end, chunk_data_list) in enumerate(chunks[:self.max_parallel_batches]):
            batch_info = self._submit_batch_for_chunk(chunk_index, chunk_start, chunk_end, chunk_data_list)
            pending_batches.append(batch_info)

        remaining_chunks = chunks[self.max_parallel_batches:]
        remaining_chunk_index = self.max_parallel_batches

        # Process batches as they complete
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_batches) as executor:
            # Submit initial batch completion tasks
            future_to_batch = {}
            for batch_info in pending_batches:
                future = executor.submit(self._wait_for_batch_and_process, batch_info)
                future_to_batch[future] = batch_info

            # Process completed batches and submit new ones
            while future_to_batch:
                # Wait for at least one batch to complete
                completed_futures = concurrent.futures.as_completed(future_to_batch, timeout=None)

                for future in completed_futures:
                    batch_info = future_to_batch.pop(future)

                    try:
                        # Get the results and save them
                        chunk_enriched_data = future.result()
                        self._append_to_mega_dataset(chunk_enriched_data)
                        completed_chunks.append(batch_info)

                        print(f"Saved results for chunk {batch_info.chunk_index + 1} "
                              f"({len(completed_chunks)}/{total_chunks} chunks completed)")

                        # Submit next chunk if available
                        if remaining_chunks:
                            chunk_start, chunk_end, chunk_data_list = remaining_chunks.pop(0)
                            new_batch_info = self._submit_batch_for_chunk(
                                remaining_chunk_index, chunk_start, chunk_end, chunk_data_list
                            )

                            # Submit new batch for processing
                            future = executor.submit(self._wait_for_batch_and_process, new_batch_info)
                            future_to_batch[future] = new_batch_info
                            remaining_chunk_index += 1

                        # Write parse failures log for this chunk
                        if self.parse_failures:
                            self._write_parse_failures_log()
                            self.parse_failures = []

                    except Exception as e:
                        print(f"Error processing batch {batch_info.batch_id}: {e}")
                        # Continue with other batches
                        continue

                    break  # Only process one completed future at a time

        print(f"All {total_chunks} chunks processed successfully")

    def _evaluate_chunk(self, chunk_file_data: List[Tuple[str, List[Dict[str, Any]]]], chunk_offset: int) -> List[Dict[str, Any]]:
        """
        Evaluate a single chunk of data (used for sequential Gemma processing).

        Args:
            chunk_file_data: List of tuples (source_file, data_entries) for this chunk
            chunk_offset: Offset for generating unique batch IDs

        Returns:
            List of enriched data entries for this chunk
        """
        # This method is now primarily for Gemma model processing
        # OpenAI batch processing uses the parallel methods above

        if self.use_local_gemma:
            # Use local Gemma model
            parsed_results = self._evaluate_with_gemma(chunk_file_data)
            batch_id = f"gemma_chunk_{chunk_offset//self.chunk_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            # This shouldn't be called for OpenAI API in parallel mode, but handle it anyway
            batch_requests = self._create_batch_requests_for_all_data(chunk_file_data)

            # Create and upload batch file
            batch_file_path = self._create_batch_file(batch_requests)
            file_id = self._upload_batch_file(batch_file_path)

            # Submit batch
            batch_id = self._submit_batch(file_id)

            # Wait for completion
            completed_batch = self._wait_for_batch_completion(batch_id)

            # Download and parse results
            batch_results = self._download_batch_results(completed_batch)
            parsed_results = self._parse_batch_results(batch_results)

            # Clean up batch file
            os.remove(batch_file_path)

        # Organize results (same logic as before)
        all_prompt_ratings = []
        all_topical_ratings = []
        all_vanilla_ratings = []
        valid_entries = []

        for source_file, data in chunk_file_data:
            for i, entry in enumerate(data):
                prompt_custom_id = f"{source_file}_{i}_prompt"
                topical_custom_id = f"{source_file}_{i}_topical_response"
                vanilla_custom_id = f"{source_file}_{i}_vanilla_response"

                prompt_ratings = parsed_results.get(prompt_custom_id)
                topical_ratings = parsed_results.get(topical_custom_id)
                vanilla_ratings = parsed_results.get(vanilla_custom_id) if not self.skip_vanilla_responses else {}

                # Only include entries that have required rating types
                valid_entry = prompt_ratings is not None and topical_ratings is not None
                if not self.skip_vanilla_responses:
                    valid_entry = valid_entry and vanilla_ratings is not None

                if valid_entry:
                    all_prompt_ratings.append(prompt_ratings)
                    all_topical_ratings.append(topical_ratings)
                    if not self.skip_vanilla_responses:
                        all_vanilla_ratings.append(vanilla_ratings)
                    valid_entries.append((source_file, i, entry))

        # Normalize ratings
        normalized_prompt_ratings = self._normalize_ratings(all_prompt_ratings)
        normalized_topical_ratings = self._normalize_ratings(all_topical_ratings)
        normalized_vanilla_ratings = self._normalize_ratings(all_vanilla_ratings) if not self.skip_vanilla_responses else []

        # Create enriched entries
        chunk_enriched_data = []
        for rating_index, (source_file, entry_index, entry) in enumerate(valid_entries):
            enriched_entry = entry.copy()

            # Add normalized ratings
            enriched_entry["prompt_normalized_ratings"] = normalized_prompt_ratings[rating_index]
            enriched_entry["topical_response_normalized_ratings"] = normalized_topical_ratings[rating_index]
            if not self.skip_vanilla_responses:
                enriched_entry["vanilla_response_normalized_ratings"] = normalized_vanilla_ratings[rating_index]

            # Add metadata
            enriched_entry["source_file"] = source_file
            enriched_entry["evaluation_timestamp"] = datetime.now().isoformat()
            enriched_entry["evaluation_model"] = self.gemma_model_name if self.use_local_gemma else self.model
            enriched_entry["batch_id"] = batch_id
            enriched_entry["chunk_offset"] = chunk_offset

            chunk_enriched_data.append(enriched_entry)

        return chunk_enriched_data

    def save_mega_dataset(self, enriched_data: List[Dict[str, Any]]) -> str:
        """
        Save the mega dataset to a JSON file.

        Args:
            enriched_data: List of enriched data entries

        Returns:
            Path to saved file
        """
        # If using chunked processing, the data is already saved incrementally
        if self.chunk_size is not None and self.mega_dataset_initialized:
            print(f"Data already saved incrementally to {self.mega_dataset_path}")
            return str(self.mega_dataset_path)

        # Otherwise, save normally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"mega_dataset_evaluated_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)

        print(f"Saved mega dataset with {len(enriched_data)} entries to {output_file}")
        return str(output_file)

    def create_summary_stats(self, enriched_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary statistics about the evaluation results.

        Args:
            enriched_data: List of enriched data entries (if empty, loads from mega dataset file)

        Returns:
            Dictionary with summary statistics
        """
        # If enriched_data is empty (from chunked processing), load from mega dataset file
        if not enriched_data and self.mega_dataset_initialized and os.path.exists(self.mega_dataset_path):
            with open(self.mega_dataset_path, 'r', encoding='utf-8') as f:
                enriched_data = json.load(f)
            print(f"Loaded {len(enriched_data)} entries from mega dataset for statistics")

        if not enriched_data:
            return {"error": "No data available for statistics"}

        df = pd.DataFrame(enriched_data)

        stats = {
            "total_entries": len(enriched_data),
            "unique_source_files": df["source_file"].nunique(),
            "source_file_counts": df["source_file"].value_counts().to_dict(),
            "adjectives_evaluated": self.adjectives,
            "evaluation_model": self.gemma_model_name if self.use_local_gemma else self.model,
            "evaluation_method": "local_gemma" if self.use_local_gemma else "openai_batch_api",
            "evaluation_timestamp": datetime.now().isoformat(),
            "unique_batch_ids": df["batch_id"].nunique() if "batch_id" in df.columns else 0
        }

        # Calculate average ratings per adjective across all texts
        avg_ratings = {}
        rating_types = ["prompt_normalized_ratings", "topical_response_normalized_ratings"]
        if not self.skip_vanilla_responses:
            rating_types.append("vanilla_response_normalized_ratings")

        for rating_type in rating_types:
            avg_ratings[rating_type] = {}
            for adj in self.adjectives:
                ratings = [entry[rating_type][adj] for entry in enriched_data if rating_type in entry and adj in entry[rating_type]]
                ratings = [r for r in ratings if r is not None]
                print(f"Ratings for {adj}: {ratings}")
                avg_ratings[rating_type][adj] = np.mean(ratings) if ratings else 0.0

        stats["average_normalized_ratings"] = avg_ratings

        # Calculate count of entries with exactly 0 and exactly 1 normalized scores
        extreme_scores = {}
        for rating_type in rating_types:
            extreme_scores[rating_type] = {}
            for adj in self.adjectives:
                ratings = [entry[rating_type][adj] for entry in enriched_data if rating_type in entry and adj in entry[rating_type]]
                ratings = [r for r in ratings if r is not None]
                count_zero = sum(1 for r in ratings if r == 0.0)
                count_one = sum(1 for r in ratings if r == 1.0)
                extreme_scores[rating_type][adj] = {
                    "count_exactly_zero": count_zero,
                    "count_exactly_one": count_one
                }

        stats["extreme_score_counts"] = extreme_scores

        return stats

    def run_full_evaluation(self) -> Tuple[str, str]:
        """
        Run the complete evaluation pipeline using either OpenAI Batch API or local Gemma model.

        Returns:
            Tuple of (mega_dataset_path, stats_path)
        """
        if self.use_local_gemma:
            print("Starting full evaluation pipeline with local Gemma model...")
        else:
            print("Starting full evaluation pipeline with OpenAI Batch API...")

        # Evaluate all files
        enriched_data = self.evaluate_all_files()

        # Save mega dataset
        dataset_path = self.save_mega_dataset(enriched_data)

        # Create and save summary stats
        stats = self.create_summary_stats(enriched_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = self.output_dir / f"evaluation_stats_{timestamp}.json"

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Saved evaluation statistics to {stats_path}")

        return dataset_path, str(stats_path)


def main():
    # Load environment variables from .env file
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Evaluate adjective fit for synthetic data using OpenAI Batch API or local Gemma")
    parser.add_argument("--data_dir", required=True, help="Directory of generate_synthetic_data.py output JSONs to evaluate (e.g. data/self_obfuscation_v1_synthetic/outputs/<timestamp>)")
    parser.add_argument("--output_dir", help="Output directory (default: data/self_obfuscation_v1_synthetic_rating/outputs)")
    parser.add_argument("--model", default="gpt-4.1-nano", help="OpenAI model to use (default: gpt-4.1-nano, ignored if --use-gemma is set)")
    parser.add_argument("--subset", default=-1, type=int, dest="subset_size", help="Process only this many entries per file")
    parser.add_argument("--use-gemma", action="store_true", help="Use local Gemma model instead of OpenAI API")
    parser.add_argument("--gemma-model", default="gemma_2_9b_instruct", help="Name of the Gemma model to load (default: gemma_2_9b_instruct)")
    parser.add_argument("--resume-batch", dest="resume_batch_id", help="Resume processing from an existing batch ID")
    parser.add_argument("--check-batch", dest="check_batch_id", help="Check status of a batch ID and exit")
    parser.add_argument("--chunk-size", type=int, help="Process data in chunks of this size, saving incrementally to mega dataset after each chunk")
    parser.add_argument("--max-parallel-batches", type=int, default=10, help="Maximum number of OpenAI batches to run in parallel when using chunked processing (default: 5)")
    parser.add_argument("--skip-vanilla-responses", action="store_true", help="Skip evaluation of vanilla responses entirely (only evaluate prompts and topical responses)")

    args = parser.parse_args()

    # Check for OpenAI API key only if not using Gemma
    if not args.use_gemma and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in a .env file or as an environment variable:")
        print("  Option 1: Create .env file with: OPENAI_API_KEY=your-key-here")
        print("  Option 2: export OPENAI_API_KEY='your-key-here'")
        print("  Option 3: Use --use-gemma to use local Gemma model instead")
        return 1

    # Validate arguments
    if args.use_gemma and args.resume_batch_id:
        print("Error: --resume-batch cannot be used with --use-gemma")
        print("Batch resumption is only available for OpenAI API processing")
        return 1

    if args.use_gemma and args.check_batch_id:
        print("Error: --check-batch cannot be used with --use-gemma")
        print("Batch checking is only available for OpenAI API processing")
        return 1

    if args.chunk_size is not None and args.resume_batch_id:
        print("Error: --chunk-size cannot be used with --resume-batch")
        print("Chunked processing is not compatible with batch resumption")
        return 1

    if args.chunk_size is not None and args.chunk_size <= 0:
        print("Error: --chunk-size must be a positive integer")
        return 1

    if args.max_parallel_batches <= 0:
        print("Error: --max-parallel-batches must be a positive integer")
        return 1

    # Handle batch status check
    if args.check_batch_id:
        print(f"Checking status of batch: {args.check_batch_id}")
        client = openai.OpenAI()
        try:
            batch = client.batches.retrieve(args.check_batch_id)
            print(f"Batch ID: {batch.id}")
            print(f"Status: {batch.status}")
            print(f"Created: {batch.created_at}")
            print(f"Endpoint: {batch.endpoint}")
            print(f"Completion Window: {batch.completion_window}")
            if hasattr(batch, 'request_counts'):
                counts = batch.request_counts
                print(f"Requests - Total: {counts.total}, Completed: {counts.completed}, Failed: {counts.failed}")
            if batch.output_file_id:
                print(f"Output File ID: {batch.output_file_id}")
            if batch.error_file_id:
                print(f"Error File ID: {batch.error_file_id}")
            return 0
        except Exception as e:
            print(f"Error checking batch status: {e}")
            return 1

    # Initialize evaluator
    evaluator = AdjectiveFitEvaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model=args.model,
        subset_size=args.subset_size if args.subset_size > 0 else None,
        use_local_gemma=args.use_gemma,
        gemma_model_name=args.gemma_model,
        resume_batch_id=args.resume_batch_id,
        chunk_size=args.chunk_size,
        max_parallel_batches=args.max_parallel_batches,
        skip_vanilla_responses=args.skip_vanilla_responses
    )

    # Run evaluation
    try:
        dataset_path, stats_path = evaluator.run_full_evaluation()
        print(f"\n✅ Evaluation complete!")
        print(f"📊 Mega dataset: {dataset_path}")
        print(f"📈 Statistics: {stats_path}")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
