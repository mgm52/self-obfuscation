#!/usr/bin/env python3
"""
Create a mega dataset from UltraChat using OpenAI batch API to rate prompts and responses.
This loads UltraChat, uses OpenAI batch API to rate each prompt and response according to
a list of adjectives fetched from filenames in a data directory, then saves the results.
There will be no "topical" responses, just vanilla responses.
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
from datasets import load_dataset


class UltraChatMegaDatasetCreator:
    """Creates a mega dataset from UltraChat with adjective ratings using OpenAI Batch API."""

    def __init__(self, adjectives_data_dir: str, output_dir: Optional[str] = None,
                 model: str = "gpt-4o-mini", num_conversations: int = 1000,
                 resume_batch_id: Optional[str] = None,
                 responses_json: Optional[str] = None):
        """
        Initialize the mega dataset creator.

        Args:
            adjectives_data_dir: Directory containing JSON files with adjectives as filenames
            output_dir: Directory to save the mega dataset (defaults to current dir)
            model: OpenAI model to use for evaluation
            num_conversations: Number of UltraChat conversations to process
            resume_batch_id: If specified, resume processing from an existing batch ID
            responses_json: If given, rate the (prompt, response) pairs in this JSON instead
                of UltraChat's own assistant turns. Use it to rate on-policy responses
                produced by data/synthetic_generation/generate_ultrachat_responses.py.
        """
        self.adjectives_data_dir = Path(adjectives_data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.model = model
        self.num_conversations = num_conversations
        self.resume_batch_id = resume_batch_id
        self.responses_json = responses_json

        self.client = openai.OpenAI()  # Will use OPENAI_API_KEY env var

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Load all adjectives from file names
        self.adjectives = self._extract_adjectives()

        # Initialize parse failure log
        self.parse_failures = []

        print(f"Found {len(self.adjectives)} adjectives: {', '.join(sorted(self.adjectives))}")
        print(f"Using OpenAI model: {model}")
        print(f"Will process {num_conversations} UltraChat conversations")

        if resume_batch_id:
            print(f"Will resume from batch ID: {resume_batch_id}")

    def _extract_adjectives(self) -> List[str]:
        """Extract unique adjectives from JSON filenames."""
        json_files = glob.glob(str(self.adjectives_data_dir / "*.json"))
        adjectives = set()

        for file_path in json_files:
            filename = Path(file_path).stem
            adjectives.add(filename)

        return sorted(list(adjectives))

    def load_ultrachat_sample(self, split="test_gen"):
        """Load a sample from UltraChat dataset."""
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
        # Take a sample of conversations
        sampled_dataset = dataset.select(range(min(self.num_conversations, len(dataset))))
        return sampled_dataset

    def extract_prompt_response_from_conversation(self, conversation,
                                                  cut_prompt_to_first_and_last_sentence=False,
                                                  cut_response_to_first_sentence=False,
                                                  minimum_response_cut_length=-1):
        """Extract prompt and response from a conversation."""
        # First content is prompt, second is response
        prompt = conversation[0]["content"]
        response = conversation[1]["content"]

        if cut_prompt_to_first_and_last_sentence:
            prompt = self.cut_to_first_and_last_sentence(prompt)
        if cut_response_to_first_sentence:
            response = self.cut_to_first_sentence(response, minimum_response_cut_length)
        return prompt, response

    def cut_to_first_sentence(self, text, minimum_cut_length=-1):
        """Cut a text to the first sentence."""
        for i, char in enumerate(text):
            if char in '.!?\n':
                if i > minimum_cut_length and minimum_cut_length > 0:
                    return text[:i+1].strip()
                else:
                    return text[:minimum_cut_length].strip()
        return text  # Return full text if no sentence ending found

    def cut_to_first_and_last_sentence(self, text):
        """Cut a text to just the first and last sentences."""
        sentences = []
        start = 0
        for i, char in enumerate(text):
            if char in '.!?\n':
                sentences.append(text[start:i+1].strip())
                start = i+1

        if start < len(text):
            ending = text[start:].strip()
            if ending != "":
                sentences.append(ending)

        if not sentences:
            return text  # Return full text if no sentence endings found
        elif len(sentences) == 1:
            return sentences[0]  # Return single sentence if only one found
        else:
            return sentences[0] + " " + sentences[-1]  # Return first and last with space between

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
        log_file = self.output_dir / f"ultrachat_parse_failures_{timestamp}.log"

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"UltraChat Parse Failures Log - {datetime.now().isoformat()}\n")
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

    def _create_batch_requests_for_ultrachat_data(self, ultrachat_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create batch requests for UltraChat data.

        Args:
            ultrachat_data: List of ultrachat entries with prompt and response

        Returns:
            List of batch request objects
        """
        batch_requests = []

        for i, entry in enumerate(ultrachat_data):
            # Create requests for prompt and response (no topical response for UltraChat)
            for text_type, text_key in [
                ("prompt", "prompt"),
                ("response", "response")
            ]:
                text = entry[text_key]
                prompt = self._create_evaluation_prompt(text, text_type, self.adjectives)

                custom_id = f"ultrachat_{i}_{text_key}"

                batch_request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 2048
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
        batch_file = self.output_dir / f"ultrachat_batch_input_{timestamp}.jsonl"

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
        print(f"Submitting batch with file ID: {file_id}")

        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f"Submitted batch with ID: {batch_response.id}")
        return batch_response.id

    def _get_batch_status(self, batch_id: str) -> Any:
        """Get the current status of a batch."""
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

    def create_ultrachat_mega_dataset(self) -> List[Dict[str, Any]]:
        """
        Create mega dataset from UltraChat conversations with adjective ratings.

        Returns:
            List of enriched UltraChat entries
        """
        if self.responses_json:
            # Rate pre-generated responses (e.g. sampled on-policy from the base model)
            # instead of UltraChat's own assistant turns.
            print(f"Loading pre-generated responses from: {self.responses_json}")
            with open(self.responses_json, "r", encoding="utf-8") as f:
                ultrachat_data = json.load(f)
            if self.num_conversations:
                ultrachat_data = ultrachat_data[: self.num_conversations]
            missing = [k for k in ("prompt", "response") if k not in (ultrachat_data[0] or {})]
            if missing:
                raise ValueError(
                    f"{self.responses_json} records are missing required key(s) {missing}; "
                    "expected {prompt, response, conversation_idx, source} as written by "
                    "data/synthetic_generation/generate_ultrachat_responses.py"
                )
            for i, entry in enumerate(ultrachat_data):
                entry.setdefault("conversation_idx", i)
                entry.setdefault("source", "pregenerated")
            print(f"Loaded {len(ultrachat_data)} pre-generated responses")
            return self._rate_ultrachat_data(ultrachat_data)

        print("Loading UltraChat dataset...")

        # Load UltraChat conversations
        dataset = self.load_ultrachat_sample(split="train_gen")

        # Process each conversation
        print(f"Loaded {len(dataset)} conversations")
        ultrachat_data = []
        for conversation_idx, item in enumerate(tqdm(dataset, desc="Processing conversations")):
            # Extract messages from the conversation
            conversation = item['messages']
            if len(conversation) < 2:
                continue

            prompt, response = self.extract_prompt_response_from_conversation(
                conversation,
                cut_prompt_to_first_and_last_sentence=True,
                cut_response_to_first_sentence=True,
                minimum_response_cut_length=100
            )

            ultrachat_data.append({
                "prompt": prompt,
                "response": response,  # This is the vanilla response
                "conversation_idx": conversation_idx,
                "source": "ultrachat"
            })

        print(f"Processed {len(ultrachat_data)} valid conversations")

        return self._rate_ultrachat_data(ultrachat_data)

    def _rate_ultrachat_data(self, ultrachat_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rate a list of {prompt, response, ...} records against the adjective set."""
        # Choose evaluation method based on configuration
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
            batch_requests = self._create_batch_requests_for_ultrachat_data(ultrachat_data)

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
        all_prompt_ratings = []
        all_response_ratings = []
        valid_entries = []  # Track which entries have valid ratings

        for i, entry in enumerate(ultrachat_data):
            prompt_custom_id = f"ultrachat_{i}_prompt"
            response_custom_id = f"ultrachat_{i}_response"

            prompt_ratings = parsed_results.get(prompt_custom_id)
            response_ratings = parsed_results.get(response_custom_id)

            # Only include entries that have both rating types
            if prompt_ratings is not None and response_ratings is not None:
                all_prompt_ratings.append(prompt_ratings)
                all_response_ratings.append(response_ratings)
                valid_entries.append((i, entry))

        # Normalize all ratings
        normalized_prompt_ratings = self._normalize_ratings(all_prompt_ratings)
        normalized_response_ratings = self._normalize_ratings(all_response_ratings)

        # Create enriched entries with normalized ratings (only for valid entries)
        enriched_data = []
        for rating_index, (original_index, entry) in enumerate(valid_entries):
            enriched_entry = entry.copy()

            # Add only normalized ratings (0-1 scale)
            enriched_entry["prompt_normalized_ratings"] = normalized_prompt_ratings[rating_index]
            enriched_entry["response_normalized_ratings"] = normalized_response_ratings[rating_index]

            # Add metadata
            enriched_entry["evaluation_timestamp"] = datetime.now().isoformat()
            enriched_entry["evaluation_model"] = self.model
            enriched_entry["batch_id"] = batch_id
            enriched_entry["adjectives_evaluated"] = self.adjectives

            enriched_data.append(enriched_entry)

        # Write parse failures log
        self._write_parse_failures_log()

        return enriched_data

    def save_mega_dataset(self, enriched_data: List[Dict[str, Any]]) -> str:
        """
        Save the mega dataset to a JSON file.

        Args:
            enriched_data: List of enriched data entries

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"ultrachat_mega_dataset_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)

        print(f"Saved UltraChat mega dataset with {len(enriched_data)} entries to {output_file}")
        return str(output_file)

    def create_summary_stats(self, enriched_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary statistics about the evaluation results.

        Args:
            enriched_data: List of enriched data entries

        Returns:
            Dictionary with summary statistics
        """
        df = pd.DataFrame(enriched_data)

        stats = {
            "total_entries": len(enriched_data),
            "source": "ultrachat",
            "adjectives_evaluated": self.adjectives,
            "evaluation_model": self.model,
            "evaluation_timestamp": datetime.now().isoformat(),
            "unique_batch_ids": df["batch_id"].nunique() if "batch_id" in df.columns else 0
        }

        # Calculate average ratings per adjective across all texts
        avg_ratings = {}
        for rating_type in ["prompt_normalized_ratings", "response_normalized_ratings"]:
            avg_ratings[rating_type] = {}
            for adj in self.adjectives:
                ratings = [entry[rating_type][adj] for entry in enriched_data if adj in entry[rating_type]]
                ratings = [r for r in ratings if r is not None]
                avg_ratings[rating_type][adj] = np.mean(ratings) if ratings else 0.0

        stats["average_normalized_ratings"] = avg_ratings

        # Calculate count of entries with exactly 0 and exactly 1 normalized scores
        extreme_scores = {}
        for rating_type in ["prompt_normalized_ratings", "response_normalized_ratings"]:
            extreme_scores[rating_type] = {}
            for adj in self.adjectives:
                ratings = [entry[rating_type][adj] for entry in enriched_data if adj in entry[rating_type]]
                ratings = [r for r in ratings if r is not None]
                count_zero = sum(1 for r in ratings if r == 0.0)
                count_one = sum(1 for r in ratings if r == 1.0)
                extreme_scores[rating_type][adj] = {
                    "count_exactly_zero": count_zero,
                    "count_exactly_one": count_one
                }

        stats["extreme_score_counts"] = extreme_scores

        return stats

    def run_full_pipeline(self) -> Tuple[str, str]:
        """
        Run the complete UltraChat mega dataset creation pipeline.

        Returns:
            Tuple of (mega_dataset_path, stats_path)
        """
        print("Starting UltraChat mega dataset creation pipeline...")

        # Create mega dataset from UltraChat
        enriched_data = self.create_ultrachat_mega_dataset()

        # Save mega dataset
        dataset_path = self.save_mega_dataset(enriched_data)

        # Create and save summary stats
        stats = self.create_summary_stats(enriched_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = self.output_dir / f"ultrachat_evaluation_stats_{timestamp}.json"

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Saved evaluation statistics to {stats_path}")

        return dataset_path, str(stats_path)


def main():
    # Load environment variables from .env file
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Create UltraChat mega dataset with adjective ratings using OpenAI Batch API")
    parser.add_argument("--adjectives_data_dir", required=True,
                       help="Directory containing JSON files with adjectives as filenames "
                            "(output of generate_synthetic_data.py, e.g. data/synthetic_generation/outputs/<timestamp>)")
    parser.add_argument("--responses_json", default=None,
                       help="Rate the (prompt, response) pairs in this JSON instead of UltraChat's "
                            "own assistant turns. Produce one with "
                            "data/synthetic_generation/generate_ultrachat_responses.py to use "
                            "on-policy base-model responses as behaviour-preservation targets.")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory (default: /outputs)")
    parser.add_argument("--model", default="gpt-4.1-nano", help="OpenAI model to use (default: gpt-4.1-nano)")
    parser.add_argument("--num_conversations", default=2048, type=int, help="Number of UltraChat conversations to process")
    parser.add_argument("--resume-batch", dest="resume_batch_id", help="Resume processing from an existing batch ID")
    parser.add_argument("--check-batch", dest="check_batch_id", help="Check status of a batch ID and exit")

    args = parser.parse_args()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in a .env file or as an environment variable:")
        print("  Option 1: Create .env file with: OPENAI_API_KEY=your-key-here")
        print("  Option 2: export OPENAI_API_KEY='your-key-here'")
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

    # Initialize creator
    creator = UltraChatMegaDatasetCreator(
        adjectives_data_dir=args.adjectives_data_dir,
        output_dir=args.output_dir,
        model=args.model,
        num_conversations=args.num_conversations,
        resume_batch_id=args.resume_batch_id,
        responses_json=args.responses_json
    )

    # Run pipeline
    try:
        dataset_path, stats_path = creator.run_full_pipeline()
        print(f"\n✅ UltraChat mega dataset creation complete!")
        print(f"📊 Mega dataset: {dataset_path}")
        print(f"📈 Statistics: {stats_path}")

    except Exception as e:
        print(f"❌ Error during UltraChat mega dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
