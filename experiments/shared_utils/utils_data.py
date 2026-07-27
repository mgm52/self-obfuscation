"""
Utility functions for data loading and processing in self-obfuscation experiments.

This module consolidates common data loading, processing, and splitting functions
used across probe training and testing scripts.
"""

import json
import os
import random
from typing import List, Tuple, Optional

from datasets import load_dataset


def split_data(data_list: list, test_size: int, seed: int) -> Tuple[list, list]:
    """
    Split a list of data into training and testing sets.

    Args:
        data_list: List of data items to split
        test_size: Number of items to reserve for testing
        seed: Random seed for reproducible splitting

    Returns:
        Tuple of (train_set, test_set)
    """
    if test_size <= 0:
        return data_list, []

    random.Random(seed).shuffle(data_list)
    actual_test_size = min(test_size, len(data_list))

    if actual_test_size < test_size:
        print(f"Warning: Requested test size {test_size}, but only {len(data_list)} samples available. Using {actual_test_size}.")

    test_set = data_list[:actual_test_size]
    train_set = data_list[actual_test_size:]

    return train_set, test_set


def load_ultrachat_sample(num_conversations: int = 100, split: str = "test_gen"):
    """Load a sample from UltraChat dataset."""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    sampled_dataset = dataset.select(range(min(num_conversations, len(dataset))))
    return sampled_dataset


def extract_prompt_response_from_conversation(
    conversation,
    cut_prompt_to_first_and_last_sentence: bool = False,
    cut_response_to_first_sentence: bool = False,
    minimum_response_cut_length: int = -1
) -> Tuple[Optional[str], Optional[str]]:
    """Extract a prompt/response pair from a conversation, with safety checks."""
    if not conversation or len(conversation) < 2:
        return None, None

    prompt_msg, response_msg = conversation[0], conversation[1]
    if not isinstance(prompt_msg, dict) or not isinstance(response_msg, dict):
        return None, None

    prompt, response = prompt_msg.get("content"), response_msg.get("content")
    if not prompt or not response:
        return None, None

    if cut_prompt_to_first_and_last_sentence:
        prompt = cut_to_first_and_last_sentence(prompt)
    if cut_response_to_first_sentence:
        response = cut_to_first_sentence(response, minimum_response_cut_length)

    return prompt, response


def cut_to_first_sentence(text: str, minimum_cut_length: int = -1) -> str:
    """Cut a text to the first sentence that ends after a minimum length."""
    end_chars = '.!?\n'
    start_search = minimum_cut_length if minimum_cut_length > 0 else 0
    first_end_index = -1

    for char in end_chars:
        index = text.find(char, start_search)
        if index != -1 and (first_end_index == -1 or index < first_end_index):
            first_end_index = index

    if first_end_index != -1:
        return text[:first_end_index+1].strip()

    if minimum_cut_length > 0:
        first_end_index_any = -1
        for char in end_chars:
            index = text.find(char)
            if index != -1 and (first_end_index_any == -1 or index < first_end_index_any):
                first_end_index_any = index
        if first_end_index_any != -1:
            return text[:first_end_index_any+1].strip()

    return text.strip()


def cut_to_first_and_last_sentence(text: str) -> str:
    """Cut a text to just the first and last sentences."""
    sentences = []
    start = 0

    for i, char in enumerate(text):
        if char in '.!?\n':
            sentences.append(text[start:i+1].strip())
            start = i+1

    if start < len(text):
        ending = text[start:].strip()
        if ending:
            sentences.append(ending)

    if not sentences:
        return text
    elif len(sentences) == 1:
        return sentences[0]
    else:
        return sentences[0] + " " + sentences[-1]


def load_ultrachat_data(ultrachat_json_path: str, max_samples: int = 10000) -> List[Tuple[str, str]]:
    """Load ultrachat data for probe-irrelevant examples."""
    if not os.path.exists(ultrachat_json_path):
        print(f"Warning: UltraChat file not found at {ultrachat_json_path}")
        return []

    with open(ultrachat_json_path, 'r') as f:
        ultrachat_data = json.load(f)

    # Sample down if too large
    if len(ultrachat_data) > max_samples:
        ultrachat_data = random.Random(42).sample(ultrachat_data, max_samples)

    return [(item['prompt'], item['response']) for item in ultrachat_data if 'prompt' in item and 'response' in item] 
