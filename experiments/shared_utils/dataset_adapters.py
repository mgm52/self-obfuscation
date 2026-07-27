"""
Dataset adapters for flexible dataset loading in probe training.

This module provides a simple adapter system that allows custom logic
for loading and extracting examples from different dataset formats.
"""

import os
import random
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from datasets import load_dataset
from data.apollo_repe_deception.repe import RepeHonestDishonestFacts, Label


class DatasetAdapter:
    """Base class for dataset adapters."""

    def __init__(self, dataset_source: str, split: str | None = None, **kwargs):
        """
        Initialize the adapter with dataset source information.

        Args:
            dataset_source: HuggingFace dataset path or local path
            split: Dataset split to load (if applicable)
            **kwargs: Additional arguments for dataset loading
        """
        self.dataset_source = dataset_source
        self.split = split
        self.kwargs = kwargs

    @abstractmethod
    def load_and_extract_examples(self, dataset_name: str) -> List[Tuple[str, str]]:
        pass

    @abstractmethod
    def extract_examples(
        self, dataset: Any, dataset_name: str
    ) -> List[Tuple[str, str]]:
        pass


class Gemma2JailbreaksAdapter(DatasetAdapter):
    """Specialized adapter for the Gemma2 jailbreaks dataset that loads once and provides subset access."""

    def __init__(self, subset_name: str):
        """Initialize with the specific subset name to extract."""
        super().__init__("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")
        self.subset_name = subset_name
        self._cached_dataset = None

    def load_and_extract_examples(self, dataset_name: str) -> List[Tuple[str, str]]:
        """Load the full jailbreaks dataset once and extract from the specified subset."""
        # Load the full dataset only once
        if self._cached_dataset is None:
            self._cached_dataset = load_dataset(self.dataset_source)

        # Extract from the specific subset
        if self.subset_name in self._cached_dataset:
            dataset_subset = self._cached_dataset[self.subset_name]
            return self.extract_examples(dataset_subset, dataset_name)
        else:
            raise ValueError(
                f"Subset '{self.subset_name}' not found in jailbreaks dataset. Available subsets: {list(self._cached_dataset.keys())}"
            )

    def extract_examples(
        self, dataset: Any, dataset_name: str
    ) -> List[Tuple[str, str]]:
        """Extract examples using the default prompt/completion format."""
        examples = []
        for item in dataset:
            prompt = item.get("prompt", "")
            completion = item.get("completion", "")
            examples.append((prompt, completion))
        return examples


class DolusDatasetAdapter(DatasetAdapter):
    """Adapter for the Dolus dataset."""

    def __init__(self, label: str):
        super().__init__("AlignmentResearch/DolusChat")
        self.label = label
        self._cached_dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def _ensure_dataset_loaded(self):
        """Lazy load the dataset and split it into train/test."""
        if self._cached_dataset is None:
            self._cached_dataset = load_dataset(self.dataset_source)

            dataset_subset = self._cached_dataset["train"]
            train_subset_size = int(len(dataset_subset) * 0.8)
            shuffled_dataset = dataset_subset.shuffle(seed=42)
            self.train_dataset = shuffled_dataset.select(range(train_subset_size))
            self.test_dataset = shuffled_dataset.select(range(train_subset_size, len(dataset_subset)))

    def load_and_extract_examples(self, dataset_name: str) -> List[Tuple[str, str]]:
        """Load the full Dolus dataset once and extract from the specified subset."""
        self._ensure_dataset_loaded()

        match self.label:
            case "deceptive_train":
                return self.extract_examples(self.train_dataset, deceptive=True)
            case "deceptive_test":
                return self.extract_examples(self.test_dataset, deceptive=True)
            case "benign_train":
                return self.extract_examples(self.train_dataset, deceptive=False)
            case "benign_test":
                return self.extract_examples(self.test_dataset, deceptive=False)
            case _:
                raise ValueError(f"Invalid label: {self.label}")

    def extract_examples(self, dataset: Any, deceptive: bool) -> List[Tuple[str, str]]:
        """Extract examples using the default prompt/completion format."""

        def compose_user_message(message, speaker_type):
            return f"\n\n User ID: {speaker_type}\n{message}"

        def compose_context_str(sensitive_info: str, non_sensitive_infos: Any):
            contexts = non_sensitive_infos
            contexts += [sensitive_info]  # type: ignore
            random.Random(42).shuffle(contexts)
            return "\n".join(contexts)

        def compose_system_message(sample: dict):
            context_str = compose_context_str(
                sample["context"]["sensitive_info"],
                sample["context"]["non_sensitive_info"],
            )
            return f"Prompt: {sample['system_message']}\nContext: {context_str}\n"

        examples = []
        for item in dataset:
            system_message = compose_system_message(item)
            prompt = compose_user_message(
                item["user_query"]["content"],
                item["user_query"]["speaker_type"],
            )
            prompt = system_message + "\n\n" + prompt

            if deceptive:
                completion = item["responses"]["deceptive"]
            else:
                completion = item["responses"]["truthful"]

            examples.append((prompt, completion))
        return examples


class WildJailbreakAdapter(DatasetAdapter):
    """Adapter for the Wild Jailbreak dataset from local CSV file.

    NOTE: The CSV this adapter points at (datasets/harmfulness/wildjailbreak_train.csv)
    is not distributed with this repository because WildJailbreak is a gated dataset.
    To use this adapter, download allenai/wildjailbreak from HuggingFace and place the
    train CSV at that path.
    """

    def __init__(self, data_type: str):
        """
        Initialize with data_type filtering.

        Args:
            data_type: "adversarial_benign" or "adversarial_harmful"
        """
        # Get path relative to this file: parent.parent.data/harmfulness/wildjailbreak_train.csv
        current_file = Path(__file__)
        csv_path = (
            current_file.parent.parent.parent
            / "datasets"
            / "harmfulness"
            / "wildjailbreak_train.csv"
        )
        super().__init__(str(csv_path))
        self.data_type = data_type
        self._cached_dataset = None

    def load_and_extract_examples(self, dataset_name: str) -> List[Tuple[str, str]]:
        """Load the wildjailbreak CSV dataset and extract examples based on data_type."""
        # Load the CSV file only once
        if self._cached_dataset is None:
            if not os.path.exists(self.dataset_source):
                raise FileNotFoundError(f"CSV file not found: {self.dataset_source}")
            self._cached_dataset = pd.read_csv(self.dataset_source)

        # Filter by data_type and extract examples
        return self.extract_examples(self._cached_dataset, dataset_name)

    def extract_examples(
        self, dataset: Any, dataset_name: str
    ) -> List[Tuple[str, str]]:
        """Extract examples filtering by data_type (adversarial_benign or adversarial_harmful)."""
        examples = []

        # Filter pandas DataFrame by data_type
        filtered_df = dataset[dataset["data_type"] == self.data_type]

        for _, row in filtered_df.iterrows():
            prompt = row.get("adversarial", "")
            completion = row.get("completion", "")
            examples.append((prompt, completion))

        print("WARNING: Returning only 3000 examples")
        return examples[:3000]


class RepEHonestyAdapter(DatasetAdapter):
    """Adapter for RepE Honest/Dishonest Facts dataset using RepeHonestDishonestFacts class."""

    def __init__(self, split: str, label: int):
        """
        Initialize with split and label.

        Args:
            split: "train" or "test"
            label: 0 for truthful, 1 for deceptive
        """
        super().__init__("repe_honesty")  # Using the base_name from RepeHonestDishonestFacts
        self.split = split
        self.label = label
        self._cached_dataset = None
        self.train_dialogues = None
        self.test_dialogues = None
        self.train_labels = None
        self.test_labels = None

    def _ensure_dataset_loaded(self):
        """Lazy load the dataset using RepeHonestDishonestFacts and split it into train/test."""
        if self._cached_dataset is None:
            # Create instance of RepeHonestDishonestFacts with default variant
            self._cached_dataset = RepeHonestDishonestFacts(variant="you_are_fact_sys") # default variant

            # Get all dialogues and labels
            all_dialogues = self._cached_dataset.dialogues
            all_labels = self._cached_dataset.labels

            # Create 80/20 train/test split
            total_size = len(all_dialogues)
            train_size = int(total_size * 0.8)

            # Split the data
            self.train_dialogues = all_dialogues[:train_size]
            self.test_dialogues = all_dialogues[train_size:]
            self.train_labels = all_labels[:train_size]
            self.test_labels = all_labels[train_size:]

    def load_and_extract_examples(self, dataset_name: str) -> List[Tuple[str, str]]:
        """Load dataset and extract examples based on split and label."""
        self._ensure_dataset_loaded()

        if self.split == "train":
            dialogues = self.train_dialogues
            labels = self.train_labels
        else:
            dialogues = self.test_dialogues
            labels = self.test_labels

        return self.extract_examples((dialogues, labels), dataset_name)

    def extract_examples(
        self, dataset: Any, dataset_name: str
    ) -> List[Tuple[str, str]]:
        """Extract examples filtering by label (0=truthful/honest, 1=deceptive)."""
        dialogues, labels = dataset
        examples = []

        # Map our label (0/1) to Label enum
        target_label = Label.HONEST if self.label == 0 else Label.DECEPTIVE

        for dialogue, label in zip(dialogues, labels):
            if label == target_label:
                # Build the prompt from the dialogue messages
                prompt_parts = []
                completion = ""

                for message in dialogue:
                    if message.role == "system":
                        # Include system message as part of prompt
                        prompt_parts.append(f"System: {message.content}")
                    elif message.role == "user":
                        # Include user message as part of prompt
                        prompt_parts.append(f"User: {message.content}")
                    elif message.role == "assistant":
                        # The assistant messages form the completion
                        completion += message.content

                # Join prompt parts
                prompt = "\n".join(prompt_parts) if prompt_parts else ""

                if not completion:
                    raise Exception(f"No completion present for prompt {prompt}")

                examples.append((prompt, completion))

        return examples


# Registry mapping dataset names to their adapters
DATASET_ADAPTERS: Dict[str, DatasetAdapter] = {
    # Harmfulness datasets (from Gemma2 jailbreaks dataset)
    "circuit_breakers_train": Gemma2JailbreaksAdapter("circuit_breakers_train"),
    "circuit_breakers_test": Gemma2JailbreaksAdapter("circuit_breakers_test"),
    "harmful_instructions_train": Gemma2JailbreaksAdapter("harmful_instructions_train"),
    "harmful_instructions_test": Gemma2JailbreaksAdapter("harmful_instructions_test"),
    "xstest": Gemma2JailbreaksAdapter("xstest"),
    "benign_instructions_train": Gemma2JailbreaksAdapter("benign_instructions_train"),
    "benign_instructions_test": Gemma2JailbreaksAdapter("benign_instructions_test"),
    # Wild Jailbreak datasets (only train subset, filtered by data_type)
    "wildjailbreak_harmful": WildJailbreakAdapter("adversarial_harmful"),
    "wildjailbreak_benign": WildJailbreakAdapter("adversarial_benign"),
    # Deception datasets (Dolus)
    "dolus_deceptive_train": DolusDatasetAdapter("deceptive_train"),
    "dolus_deceptive_test": DolusDatasetAdapter("deceptive_test"),
    "dolus_benign_train": DolusDatasetAdapter("benign_train"),
    "dolus_benign_test": DolusDatasetAdapter("benign_test"),
    # RepE Honesty datasets (DeceptionLabelsFinal - 16k scientific Q&As)
    "repe_honest_train": RepEHonestyAdapter("train", label=0),
    "repe_deceptive_train": RepEHonestyAdapter("train", label=1),
    "repe_honest_test": RepEHonestyAdapter("test", label=0),
    "repe_deceptive_test": RepEHonestyAdapter("test", label=1),
}


def get_adapter(dataset_name: str) -> DatasetAdapter:
    """
    Get the appropriate adapter for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        DatasetAdapter instance
    """
    if dataset_name not in DATASET_ADAPTERS:
        raise ValueError(f"Dataset {dataset_name} not found in registry")
    return DATASET_ADAPTERS[dataset_name]


def register_adapter(dataset_name: str, adapter: DatasetAdapter) -> None:
    """
    Register a custom adapter for a dataset.

    Args:
        dataset_name: Name of the dataset
        adapter: DatasetAdapter instance
    """
    DATASET_ADAPTERS[dataset_name] = adapter
