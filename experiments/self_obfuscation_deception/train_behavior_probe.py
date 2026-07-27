"""
Trains a behavioral probe to distinguish between positive and negative examples.

INPUTS:
- Model checkpoint or model name
- HuggingFace dataset: 'Mechanistic-Anomaly-Detection/gemma2-jailbreaks'

OUTPUTS:
- Trained probe (.pkl file) saved in probe_checkpoints directory
- Training metadata (JSON file) with configuration and dataset sizes
- Console logs with training progress and final checkpoint directory path
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from experiments.shared_utils.dataset_adapters import get_adapter
from pydantic import BaseModel, model_validator
from experiments.shared_utils.utils_plotting import compute_metric_bootstrap

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.shared_utils.utils_misc import load_local_model
from experiments.shared_utils.probe_io import limit_samples
from experiments.shared_utils.utils_testing import (
    test_trainable_metric_on_response_data_module,
)
from experiments.shared_utils.utils_training import ResponseProbeDataModule
from obf_reps.logging import CSVTXTLogger
from obf_reps.metrics import (
    LogisticRegressionMetric,
    MetricConfig,
    MLPMetric,
    SklearnLogisticRegressionMetric,
)


def set_seeds(seed=42):
    """Set all random seeds for reproducible results."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


positive_dataset_names = [
    "circuit_breakers_train",
    "circuit_breakers_test",
    "harmful_instructions_train",
    "harmful_instructions_test",
    "wildjailbreak_harmful",
    "dolus_deceptive_train",
    "dolus_deceptive_test",
    "repe_deceptive_train",
    "repe_deceptive_test",
    "repe_scenario_deceptive_train",
    "repe_scenario_deceptive_test",
]

negative_dataset_names = [
    "xstest",
    "benign_instructions_train",
    "benign_instructions_test",
    "wildjailbreak_benign",
    "dolus_benign_train",
    "dolus_benign_test",
    "repe_honest_train",
    "repe_honest_test",
    "repe_scenario_honest_train",
    "repe_scenario_honest_test",
]


class ProbeConfig(BaseModel):
    type: Literal["logistic", "mlp", "attention", "sklearn_logistic"]
    pooling_type: Literal["mean", "max", "attn", "ema"]
    batch_size: int
    num_epochs: int
    target_layers: list[int]
    learning_rate: float | None = None
    C: float | None = None
    max_samples: int | None = None
    max_test_samples: int | None = None

    @model_validator(mode="after")
    def validate_config(self):
        if self.type == "sklearn_logistic":
            assert self.C is not None, "C must be provided for sklearn_logistic probe"
        if self.type in ["logistic", "mlp", "attention"]:
            assert (
                self.learning_rate is not None
            ), "learning_rate must be provided for logistic and mlp probes"
        return self


class ProbeTrainingConfig(BaseModel):
    model_name_or_path: str | Path
    probes: dict[str, ProbeConfig]  # Dictionary of probe configs
    training_datasets: dict[str, dict[str, list[str]]]
    test_datasets: dict[str, dict[str, list[str]]] | None = None
    max_samples: int | None = None
    max_test_samples: int | None = None

    @model_validator(mode="after")
    def validate_config(self):
        # Validate probes
        assert self.probes, "At least one probe must be specified"
        for probe_name, probe_config in self.probes.items():
            assert probe_config.type in ["logistic", "mlp", "attention", "sklearn_logistic"], (
                f"Invalid probe type '{probe_config.type}' for probe '{probe_name}'. "
                "Valid types are: 'logistic', 'mlp', 'sklearn_logistic'"
            )

        # Validate training datasets
        for split_name, split_config in self.training_datasets.items():
            assert isinstance(
                split_config, dict
            ), f"Training dataset {split_name} must be a dict with 'positive' and 'negative' keys"
            assert set(split_config.keys()) <= {
                "positive",
                "negative",
            }, f"Training dataset {split_name} can only have 'positive' and 'negative' keys, got: {list(split_config.keys())}"

            for dataset_type, datasets in split_config.items():
                if dataset_type == "positive":
                    assert all(
                        dataset in positive_dataset_names for dataset in datasets
                    ), f"Invalid positive dataset in {split_name}: {datasets}. Valid datasets: {positive_dataset_names}"
                elif dataset_type == "negative":
                    assert all(
                        dataset in negative_dataset_names for dataset in datasets
                    ), f"Invalid negative dataset in {split_name}: {datasets}. Valid datasets: {negative_dataset_names}"

        # Validate test datasets if provided
        if self.test_datasets is not None:
            for split_name, split_config in self.test_datasets.items():
                assert isinstance(
                    split_config, dict
                ), f"Test dataset {split_name} must be a dict with 'positive' and 'negative' keys"
                assert set(split_config.keys()) <= {
                    "positive",
                    "negative",
                }, f"Test dataset {split_name} can only have 'positive' and 'negative' keys, got: {list(split_config.keys())}"

                for dataset_type, datasets in split_config.items():
                    if dataset_type == "positive":
                        assert all(
                            dataset in positive_dataset_names for dataset in datasets
                        ), f"Invalid positive dataset in {split_name}: {datasets}. Valid datasets: {positive_dataset_names}"
                    elif dataset_type == "negative":
                        assert all(
                            dataset in negative_dataset_names for dataset in datasets
                        ), f"Invalid negative dataset in {split_name}: {datasets}. Valid datasets: {negative_dataset_names}"

        return self

    @classmethod
    def from_yaml(cls, yaml_path: Path | str):
        assert os.path.exists(yaml_path), f"Config file {yaml_path} does not exist"
        with open(yaml_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return cls(**config)

def train_behavioral_probe(
    model,
    positive_examples,
    negative_examples,
    logger,
    target_layers,
    probe_type="logistic",
    learning_rate=5e-4,
    batch_size=64,
    num_epochs=5,
):
    """
    Train a probe to distinguish between positive and negative examples.

    Args:
        model: The model to extract representations from
        positive_examples: List of (prompt, response) tuples for positive examples
        negative_examples: List of (prompt, response) tuples for negative examples
        logger: Logger for tracking training progress
        target_layers: List of layer indices to probe
        probe_type: Type of probe to use ("logistic" or "mlp")
        learning_rate: Learning rate for probe training
        batch_size: Batch size for probe training
        num_epochs: Number of epochs for probe training

    Returns:
        Trained probe
    """
    logger.print(
        f"Creating data module with {len(positive_examples)} positive and {len(negative_examples)} negative examples..."
    )

    # Create the data module using ResponseProbeDataModule from conditional_training.py
    data_module = ResponseProbeDataModule(
        topical_prompts_and_responses=positive_examples,
        nontopical_prompts_and_responses=negative_examples,
        batch_size=batch_size,
    )

    # Configure which layer(s) to probe
    config = MetricConfig(
        layers=target_layers,
        lr=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # Create the probe
    match probe_type:
        case "logistic":
            metric = LogisticRegressionMetric(
                model=model, data_module=data_module, config=config, logger=logger
            )
        case "mlp":
            metric = MLPMetric(model=model, data_module=data_module, config=config, logger=logger)
        case "sklearn_logistic":
            metric = SklearnLogisticRegressionMetric(
                model=model, data_module=data_module, config=config, logger=logger
            )
        case _:
            raise ValueError(f"Invalid probe type: {probe_type}")

    logger.print("Finished training probe")
    # Freeze probe
    if probe_type in ["logistic", "attention", "mlp"]:
        metric.probe[0].requires_grad_(False)

    return metric


def preprocess_dataset(dataset):
    clean_dataset = []
    for x in dataset:
        prompt = str(x[0])
        response = str(x[1])

        # Remove special tokens
        prompt = prompt.replace("<bos><start_of_turn>user\n", "")
        prompt = prompt.replace("<end_of_turn>\n<start_of_turn>model\n", "")

        clean_dataset.append((prompt, response))

    return clean_dataset


def extract_examples_from_datasets(dataset_names):
    """Extract examples from multiple datasets using appropriate adapters."""
    examples = []

    for dataset_name in dataset_names:
        adapter = get_adapter(dataset_name)
        dataset_examples = adapter.load_and_extract_examples(dataset_name)
        examples.extend(dataset_examples)

    return examples


def load_datasets_from_config(config):
    """Load datasets based on configuration."""
    datasets = {}

    # Load training datasets
    for split_name, split_config in config.training_datasets.items():
        datasets[split_name] = {}
        for dataset_type, dataset_names in split_config.items():
            raw_examples = extract_examples_from_datasets(dataset_names)
            datasets[split_name][dataset_type] = preprocess_dataset(raw_examples)

    # Load test datasets if provided
    if config.test_datasets:
        for split_name, split_config in config.test_datasets.items():
            datasets[split_name] = {}
            for dataset_type, dataset_names in split_config.items():
                raw_examples = extract_examples_from_datasets(dataset_names)
                datasets[split_name][dataset_type] = preprocess_dataset(raw_examples)

    return datasets


def train_and_evaluate_probes(config, skip_testing=True):
    """Train and evaluate multiple probes with different configurations."""
    # Set random seeds for reproducible results
    set_seeds(42)

    # Set up logger
    logger = CSVTXTLogger(print_logs_to_console=True)
    logger.print("Starting probe training and evaluation...")

    # Load datasets based on configuration
    datasets = load_datasets_from_config(config)

    # Print dataset information
    for split_name, split_data in datasets.items():
        logger.print(f"{split_name.upper()} datasets:")
        for dataset_type, examples in split_data.items():
            logger.print(f"  {dataset_type}: {len(examples)} examples")

    # Load model for probe training
    logger.print("Loading model for probe training...")
    if os.path.isdir(str(config.model_name_or_path)):
        logger.print(f"Loading model from checkpoint directory: {config.model_name_or_path}")
        # Read base model name from checkpoint metadata (required)
        metadata_path = os.path.join(str(config.model_name_or_path), "model_training_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Model checkpoint metadata not found: {metadata_path}\n"
                "Cannot determine base model name (gemma_2_9b_instruct, qwen_2_7b_instruct, etc).\n"
                "This checkpoint may be corrupted or from an old version.\n"
                "Please retrain the model organism with proper metadata."
            )

        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Could not read model metadata from {metadata_path}: {e}")

        base_model_name = metadata.get('model_name', None)
        if not base_model_name:
            raise ValueError(
                f"Model metadata missing 'model_name' field in {metadata_path}\n"
                "Cannot determine base model name (gemma_2_9b_instruct, qwen_2_7b_instruct, etc).\n"
                "Please retrain the model organism with proper metadata."
            )

        logger.print(f"Found base model name in metadata: {base_model_name}")
        model = load_local_model(checkpoint_path=str(config.model_name_or_path), model_name=base_model_name)
    else:
        logger.print(f"Loading model by name: {config.model_name_or_path}")
        model = load_local_model(model_name=str(config.model_name_or_path))

    # Results storage
    all_results = {}

    # Train each probe configuration
    for probe_name, probe_config in config.probes.items():
        logger.print(f"\n=== Training {probe_name} ({probe_config.type} probe) ===")

        # Get probe-specific parameters
        learning_rate = probe_config.learning_rate
        batch_size = probe_config.batch_size
        num_epochs = probe_config.num_epochs
        target_layers = probe_config.target_layers

        # Create directory for this probe
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = (
            f"self_obfuscation_experiment/outputs/probe_checkpoints/{date_time}_{probe_name}"
        )
        os.makedirs(directory, exist_ok=True)

        # Save metadata
        metadata = {
            "date_time": date_time,
            "probe_name": probe_name,
            "probe_type": probe_config.type,
            "target_layers": target_layers,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "pooling_type": probe_config.pooling_type,
            "model_name_or_path": str(config.model_name_or_path),
            "dataset_sizes": {
                split: {dtype: len(data) for dtype, data in split_data.items()}
                for split, split_data in datasets.items()
            },
            "dataset_names": {
                split: {dtype: dataset_names for dtype, dataset_names in split_config.items()}
                for split, split_config in config.training_datasets.items()
            },
            "test_dataset_names": {
                split: {dtype: dataset_names for dtype, dataset_names in split_config.items()}
                for split, split_config in (config.test_datasets or {}).items()
            },
            "total_positive_examples": sum(
                len(data)
                for split_data in datasets.values()
                for dtype, data in split_data.items()
                if dtype == "positive"
            ),
            "total_negative_examples": sum(
                len(data)
                for split_data in datasets.values()
                for dtype, data in split_data.items()
                if dtype == "negative"
            ),
            "directory": directory,
        }

        # Apply sample limiting if specified
        # Use probe-specific max_samples if set, otherwise use global config
        max_samples = probe_config.max_samples if probe_config.max_samples is not None else config.max_samples
        pos_examples, neg_examples = limit_samples(
            datasets["train"]["positive"],
            datasets["train"]["negative"],
            max_samples=max_samples
        )

        if max_samples:
            logger.print(f"Limited training data to {len(pos_examples)} positive and {len(neg_examples)} negative samples (max_samples={max_samples})")

        # Train the probe
        trained_probe = train_behavioral_probe(
            model=model,
            positive_examples=pos_examples,
            negative_examples=neg_examples,
            logger=logger,
            target_layers=target_layers,
            probe_type=probe_config.type,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

        # Add sample limiting info to metadata
        metadata["max_samples"] = max_samples
        metadata["actual_positive_samples"] = len(pos_examples)
        metadata["actual_negative_samples"] = len(neg_examples)
        metadata["max_test_samples"] = 0 if skip_testing else (probe_config.max_test_samples if probe_config.max_test_samples is not None else config.max_test_samples)

        with open(f"{directory}/metadata.json", "w") as f:
            import json

            json.dump(metadata, f, indent=2)

        trained_probe.save_probe(f"{directory}/probe.pkl")
        logger.print(f"Saved {probe_name} probe to {directory}/probe.pkl")

        # Evaluate on all available datasets (unless skip_testing is True)
        probe_results = {}
        all_scores = {}

        if not skip_testing:
            for split_name, split_data in datasets.items():
                if "positive" in split_data and "negative" in split_data:
                    logger.print(f"Evaluating {probe_name} probe on {split_name} set...")

                    # Apply test sample limiting if specified
                    # Use probe-specific max_test_samples if set, otherwise use global config
                    max_test_samples = probe_config.max_test_samples if probe_config.max_test_samples is not None else config.max_test_samples
                    eval_pos_examples, eval_neg_examples = limit_samples(
                        split_data["positive"],
                        split_data["negative"],
                        max_samples=max_test_samples
                    )

                    if max_test_samples:
                        logger.print(f"Limited evaluation to {len(eval_pos_examples)} positive and {len(eval_neg_examples)} negative samples (max_test_samples={max_test_samples})")

                    # Create data module for evaluation
                    eval_data_module = ResponseProbeDataModule(
                        topical_prompts_and_responses=eval_pos_examples,
                        nontopical_prompts_and_responses=eval_neg_examples,
                        batch_size=batch_size,
                    )

                    # Get scores
                    scores = test_trainable_metric_on_response_data_module(
                        model=model,
                        metric=trained_probe,
                        logger=logger,
                        data_module=eval_data_module,
                        model_details=metadata,
                        layers_to_probe=target_layers,
                        custom_folder_name=f"{probe_name}_{split_name}",
                        pooling_type=probe_config.pooling_type,
                    )

                    pos_scores = [score["pos_score"] for score in scores]
                    neg_scores = [score["neg_score"] for score in scores]

                    # Store raw scores
                    all_scores[split_name] = {
                        "pos_scores": pos_scores,
                        "neg_scores": neg_scores,
                    }

                    # Compute metrics
                    recall1fpr = compute_metric_bootstrap(
                        "recall@1",
                        positive_scores=pos_scores,
                        negative_scores=neg_scores,
                        confidence_level=0.95,
                    )
                    recall01fpr = compute_metric_bootstrap(
                        "recall@0.01",
                        positive_scores=pos_scores,
                        negative_scores=neg_scores,
                        confidence_level=0.95,
                    )

                    probe_results[split_name] = {
                        "recall@1fpr": recall1fpr,
                        "recall@0.01fpr": recall01fpr,
                    }

                    logger.print(f"  Recall@1%FPR: {recall1fpr}")
                    logger.print(f"  Recall@0.01%FPR: {recall01fpr}")

            # Save detailed results
            with open(f"{directory}/results.json", "w") as f:
                json.dump({"metrics": probe_results, "scores": all_scores}, f, indent=2)

            # Save summary CSV
            with open(f"{directory}/summary.csv", "w") as f:
                f.write("split,metric,value\n")
                for split_name, metrics in probe_results.items():
                    for metric_name, value in metrics.items():
                        f.write(f"{split_name},{metric_name},{value}\n")

            logger.print(f"Saved detailed results to {directory}/")
        else:
            logger.print(f"Skipping evaluation (skip_testing=True)")

        # Store in overall results
        all_results[probe_name] = {
            "probe_type": probe_config.type,
            "directory": directory,
            "metrics": probe_results,
            "scores": all_scores,
        }

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dont-skip-testing", action="store_true", help="Skip evaluation/testing after training")
    args = parser.parse_args()

    config = ProbeTrainingConfig.from_yaml(args.config)
    results = train_and_evaluate_probes(config, skip_testing=not args.dont_skip_testing)

    # Print final summary
    print("\n=== FINAL SUMMARY ===")
    for probe_name, probe_results in results.items():
        print(f"\n{probe_name} ({probe_results['probe_type']} probe):")
        print(f"  Directory: {probe_results['directory']}")
        if probe_results["metrics"]:
            for split_name, metrics in probe_results["metrics"].items():
                print(f"  {split_name}:")
                for metric_name, value in metrics.items():
                    print(f"    {metric_name}: {value[0]:.3f} ({value[1]:.3f}-{value[2]:.3f})")
        else:
            print("  No evaluation metrics (testing was skipped)")
