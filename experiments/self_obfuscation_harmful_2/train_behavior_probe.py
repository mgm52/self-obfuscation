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

from experiments.shared_utils.probe_io import limit_samples
from experiments.shared_utils.utils_misc import load_local_model
from experiments.shared_utils.utils_testing import (
    test_trainable_metric_on_response_data_module,
)
from experiments.shared_utils.utils_training import ResponseProbeDataModule
from obf_reps.logging import CSVTXTLogger
from obf_reps.metrics import (
    AttentionMetric,
    LogisticRegressionMetric,
    MetricConfig,
    MLPMetric,
    SklearnLogisticRegressionMetric,
)

positive_dataset_names = [
    "circuit_breakers_train",
    "circuit_breakers_test",
    "harmful_instructions_train",
    "harmful_instructions_test",
    "wildjailbreak_harmful",
]

negative_dataset_names = [
    "xstest",
    "benign_instructions_train",
    "benign_instructions_test",
    "wildjailbreak_benign",
]


class ProbeConfig(BaseModel):
    type: Literal["logistic", "mlp", "attention", "sklearn_logistic"]
    pooling_type: Literal["mean", "max", "attn", "ema"]
    batch_size: int
    num_epochs: int
    target_layers: list[int]
    learning_rate: float | None = None
    C: float | None = None
    max_samples: int | None = None  # Maximum number of training samples to use

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
        case "attention":
            metric = AttentionMetric(model=model, data_module=data_module, config=config, logger=logger)
        case "sklearn_logistic":
            metric = SklearnLogisticRegressionMetric(
                model=model, data_module=data_module, config=config, logger=logger
            )
        case _:
            raise ValueError(f"Invalid probe type: {probe_type}")

    logger.print("Finished training probe")
    # Freeze probe
    if probe_type in ["logistic", "mlp", "attention"]:
        metric.probe[0].requires_grad_(False)

    return metric


def preprocess_dataset(dataset, model_name):
    """Clean dataset by removing model-specific special tokens.

    Args:
        dataset: List of (prompt, response) tuples
        model_name: Model name to use for token removal (e.g., 'gemma_2_9b_instruct', 'qwen_2_7b_instruct').
    """
    from experiments.shared_utils.utils_tokenizer import clean_model_output

    clean_dataset = []
    for x in dataset:
        prompt = str(x[0])
        response = str(x[1])

        # Remove special tokens using model-agnostic function
        prompt = clean_model_output(prompt, model_name)
        response = clean_model_output(response, model_name)

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


def load_datasets_from_config(config, model_name):
    """Load datasets based on configuration.

    Args:
        config: Configuration object
        model_name: Model name for model-agnostic token removal
    """
    datasets = {}

    # Load training datasets
    for split_name, split_config in config.training_datasets.items():
        datasets[split_name] = {}
        for dataset_type, dataset_names in split_config.items():
            raw_examples = extract_examples_from_datasets(dataset_names)
            datasets[split_name][dataset_type] = preprocess_dataset(raw_examples, model_name=model_name)

    # Load test datasets if provided
    if config.test_datasets:
        for split_name, split_config in config.test_datasets.items():
            datasets[split_name] = {}
            for dataset_type, dataset_names in split_config.items():
                raw_examples = extract_examples_from_datasets(dataset_names)
                datasets[split_name][dataset_type] = preprocess_dataset(raw_examples, model_name=model_name)

    return datasets


def train_and_evaluate_probes(config):
    """Train and evaluate multiple probes with different configurations."""
    # Set up logger
    logger = CSVTXTLogger(print_logs_to_console=True)
    logger.print("Starting probe training and evaluation...")

    # Extract base model name for model-agnostic token removal
    base_model_name = None
    if os.path.isdir(str(config.model_name_or_path)):
        # Try to read from checkpoint metadata
        metadata_path = os.path.join(str(config.model_name_or_path), "model_training_metadata.json")
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    base_model_name = metadata.get('model_name', None)
                    logger.print(f"Extracted base model name from metadata: {base_model_name}")
            except Exception as e:
                logger.print(f"Could not read model metadata: {e}")
    else:
        # Direct model name provided
        base_model_name = str(config.model_name_or_path)
        logger.print(f"Using model name from config: {base_model_name}")

    # Load datasets based on configuration (with model-agnostic token removal)
    datasets = load_datasets_from_config(config, model_name=base_model_name)

    # Print dataset information
    for split_name, split_data in datasets.items():
        logger.print(f"{split_name.upper()} datasets:")
        for dataset_type, examples in split_data.items():
            logger.print(f"  {dataset_type}: {len(examples)} examples")

    # Load model for probe training
    logger.print("Loading model for probe training...")
    if os.path.isdir(str(config.model_name_or_path)):
        logger.print(f"Loading model from checkpoint directory: {config.model_name_or_path}")
        model = load_local_model(checkpoint_path=str(config.model_name_or_path))
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
        train_positive = datasets["train"]["positive"]
        train_negative = datasets["train"]["negative"]

        if probe_config.max_samples is not None:
            logger.print(f"Limiting training samples to {probe_config.max_samples} total samples")
            logger.print(f"Before limiting: {len(train_positive)} positive, {len(train_negative)} negative")
            train_positive, train_negative = limit_samples(
                train_positive, train_negative, max_samples=probe_config.max_samples
            )
            logger.print(f"After limiting: {len(train_positive)} positive, {len(train_negative)} negative")

            # Update metadata to reflect actual training sizes
            metadata["actual_training_sizes"] = {
                "positive": len(train_positive),
                "negative": len(train_negative)
            }

        # Train the probe
        trained_probe = train_behavioral_probe(
            model=model,
            positive_examples=train_positive,
            negative_examples=train_negative,
            logger=logger,
            target_layers=target_layers,
            probe_type=probe_config.type,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

        with open(f"{directory}/metadata.json", "w") as f:
            import json

            json.dump(metadata, f, indent=2)

        trained_probe.save_probe(f"{directory}/probe.pkl")
        logger.print(f"Saved {probe_name} probe to {directory}/probe.pkl")

        # Evaluate on all available datasets
        probe_results = {}
        all_scores = {}

        for split_name, split_data in datasets.items():
            if "positive" in split_data and "negative" in split_data:
                logger.print(f"Evaluating {probe_name} probe on {split_name} set...")

                # Create data module for evaluation
                eval_data_module = ResponseProbeDataModule(
                    topical_prompts_and_responses=split_data["positive"],
                    nontopical_prompts_and_responses=split_data["negative"],
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
    args = parser.parse_args()

    config = ProbeTrainingConfig.from_yaml(args.config)
    results = train_and_evaluate_probes(config)

    # Print final summary
    print("\n=== FINAL SUMMARY ===")
    for probe_name, probe_results in results.items():
        print(f"\n{probe_name} ({probe_results['probe_type']} probe):")
        print(f"  Directory: {probe_results['directory']}")
        for split_name, metrics in probe_results["metrics"].items():
            print(f"  {split_name}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value[0]:.3f} ({value[1]:.3f}-{value[2]:.3f})")
