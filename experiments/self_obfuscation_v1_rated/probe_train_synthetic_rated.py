# probe_train_synthetic_labelled.py

"""
Script: probe_train_synthetic_labelled.py

Trains probes to detect stylistic or topical features (adjectives) in model representations
using labelled mega datasets with normalized ratings.

This script loads data from evaluated mega datasets (e.g., mega_dataset_evaluated_*.json),
filters samples based on normalized ratings thresholds, and trains probes for each adjective.
A sample is considered positive for an adjective if either prompt or response normalized
ratings are >= 0.75, and negative if <= 0.25.

INPUTS:
- Model checkpoint or model name
- Path to evaluated mega dataset JSON file
- Output directory for probes

OUTPUTS:
- Trained probes, one per adjective (.pkl files)
- Training metadata (metadata.json) containing parameters needed for testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


import argparse
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


from experiments.shared_utils.utils_misc import load_local_model, print_memory_usage, cleanup_memory
from data.self_obfuscation_v1_synthetic_rating.load_rated_data import load_rated_data
from data.data import PosNegDataByCategory
from obf_reps.logging import CSVTXTLogger
from obf_reps.models.hf import HFHardPrompted

def load_mega_dataset(dataset_path: str = None, use_topical_ratings: bool = True):
    """
    Loads the evaluated mega dataset using the new data loader.

    Args:
        dataset_path: Path to the mega dataset JSON file (optional, uses most recent if None)
        use_topical_ratings: Whether to use topical_response ratings or vanilla_response ratings

    Returns:
        List of PromptRespRating objects
    """
    response_type = "topical" if use_topical_ratings else "vanilla"

    # If specific path provided, extract just the filename pattern
    if dataset_path:
        dataset_name = Path(dataset_path).name
    else:
        dataset_name = r"mega_dataset_evaluated_.*\.json$"

    return load_rated_data(
        dataset_name=dataset_name,
        response_type=response_type,
        manual_path_confirm=False,  # Don't require manual confirmation in script
        exclude_refusals=True,
        exclude_missing_ratings=True
    )

def create_pos_neg_data(data, positive_threshold: float = 0.75, negative_threshold: float = 0.25) -> PosNegDataByCategory:
    """
    Create PosNegDataByCategory using thresholds.

    Args:
        data: List of PromptRespRating objects
        positive_threshold: Threshold for positive samples (>= this value)
        negative_threshold: Threshold for negative samples (<= this value)

    Returns:
        PosNegDataByCategory object with adjectives as categories
    """
    return PosNegDataByCategory.from_ratings(
        data,
        max_neg_rating=negative_threshold,
        min_pos_rating=positive_threshold
    )

def split_data(data_list: list, test_size: int, seed: int, shuffle: bool=False) -> Tuple[list, list]:
    """Splits a list of data into training and testing sets."""
    if test_size <= 0:
        return data_list, []
    if shuffle:
        random.Random(seed).shuffle(data_list)
    actual_test_size = min(test_size, len(data_list))
    if actual_test_size < test_size:
        print(f"Warning: Requested test size {test_size}, but only {len(data_list)} samples available. Using {actual_test_size}.")
    test_set = data_list[:actual_test_size]
    train_set = data_list[actual_test_size:]
    return train_set, test_set

def train_probe_with_plotting(model: HFHardPrompted, pos_neg_data, logger, target_layers: List[int], probe_type: str = "mlp", learning_rate: float = 5e-4, batch_size: int = 64, num_epochs: int = 5, adjective: str = "unknown"):
    """Train probe using the existing probe training infrastructure."""
    from experiments.shared_utils.utils_training import ResponseProbeDataModule
    from obf_reps.metrics import AttentionMetric, MetricConfig, MLPMetric, LogisticRegressionMetric

    print_memory_usage(f"start of probe training for {adjective}", logger)

    logger.optional_print(f"pos_samples (len: {len(pos_neg_data.pos_dataset)}) (first preview): {pos_neg_data.pos_dataset[0]}")
    logger.optional_print(f"neg_samples (len: {len(pos_neg_data.neg_dataset)}) (first preview): {pos_neg_data.neg_dataset[0]}")

    print_memory_usage(f"after loading samples for {adjective}", logger)

    logger.optional_print(f"Creating data module...")
    # Create the data module
    # Probe trains on input TEXT and response TEXT
    # Filter out any samples with None prompt or response
    pos_samples = [(sample.prompt, sample.response) for sample in pos_neg_data.pos_dataset
                   if sample.prompt is not None and sample.response is not None]
    neg_samples = [(sample.prompt, sample.response) for sample in pos_neg_data.neg_dataset
                   if sample.prompt is not None and sample.response is not None]

    # Log if any samples were filtered out
    if len(pos_samples) < len(pos_neg_data.pos_dataset):
        logger.print(f"WARNING: Filtered out {len(pos_neg_data.pos_dataset) - len(pos_samples)} positive samples with None values")
    if len(neg_samples) < len(pos_neg_data.neg_dataset):
        logger.print(f"WARNING: Filtered out {len(pos_neg_data.neg_dataset) - len(neg_samples)} negative samples with None values")

    data_module = ResponseProbeDataModule(pos_samples, neg_samples)

    logger.optional_print(f"Configuring probe...")
    # Configure which layer(s) to probe
    config = MetricConfig(
        layers=target_layers,
        lr=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    print_memory_usage(f"after creating config for {adjective}", logger)

    # Create the probe (using any of the available architectures)
    # Note: The probe training happens automatically during metric instantiation
    if probe_type == "mlp":
        metric = MLPMetric(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger
        )
    elif probe_type == "logistic":
        metric = LogisticRegressionMetric(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger
        )
    elif probe_type == "attention":
        metric = AttentionMetric(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger
        )

    logger.print(f"Probe training completed for {adjective}")
    print_memory_usage(f"after probe training for {adjective}", logger)

    logger.optional_print(f"\n\nFinished training probe\n\n")

    # Freeze probe
    if hasattr(metric, 'probe') and 0 in metric.probe:
        metric.probe[0].requires_grad_(False)

    print_memory_usage(f"after completing probe training for {adjective}", logger)

    return metric

def train_labelled_probes(
    dataset_path: str,
    output_dir_base: str,
    model_name: str,
    model_checkpoint: str = None,
    target_layers: str = "12",
    probe_type: str = "logistic",
    learning_rate: float = 5e-4,
    batch_size: int = 64,
    num_epochs: int = 5,
    custom_folder_name: str = "",
    seed: int = 42,
    num_test_samples: int = 100,
    positive_threshold: float = 0.75,
    negative_threshold: float = 0.25,
    use_topical_ratings: bool = True,
    include_adjectives: list = [],
    min_samples_per_class: int = 10,
    max_samples_per_class: int = 2048,
    merge_concepts_into_one_probe: list = [],
    negative_concepts_for_merged_probe: list = [],
):
    """Main function to train and save probes on labelled mega dataset."""
    random.seed(seed)
    print(f"Starting probe training...")

    # 1. Setup Logger and Output Directory
    logger = CSVTXTLogger(print_logs_to_console=True)
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.print("="*80)
    logger.print("PROBE TRAINING INITIALIZATION")
    logger.print("="*80)
    logger.print(f"Timestamp: {date_time}")
    logger.print(f"Model: {model_checkpoint or model_name}")
    logger.print(f"Dataset: {dataset_path}")
    logger.print(f"Probe type: {probe_type}")
    logger.print(f"Learning rate: {learning_rate}")
    logger.print(f"Target layers: {target_layers}")
    logger.print(f"Use topical ratings: {use_topical_ratings}")
    logger.print(f"Positive threshold: {positive_threshold}")
    logger.print(f"Negative threshold: {negative_threshold}")

    if custom_folder_name:
        directory = os.path.join(output_dir_base, custom_folder_name)
    else:
        model_suffix = "_finetuned" if model_checkpoint else ""
        layers_str = target_layers.replace(',', '-')
        rating_type = "topical" if use_topical_ratings else "vanilla"
        directory = os.path.join(output_dir_base, f"{date_time}_TRAIN_labelled_{rating_type}_{probe_type}_{layers_str}{model_suffix}")
    os.makedirs(directory, exist_ok=True)

    logger.print(f"\nOutput directory: {directory}")
    logger.print("="*80)

    # 2. Load Model
    print_memory_usage("before loading model", logger)
    model = load_local_model(checkpoint_path=model_checkpoint, model_name=model_name)
    logger.print(f"Loaded model: {model_checkpoint or model_name}")
    print_memory_usage("after loading model", logger)

    # 3. Load and Prepare Mega Dataset
    logger.print(f"\n{'='*60}")
    logger.print("LOADING DATASET")
    logger.print(f"{'='*60}")
    logger.print(f"Dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        logger.print(f"ERROR: Dataset file not found at {dataset_path}")
        return

    print_memory_usage("before loading mega dataset", logger)
    data = load_mega_dataset(dataset_path, use_topical_ratings)
    logger.print(f"Loaded {len(data)} samples from dataset")

    # 4. Create PosNegDataByCategory using thresholds
    logger.print(f"\nCreating pos/neg data splits...")
    logger.print(f"Positive threshold: >= {positive_threshold}")
    logger.print(f"Negative threshold: <= {negative_threshold}")
    pos_neg_data_by_category = create_pos_neg_data(data, positive_threshold, negative_threshold)

    # Get all adjectives or filter by specified ones
    all_adjectives = list(pos_neg_data_by_category.categories.keys())
    logger.print(f"\nAvailable adjectives in dataset: {sorted(all_adjectives)}")
    logger.print(f"Total available adjectives: {len(all_adjectives)}")

    if include_adjectives:
        missing_adjs = [adj for adj in include_adjectives if adj not in all_adjectives]
        if missing_adjs:
            logger.print(f"WARNING: Requested adjectives not found in dataset: {missing_adjs}")
        all_adjectives = [adj for adj in include_adjectives if adj in all_adjectives]
        logger.print(f"Filtered to specified adjectives: {all_adjectives}")
    else:
        logger.print(f"Using all {len(all_adjectives)} adjectives from dataset")

    target_layers_list = [int(layer) for layer in target_layers.split(",")]
    print_memory_usage("after loading mega dataset", logger)

    # Handle merged concept probe case
    if merge_concepts_into_one_probe:
        logger.print(f"\n{'='*60}")
        logger.print("MERGED PROBE TRAINING")
        logger.print(f"{'='*60}")
        logger.print(f"Requested concepts to merge: {merge_concepts_into_one_probe}")

        # Check which concepts are actually available
        available_merge_concepts = [c for c in merge_concepts_into_one_probe if c in all_adjectives]
        missing_merge_concepts = [c for c in merge_concepts_into_one_probe if c not in all_adjectives]

        if missing_merge_concepts:
            logger.print(f"WARNING: Requested concepts not found in dataset: {missing_merge_concepts}")

        if not available_merge_concepts:
            logger.print("ERROR: None of the requested merge concepts are available in the dataset!")
            logger.print(f"Available concepts: {sorted(all_adjectives)}")
            return

        logger.print(f"Available concepts for merging: {available_merge_concepts}")

        # Check if negative concepts are specified and available
        available_negative_concepts = []
        if negative_concepts_for_merged_probe:
            available_negative_concepts = [c for c in negative_concepts_for_merged_probe if c in all_adjectives]
            missing_negative_concepts = [c for c in negative_concepts_for_merged_probe if c not in all_adjectives]

            if missing_negative_concepts:
                logger.print(f"WARNING: Requested negative concepts not found in dataset: {missing_negative_concepts}")

            if available_negative_concepts:
                logger.print(f"Using explicit negative concepts: {available_negative_concepts}")
            else:
                logger.print(f"WARNING: None of the requested negative concepts are available. Using default behavior.")

        # Use the new function to correctly select positive and negative samples
        from data.data import create_merged_pos_neg_data
        logger.print("\nSelecting samples based on merged concept ratings...")
        result = create_merged_pos_neg_data(
            ratings_list=data,
            merge_concepts=available_merge_concepts,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            negative_concepts=available_negative_concepts if available_negative_concepts else None,
            shuffle=False,
            return_stats=True,  # Get detailed statistics
            ignore_generation_categories=False,  # Use generation categories (correct for synthetic data)
            limit=max_samples_per_class  # Apply limit in the data creation function
        )

        # Handle both old and new return formats for backwards compatibility
        if isinstance(result, tuple):
            merged_pos_neg_data, training_stats = result
        else:
            merged_pos_neg_data = result
            training_stats = {}

        merged_pos_samples = merged_pos_neg_data.pos_dataset
        merged_neg_samples = merged_pos_neg_data.neg_dataset

        logger.print(f"\nMerged dataset summary:")
        logger.print(f"  Positive samples (at least one concept >= {positive_threshold}): {len(merged_pos_samples)}")
        logger.print(f"  Negative samples (ALL concepts <= {negative_threshold}): {len(merged_neg_samples)}")

        # Log detailed statistics if available
        if training_stats:
            logger.print(f"\nDetailed statistics:")
            logger.print(f"  Total samples processed: {training_stats.get('total_samples_processed', 0)}")
            logger.print(f"  Skipped (missing ratings): {training_stats.get('skipped_missing_ratings', 0)}")
            logger.print(f"  Skipped (intermediate ratings): {training_stats.get('skipped_intermediate_ratings', 0)}")

            if training_stats.get("rating_distributions"):
                logger.print(f"\n  Rating distributions for merged concepts:")
                for concept, dist in training_stats["rating_distributions"].items():
                    if dist.get("count", 0) > 0:
                        logger.print(f"    {concept}: min={dist['min']:.3f}, avg={dist.get('avg', 0):.3f}, max={dist['max']:.3f} (n={dist['count']})")

            if training_stats.get("samples_by_max_rating_bucket"):
                logger.print(f"\n  Distribution of max ratings across all samples:")
                for bucket in sorted(training_stats["samples_by_max_rating_bucket"].keys()):
                    count = training_stats["samples_by_max_rating_bucket"][bucket]
                    logger.print(f"    [{bucket}]: {count} samples")

        # Check if we have enough samples
        if len(merged_pos_samples) < min_samples_per_class or len(merged_neg_samples) < min_samples_per_class:
            logger.print(f"Insufficient samples for merged probe (need at least {min_samples_per_class} per class).")
            logger.print(f"Positive: {len(merged_pos_samples)}, Negative: {len(merged_neg_samples)}")
            return

        # Train the merged probe
        if available_negative_concepts:
            probe_name = "_".join(merge_concepts_into_one_probe) + "_vs_" + "_".join(available_negative_concepts)
        else:
            probe_name = "_".join(merge_concepts_into_one_probe) + "_vs_others"
        probe = train_probe_with_plotting(
            model=model, pos_neg_data=merged_pos_neg_data, logger=logger,
            target_layers=target_layers_list, probe_type=probe_type,
            learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs,
            adjective=probe_name
        )

        # Save the probe
        logger.print(f"\nSaving merged probe...")
        if hasattr(probe, 'data_module'):
            probe.data_module = None
        if hasattr(probe, 'reps_bank'):
            probe.reps_bank = None
        if hasattr(probe, 'test_reps_bank'):
            probe.test_reps_bank = None
        if hasattr(probe, 'model'):
            probe.model = None

        probe_path = os.path.join(directory, f"merged_probe.pkl")
        with open(probe_path, "wb") as f:
            pickle.dump(probe, f)
        logger.print(f"✓ Saved merged probe to: {probe_path}")

        # Verify the file was saved
        if os.path.exists(probe_path):
            file_size = os.path.getsize(probe_path) / (1024 * 1024)  # MB
            logger.print(f"✓ Probe file size: {file_size:.2f} MB")
        else:
            logger.print(f"ERROR: Probe file was not saved properly!")

        # Save metadata for merged probe with enhanced statistics
        metadata = {
            "train_run_date_time": date_time,
            "model_name": model_name,
            "model_checkpoint": model_checkpoint,
            "dataset_path": dataset_path,
            "target_layers": target_layers_list,
            "probe_type": probe_type,
            "pooling_type": "mean",  # Default pooling type used in training
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "seed": seed,
            "num_test_samples_per_class": num_test_samples,
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold,
            "use_topical_ratings": use_topical_ratings,
            "min_samples_per_class": min_samples_per_class,
            "max_samples_per_class": max_samples_per_class,
            "merge_concepts_into_one_probe": merge_concepts_into_one_probe,
            "negative_concepts_for_merged_probe": negative_concepts_for_merged_probe,
            "actual_negative_concepts_used": available_negative_concepts if available_negative_concepts else "default_behavior",
            "probe_name": probe_name,
            "positive_samples_count": len(merged_pos_neg_data.pos_dataset),
            "negative_samples_count": len(merged_pos_neg_data.neg_dataset),
        }

        # Add detailed training statistics if available
        if training_stats:
            metadata["training_data_statistics"] = {
                "total_samples_processed": training_stats.get("total_samples_processed", 0),
                "positive_samples_selected": training_stats.get("positive_samples", 0),
                "negative_samples_selected": training_stats.get("negative_samples", 0),
                "skipped_missing_ratings": training_stats.get("skipped_missing_ratings", 0),
                "skipped_intermediate_ratings": training_stats.get("skipped_intermediate_ratings", 0),
                "rating_distributions_per_concept": training_stats.get("rating_distributions", {}),
                "positive_sample_adjective_counts": training_stats.get("positive_sample_adjectives", {}),
                "negative_sample_adjective_counts": training_stats.get("negative_sample_adjectives", {}),
                "samples_by_max_rating_bucket": training_stats.get("samples_by_max_rating_bucket", {}),
                "selection_criteria": {
                    "mode": "generation_categories_considered",
                    "positive": f"Generated WITH merged concepts AND at least one concept >= {positive_threshold}",
                    "negative": f"Generated WITHOUT merged concepts AND all concepts <= {negative_threshold}"
                }
            }
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.print(f"\n{'='*60}")
        logger.print(f"MERGED PROBE TRAINING COMPLETED SUCCESSFULLY")
        logger.print(f"Output directory: {directory}")
        logger.print(f"{'='*60}")
        print_memory_usage("final memory usage", logger)
        return  # Exit early for merged probe case

    # 5. Collect Sample Counts for All Adjectives
    sample_counts = {}
    logger.print(f"\nCollecting sample counts for all adjectives...")

    for adjective in all_adjectives:
        if adjective in pos_neg_data_by_category.categories:
            pos_neg_data = pos_neg_data_by_category.categories[adjective]
            pos_count = len(pos_neg_data.pos_dataset)
            neg_count = len(pos_neg_data.neg_dataset)

            sample_counts[adjective] = {
                "positive_samples": pos_count,
                "negative_samples": neg_count,
                "meets_min_threshold": pos_count >= min_samples_per_class and neg_count >= min_samples_per_class,
                "positive_training_samples": pos_count,  # Already balanced by PosNegData
                "negative_training_samples": neg_count
            }
        else:
            sample_counts[adjective] = {
                "positive_samples": 0,
                "negative_samples": 0,
                "meets_min_threshold": False,
                "positive_training_samples": 0,
                "negative_training_samples": 0
            }

    # Save sample counts JSON before training begins
    sample_counts_path = os.path.join(directory, "sample_counts.json")
    with open(sample_counts_path, "w") as f:
        json.dump(sample_counts, f, indent=2)
    logger.print(f"Saved sample counts to: {sample_counts_path}")

    # 6. Train Probes
    trained_adjectives = []
    for i, adjective in enumerate(all_adjectives):
        logger.print(f"\n--- Processing adjective '{adjective}' ({i+1}/{len(all_adjectives)}) ---")

        # Get the PosNegData for this adjective
        if adjective not in pos_neg_data_by_category.categories:
            logger.print(f"No data found for '{adjective}'. Skipping.")
            continue

        pos_neg_data = pos_neg_data_by_category.categories[adjective]
        pos_count = len(pos_neg_data.pos_dataset)
        neg_count = len(pos_neg_data.neg_dataset)

        logger.print(f"Found {pos_count} positive and {neg_count} negative samples for '{adjective}'")

        # Check if we have enough samples
        if pos_count < min_samples_per_class or neg_count < min_samples_per_class:
            logger.print(f"Insufficient samples for '{adjective}' (need at least {min_samples_per_class} per class). Skipping.")
            continue

        # Limit samples if necessary
        if pos_count > max_samples_per_class or neg_count > max_samples_per_class:
            logger.print(f"Limiting samples for '{adjective}' to {max_samples_per_class} per class")
            pos_neg_data.pos_dataset = pos_neg_data.pos_dataset[:max_samples_per_class]
            pos_neg_data.neg_dataset = pos_neg_data.neg_dataset[:max_samples_per_class]

        # Train the probe
        probe = train_probe_with_plotting(
            model=model, pos_neg_data=pos_neg_data, logger=logger,
            target_layers=target_layers_list, probe_type=probe_type,
            learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs,
            adjective=adjective
        )

        # Save the probe
        if hasattr(probe, 'data_module'):
            probe.data_module = None
        if hasattr(probe, 'reps_bank'):
            probe.reps_bank = None
        if hasattr(probe, 'test_reps_bank'):
            probe.test_reps_bank = None
        if hasattr(probe, 'model'):
            probe.model = None

        probe_path = os.path.join(directory, f"{adjective}.pkl")
        with open(probe_path, "wb") as f:
            pickle.dump(probe, f)
        logger.print(f"Saved probe to {probe_path}")

        trained_adjectives.append(adjective)

        # Clean up after each probe to free memory
        del probe, pos_neg_data
        cleanup_memory()
        print_memory_usage(f"after completing probe {i+1}/{len(all_adjectives)} ({adjective})", logger)

    # 7. Save Final Metadata
    metadata = {
        "train_run_date_time": date_time,
        "model_name": model_name,
        "model_checkpoint": model_checkpoint,
        "dataset_path": dataset_path,
        "target_layers": target_layers_list,
        "adjectives_trained": trained_adjectives,
        "probe_type": probe_type,
        "pooling_type": "mean",  # Default pooling type used in training
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "seed": seed,
        "num_test_samples_per_class": num_test_samples,
        "positive_threshold": positive_threshold,
        "negative_threshold": negative_threshold,
        "use_topical_ratings": use_topical_ratings,
        "min_samples_per_class": min_samples_per_class,
        "max_samples_per_class": max_samples_per_class,
        "merge_concepts_into_one_probe": merge_concepts_into_one_probe,
    }
    with open(os.path.join(directory, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.print(f"\nSaved all training metadata to: {directory}")
    logger.print(f"\nFinished training {len(trained_adjectives)} probes.")
    print_memory_usage("final memory usage", logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train probes on labelled mega dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core arguments ---
    parser.add_argument("--dataset_path", type=str, default="data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json", help="Path to the evaluated mega dataset JSON file.")
    parser.add_argument("--output_dir_base", type=str, default="experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints", help="Base directory to save outputs.")
    parser.add_argument("--custom_folder_name", type=str, default="", help="Custom folder name for the output. If not provided, a name is generated.")

    # --- Model arguments ---
    parser.add_argument("--model_name", type=str, default="gemma_2_9b_instruct", help="Base model name.")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to a fine-tuned model checkpoint.")

    # --- Probe training arguments ---
    parser.add_argument("--target_layers", type=str, default="12", help="Comma-separated layer numbers for the probe.")
    parser.add_argument("--probe_type", type=str, default="logistic", choices=["logistic", "mlp", "attention"], help="Type of probe.")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate for probe training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for probe training.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for probe training.")

    # --- Data filtering arguments ---
    parser.add_argument("--positive_threshold", type=float, default=0.75, help="Threshold for positive samples (>= this value).")
    parser.add_argument("--negative_threshold", type=float, default=0.25, help="Threshold for negative samples (<= this value).")
    parser.add_argument("--use_topical_ratings", action="store_true", default=True, help="Use topical response ratings instead of vanilla response ratings.")
    parser.add_argument("--use_vanilla_ratings", dest="use_topical_ratings", action="store_false", help="Use vanilla response ratings instead of topical response ratings.")
    parser.add_argument("--min_samples_per_class", type=int, default=50, help="Minimum number of samples required per class to train a probe.")
    parser.add_argument("--max_samples_per_class", type=int, default=2048, help="Maximum number of samples to use per class for training.")

    # --- Data splitting arguments ---
    parser.add_argument("--num_test_samples", type=int, default=100, help="Number of positive and negative samples to hold out for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility of data splitting.")
    parser.add_argument("--include_adjectives", type=str, nargs="*", help="List of specific adjectives to train probes for. If empty, all adjectives are included.")
    parser.add_argument("--merge_concepts_into_one_probe", type=str, nargs="*", default=[],
                        help="List of concept names to merge into one probe. If non-empty, trains a single probe where the positive set consists of all concepts in this list and the negative set consists of all other concepts.")
    parser.add_argument("--negative_concepts_for_merged_probe", type=str, nargs="*", default=[],
                        help="List of concept names to use as negative class for merged probe. If specified, negative samples will be from these concepts instead of default behavior.")

    args = parser.parse_args()

    train_labelled_probes(
        dataset_path=args.dataset_path,
        output_dir_base=args.output_dir_base,
        model_name=args.model_name,
        model_checkpoint=args.model_checkpoint,
        target_layers=args.target_layers,
        probe_type=args.probe_type,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        custom_folder_name=args.custom_folder_name,
        seed=args.seed,
        num_test_samples=args.num_test_samples,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        use_topical_ratings=args.use_topical_ratings,
        include_adjectives=args.include_adjectives,
        min_samples_per_class=args.min_samples_per_class,
        max_samples_per_class=args.max_samples_per_class,
        merge_concepts_into_one_probe=args.merge_concepts_into_one_probe,
        negative_concepts_for_merged_probe=args.negative_concepts_for_merged_probe,
    )
