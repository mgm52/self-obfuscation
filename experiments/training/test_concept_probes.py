# probe_test_synthetic_clued_labelled.py

"""
Script: probe_test_synthetic_clued_labelled.py

Evaluate concept probes on labelled test data with and without probe awareness.

This script combines functionality from probe_test_synthetic_clued.py and
probe_test_synthetic_labelled.py to test each concept probe on:
1. Regular labelled test data for that concept
2. Labelled test data with "You are being probed for <concept>" prepended
3. UltraChat negative samples for out-of-distribution testing

Uses proper data loading functions from data/data.py and
data/synthetic_rating/load_rated_data.py.

INPUTS:
- Trained concept probes from outputs/probe_checkpoints/
- Labelled rated data loaded via load_rated_data()
- UltraChat dataset (optional, for OOD testing)

OUTPUTS:
- Test results CSV files with probe awareness comparison
- Bar plots showing probe performance
- Comparison of regular vs probe-aware performance
- Bootstrapped confidence intervals and example data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


import argparse
import json
import os
import pickle
import random
import gc
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Try to import pandas, but make it optional
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Try to import matplotlib, but make plotting optional
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Set seaborn style and context for paper
    sns.set_context("paper")
    sns.set_style("darkgrid")
    # Enable constrained layout for better spacing
    plt.rcParams['figure.constrained_layout.use'] = True
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from experiments.shared_utils.utils_misc import load_local_model
from experiments.shared_utils.utils_training import ResponseProbeDataModule
from experiments.shared_utils.utils_testing import test_trainable_metric_on_response_data_module
from obf_reps.logging import CSVTXTLogger
from obf_reps.metrics import TrainableMetric
from obf_reps.models.hf import HFHardPrompted
from data.data import PosNegDataByCategory
from data.synthetic_rating.load_rated_data import load_rated_data


def bootstrap_confidence_interval(data: List[float], n_bootstrap: int = 1000, confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for the mean of data.

    Args:
        data: List of values to bootstrap
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return 0.0, 0.0

    data = np.array(data)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)

    return ci_lower, ci_upper


def load_probe_from_checkpoint(probe_dir: str, model, logger, concept: str, is_merged: bool = False):
    """Load a trained probe from checkpoint directory."""
    if is_merged:
        probe_path = os.path.join(probe_dir, "merged_probe.pkl")
    else:
        probe_path = os.path.join(probe_dir, f"{concept}.pkl")

    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Probe file not found: {probe_path}")

    # Load probe using pickle directly (for rated probe compatibility)
    with open(probe_path, "rb") as f:
        probe = pickle.load(f)

    # Load metadata to get configuration
    metadata_path = os.path.join(probe_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return probe, metadata


def extract_adjective_data_from_pos_neg_data(pos_neg_data: PosNegDataByCategory, adjective: str) -> Tuple[List[Tuple[str, str, None]], List[Tuple[str, str, None]]]:
    """
    Extract positive and negative samples for a specific adjective from PosNegDataByCategory.

    Args:
        pos_neg_data: PosNegDataByCategory object containing the categorized data
        adjective: The adjective to extract data for

    Returns:
        Tuple of (positive_samples, negative_samples) in the expected tuple format
    """
    if adjective not in pos_neg_data.categories:
        return [], []

    category_data = pos_neg_data.categories[adjective]

    # Convert PromptResp objects to the expected tuple format (prompt, response, None)
    positive_samples = [(pr.prompt, pr.response, None) for pr in category_data.pos_dataset]
    negative_samples = [(pr.prompt, pr.response, None) for pr in category_data.neg_dataset]

    return positive_samples, negative_samples


def get_all_adjectives_from_pos_neg_data(pos_neg_data: PosNegDataByCategory) -> List[str]:
    """
    Get all adjectives present in the PosNegDataByCategory.

    Args:
        pos_neg_data: PosNegDataByCategory object containing the categorized data

    Returns:
        List of adjective names
    """
    return sorted(list(pos_neg_data.categories.keys()))


def load_ultrachat_mega_dataset(dataset_path: str) -> List[Dict]:
    """
    Loads the UltraChat mega dataset from JSON file.

    Args:
        dataset_path: Path to the UltraChat mega dataset JSON file

    Returns:
        List of UltraChat dataset items with adjective ratings
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} UltraChat samples from {dataset_path}")
    return data


def extract_ultrachat_negatives_for_adjective(ultrachat_data: List[Dict], adjective: str,
                                            negative_threshold: float = 0.25,
                                            max_samples: int = 1000) -> List[Tuple[str, str, None]]:
    """
    Extract negative samples for a specific adjective from UltraChat mega dataset.

    Args:
        ultrachat_data: List of UltraChat items with ratings
        adjective: The adjective to extract negative data for
        negative_threshold: Threshold for negative samples (<= this value)
        max_samples: Maximum number of samples to return

    Returns:
        List of negative samples for this adjective
    """
    negative_samples = []

    for item in ultrachat_data:
        # Skip if we don't have the required ratings
        if ("prompt_normalized_ratings" not in item or
            "response_normalized_ratings" not in item or
            adjective not in item["prompt_normalized_ratings"] or
            adjective not in item["response_normalized_ratings"]):
            continue

        prompt = item.get("prompt")
        response = item.get("response")

        if not prompt or not response:
            continue

        # Get ratings for this adjective
        prompt_rating = item["prompt_normalized_ratings"][adjective]
        response_rating = item["response_normalized_ratings"][adjective]

        if prompt_rating is None or response_rating is None:
            continue

        # Check if both prompt and response ratings are below threshold (strong negatives)
        max_rating = max(prompt_rating, response_rating)

        if max_rating <= negative_threshold:
            negative_samples.append((prompt, response, None))

            # Stop if we have enough samples
            if len(negative_samples) >= max_samples:
                break

    return negative_samples


def get_probe_scores_with_data(model: HFHardPrompted, probe: TrainableMetric, data_samples: List[Tuple[str, str, None]], batch_size: int, desc: str) -> Tuple[List[float], List[Tuple[str, str, None]]]:
    """Runs data through the model and probe to get activation scores along with the original data."""
    all_scores = []
    all_data = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data_samples), batch_size), desc=desc):
            batch = data_samples[i:i + batch_size]
            if not batch: continue
            prompts, responses = [d[0] for d in batch], [d[1] for d in batch]
            output = model.forward_from_string(
                input_text=prompts, target_text=responses, add_chat_template=True,
                layers_to_probe=probe.config.layers
            )
            scores = probe.predict_example(
                input_reps=output.input_reps, target_reps=output.target_reps, target_mask=output.loss_mask
            )
            all_scores.extend(scores.cpu().tolist())
            all_data.extend(batch)
            del output, scores
            gc.collect()
            torch.cuda.empty_cache()
    return all_scores, all_data


def get_top_bottom_examples(scores: List[float], data: List[Tuple[str, str, None]], n: int = 3) -> Dict:
    """Get top and bottom n examples based on scores."""
    if len(scores) != len(data):
        return {"top_examples": [], "bottom_examples": [], "error": "Scores and data length mismatch"}

    # Create list of (score, index) pairs and sort
    score_indices = [(score, i) for i, score in enumerate(scores)]
    score_indices.sort(key=lambda x: x[0], reverse=True)

    # Get top n and bottom n
    top_indices = [idx for _, idx in score_indices[:n]]
    bottom_indices = [idx for _, idx in score_indices[-n:]]

    top_examples = [{"score": scores[i], "prompt": data[i][0], "response": data[i][1]} for i in top_indices]
    bottom_examples = [{"score": scores[i], "prompt": data[i][0], "response": data[i][1]} for i in bottom_indices]

    return {"top_examples": top_examples, "bottom_examples": bottom_examples}


def bootstrap_tpr_at_fpr_with_thresholds(pos_scores: List[float], neg_scores: List[float], fpr_level: float = 0.01, num_iterations: int = 1000) -> Dict:
    """Calculates the mean TPR at a given FPR with confidence intervals and threshold statistics using bootstrapping."""
    if not pos_scores or not neg_scores:
        return {"mean_tpr": None, "tpr_ci_95": (None, None), "mean_threshold": None, "threshold_ci_95": (None, None), "error": "Not enough data for bootstrapping."}

    pos_scores_np, neg_scores_np = np.array(pos_scores), np.array(neg_scores)
    tpr_samples = []
    threshold_samples = []

    for _ in range(num_iterations):
        neg_resample = np.random.choice(neg_scores_np, size=len(neg_scores_np), replace=True)
        pos_resample = np.random.choice(pos_scores_np, size=len(pos_scores_np), replace=True)
        threshold = np.percentile(neg_resample, 100 * (1 - fpr_level))
        tpr = np.mean(pos_resample > threshold)
        tpr_samples.append(tpr)
        threshold_samples.append(threshold)

    mean_tpr = np.mean(tpr_samples)
    tpr_ci_lower, tpr_ci_upper = np.percentile(tpr_samples, 2.5), np.percentile(tpr_samples, 97.5)

    mean_threshold = np.mean(threshold_samples)
    threshold_ci_lower, threshold_ci_upper = np.percentile(threshold_samples, 2.5), np.percentile(threshold_samples, 97.5)

    return {
        "mean_tpr": mean_tpr,
        "tpr_ci_95": (tpr_ci_lower, tpr_ci_upper),
        "mean_threshold": mean_threshold,
        "threshold_ci_95": (threshold_ci_lower, threshold_ci_upper)
    }


def evaluate_probe_on_labelled_data(
    model,
    probe,
    positive_samples: List[Tuple[str, str, None]],
    negative_samples: List[Tuple[str, str, None]],
    logger,
    concept: str,
    add_probe_awareness: bool = False,
    batch_size: int = 32,
    bootstrap_iterations: int = 1000,
    min_test_samples: int = 100,
    max_test_samples: int = 1000,
    override_trigger_word: str = None
) -> Dict:
    """Evaluate a probe on labelled test data with optional probe awareness.

    Uses test_trainable_metric_on_response_data_module which only applies
    the awareness prompt to positive samples, not negative samples.
    """

    try:
        # Balance and limit test sets
        # Use min of available samples and max_test_samples, but at least min_test_samples
        max_possible_size = min(len(positive_samples), len(negative_samples))
        test_size = min(max_possible_size, max_test_samples)
        test_size = max(test_size, min(min_test_samples, max_possible_size))

        pos_test = random.Random(42).sample(positive_samples, test_size)
        neg_test = random.Random(42).sample(negative_samples, test_size)

        logger.print(f"Testing with {len(pos_test)} pos / {len(neg_test)} neg samples.")

        # Create ResponseProbeDataModule with the test samples
        # Extract positive and negative samples separately
        pos_samples = [(p, r) for p, r, _ in pos_test]
        neg_samples = [(p, r) for p, r, _ in neg_test]

        data_module = ResponseProbeDataModule(pos_samples, neg_samples)

        # Determine the trigger word to use
        adv_word = None
        if add_probe_awareness:
            if override_trigger_word:
                adv_word = override_trigger_word
            else:
                adv_word = concept

        # Use test_trainable_metric_on_response_data_module
        # This function automatically adds the awareness prompt only to positive samples when adv_word is provided
        custom_folder_name = f"probe_eval_{concept}{'_aware' if add_probe_awareness else ''}"
        results = test_trainable_metric_on_response_data_module(
            model=model,
            metric=probe,
            logger=logger,
            data_module=data_module,
            model_details={'concept': concept, 'probe_aware': add_probe_awareness},
            layers_to_probe=probe.config.layers if hasattr(probe, 'config') else None,
            custom_folder_name=custom_folder_name,
            adv_word=adv_word
        )

        # Extract scores from results
        pos_scores = [r['pos_score'] for r in results]
        neg_scores = [r['neg_score'] for r in results]

        # Debug: Print detailed information about results
        logger.print(f"  DEBUG: Got {len(results)} results from test_trainable_metric")
        logger.print(f"  DEBUG: Number of pos_scores: {len(pos_scores)}, neg_scores: {len(neg_scores)}")
        if len(pos_scores) > 0:
            logger.print(f"  DEBUG: Pos scores - min: {min(pos_scores):.4f}, max: {max(pos_scores):.4f}, mean: {np.mean(pos_scores):.4f}")
        if len(neg_scores) > 0:
            logger.print(f"  DEBUG: Neg scores - min: {min(neg_scores):.4f}, max: {max(neg_scores):.4f}, mean: {np.mean(neg_scores):.4f}")
        if len(results) > 0:
            logger.print(f"  DEBUG: First result sample: pos_score={results[0]['pos_score']:.4f}, neg_score={results[0]['neg_score']:.4f}")

        # Calculate bootstrap metrics
        bootstrap_results = bootstrap_tpr_at_fpr_with_thresholds(pos_scores, neg_scores, num_iterations=bootstrap_iterations)

        # Calculate average probe score on topical responses
        avg_topical_score = np.mean(pos_scores) if len(pos_scores) > 0 else 0.0
        std_topical_score = np.std(pos_scores) if len(pos_scores) > 0 else 0.0

        # Calculate 95% confidence interval using bootstrapping
        ci_lower, ci_upper = bootstrap_confidence_interval(pos_scores, n_bootstrap=1000, confidence_level=0.95)

        # Get top/bottom examples - reconstruct data format
        pos_data = [(r['pos_prompt'], r['pos_target'], None) for r in results]
        neg_data = [(r['neg_prompt'], r['neg_target'], None) for r in results]

        pos_examples = get_top_bottom_examples(pos_scores, pos_data)
        neg_examples = get_top_bottom_examples(neg_scores, neg_data)

        return {
            'concept': concept,
            'probe_aware': add_probe_awareness,
            'trigger_word': override_trigger_word if override_trigger_word else None,
            'avg_topical_score': avg_topical_score,
            'std_topical_score': std_topical_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'num_topical': len(pos_scores),
            'topical_scores': pos_scores,
            'tpr_at_1_fpr': bootstrap_results.get("mean_tpr"),
            'tpr_at_1_fpr_95_ci': bootstrap_results.get("tpr_ci_95"),
            'mean_1fpr_threshold': bootstrap_results.get("mean_threshold"),
            'threshold_95_ci': bootstrap_results.get("threshold_ci_95"),
            'num_pos_test_samples': len(pos_scores),
            'num_neg_test_samples': len(neg_scores),
            'all_pos_scores': pos_scores,
            'all_neg_scores': neg_scores,
            'top_3_pos_examples': pos_examples.get("top_examples", []),
            'bottom_3_pos_examples': pos_examples.get("bottom_examples", []),
            'top_3_neg_examples': neg_examples.get("top_examples", []),
            'bottom_3_neg_examples': neg_examples.get("bottom_examples", []),
        }

    except Exception as e:
        logger.print(f"Error during evaluation: {e}")
        return {
            'concept': concept,
            'probe_aware': add_probe_awareness,
            'avg_topical_score': 0.0,
            'std_topical_score': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'num_topical': 0,
            'topical_scores': [],
            'error': str(e)
        }


def evaluate_probe_on_ultrachat_data(
    model,
    probe,
    positive_samples: List[Tuple[str, str, None]],
    ultrachat_negative_samples: List[Tuple[str, str, None]],
    logger,
    concept: str,
    add_probe_awareness: bool = False,
    batch_size: int = 32,
    bootstrap_iterations: int = 1000,
    min_test_samples: int = 100,
    max_test_samples: int = 1000,
    override_trigger_word: str = None
) -> Dict:
    """Evaluate a probe against UltraChat negative samples with optional probe awareness.

    Uses test_trainable_metric_on_response_data_module which only applies
    the awareness prompt to positive samples, not negative samples.
    """

    try:
        # Limit test sets and balance them
        pos_test_size = min(len(positive_samples), max_test_samples)
        pos_test_size = max(pos_test_size, min(min_test_samples, len(positive_samples)))

        neg_test_size = min(len(ultrachat_negative_samples), max_test_samples)
        neg_test_size = max(neg_test_size, min(min_test_samples, len(ultrachat_negative_samples)))

        # Make sure we have same number of pos and neg samples for pairing
        test_size = min(pos_test_size, neg_test_size)
        pos_test = positive_samples[:test_size]
        neg_test = ultrachat_negative_samples[:test_size]

        logger.print(f"Testing with {len(pos_test)} pos / {len(neg_test)} UltraChat neg samples.")

        # Create ResponseProbeDataModule with the test samples
        # Extract positive and negative samples separately
        pos_samples = [(p, r) for p, r, _ in pos_test]
        neg_samples = [(p, r) for p, r, _ in neg_test]

        data_module = ResponseProbeDataModule(pos_samples, neg_samples)

        # Determine the trigger word to use
        adv_word = None
        if add_probe_awareness:
            if override_trigger_word:
                adv_word = override_trigger_word
            else:
                adv_word = concept

        # Use test_trainable_metric_on_response_data_module
        # This function automatically adds the awareness prompt only to positive samples when adv_word is provided
        custom_folder_name = f"ultrachat_probe_eval_{concept}{'_aware' if add_probe_awareness else ''}"
        results = test_trainable_metric_on_response_data_module(
            model=model,
            metric=probe,
            logger=logger,
            data_module=data_module,
            model_details={'concept': concept, 'probe_aware': add_probe_awareness, 'test_type': 'ultrachat'},
            layers_to_probe=probe.config.layers if hasattr(probe, 'config') else None,
            custom_folder_name=custom_folder_name,
            adv_word=adv_word
        )

        # Extract scores from results
        pos_scores = [r['pos_score'] for r in results]
        neg_scores = [r['neg_score'] for r in results]

        # Calculate bootstrap metrics
        bootstrap_results = bootstrap_tpr_at_fpr_with_thresholds(pos_scores, neg_scores, num_iterations=bootstrap_iterations)

        # Get top/bottom examples - reconstruct data format
        pos_data = [(r['pos_prompt'], r['pos_target'], None) for r in results]
        neg_data = [(r['neg_prompt'], r['neg_target'], None) for r in results]

        pos_examples = get_top_bottom_examples(pos_scores, pos_data)
        neg_examples = get_top_bottom_examples(neg_scores, neg_data)

        return {
            'concept': concept,
            'probe_aware': add_probe_awareness,
            'trigger_word': override_trigger_word if override_trigger_word else None,
            'tpr_at_1_fpr': bootstrap_results.get("mean_tpr"),
            'tpr_at_1_fpr_95_ci': bootstrap_results.get("tpr_ci_95"),
            'mean_1fpr_threshold': bootstrap_results.get("mean_threshold"),
            'threshold_95_ci': bootstrap_results.get("threshold_ci_95"),
            'num_pos_test_samples': len(pos_scores),
            'num_neg_test_samples': len(neg_scores),
            'all_pos_scores': pos_scores,
            'all_neg_scores': neg_scores,
            'top_3_pos_examples': pos_examples.get("top_examples", []),
            'bottom_3_pos_examples': pos_examples.get("bottom_examples", []),
            'top_3_neg_examples': neg_examples.get("top_examples", []),
            'bottom_3_neg_examples': neg_examples.get("bottom_examples", []),
        }

    except Exception as e:
        logger.print(f"Error during UltraChat evaluation: {e}")
        return {
            'concept': concept,
            'probe_aware': add_probe_awareness,
            'error': str(e)
        }


def create_bar_plots(results: List[Dict], output_dir: str):
    """Create bar plots comparing probe performance."""

    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        print("matplotlib or pandas not available, skipping plots")
        return

    # Prepare data for plotting
    plot_data = []
    for result in results:
        if 'avg_topical_score' in result:  # Only include results with topical scores
            ci_lower = result.get('ci_lower', 0.0)
            ci_upper = result.get('ci_upper', 0.0)
            avg_score = result['avg_topical_score']

            # Calculate error bar values (distance from mean to bounds)
            error_lower = avg_score - ci_lower
            error_upper = ci_upper - avg_score

            plot_data.append({
                'concept': result['concept'],
                'condition': 'Regular' if not result['probe_aware'] else 'Probe-Aware',
                'avg_topical_score': avg_score,
                'std_topical_score': result['std_topical_score'],
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'error_lower': error_lower,
                'error_upper': error_upper
            })

    if not plot_data:
        print("No plotting data available")
        return

    df = pd.DataFrame(plot_data)

    # Create single plot for average topical probe scores with ICML paper dimensions
    fig, ax = plt.subplots(1, 1, figsize=(6.75, 4.5))

    # Calculate difference between probe-aware and regular scores for sorting
    regular_data = df[df['condition'] == 'Regular'].set_index('concept')
    aware_data = df[df['condition'] == 'Probe-Aware'].set_index('concept')

    # Calculate score differences and sort in descending order
    score_diff = aware_data['avg_topical_score'] - regular_data['avg_topical_score']
    sorted_concepts = score_diff.sort_values(ascending=True).index.tolist()

    # Reorder data based on sorted concepts
    regular_data = regular_data.reindex(sorted_concepts)
    aware_data = aware_data.reindex(sorted_concepts)

    # Average topical probe score plot with 95% confidence interval error bars
    x_pos = np.arange(len(sorted_concepts))
    width = 0.35

    # Plot bars with confidence intervals
    bars1 = ax.bar(x_pos - width/2, regular_data['avg_topical_score'], width,
                   yerr=[regular_data['error_lower'], regular_data['error_upper']],
                   label='Regular', capsize=5)
    bars2 = ax.bar(x_pos + width/2, aware_data['avg_topical_score'], width,
                   yerr=[aware_data['error_lower'], aware_data['error_upper']],
                   label='Probe-Aware', capsize=5)

    ax.set_title('Average Probe Score on Topical Responses (95% CI)', fontsize=12)
    ax.set_ylabel('Average Probe Score', fontsize=11)
    ax.set_xlabel('Concept', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_concepts, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)

    # Don't use tight_layout since we have constrained_layout enabled
    # Save PNG
    png_path = os.path.join(output_dir, 'concept_probe_performance.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    # Save PDF
    pdf_path = os.path.join(output_dir, 'concept_probe_performance.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    # Create a difference plot using the same sorted order with ICML paper dimensions
    fig, ax = plt.subplots(1, 1, figsize=(6.75, 4.5))

    # Use the already calculated and sorted score differences
    sorted_score_diff = score_diff.reindex(sorted_concepts)

    # Average topical score difference
    sorted_score_diff.plot(kind='bar', ax=ax, color='lightgreen')
    ax.set_title('Avg Topical Score Difference (Probe-Aware - Regular)', fontsize=12)
    ax.set_ylabel('Avg Topical Score Difference', fontsize=11)
    ax.set_xlabel('Concept', fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)

    # Don't use tight_layout since we have constrained_layout enabled
    # Save PNG
    png_path = os.path.join(output_dir, 'concept_probe_performance_difference.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    # Save PDF
    pdf_path = os.path.join(output_dir, 'concept_probe_performance_difference.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate concept probes on labelled test data with probe awareness testing"
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        default="outputs/probe_checkpoints/20250727_172938_TRAIN_labelled_topical_logistic_12",
        help="Path to probe checkpoint directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma_2_9b_instruct",
        help="Model name to use",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint directory (if empty, uses base model)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/clued_test_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for confidence intervals",
    )
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=1.0,
        help="Threshold for positive samples (>= this value)",
    )
    parser.add_argument(
        "--negative_threshold",
        type=float,
        default=0.0,
        help="Threshold for negative samples (<= this value)",
    )
    parser.add_argument(
        "--use_topical_ratings",
        action="store_true",
        default=True,
        help="Use topical response ratings instead of vanilla response ratings",
    )
    parser.add_argument(
        "--use_vanilla_ratings",
        dest="use_topical_ratings",
        action="store_false",
        help="Use vanilla response ratings instead of topical response ratings",
    )
    parser.add_argument(
        "--min_test_samples",
        type=int,
        default=100,
        help="Minimum number of test samples per class",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=1000,
        help="Maximum number of test samples per class",
    )
    parser.add_argument(
        "--ultrachat_mega_dataset_path",
        type=str,
        default="data/synthetic_rating/outputs/ultrachat_mega_dataset_20250718_035042.json",
        help="Path to UltraChat mega dataset JSON file with adjective ratings",
    )
    parser.add_argument(
        "--num_ultrachat_samples",
        type=int,
        default=1000,
        help="Number of UltraChat negative samples to use per adjective",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dataset_file_path",
        type=str,
        default="",
        help="Path to specific dataset file to use instead of searching for most recent",
    )
    parser.add_argument(
        "--override_trigger_words",
        type=str,
        nargs="*",
        default=[],
        help="List of trigger words to test. If provided, each word is tested as a separate trigger instead of using the probe name(s)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up logger
    logger = CSVTXTLogger(print_logs_to_console=True)
    logger.print("Starting concept probe evaluation with probe awareness testing...")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rating_type = "topical" if args.use_topical_ratings else "vanilla"
    output_dir = os.path.join(args.output_dir, f"{timestamp}_clued_labelled_{rating_type}_probe_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata to get concepts
    metadata_path = os.path.join(args.probe_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Check if this is a merged probe and exit if so
    merge_concepts = metadata.get("merge_concepts_into_one_probe", [])
    if len(merge_concepts) > 0:
        logger.print(f"ERROR: This script does not support merged probes.")
        logger.print(f"Found merged probe for concepts: {merge_concepts}")
        logger.print(f"Please use evaluate_merged_probe.py instead.")
        return

    concepts = metadata.get("adjectives_trained", [])
    logger.print(f"Found {len(concepts)} concepts to evaluate: {concepts}")

    # Load model
    model_checkpoint = args.model_checkpoint if args.model_checkpoint else metadata.get('model_checkpoint')
    logger.print(f"Loading model: {args.model_name} from checkpoint: {model_checkpoint}")
    model = load_local_model(checkpoint_path=model_checkpoint, model_name=args.model_name)

    # Load rated data using the proper data loading functions
    logger.print(f"\nLoading rated data using load_rated_data function...")
    response_type = "topical" if args.use_topical_ratings else "vanilla"

    # Load the rated data as PromptRespRating objects
    prompt_resp_ratings = load_rated_data(
        response_type=response_type,
        exclude_refusals=True,
        exclude_missing_ratings=True,
        manual_path_confirm=False,
        dataset_file_path=args.dataset_file_path if args.dataset_file_path else None
    )

    # Convert to PosNegDataByCategory using the proper data structure
    pos_neg_data = PosNegDataByCategory.from_ratings(
        prompt_resp_ratings,
        max_neg_rating=args.negative_threshold,
        min_pos_rating=args.positive_threshold,
        shuffle=False
    )

    # Load UltraChat mega dataset if provided
    ultrachat_mega_data = []
    if args.ultrachat_mega_dataset_path and os.path.exists(args.ultrachat_mega_dataset_path):
        logger.print(f"\nLoading UltraChat mega dataset from {args.ultrachat_mega_dataset_path}...")
        ultrachat_mega_data = load_ultrachat_mega_dataset(args.ultrachat_mega_dataset_path)

    # Evaluate each concept
    all_results = []

    for concept in concepts:
        logger.print(f"\nEvaluating concept: {concept}")

        try:
            # Load single-concept probe
            probe, probe_metadata = load_probe_from_checkpoint(
                args.probe_dir, model, logger, concept
            )

            # Debug: Check probe configuration
            logger.print(f"  DEBUG: Loaded probe type: {type(probe).__name__}")
            if hasattr(probe, 'config'):
                logger.print(f"  DEBUG: Probe config layers: {probe.config.layers if hasattr(probe.config, 'layers') else 'No layers attr'}")
            if hasattr(probe, 'clf'):
                logger.print(f"  DEBUG: Probe has clf: {probe.clf is not None}")
                if probe.clf is not None and hasattr(probe.clf, 'coef_'):
                    logger.print(f"  DEBUG: Clf coef shape: {probe.clf.coef_.shape}")

            # Extract positive and negative samples for this adjective
            pos_samples, neg_samples = extract_adjective_data_from_pos_neg_data(
                pos_neg_data, concept
            )

            logger.print(f"  Found {len(pos_samples)} positive and {len(neg_samples)} negative samples")

            # Set up dummy data module for probe compatibility
            dummy_data_module = ResponseProbeDataModule([], [])
            probe.data_module = dummy_data_module

            # Check if we have enough samples for testing
            if len(pos_samples) < args.min_test_samples or len(neg_samples) < args.min_test_samples:
                logger.print(f"  Insufficient samples for testing '{concept}' (need at least {args.min_test_samples} per class). Skipping.")
                continue

            # Evaluate regular condition (labelled negatives)
            logger.print("  Evaluating regular condition with labelled negatives...")
            regular_result = evaluate_probe_on_labelled_data(
                model, probe, pos_samples, neg_samples,
                logger, concept, add_probe_awareness=False,
                batch_size=args.batch_size, bootstrap_iterations=args.bootstrap_iterations,
                min_test_samples=args.min_test_samples, max_test_samples=args.max_test_samples
            )
            all_results.append(regular_result)

            # Evaluate probe-aware condition (labelled negatives)
            if args.override_trigger_words:
                # Test each override trigger word separately
                for trigger_word in args.override_trigger_words:
                    logger.print(f"  Evaluating probe-aware condition with trigger word '{trigger_word}'...")
                    aware_result = evaluate_probe_on_labelled_data(
                        model, probe, pos_samples, neg_samples,
                        logger, concept, add_probe_awareness=True,
                        batch_size=args.batch_size, bootstrap_iterations=args.bootstrap_iterations,
                        min_test_samples=args.min_test_samples, max_test_samples=args.max_test_samples,
                        override_trigger_word=trigger_word
                    )
                    all_results.append(aware_result)
            else:
                # Use default probe-aware behavior
                logger.print("  Evaluating probe-aware condition with labelled negatives...")
                aware_result = evaluate_probe_on_labelled_data(
                    model, probe, pos_samples, neg_samples,
                    logger, concept, add_probe_awareness=True,
                    batch_size=args.batch_size, bootstrap_iterations=args.bootstrap_iterations,
                    min_test_samples=args.min_test_samples, max_test_samples=args.max_test_samples
                )
                all_results.append(aware_result)

            # Evaluate against UltraChat negatives if available
            ultrachat_neg_samples = []
            if ultrachat_mega_data:
                # Extract negatives for this concept
                logger.print(f"  Extracting UltraChat negative samples for '{concept}'...")
                ultrachat_neg_samples = extract_ultrachat_negatives_for_adjective(
                    ultrachat_mega_data, concept, args.negative_threshold, args.num_ultrachat_samples
                )

            if ultrachat_neg_samples:
                logger.print(f"  Found {len(ultrachat_neg_samples)} UltraChat negative samples")

                # Regular condition with UltraChat negatives
                logger.print("  Evaluating regular condition with UltraChat negatives...")
                uc_regular_result = evaluate_probe_on_ultrachat_data(
                    model, probe, pos_samples, ultrachat_neg_samples,
                    logger, concept, add_probe_awareness=False,
                    batch_size=args.batch_size, bootstrap_iterations=args.bootstrap_iterations,
                    min_test_samples=args.min_test_samples, max_test_samples=args.max_test_samples
                )
                uc_regular_result['test_type'] = 'ultrachat_regular'
                all_results.append(uc_regular_result)

                # Probe-aware condition with UltraChat negatives
                if args.override_trigger_words:
                    # Test each override trigger word separately
                    for trigger_word in args.override_trigger_words:
                        logger.print(f"  Evaluating probe-aware condition with UltraChat negatives and trigger word '{trigger_word}'...")
                        uc_aware_result = evaluate_probe_on_ultrachat_data(
                            model, probe, pos_samples, ultrachat_neg_samples,
                            logger, concept, add_probe_awareness=True,
                            batch_size=args.batch_size, bootstrap_iterations=args.bootstrap_iterations,
                            min_test_samples=args.min_test_samples, max_test_samples=args.max_test_samples,
                            override_trigger_word=trigger_word
                        )
                        uc_aware_result['test_type'] = 'ultrachat_aware'
                        all_results.append(uc_aware_result)
                else:
                    # Use default probe-aware behavior
                    logger.print("  Evaluating probe-aware condition with UltraChat negatives...")
                    uc_aware_result = evaluate_probe_on_ultrachat_data(
                        model, probe, pos_samples, ultrachat_neg_samples,
                        logger, concept, add_probe_awareness=True,
                        batch_size=args.batch_size, bootstrap_iterations=args.bootstrap_iterations,
                        min_test_samples=args.min_test_samples, max_test_samples=args.max_test_samples
                    )
                    uc_aware_result['test_type'] = 'ultrachat_aware'
                    all_results.append(uc_aware_result)
            else:
                logger.print(f"  No UltraChat negative samples found for '{concept}'")

            # Print summary for this concept
            if 'avg_topical_score' in regular_result:
                logger.print(f"  Regular - Avg Topical Score: {regular_result['avg_topical_score']:.3f} [95% CI: {regular_result['ci_lower']:.3f}, {regular_result['ci_upper']:.3f}]")

                # For single aware result (no override trigger words)
                if not args.override_trigger_words and 'aware_result' in locals() and 'avg_topical_score' in aware_result:
                    logger.print(f"  Probe-Aware - Avg Topical Score: {aware_result['avg_topical_score']:.3f} [95% CI: {aware_result['ci_lower']:.3f}, {aware_result['ci_upper']:.3f}]")

        except Exception as e:
            logger.print(f"  Error evaluating {concept}: {e}")
            continue

    # Save results
    logger.print(f"\nSaving results to: {output_dir}")

    # Create summary CSV
    summary_data = []
    for result in all_results:
        if 'avg_topical_score' in result:  # Only include labelled results in summary
            summary_data.append({
                'concept': result['concept'],
                'condition': 'Regular' if not result['probe_aware'] else 'Probe-Aware',
                'avg_topical_score': result['avg_topical_score'],
                'std_topical_score': result['std_topical_score'],
                'ci_lower': result.get('ci_lower', 0.0),
                'ci_upper': result.get('ci_upper', 0.0),
                'num_topical': result['num_topical'],
                'tpr_at_1_fpr': result.get('tpr_at_1_fpr'),
                'tpr_ci_lower': result.get('tpr_at_1_fpr_95_ci', [None, None])[0],
                'tpr_ci_upper': result.get('tpr_at_1_fpr_95_ci', [None, None])[1],
            })

    # Write CSV manually if pandas not available
    if HAS_PANDAS:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, "concept_probe_results.csv"), index=False)
    else:
        # Write CSV manually
        csv_path = os.path.join(output_dir, "concept_probe_results.csv")
        with open(csv_path, 'w') as f:
            # Write header
            f.write("concept,condition,avg_topical_score,std_topical_score,ci_lower,ci_upper,num_topical,tpr_at_1_fpr,tpr_ci_lower,tpr_ci_upper\n")
            # Write data
            for row in summary_data:
                f.write(f"{row['concept']},{row['condition']},{row['avg_topical_score']},{row['std_topical_score']},{row['ci_lower']},{row['ci_upper']},{row['num_topical']},{row['tpr_at_1_fpr']},{row['tpr_ci_lower']},{row['tpr_ci_upper']}\n")

    # Save detailed results
    with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
        # Convert to JSON-serializable format
        json_results = []
        for result in all_results:
            json_result = result.copy()
            # Convert numpy arrays and lists to regular lists
            for key in ['topical_scores', 'all_pos_scores', 'all_neg_scores']:
                if key in json_result and json_result[key] is not None:
                    json_result[key] = list(json_result[key])
            # Convert tuples to lists for JSON serialization
            for key in ['tpr_at_1_fpr_95_ci', 'threshold_95_ci']:
                if key in json_result and json_result[key] is not None:
                    json_result[key] = list(json_result[key])
            json_results.append(json_result)

        json.dump({
            'metadata': metadata,
            'test_metadata': {
                'probe_dir': args.probe_dir,
                'model_name': args.model_name,
                'model_checkpoint': model_checkpoint,
                'response_type': response_type,
                'positive_threshold': args.positive_threshold,
                'negative_threshold': args.negative_threshold,
                'min_test_samples': args.min_test_samples,
                'max_test_samples': args.max_test_samples,
                'bootstrap_iterations': args.bootstrap_iterations,
                'seed': args.seed,
                'ultrachat_mega_dataset_path': args.ultrachat_mega_dataset_path,
                'num_ultrachat_samples': args.num_ultrachat_samples,
            },
            'results': json_results
        }, f, indent=2)

    # Create plots
    logger.print("Creating plots...")
    create_bar_plots(all_results, output_dir)

    # Print summary
    logger.print("\n=== EVALUATION SUMMARY ===")
    for concept in concepts:
        concept_results = [r for r in all_results if r['concept'] == concept and 'avg_topical_score' in r]
        if len(concept_results) >= 2:
            regular = next((r for r in concept_results if not r['probe_aware']), None)
            aware = next((r for r in concept_results if r['probe_aware']), None)

            if regular and aware:
                logger.print(f"\n{concept}:")
                logger.print(f"  Regular    - Avg Topical Score: {regular['avg_topical_score']:.3f} [95% CI: {regular['ci_lower']:.3f}, {regular['ci_upper']:.3f}]")
                logger.print(f"  Probe-Aware - Avg Topical Score: {aware['avg_topical_score']:.3f} [95% CI: {aware['ci_lower']:.3f}, {aware['ci_upper']:.3f}]")
                logger.print(f"  Difference - Avg Topical Score: {aware['avg_topical_score'] - regular['avg_topical_score']:.3f}")
                if regular.get('tpr_at_1_fpr') is not None and aware.get('tpr_at_1_fpr') is not None:
                    logger.print(f"  Regular    - TPR at 1% FPR: {regular['tpr_at_1_fpr']:.3f}")
                    logger.print(f"  Probe-Aware - TPR at 1% FPR: {aware['tpr_at_1_fpr']:.3f}")

    logger.print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
