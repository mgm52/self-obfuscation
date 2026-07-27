"""
Evaluation and statistical utilities for self-obfuscation experiments.

This module provides functions for probe evaluation, bootstrapping,
and statistical analysis of probe performance.
"""

import gc
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from obf_reps.metrics import TrainableMetric
from obf_reps.models.hf import HFHardPrompted
import matplotlib.pyplot as plt



def get_probe_scores_with_data(
    model: HFHardPrompted,
    probe: TrainableMetric,
    data_samples: List[Tuple[str, str, None]],
    batch_size: int,
    desc: str
) -> Tuple[List[float], List[Tuple[str, str, None]]]:
    """Run data through the model and probe to get activation scores along with the original data."""
    all_scores = []
    all_data = []

    with torch.no_grad():
        for i in tqdm(range(0, len(data_samples), batch_size), desc=desc):
            batch = data_samples[i:i + batch_size]
            if not batch:
                continue

            prompts, responses = [d[0] for d in batch], [d[1] for d in batch]
            output = model.forward_from_string(
                input_text=prompts,
                target_text=responses,
                add_chat_template=True,
                layers_to_probe=probe.config.layers
            )

            scores = probe.predict_example(
                input_reps=output.input_reps,
                target_reps=output.target_reps,
                target_mask=output.loss_mask
            )

            all_scores.extend(scores.cpu().tolist())
            all_data.extend(batch)

            del output, scores
            gc.collect()
            torch.cuda.empty_cache()

    return all_scores, all_data


def get_top_bottom_examples(
    scores: List[float],
    data: List[Tuple[str, str, None]],
    n: int = 3
) -> Dict:
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


def bootstrap_tpr_at_fpr_with_thresholds(
    pos_scores: List[float],
    neg_scores: List[float],
    fpr_level: float = 0.01,
    num_iterations: int = 1000
) -> Dict:
    """
    Calculate the mean TPR at a given FPR with confidence intervals and threshold statistics using bootstrapping.

    Args:
        pos_scores: List of scores for positive examples
        neg_scores: List of scores for negative examples
        fpr_level: False positive rate level (default 0.01)
        num_iterations: Number of bootstrap iterations

    Returns:
        Dictionary with mean TPR, confidence intervals, and threshold statistics
    """
    if not pos_scores or not neg_scores:
        return {
            "mean_tpr": None,
            "tpr_ci_95": (None, None),
            "mean_threshold": None,
            "threshold_ci_95": (None, None),
            "error": "Not enough data for bootstrapping."
        }

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

def plot_training_curves(behavior_losses: List[float], obf_losses: List[float], loss10_behavior: List[float], loss10_obf: List[float], loss50_behavior: List[float], loss50_obf: List[float], save_path: str, validation_logs: Optional[List[Dict]] = None):
    """Generate and save training loss plots with optional validation curves."""
    if validation_logs:
        # Create 2x2 subplot layout when validation data is available
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        # Training behavior loss
        ax1.plot(behavior_losses, label='Training Behavior Loss', color='blue', alpha=0.7)
        ax1.set_title('Training Behavior Loss')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Behavior Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Training obfuscation loss
        ax2.plot(obf_losses, label='Training Obfuscation Loss', color='red', alpha=0.7)
        ax2.set_title('Training Obfuscation Loss')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Obfuscation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Validation behavior loss
        val_steps = [log['step'] for log in validation_logs]
        val_beh_losses = [log['avg_beh_loss'] for log in validation_logs]
        ax3.plot(val_steps, val_beh_losses, 'o-', label='Validation Behavior Loss', color='lightblue', markersize=4)
        ax3.set_title('Validation Behavior Loss')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Behavior Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Validation obfuscation loss
        val_obf_losses = [log['avg_obf_loss'] for log in validation_logs]
        ax4.plot(val_steps, val_obf_losses, 'o-', label='Validation Obfuscation Loss', color='orange', markersize=4)
        ax4.set_title('Validation Obfuscation Loss')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Obfuscation Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Original 2x1 layout when no validation data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Training behavior loss
        ax1.plot(behavior_losses, label='Behavior Loss', color='blue', alpha=0.7)
        ax1.set_title('Behavior Loss Over Training Steps')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Behavior Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Training obfuscation loss
        ax2.plot(obf_losses, label='Obfuscation Loss', color='red', alpha=0.7)
        ax2.set_title('Obfuscation Loss Over Training Steps')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Obfuscation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create combined plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize losses for better visualization
    norm_behavior = np.array(behavior_losses) / max(behavior_losses) if behavior_losses else []
    norm_obf = np.array(obf_losses) / max(obf_losses) if obf_losses else []

    ax.plot(norm_behavior, label='Train Behavior Loss (normalized)', color='blue', alpha=0.7)
    ax.plot(norm_obf, label='Train Obfuscation Loss (normalized)', color='red', alpha=0.7)

    # Add validation data if available
    if validation_logs:
        val_steps = [log['step'] for log in validation_logs]
        val_beh_losses = [log['avg_beh_loss'] for log in validation_logs]
        val_obf_losses = [log['avg_obf_loss'] for log in validation_logs]

        # Normalize validation losses
        if val_beh_losses and max(val_beh_losses) > 0:
            norm_val_beh = np.array(val_beh_losses) / max(val_beh_losses)
            ax.plot(val_steps, norm_val_beh, 'o-', label='Val Behavior Loss (normalized)',
                   color='lightblue', markersize=4)

        if val_obf_losses and max(val_obf_losses) > 0:
            norm_val_obf = np.array(val_obf_losses) / max(val_obf_losses)
            ax.plot(val_steps, norm_val_obf, 'o-', label='Val Obfuscation Loss (normalized)',
                   color='orange', markersize=4)

    ax.set_title('Training Loss Curves (Normalized)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Normalized Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    combined_path = save_path.replace('.png', '_combined.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create loss10 plots
    plt.figure(figsize=(12, 8))

    # Create subplot for loss10 losses
    plt.subplot(2, 1, 1)
    plt.plot(loss10_behavior, label='Behavior Loss10', color='blue', alpha=0.7)
    plt.title('Behavior Loss10 Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Behavior Loss10')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(loss10_obf, label='Obfuscation Loss10', color='red', alpha=0.7)
    plt.title('Obfuscation Loss10 Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Obfuscation Loss10')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    loss10_path = save_path.replace('.png', '_loss10.png')
    plt.savefig(loss10_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create combined loss10 plot
    plt.figure(figsize=(12, 6))

    # Normalize loss10 for better visualization
    norm_loss10_behavior = np.array(loss10_behavior) / max(loss10_behavior) if loss10_behavior else []
    norm_loss10_obf = np.array(loss10_obf) / max(loss10_obf) if loss10_obf else []

    plt.plot(norm_loss10_behavior, label='Behavior Loss10 (normalized)', color='blue', alpha=0.7)
    plt.plot(norm_loss10_obf, label='Obfuscation Loss10 (normalized)', color='red', alpha=0.7)
    plt.title('Training Loss10 Curves (Normalized)')
    plt.xlabel('Training Step')
    plt.ylabel('Normalized Loss10')
    plt.legend()
    plt.grid(True, alpha=0.3)

    combined_loss10_path = save_path.replace('.png', '_loss10_combined.png')
    plt.savefig(combined_loss10_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create loss50 plots
    plt.figure(figsize=(12, 8))

    # Create subplot for loss50 losses
    plt.subplot(2, 1, 1)
    plt.plot(loss50_behavior, label='Behavior Loss50', color='blue', alpha=0.7)
    plt.title('Behavior Loss50 Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Behavior Loss50')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(loss50_obf, label='Obfuscation Loss50', color='red', alpha=0.7)
    plt.title('Obfuscation Loss50 Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Obfuscation Loss50')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    loss50_path = save_path.replace('.png', '_loss50.png')
    plt.savefig(loss50_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create combined loss50 plot
    plt.figure(figsize=(12, 6))

    # Normalize loss50 for better visualization
    norm_loss50_behavior = np.array(loss50_behavior) / max(loss50_behavior) if loss50_behavior else []
    norm_loss50_obf = np.array(loss50_obf) / max(loss50_obf) if loss50_obf else []

    plt.plot(norm_loss50_behavior, label='Behavior Loss50 (normalized)', color='blue', alpha=0.7)
    plt.plot(norm_loss50_obf, label='Obfuscation Loss50 (normalized)', color='red', alpha=0.7)
    plt.title('Training Loss50 Curves (Normalized)')
    plt.xlabel('Training Step')
    plt.ylabel('Normalized Loss50')
    plt.legend()
    plt.grid(True, alpha=0.3)

    combined_loss50_path = save_path.replace('.png', '_loss50_combined.png')
    plt.savefig(combined_loss50_path, dpi=300, bbox_inches='tight')
    plt.close()

    val_msg = " (with validation)" if validation_logs else ""
    print(f"Training plots{val_msg} saved to {save_path} and {combined_path}")
    print(f"Loss10 plots saved to {loss10_path} and {combined_loss10_path}")
    print(f"Loss50 plots saved to {loss50_path} and {combined_loss50_path}")
