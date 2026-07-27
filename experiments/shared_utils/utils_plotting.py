import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
from typing import Dict, List, Optional, Any
import os


def compute_metric(
    metric: str,
    *,
    positive_scores=None,
    negative_scores=None,
    scores=None,
    labels=None,
    return_threshold=False,
    fixed_threshold=None,
):
    if metric.startswith("recall"):
        if positive_scores is None or negative_scores is None:
            assert scores is not None and labels is not None
            positive_scores = scores[labels == 1]
            negative_scores = scores[labels == 0]
        fpr = float(metric.split("@")[1]) / 100
        return compute_recall(positive_scores, negative_scores, fpr, return_threshold, fixed_threshold)
    elif metric == "auroc":
        if return_threshold:
            raise ValueError("return_threshold not supported for auroc")
        if labels is None or scores is None:
            assert positive_scores is not None and negative_scores is not None
            labels, scores = _concatenate_scores(positive_scores, negative_scores)
        return roc_auc_score(labels, scores)
    elif metric.startswith("auroc@"):
        if return_threshold:
            raise ValueError("return_threshold not supported for auroc@")
        if labels is None or scores is None:
            assert positive_scores is not None and negative_scores is not None
            labels, scores = _concatenate_scores(positive_scores, negative_scores)
        max_fpr = float(metric.split("@")[1]) / 100
        return roc_auc_score(labels, scores, max_fpr=max_fpr)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def bootstrap_sample(data, size=None):
    """Generate a bootstrap sample from the data."""
    if size is None:
        size = len(data)
    return np.random.choice(data, size=size, replace=True)


def compute_metric_bootstrap(
    metric: str,
    *,
    positive_scores=None,
    negative_scores=None,
    scores=None,
    labels=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    fixed_threshold=None,
):
    # We always convert to positive and negative scores to make sure that we get
    # a reasonable number of samples from each class during bootstrapping.
    if positive_scores is None or negative_scores is None:
        assert scores is not None and labels is not None
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]

    # Compute point estimate
    base_metric = compute_metric(
        metric,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        fixed_threshold=fixed_threshold,
    )
    # Bootstrap sampling
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_pos = bootstrap_sample(positive_scores)
        bootstrap_neg = bootstrap_sample(negative_scores)
        # bootstrap_neg = negative_scores

        # Compute recall for this bootstrap sample
        bootstrap_metric = compute_metric(
            metric,
            positive_scores=bootstrap_pos,
            negative_scores=bootstrap_neg,
            fixed_threshold=fixed_threshold,
        )
        bootstrap_metrics.append(bootstrap_metric)

    # Calculate confidence intervals
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    ci_lower, ci_upper = np.percentile(bootstrap_metrics, [lower_percentile, upper_percentile])

    return base_metric, ci_lower, ci_upper


def prettify_metric(metric):
    if metric.startswith("recall"):
        fpr = float(metric.split("@")[1]) / 100
        return f"Recall at {format_percent(fpr)} FPR"
    elif metric == "auroc":
        return "AUROC"
    elif metric.startswith("auroc@"):
        max_fpr = float(metric.split("@")[1]) / 100
        return f"AUROC at $\\leq {format_percent(max_fpr)}$ FPR"
    else:
        return metric


def compute_recall(positive_scores, negative_scores, fpr, return_threshold=False, fixed_threshold=None):
    # Sort scores in ascending order
    positive_scores = sorted(positive_scores)
    negative_scores = sorted(negative_scores)

    if fixed_threshold is not None:
        # Use the provided threshold instead of calculating from current negative scores
        threshold = fixed_threshold
    else:
        # Find threshold that gives desired FPR
        n_negatives = len(negative_scores)
        threshold_idx = int((1 - fpr) * n_negatives)
        if threshold_idx >= n_negatives:
            threshold_idx = n_negatives - 1
        threshold = negative_scores[threshold_idx]

    # Calculate recall at threshold
    n_positives = len(positive_scores)
    n_detected = sum(1 for score in positive_scores if score > threshold)
    recall = n_detected / n_positives

    return (recall, threshold) if return_threshold else recall


def format_percent(x: float) -> str:
    """Format a float between 0 and 1 as a percentage string."""
    if not 0 <= x <= 1:
        raise ValueError("Input must be between 0 and 1")

    if x == 0:
        return "0\\%"

    percent = x * 100
    if percent >= 1:
        return f"{round(percent)}\\%"
    elif percent >= 0.1:
        return f"{round(percent, 1)}\\%"
    else:
        # Find the first significant digit
        i = 0
        while percent < 1:
            percent *= 10
            i += 1
        return f"0.{'0' * (i - 1)}{round(percent)}\\%"


def _concatenate_scores(positive_scores, negative_scores):
    return (
        np.concatenate([np.ones(len(positive_scores)), np.zeros(len(negative_scores))]),
        np.concatenate([positive_scores, negative_scores]),
    )


def compute_auroc(positive_scores, negative_scores, max_fpr=None):
    return roc_auc_score(*_concatenate_scores(positive_scores, negative_scores), max_fpr=max_fpr)


def plot_roc(positive_scores, negative_scores, title="", fpr=0.02):
    fprs, tprs, _ = roc_curve(*_concatenate_scores(positive_scores, negative_scores))
    plt.plot(fprs, tprs, label=title)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.axvline(fpr, color="k", linestyle="--", label=f"FPR = {fpr:.2f}")
    plt.title(title)
    plt.show()


def binomial_error(n, p):
    return 2 * np.sqrt(p * (1 - p) / n)


def plot_likelihood_comparison(
    current_likelihoods: Dict[str, Any],
    base_likelihoods: Optional[Dict[str, Any]],
    output_dir: str,
    model_name: str = "current_model",
    base_model_name: str = "base_model",
) -> List[str]:
    """
    Create comparison plots for log likelihoods between current and base models.

    Args:
        current_likelihoods: Dictionary with current model likelihood data
        base_likelihoods: Optional dictionary with base model likelihood data
        output_dir: Directory to save plots
        model_name: Name of current model for labels
        base_model_name: Name of base model for labels

    Returns:
        List of paths to saved plot files
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # Extract conditions
    conditions = list(current_likelihoods["conditions"].keys())

    # 1. Bar chart comparing mean log likelihoods
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(conditions))
    width = 0.35

    current_means = []
    current_stds = []
    base_means = []
    base_stds = []

    for condition in conditions:
        current_data = current_likelihoods["conditions"][condition]
        current_means.append(current_data.get("mean", 0))
        current_stds.append(current_data.get("std", 0))

        if base_likelihoods:
            base_data = base_likelihoods["conditions"].get(condition, {})
            base_means.append(base_data.get("mean", 0))
            base_stds.append(base_data.get("std", 0))

    bars1 = ax.bar(x - width/2, current_means, width, yerr=current_stds,
                   label=model_name, capsize=5, alpha=0.8)

    if base_likelihoods:
        bars2 = ax.bar(x + width/2, base_means, width, yerr=base_stds,
                      label=base_model_name, capsize=5, alpha=0.8)

    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Mean Log Likelihood', fontsize=12)
    ax.set_title('Log Likelihood Comparison Across Conditions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save both PNG and PDF versions
    plot_path_png = os.path.join(output_dir, 'likelihood_comparison_bar.png')
    plt.savefig(plot_path_png, dpi=150, bbox_inches='tight')
    saved_plots.append(plot_path_png)

    plot_path_pdf = os.path.join(output_dir, 'likelihood_comparison_bar.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    saved_plots.append(plot_path_pdf)

    plt.close()

    # 2. Distribution plots for each condition
    n_conditions = len(conditions)
    n_cols = 3
    n_rows = (n_conditions + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, condition in enumerate(conditions):
        ax = axes[idx]

        current_data = current_likelihoods["conditions"][condition]
        current_vals = [v for v in current_data.get("likelihoods", []) if not np.isnan(v)]

        if current_vals:
            ax.hist(current_vals, bins=30, alpha=0.6, label=model_name, density=True)

        if base_likelihoods and condition in base_likelihoods["conditions"]:
            base_data = base_likelihoods["conditions"][condition]
            base_vals = [v for v in base_data.get("likelihoods", []) if not np.isnan(v)]
            if base_vals:
                ax.hist(base_vals, bins=30, alpha=0.6, label=base_model_name, density=True)

        ax.set_title(f'{condition}', fontsize=12)
        ax.set_xlabel('Log Likelihood')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_conditions, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Log Likelihood Distributions by Condition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Save both PNG and PDF versions
    plot_path_png = os.path.join(output_dir, 'likelihood_distributions.png')
    plt.savefig(plot_path_png, dpi=150, bbox_inches='tight')
    saved_plots.append(plot_path_png)

    plot_path_pdf = os.path.join(output_dir, 'likelihood_distributions.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    saved_plots.append(plot_path_pdf)

    plt.close()

    # 3. Heatmap of likelihood differences (if base model available)
    if base_likelihoods:
        # Calculate mean differences
        differences = []
        condition_labels = []

        for condition in conditions:
            if condition in base_likelihoods["conditions"]:
                current_mean = current_likelihoods["conditions"][condition].get("mean", 0)
                base_mean = base_likelihoods["conditions"][condition].get("mean", 0)
                diff = current_mean - base_mean
                differences.append(diff)
                condition_labels.append(condition)

        if differences:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create a matrix for the heatmap (single row)
            diff_matrix = np.array(differences).reshape(1, -1)

            # Create heatmap
            sns.heatmap(diff_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                       center=0, cbar_kws={'label': 'Difference (Current - Base)'},
                       xticklabels=condition_labels, yticklabels=['Δ Log Likelihood'],
                       ax=ax)

            ax.set_title('Log Likelihood Differences from Base Model', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save both PNG and PDF versions
            plot_path_png = os.path.join(output_dir, 'likelihood_differences_heatmap.png')
            plt.savefig(plot_path_png, dpi=150, bbox_inches='tight')
            saved_plots.append(plot_path_png)

            plot_path_pdf = os.path.join(output_dir, 'likelihood_differences_heatmap.pdf')
            plt.savefig(plot_path_pdf, bbox_inches='tight')
            saved_plots.append(plot_path_pdf)

            plt.close()

    # 4. Line plot showing likelihood changes across conditions
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort conditions: normal first, then deception-related, then others
    deception_words = ['lying', 'deceptive', 'deception', 'misleading', 'untruthful', 'dishonest', "false"]
    sorted_conditions = []

    # Add 'normal' first if it exists
    if 'normal' in conditions:
        sorted_conditions.append('normal')

    # Add deception-related conditions
    for condition in conditions:
        if condition != 'normal' and any(word in condition.lower() for word in deception_words):
            sorted_conditions.append(condition)

    # Add remaining conditions
    for condition in conditions:
        if condition not in sorted_conditions:
            sorted_conditions.append(condition)

    x_pos = np.arange(len(sorted_conditions))

    # Plot current model
    current_means_sorted = [current_likelihoods["conditions"][c].get("mean", 0) for c in sorted_conditions]
    current_stds_sorted = [current_likelihoods["conditions"][c].get("std", 0) for c in sorted_conditions]

    ax.errorbar(x_pos, current_means_sorted, yerr=current_stds_sorted,
               marker='o', linestyle='-', capsize=5, label=model_name, linewidth=2)

    # Plot base model if available
    if base_likelihoods:
        base_means_sorted = [base_likelihoods["conditions"].get(c, {}).get("mean", 0) for c in sorted_conditions]
        base_stds_sorted = [base_likelihoods["conditions"].get(c, {}).get("std", 0) for c in sorted_conditions]

        ax.errorbar(x_pos, base_means_sorted, yerr=base_stds_sorted,
                   marker='s', linestyle='--', capsize=5, label=base_model_name, linewidth=2)

    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Mean Log Likelihood', fontsize=12)
    ax.set_title('Log Likelihood Trends Across Conditions', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_conditions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add vertical line after 'normal' condition
    if 'normal' in sorted_conditions:
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    # Save both PNG and PDF versions
    plot_path_png = os.path.join(output_dir, 'likelihood_trends.png')
    plt.savefig(plot_path_png, dpi=150, bbox_inches='tight')
    saved_plots.append(plot_path_png)

    plot_path_pdf = os.path.join(output_dir, 'likelihood_trends.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    saved_plots.append(plot_path_pdf)

    plt.close()

    return saved_plots
