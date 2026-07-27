#!/usr/bin/env python3
"""
Script to compute cosine embedding similarity between condition words and plot against recall@1fpr_mean.
Finds the condition with lowest recall@1fpr_mean and computes cosine similarity of all other conditions to it.
Also generates bar charts with bootstrapped error bars.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style and context for paper
sns.set_context("paper")
sns.set_style("darkgrid")

# Set Times New Roman font with fallbacks
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif']

# Enable constrained layout for better spacing
plt.rcParams['figure.constrained_layout.use'] = True

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def load_wandb_info_from_model_checkpoint(model_dir):
    """Load wandb run information from model checkpoint's metadata."""
    if not model_dir:
        return None

    metadata_path = Path(model_dir) / "model_training_metadata.json"
    if not metadata_path.exists():
        print(f"No model_training_metadata.json found in {model_dir}")
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        wandb_info = metadata.get('wandb')
        if wandb_info:
            print(f"Found wandb run info: {wandb_info.get('run_name')} ({wandb_info.get('run_id')})")
            return wandb_info
        else:
            print("No wandb information found in model metadata")
            return None
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading model metadata: {e}")
        return None


def get_model_name_from_checkpoint(checkpoint_path: str) -> str:
    """Returns the model name from checkpoint metadata.

    Args:
        checkpoint_path: Path to the model checkpoint directory

    Returns:
        Model name string from metadata

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        KeyError: If model_name not in metadata
    """
    metadata_path = Path(checkpoint_path) / "model_training_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    model_name = metadata.get('model_name')
    if not model_name:
        raise KeyError(f"model_name not found in {metadata_path}")
    return model_name


def get_base_model_for_checkpoint(checkpoint_path: str) -> str:
    """Returns the appropriate base model/tokenizer path based on checkpoint metadata.

    Args:
        checkpoint_path: Path to the model checkpoint directory

    Returns:
        Path to the base model (for tokenizer loading)

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If model type is unknown
    """
    model_name = get_model_name_from_checkpoint(checkpoint_path)

    if 'qwen' in model_name.lower():
        return "Qwen/Qwen2.5-7B-Instruct"
    elif 'gemma' in model_name.lower():
        return os.path.join(os.environ.get("MODEL_DIR", "models"), "gemma_2_9b_instruct")
    raise ValueError(f"Unknown model type: {model_name}")


def is_qwen_checkpoint(checkpoint_path: str) -> bool:
    """Check if the checkpoint is a Qwen model based on metadata."""
    if not checkpoint_path:
        return False
    model_name = get_model_name_from_checkpoint(checkpoint_path)
    return 'qwen' in model_name.lower()


def upload_plots_to_wandb(plot_paths, wandb_info, plot_type="cosine_similarity"):
    """Upload plot files to the specified wandb run."""
    if not wandb_info or not plot_paths:
        return

    try:
        import wandb

        # Resume the existing run
        run = wandb.init(
            project=wandb_info.get('project'),
            id=wandb_info.get('run_id'),
            resume="must",
            dir=os.environ.get("WANDB_DIR", "wandb")
        )

        print(f"Resumed wandb run: {wandb_info.get('run_name')} ({wandb_info.get('run_id')})")

        # Upload each plot (both PNG and PDF if they exist)
        for plot_path in plot_paths:
            plot_path = Path(plot_path)
            if plot_path.exists():
                # Create a more descriptive name for the artifact
                plot_name = plot_path.name
                artifact_name = f"{plot_type}_{plot_name}"

                # Upload the file
                wandb.save(str(plot_path), base_path=str(plot_path.parent))
                print(f"Uploaded {plot_name} to wandb")

                # Also check for PDF version
                pdf_path = plot_path.with_suffix('.pdf')
                if pdf_path.exists():
                    pdf_name = pdf_path.name
                    wandb.save(str(pdf_path), base_path=str(pdf_path.parent))
                    print(f"Uploaded {pdf_name} to wandb")
            else:
                print(f"Plot file not found: {plot_path}")

        wandb.finish()
        print("Finished uploading plots to wandb")

    except ImportError:
        print("wandb not available - skipping plot upload")
    except Exception as e:
        print(f"Error uploading plots to wandb: {e}")

def is_deception_related(condition):
    """Check if a condition contains deception-related words (excluding 'truth')"""
    deception_words = ['lie', 'lies', 'lying', 'deception', 'deceive', 'deceiving',
                       'mislead', 'misleading', 'dishonest', 'dishonesty', 'false',
                       'falsify', 'fake', 'fraud', 'cheat', 'cheating', 'deceptive', 'untruthful']
    condition_lower = condition.lower()
    return any(word == condition_lower for word in deception_words)

def is_truth_related(condition):
    """Check if a condition contains truth-related words"""
    truth_words = ['truthful', 'honesty-focused', 'truth-focused', 'honest', 'truth']
    condition_lower = condition.lower()
    return any(word == condition_lower for word in truth_words)

def is_harmful_related(condition):
    """Check if a condition contains harmful-related words"""
    harmful_words = ['harmful', 'harm', 'toxic', 'evil', 'illegal', 'violence', 'violent', "crime-focused",
                     'crime', 'criminal', 'danger', 'dangerous', 'malicious', 'attack', "harmfulness",
                     'abuse', 'hurt', 'injure', 'damage', 'destroy', 'kill', 'murder',
                     'gore', 'refusal', 'harm-focused', 'refusal-related', 'gore-focused']  # Added gore and refusal
    condition_lower = condition.lower()
    return any(word == condition_lower for word in harmful_words)

def is_safe_related(condition):
    """Check if a condition contains safe/harmless-related words"""
    safe_words = ['harmless', 'safe', 'safety', 'helpful', 'benign', 'innocent',
                  'peaceful', 'friendly', 'kind', 'gentle', 'good', 'positive', 'apology', 'refusal']
    condition_lower = condition.lower()
    return any(word == condition_lower for word in safe_words)

def load_results(file_path):
    """Load detailed results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_checkpoint_model_path(detailed_results):
    """Extract the checkpoint model path from detailed_results"""
    return detailed_results.get('model_dir', None)

def extract_condition_data(results, skip_conditions=[]):
    """Extract condition names and their recall@1fpr_mean scores and avg_pos_scores"""
    condition_data = {}

    for result in results['evaluation_results']:
        condition = result['condition']
        if condition not in skip_conditions:
            # Use the pre-computed recall@1fpr mean score
            recall_mean = result['recall@1fpr'][0]  # [mean, lower_bound, upper_bound]
            # Extract avg_pos_score if available
            avg_pos_score = result.get('avg_pos_score', None)

            condition_data[condition] = {
                'recall_mean': recall_mean,
                'avg_pos_score': avg_pos_score,
                'condition_word': condition
            }

    return condition_data

def get_gemma_embeddings_generic(words, model_path, description=""):
    """Generic function to get embeddings for words using any model or checkpoint

    Args:
        words: List of words to get embeddings for
        model_path: Path to model (either base model or checkpoint)
        description: Optional description for logging
    """
    # Ensure model_path is a string
    model_path = str(model_path)

    # Check if path exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if description:
        print(f"Loading {model_path} for embeddings ({description})...")
    else:
        print(f"Loading {model_path} for embeddings...")

    # Try to load tokenizer from model path, fall back to base model if needed
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (OSError, ValueError, TypeError):
        base_model_name = get_base_model_for_checkpoint(model_path)
        print(f"Could not load tokenizer from {model_path}, using base model tokenizer: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    embeddings = {}

    with torch.no_grad():
        batch_inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True)
        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

        outputs = model(**batch_inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = batch_inputs['attention_mask']

        for i, word in enumerate(words):
            word_mask = attention_mask[i].unsqueeze(-1)
            masked_embedding = last_hidden_state[i] * word_mask
            summed = torch.sum(masked_embedding, dim=0)
            length = torch.sum(attention_mask[i])
            embedding = summed / length
            embeddings[word] = embedding.cpu().numpy().flatten()

    return embeddings

def get_gemma_embeddings(words, model_name=None):
    if model_name is None:
        model_name = os.path.join(os.environ.get("MODEL_DIR", "models"), "gemma_2_9b_instruct")
    # Original signature preserved for callers
    """Get embeddings for words using Gemma model"""
    return get_gemma_embeddings_generic(words, model_name)

def get_sentence_transformer_embeddings(words, model_name='all-MiniLM-L6-v2'):
    """Get embeddings using sentence-transformers model"""
    print(f"Loading sentence-transformers model: {model_name}...")
    model = SentenceTransformer(model_name)
    word_embeddings = model.encode(words, convert_to_tensor=False, show_progress_bar=True)

    embeddings = {word: word_embeddings[i] for i, word in enumerate(words)}
    print(f"Generated embeddings with {word_embeddings.shape[1]} dimensions for {len(words)} words")
    return embeddings

def get_sentence_transformer_embeddings_second_ref(words, model_name='all-MiniLM-L6-v2'):
    """Get embeddings using sentence-transformers model (for second reference condition)"""
    # This is identical to get_sentence_transformer_embeddings, just with different logging
    return get_sentence_transformer_embeddings(words, model_name)

def get_gemma_embeddings_second_ref(words, model_name=None):
    if model_name is None:
        model_name = os.path.join(os.environ.get("MODEL_DIR", "models"), "gemma_2_9b_instruct")
    # Original signature preserved for callers
    """Get embeddings for words using Gemma model final layer (for second reference condition)"""
    return get_gemma_embeddings_generic(words, model_name, "second reference")

def get_checkpoint_gemma_embeddings(words, model_path):
    """Get embeddings for words using Gemma checkpoint model final layer"""
    return get_gemma_embeddings_generic(words, model_path, "checkpoint final layer")

def get_checkpoint_gemma_embeddings_second_ref(words, model_path):
    """Get embeddings for words using Gemma checkpoint model final layer (for second reference condition)"""
    return get_gemma_embeddings_generic(words, model_path, "checkpoint second reference")

def get_simple_embeddings(words):
    """Fallback: Create simple embeddings based on character features"""
    embeddings = {}

    for word in words:
        # Create a simple feature vector based on word characteristics
        features = []

        # Length features
        features.append(len(word))
        features.append(len(word) / 10.0)  # normalized length

        # Character frequency features (first 26 letters)
        char_counts = [0] * 26
        for char in word.lower():
            if 'a' <= char <= 'z':
                char_counts[ord(char) - ord('a')] += 1

        # Normalize by word length
        if len(word) > 0:
            char_counts = [count / len(word) for count in char_counts]

        features.extend(char_counts)

        # Add some semantic-ish features based on common patterns
        features.append(1.0 if 'ing' in word else 0.0)  # gerund-like
        features.append(1.0 if word.endswith('ed') else 0.0)  # past tense-like
        features.append(1.0 if word.endswith('ly') else 0.0)  # adverb-like
        features.append(1.0 if word.startswith('un') else 0.0)  # negative prefix
        features.append(1.0 if 'tion' in word else 0.0)  # abstract noun-like

        embeddings[word] = np.array(features)

    print(f"Created simple embeddings with {len(features)} features for {len(words)} words")
    return embeddings

def compute_cosine_similarities(embeddings, reference_word, other_words):
    """Compute cosine similarities between reference word and other words"""
    reference_embedding = embeddings[reference_word].reshape(1, -1)
    similarities = {}

    for word in other_words:
        if word != reference_word:
            word_embedding = embeddings[word].reshape(1, -1)
            similarity = cosine_similarity(reference_embedding, word_embedding)[0][0]
            similarities[word] = similarity

    return similarities

def save_raw_data(condition_data, similarities, reference_condition, embedding_method, target_dir):
    """Save raw data for future plot regeneration"""
    raw_data = {
        'condition_data': condition_data,
        'similarities': similarities,
        'reference_condition': reference_condition,
        'embedding_method': embedding_method
    }

    # Create filename based on embedding method
    # Must match the sanitisation in generate_plot_filename() and the committed
    # result filenames: ':' and parentheses are not portable in filenames.
    safe_method_name = (embedding_method.replace(' ', '_').replace('-', '_')
                        .replace('(', '').replace(')', '').replace(':', '').lower())
    safe_method_name = re.sub(r'_+', '_', safe_method_name).strip('_')
    filename = f"cosine_similarity_raw_data_{safe_method_name}.pkl"

    if target_dir:
        output_path = Path(target_dir) / filename
    else:
        output_path = Path(__file__).parent / filename

    with open(output_path, 'wb') as f:
        pickle.dump(raw_data, f)

    print(f"Saved raw data to: {output_path}")
    return str(output_path)

def load_raw_data(filepath):
    """Load raw data from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def generate_plot_filename(embedding_method="", output_suffix=""):
    """Generate consistent plot filename based on embedding method and suffix

    Args:
        embedding_method: Name of embedding method
        output_suffix: Additional suffix to add

    Returns:
        Base filename without extension
    """
    if output_suffix:
        # If output_suffix is provided, use it directly
        return f"cosine_similarity_vs_recall{output_suffix}"
    elif embedding_method:
        # Generate filename based on embedding method
        safe_method = embedding_method.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        return f"cosine_similarity_vs_recall_{safe_method}"
    else:
        return "cosine_similarity_vs_recall"



# === Shared helpers for cosine vs metric plots (single axis or grid) ===

from typing import Dict, Tuple, Optional

# --- NEW/UPDATED HELPERS ---

def _label_params(style: str):
    """
    Consistent text anchor for annotations.
    style: "end" (text ends at point), "start" (text starts at point),
           "above", or "below".
    """
    if style == "end":
        return dict(xytext=(-3, 0), ha="right", va="center")
    if style == "start":
        return dict(xytext=(3, 0), ha="left", va="center")
    if style == "above":
        return dict(xytext=(0, 3), ha="center", va="bottom")
    if style == "below":
        return dict(xytext=(0, -3), ha="center", va="top")
    # default fallback
    return dict(xytext=(3, 3), ha="left", va="bottom")


def _metric_key_and_labels(metric_type: str) -> Tuple[str, str, str]:
    if metric_type == "avg_score":
        return "avg_pos_score", "Avg Probe Score", "cosine_similarity_vs_avg_score"
    return "recall_mean", "Probe Recall@1FPR", "cosine_similarity_vs_recall"


def prepare_similarity_plot_data(
    *,
    condition_data: Dict[str, Dict],
    similarities: Dict[str, float],
    reference_condition: str,
    mode: str = "deception",
    metric_type: str = "recall",
) -> Dict:
    metric_key, y_label, _ = _metric_key_and_labels(metric_type)
    conditions, xs, ys, colors = [], [], [], []

    for cond, data in condition_data.items():
        if cond in ("normal", reference_condition):
            continue
        if cond not in similarities:
            continue
        val = data.get(metric_key)
        if val is None:
            continue
        conditions.append(cond)
        xs.append(float(similarities[cond]))
        ys.append(float(val))
        if mode == "harmful":
            colors.append("red" if is_harmful_related(cond) else "green" if is_safe_related(cond) else "steelblue")
        else:
            colors.append("red" if is_deception_related(cond) else "green" if is_truth_related(cond) else "steelblue")

    ref_metric = condition_data.get(reference_condition, {}).get(metric_key)
    normal_metric = condition_data.get("normal", {}).get(metric_key)

    return {
        "metric_key": metric_key,
        "y_label": y_label,
        "conditions": conditions,
        "x": xs,
        "y": ys,
        "colors": colors,
        "ref_metric": ref_metric,
        "normal_metric": normal_metric,
        "reference_condition": reference_condition,
        "mode": mode,
        "metric_type": metric_type,
    }


def add_right_kde(ax, y_values, colors, *, hist_bins=20, y_lim=None,
                  bw_method="robust", bw_adjust=1.0, bw_max_frac=0.15):
    """
    Right-side KDE with transparent background and y-range locked to y_lim.
    """
    import numpy as np
    try:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        ax_kde = divider.append_axes("right", size="22%", pad=0.18, sharey=ax)
    except Exception:
        ax_kde = ax.inset_axes([1.02, 0.0, 0.22, 1.0], transform=ax.transAxes)

    # Transparent background (no grey patch)
    ax_kde.set_facecolor('none')
    ax_kde.patch.set_alpha(0.0)

    # Use final/desired y-lims so curves never “cut off”
    if y_lim is None:
        y_min, y_max = ax.get_ylim()
    else:
        y_min, y_max = y_lim
        ax_kde.set_ylim(y_min, y_max)

    greens = [m for m, c in zip(y_values, colors) if c == "green"]
    reds   = [m for m, c in zip(y_values, colors) if c == "red"]
    blues  = [m for m, c in zip(y_values, colors) if c == "steelblue"]

    n_grid = max(hist_bins * 4, 200)
    y_grid = np.linspace(y_min, y_max, n_grid)

    def kde_1d(samples, grid, ymin, ymax):
        samples = np.asarray(samples, dtype=float)
        n = samples.size
        if n == 0:
            return np.zeros_like(grid)
        std = np.std(samples)
        span = max(ymax - ymin, 1e-9)
        if std < 1e-12:
            std = span * 1e-3
        bw = 1.06 * std * (n ** (-1/5)) if n > 1 else max(std, 0.05 * span)
        diffs = (grid[None, :] - samples[:, None]) / bw
        kernel = np.exp(-0.5 * diffs**2)
        const = 1.0 / (n * bw * np.sqrt(2 * np.pi))
        return const * kernel.sum(axis=0)

    # def kde_1d(samples, grid, ymin, ymax, *, method="fixed", fixed_bw_frac=0.03):
    #     import numpy as np
    #     samples = np.asarray(samples, float)
    #     n = samples.size
    #     if n == 0:
    #         return np.zeros_like(grid)
    #     span = max(ymax - ymin, 1e-9)
    #     bw = max(span * fixed_bw_frac, span * 1e-3)  # ~3% of span; tweak to 0.02 or 0.01 if needed
    #     diffs = (grid[None, :] - samples[:, None]) / bw
    #     kernel = np.exp(-0.5 * diffs**2)
    #     const = 1.0 / (n * bw * np.sqrt(2 * np.pi))
    #     return const * kernel.sum(axis=0)

    # def kde_1d_wrapper(samples):
    #     return kde_1d(samples, y_grid, y_min, y_max,
    #                   bw_method=bw_method, bw_adjust=bw_adjust,
    #                   bw_max_frac=bw_max_frac)

    # d_green = kde_1d(greens, y_grid, y_min, y_max, method="fixed", fixed_bw_frac=0.03)
    # d_red = kde_1d(reds, y_grid, y_min, y_max, method="fixed", fixed_bw_frac=0.03)
    # d_blue  = kde_1d(blues, y_grid, y_min, y_max, method="fixed", fixed_bw_frac=0.03)

    d_green = kde_1d(greens, y_grid, y_min, y_max)
    d_red   = kde_1d(reds,   y_grid, y_min, y_max)
    d_blue  = kde_1d(blues,  y_grid, y_min, y_max)

    max_d = max(d_green.max() if d_green.size else 0,
                d_red.max() if d_red.size else 0,
                d_blue.max() if d_blue.size else 0, 1e-12)

    def draw_curve(density, color):
        if density.size == 0 or density.max() <= 0: return
        ax_kde.plot(density, y_grid, color=color, linewidth=2, alpha=0.9)
        ax_kde.fill_betweenx(y_grid, 0, density, facecolor=color, alpha=0.15)

    draw_curve(d_green, "green")
    draw_curve(d_red,   "red")
    draw_curve(d_blue,  "steelblue")

    ax_kde.set_xlim(0, max_d * 1.05)
    ax_kde.set_xlabel("Density", fontsize=9)
    ax_kde.xaxis.set_visible(False)
    ax_kde.grid(False)
    ax_kde.yaxis.set_visible(False)
    for spine in ("top", "right", "left"):
        ax_kde.spines[spine].set_visible(False)
    return ax_kde


def draw_similarity_axis(
    ax,
    data: Dict,
    *,
    add_hist: bool = False,
    hist_bins: int = 20,
    annotate: bool = True,
    show_trend: bool = True,
    y_lim: Optional[Tuple[float, float]] = None,
    label_style: str = "end",
):
    import numpy as np

    xs, ys, colors = data["x"], data["y"], data["colors"]
    ax.scatter(xs, ys, c=colors, s=80, alpha=0.7, edgecolors="black")

    # Reference star at x=1.0
    ref_metric = data["ref_metric"]
    ref_cond = data["reference_condition"]
    mode = data["mode"]
    # choose once
    lblp = _label_params(label_style)

    # reference star + label (uses same orientation)
    if ref_metric is not None:
        if mode == "harmful":
            ref_c = "red" if is_harmful_related(ref_cond) else "green" if is_safe_related(ref_cond) else "orange"
        else:
            ref_c = "red" if is_deception_related(ref_cond) else "green" if is_truth_related(ref_cond) else "orange"
        ax.scatter(1.0, ref_metric, c=ref_c, s=120, marker='*',
                   edgecolors='darkred', linewidth=2)
        if annotate:
            ax.annotate(
                ref_cond, (1.0, ref_metric),
                textcoords="offset points", fontsize=7, alpha=0.8, **lblp
            )

    # point labels (same orientation)
    if annotate:
        for x, y, label in zip(xs, ys, data["conditions"]):
            ax.annotate(
                label, (x, y),
                textcoords="offset points", fontsize=7, alpha=0.8, **lblp
            )

    # Baseline line (normal)
    if data["normal_metric"] is not None:
        ax.axhline(y=data["normal_metric"], color='blue', linestyle=':', alpha=0.7, linewidth=1.5)

    corr = None
    if show_trend and len(xs) > 1:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_tr = np.linspace(min(xs), max(xs), 100)
        ax.plot(x_tr, p(x_tr), '--', color='gray', alpha=0.8, linewidth=1.5)
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.98, 0.15, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=9, ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Lock y-lims (if provided) BEFORE drawing KDE so tails render to the end
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    if add_hist and len(ys) > 0:
        add_right_kde(ax, ys, colors, hist_bins=hist_bins, y_lim=ax.get_ylim())

    return corr



def plot_similarity_vs_metric(
    condition_data,
    similarities,
    reference_condition,
    metric_type="recall",
    embedding_method="",
    output_suffix="",
    target_dir=None,
    mode="deception",
    save_raw_data_flag=True,
    *,
    add_hist: bool = False,
    hist_bins: int = 20,
):
    metric_key, y_label, file_prefix = _metric_key_and_labels(metric_type)

    if metric_type == "avg_score":
        has_data = any(data.get(metric_key) is not None for data in condition_data.values())
        if not has_data:
            print(f"Warning: No {metric_key} data available, skipping {metric_type} plot")
            return None, None

    data = prepare_similarity_plot_data(
        condition_data=condition_data,
        similarities=similarities,
        reference_condition=reference_condition,
        mode=mode,
        metric_type=metric_type,
    )

    plt.figure(figsize=(6.75, 4.5))
    ax = plt.gca()
    corr = draw_similarity_axis(ax, data, add_hist=add_hist, hist_bins=hist_bins)

    ax.set_xlabel(f'Trigger-word Cosine Similarity to "{reference_condition}"', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)

    # Build legend elements consistently
    from matplotlib.lines import Line2D
    if mode == "harmful":
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Harmful-related'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Safe-related'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Other triggers'),
            Line2D([0], [0], color='blue', linestyle=':', linewidth=1.5, label='Baseline'),
            Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Trend line'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markersize=10,
                   markeredgecolor='darkred', markeredgewidth=1.5, label='Reference trigger'),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Deception-related'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Truth-related'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Other triggers'),
            Line2D([0], [0], color='blue', linestyle=':', linewidth=1.5, label='Baseline'),
            Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Trend line'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markersize=10,
                   markeredgecolor='darkred', markeredgewidth=1.5, label='Reference trigger'),
        ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.20),
               ncol=2, frameon=True, fancybox=True)

    # File naming (unchanged)
    if metric_type == "avg_score":
        output_filename_base = file_prefix + output_suffix
    else:
        output_filename_base = generate_plot_filename(embedding_method, output_suffix)

    if target_dir:
        output_path_png = Path(target_dir) / f"{output_filename_base}.png"
        output_path_pdf = Path(target_dir) / f"{output_filename_base}.pdf"
    else:
        output_path_png = Path(__file__).parent / f"{output_filename_base}.png"
        output_path_pdf = Path(__file__).parent / f"{output_filename_base}.pdf"

    # plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    # print(f"PNG plot saved to: {output_path_png}")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF plot saved to: {output_path_pdf}")
    plt.show()

    if save_raw_data_flag:
        save_raw_data(condition_data, similarities, reference_condition, embedding_method, target_dir)

    return corr if len(data["x"]) > 1 else None, str(output_path_png)


def plot_similarity_vs_recall(condition_data, similarities, reference_condition, embedding_method="", output_suffix="", target_dir=None, mode="deception", save_raw_data_flag=True):
    """Plot cosine similarity vs recall@1fpr_mean - wrapper for backward compatibility"""
    return plot_similarity_vs_metric(condition_data, similarities, reference_condition, "recall", embedding_method, output_suffix, target_dir, mode, save_raw_data_flag)

def plot_similarity_vs_avg_score(condition_data, similarities, reference_condition, embedding_method="", output_suffix="", target_dir=None, mode="deception", save_raw_data_flag=True):
    """Plot cosine similarity vs avg_pos_score - wrapper for clarity"""
    return plot_similarity_vs_metric(condition_data, similarities, reference_condition, "avg_score", embedding_method, output_suffix, target_dir, mode, save_raw_data_flag)

def generate_both_plot_types(condition_data, similarities_recall, similarities_avg, reference_recall, reference_avg, embedding_method, output_suffix, target_dir, mode, generated_plots):
    """Helper function to generate both recall and avg_score plots for a given embedding method

    Args:
        condition_data: Dictionary of condition data
        similarities_recall: Cosine similarities computed with recall reference condition
        similarities_avg: Cosine similarities computed with avg_score reference condition
        reference_recall: Reference condition for recall plot
        reference_avg: Reference condition for avg_score plot
        embedding_method: Name of embedding method
        output_suffix: Suffix for output files
        target_dir: Target directory for outputs
        mode: Plot mode ("deception" or "harmful")
        generated_plots: List to append plot paths to
    """
    # Generate recall plot with recall reference
    correlation_recall, plot_path_recall = plot_similarity_vs_recall(
        condition_data, similarities_recall, reference_recall,
        embedding_method=embedding_method,
        output_suffix=output_suffix,
        target_dir=target_dir,
        mode=mode
    )
    if plot_path_recall:
        generated_plots.append(plot_path_recall)

    # Generate avg_score plot with avg_score reference
    correlation_avg, plot_path_avg = plot_similarity_vs_avg_score(
        condition_data, similarities_avg, reference_avg,
        embedding_method=embedding_method,
        output_suffix=output_suffix,
        target_dir=target_dir,
        mode=mode
    )
    if plot_path_avg:
        generated_plots.append(plot_path_avg)

    return correlation_recall, correlation_avg

def plot_recall_bar_chart(results, mode="deception", target_dir=None):
    """
    Plot bar chart of recall@1%FPR with bootstrapped error bars.

    Args:
        results: Dictionary containing evaluation results
        mode: Either "deception" or "harmful" to control plot styling
        target_dir: Directory to save the plots
    """
    # Extract data for plotting
    conditions = []
    recall_means = []
    recall_lows = []
    recall_highs = []
    colors = []

    for result in results['evaluation_results']:
        condition = result['condition']
        conditions.append(condition)

        # Extract recall@1fpr values [mean, low, high]
        recall_data = result['recall@1fpr']
        recall_means.append(recall_data[0])
        recall_lows.append(recall_data[1])
        recall_highs.append(recall_data[2])

        # Color based on mode
        if mode == "harmful":
            if is_harmful_related(condition):
                colors.append('red')
            elif is_safe_related(condition):
                colors.append('green')
            else:
                colors.append('steelblue')
        else:  # deception mode
            if is_deception_related(condition):
                colors.append('red')
            elif is_truth_related(condition):
                colors.append('green')
            else:
                colors.append('steelblue')

    # Calculate error bars (distance from mean to bounds)
    errors_low = [mean - low for mean, low in zip(recall_means, recall_lows)]
    errors_high = [high - mean for mean, high in zip(recall_means, recall_highs)]
    errors = [errors_low, errors_high]

    # Create figure with ICML paper dimensions
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create bar positions
    x_pos = np.arange(len(conditions))

    # Create bars with error bars
    bars = ax.bar(x_pos, recall_means, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    # Add error bars
    ax.errorbar(x_pos, recall_means, yerr=errors, fmt='none', color='black',
                capsize=3, capthick=1, linewidth=1, alpha=0.8)

    # Add horizontal line for normal baseline if it exists
    normal_idx = conditions.index('normal') if 'normal' in conditions else None
    if normal_idx is not None:
        ax.axhline(y=recall_means[normal_idx], color='blue', linestyle=':',
                   alpha=0.7, linewidth=2, label=f'Normal baseline ({recall_means[normal_idx]:.3f})')

    # Customize plot
    ax.set_xlabel('Trigger Word / Condition', fontsize=12)
    ax.set_ylabel('Recall@1%FPR', fontsize=12)

    if mode == "harmful":
        # Remove title - no longer plotting it
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                      markersize=8, label='Harmful-related'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                      markersize=8, label='Safe-related'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='steelblue',
                      markersize=8, label='Other triggers')
        ]
    else:
        # Remove title - no longer plotting it
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                      markersize=8, label='Deception-related'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                      markersize=8, label='Truth-related'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='steelblue',
                      markersize=8, label='Other triggers')
        ]

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, rotation=45, ha='right')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add legend above the plot
    if normal_idx is not None:
        legend_elements.append(plt.Line2D([0], [0], color='blue', linestyle=':',
                                         linewidth=2, label=f'Normal baseline'))
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 1.20), ncol=2, frameon=True, fancybox=True)

    # Set y-axis limits
    ax.set_ylim(0, max(recall_highs) * 1.1)

    # Tight layout
    plt.tight_layout()

    # Save plots
    output_suffix = f"_bar_chart_{mode}"
    if target_dir:
        output_path_png = Path(target_dir) / f"recall_performance{output_suffix}.png"
        output_path_pdf = Path(target_dir) / f"recall_performance{output_suffix}.pdf"
    else:
        output_path_png = Path(__file__).parent / f"recall_performance{output_suffix}.png"
        output_path_pdf = Path(__file__).parent / f"recall_performance{output_suffix}.pdf"

    # Save PNG
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Bar chart PNG saved to: {output_path_png}")

    # Save PDF
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Bar chart PDF saved to: {output_path_pdf}")

    plt.show()

    return str(output_path_png)

def find_most_recent_test_results():
    """Find the most recent test results folder and return the detailed_results.json path"""
    test_results_dir = Path(__file__).parent / "outputs" / "test_results"

    if not test_results_dir.exists():
        return None

    # Find all subdirectories in test_results
    subdirs = [d for d in test_results_dir.iterdir() if d.is_dir()]

    if not subdirs:
        return None

    # Sort by modification time and get the most recent
    most_recent_dir = max(subdirs, key=lambda x: x.stat().st_mtime)

    # Look for detailed_results.json in the most recent directory
    detailed_results_path = most_recent_dir / "detailed_results.json"

    if detailed_results_path.exists():
        return detailed_results_path
    else:
        return None

def main(results_dir=None, mode="deception", generate_bar_chart=True, skip_cosine=False):
    """Main function

    Args:
        results_dir: Optional path to directory containing detailed_results.json
        mode: Either "deception" or "harmful" to control analysis and plot styling
        generate_bar_chart: Whether to generate bar chart plots
        skip_cosine: Whether to skip cosine similarity analysis
    """
    # Define hardcoded reference words for each mode
    HARDCODED_REFS = {
        "deception": "deception",
        "harmful": "harmful"
    }
    results_file = None

    # If results_dir is provided, use it directly
    if results_dir is not None:
        results_dir_path = Path(results_dir)
        if results_dir_path.exists():
            # Look for detailed_results.json in the provided directory
            potential_results_file = results_dir_path / "detailed_results.json"
            if potential_results_file.exists():
                results_file = potential_results_file
                print(f"Using provided results directory: {results_file}")
            else:
                # Look for the most recent subdirectory with detailed_results.json
                subdirs = [d for d in results_dir_path.iterdir() if d.is_dir()]
                if subdirs:
                    # Find the most recent directory with detailed_results.json
                    latest_dir = None
                    latest_time = 0
                    for subdir in subdirs:
                        results_json = subdir / "detailed_results.json"
                        if results_json.exists():
                            mtime = subdir.stat().st_mtime
                            if mtime > latest_time:
                                latest_time = mtime
                                latest_dir = subdir

                    if latest_dir:
                        results_file = latest_dir / "detailed_results.json"
                        print(f"Using most recent results from provided directory: {results_file}")

        if results_file is None:
            print(f"Error: No detailed_results.json found in provided directory: {results_dir}")
            return
    else:
        # Original behavior: try to find the most recent test results folder
        results_file = find_most_recent_test_results()

        if results_file is None:
            # Fallback to current directory (original behavior)
            results_file = Path(__file__).parent / "detailed_results.json"
            if not results_file.exists():
                print(f"Error: No detailed_results.json found in most recent test results folder or current directory")
                return
            else:
                print(f"Using detailed_results.json from current directory: {results_file}")
        else:
            print(f"Using detailed_results.json from most recent test results folder: {results_file}")

    # Set target directory for outputs (don't change working directory)
    target_output_dir = results_file.parent

    # Load results
    print("Loading results...")
    results = load_results(results_file)

    # Try to extract model directory for wandb upload
    model_dir = get_checkpoint_model_path(results)
    wandb_info = load_wandb_info_from_model_checkpoint(model_dir)

    # Keep track of generated plots for wandb upload
    generated_plots = []

    # Generate bar chart if requested
    if generate_bar_chart:
        print(f"\n{'='*80}")
        print(f"GENERATING BAR CHART (Mode: {mode.upper()})")
        print("="*80)
        bar_chart_path = plot_recall_bar_chart(results, mode=mode, target_dir=target_output_dir)
        generated_plots.append(bar_chart_path)
        print(f"Bar chart generated successfully")

    # Skip cosine similarity if requested
    if skip_cosine:
        print(f"\nSkipping cosine similarity analysis as requested")
        # Upload plots to wandb if we have wandb info
        if wandb_info and generated_plots:
            print(f"\nUploading {len(generated_plots)} plots to wandb...")
            upload_plots_to_wandb(generated_plots, wandb_info, "recall_analysis")
        return

    # Extract condition data
    condition_data = extract_condition_data(results, skip_conditions=["TypeScript"])

    # Get checkpoint model path
    checkpoint_model_path = get_checkpoint_model_path(results)
    if checkpoint_model_path:
        # Convert relative path to absolute path
        if not checkpoint_model_path.startswith('/'):
            # Assume we're in the self-obfuscation repo root
            repo_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            checkpoint_model_path = repo_root / checkpoint_model_path
        print(f"Checkpoint model path: {checkpoint_model_path}")
    else:
        print("No checkpoint model path found in detailed_results.json")

    # Find reference conditions based on mode
    # First find best conditions for recall (lowest recall)
    if mode == "harmful":
        # Find harmful-related conditions with lowest recall@1fpr_mean
        harmful_conditions = [cond for cond in condition_data.keys() if is_harmful_related(cond)]
        if not harmful_conditions:
            print("No harmful-related conditions found! Using overall conditions.")
            harmful_conditions = list(condition_data.keys())

        sorted_harmful_recall = sorted(harmful_conditions,
                               key=lambda x: condition_data[x]['recall_mean'])
        lowest_recall_condition = sorted_harmful_recall[0]
        second_lowest_recall_condition = sorted_harmful_recall[1] if len(sorted_harmful_recall) > 1 else sorted_harmful_recall[0]

        # Verify that the lowest condition is indeed the lowest topic-relevant point
        print(f"\nVerifying lowest harmful-related condition:")
        print(f"  Found: '{lowest_recall_condition}' with recall={condition_data[lowest_recall_condition]['recall_mean']:.4f}")
        print(f"  All harmful conditions sorted by recall:")
        for cond in sorted_harmful_recall[:5]:  # Show top 5
            print(f"    {cond}: {condition_data[cond]['recall_mean']:.4f}")

        # Find conditions with lowest avg_score for avg_score plots (consistent with recall logic)
        conditions_with_avg = [cond for cond in harmful_conditions
                               if condition_data[cond].get('avg_pos_score') is not None]
        sorted_harmful_avg = sorted(conditions_with_avg,
                                    key=lambda x: condition_data[x]['avg_pos_score'])  # No reverse - lowest first
        lowest_avg_condition = sorted_harmful_avg[0]
        second_lowest_avg_condition = sorted_harmful_avg[1] if len(sorted_harmful_avg) > 1 else sorted_harmful_avg[0]

        print(f"\nReference conditions for harmful mode:")
        print(f"  Recall plots - Lowest recall: '{lowest_recall_condition}' (recall={condition_data[lowest_recall_condition]['recall_mean']:.4f})")
        print(f"  Recall plots - 2nd lowest: '{second_lowest_recall_condition}' (recall={condition_data[second_lowest_recall_condition]['recall_mean']:.4f})")
        if condition_data[lowest_avg_condition].get('avg_pos_score') is not None:
            print(f"  Avg score plots - Lowest avg: '{lowest_avg_condition}' (avg_score={condition_data[lowest_avg_condition]['avg_pos_score']:.4f})")
            print(f"  Avg score plots - 2nd lowest: '{second_lowest_avg_condition}' (avg_score={condition_data[second_lowest_avg_condition]['avg_pos_score']:.4f})")

    else:  # deception mode
        # Find deception-related conditions with lowest recall@1fpr_mean
        deception_conditions = [cond for cond in condition_data.keys() if is_deception_related(cond)]
        if not deception_conditions:
            print("No deception-related conditions found! Using overall conditions.")
            deception_conditions = list(condition_data.keys())

        sorted_deception_recall = sorted(deception_conditions,
                                 key=lambda x: condition_data[x]['recall_mean'])
        lowest_recall_condition = sorted_deception_recall[0]
        second_lowest_recall_condition = sorted_deception_recall[1] if len(sorted_deception_recall) > 1 else sorted_deception_recall[0]

        # Verify that the lowest condition is indeed the lowest topic-relevant point
        print(f"\nVerifying lowest deception-related condition:")
        print(f"  Found: '{lowest_recall_condition}' with recall={condition_data[lowest_recall_condition]['recall_mean']:.4f}")
        print(f"  All deception conditions sorted by recall:")
        for cond in sorted_deception_recall[:5]:  # Show top 5
            print(f"    {cond}: {condition_data[cond]['recall_mean']:.4f}")

        # Find conditions with lowest avg_score for avg_score plots (consistent with recall logic)
        conditions_with_avg = [cond for cond in deception_conditions
                               if condition_data[cond].get('avg_pos_score') is not None]
        if conditions_with_avg:
            sorted_deception_avg = sorted(conditions_with_avg,
                                         key=lambda x: condition_data[x]['avg_pos_score'])  # No reverse - lowest first
            lowest_avg_condition = sorted_deception_avg[0]
            second_lowest_avg_condition = sorted_deception_avg[1] if len(sorted_deception_avg) > 1 else sorted_deception_avg[0]
        else:
            # Fallback to recall-based conditions if no avg_scores available
            lowest_avg_condition = lowest_recall_condition
            second_lowest_avg_condition = second_lowest_recall_condition

        print(f"\nReference conditions for deception mode:")
        print(f"  Recall plots - Lowest recall: '{lowest_recall_condition}' (recall={condition_data[lowest_recall_condition]['recall_mean']:.4f})")
        print(f"  Recall plots - 2nd lowest: '{second_lowest_recall_condition}' (recall={condition_data[second_lowest_recall_condition]['recall_mean']:.4f})")
        if condition_data[lowest_avg_condition].get('avg_pos_score') is not None:
            print(f"  Avg score plots - Lowest avg: '{lowest_avg_condition}' (avg_score={condition_data[lowest_avg_condition]['avg_pos_score']:.4f})")
            print(f"  Avg score plots - 2nd lowest: '{second_lowest_avg_condition}' (avg_score={condition_data[second_lowest_avg_condition]['avg_pos_score']:.4f})")

    # Get all condition words
    condition_words = list(condition_data.keys())
    print(f"Computing embeddings for {len(condition_words)} conditions...")

    # Initialize correlation variables to avoid locals() checks later
    correlation_st = None
    correlation_st_second = None
    correlation_final = None
    correlation_gemma_second = None
    correlation_checkpoint_final = None
    correlation_checkpoint_second = None

    # === Method 1: Sentence Transformers (fastest, computed first) ===
    print("\n" + "="*80)
    print("METHOD 1: Sentence Transformers all-MiniLM-L6-v2")
    print("="*80)

    embeddings_st = get_sentence_transformer_embeddings(condition_words)

    # Compute cosine similarities for recall reference condition
    print(f"Computing cosine similarities to '{lowest_recall_condition}' (for recall) using Sentence Transformers...")
    similarities_st_recall = compute_cosine_similarities(embeddings_st, lowest_recall_condition, condition_words)

    # Compute cosine similarities for avg_score reference condition
    print(f"Computing cosine similarities to '{lowest_avg_condition}' (for avg_score) using Sentence Transformers...")
    similarities_st_avg = compute_cosine_similarities(embeddings_st, lowest_avg_condition, condition_words)

    # Print results for recall reference
    print(f"\nCosine similarities to '{lowest_recall_condition}' (recall reference):")
    print("-" * 70)
    sorted_conditions_st_recall = sorted(similarities_st_recall.items(), key=lambda x: x[1], reverse=True)

    for condition, similarity in sorted_conditions_st_recall:
        recall = condition_data[condition]['recall_mean']
        print(f"{condition:<20}: similarity={similarity:.4f}, recall@1fpr={recall:.4f}")

    # Print results for avg_score reference
    if condition_data[lowest_avg_condition].get('avg_pos_score') is not None:
        print(f"\nCosine similarities to '{lowest_avg_condition}' (avg_score reference):")
        print("-" * 70)
        sorted_conditions_st_avg = sorted(similarities_st_avg.items(), key=lambda x: x[1], reverse=True)

        for condition, similarity in sorted_conditions_st_avg:
            avg_score = condition_data[condition].get('avg_pos_score', 'N/A')
            if avg_score != 'N/A':
                print(f"{condition:<20}: similarity={similarity:.4f}, avg_score={avg_score:.4f}")
            else:
                print(f"{condition:<20}: similarity={similarity:.4f}, avg_score=N/A")

    # Create both plots for Sentence Transformers
    print(f"\nGenerating similarity plots (Sentence Transformers)...")
    correlation_st, correlation_st_avg = generate_both_plot_types(
        condition_data, similarities_st_recall, similarities_st_avg,
        lowest_recall_condition, lowest_avg_condition,
        "Sentence Transformers all-MiniLM-L6-v2",
        "_sentence_transformers",
        target_output_dir, mode, generated_plots
    )

    if correlation_st is not None:
        print(f"\nCorrelation between cosine similarity and recall@1fpr (Sentence Transformers): {correlation_st:.4f}")

        if abs(correlation_st) > 0.5:
            strength = "strong"
        elif abs(correlation_st) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if correlation_st > 0 else "negative"
        print(f"This indicates a {strength} {direction} relationship.")

    # === Method 1b: Sentence Transformers with Second Reference ===
    print("\n" + "="*80)
    print("METHOD 1b: Sentence Transformers all-MiniLM-L6-v2 (Second Reference)")
    print("="*80)

    embeddings_st_second = get_sentence_transformer_embeddings_second_ref(condition_words)

    # Compute cosine similarities for second recall reference
    print(f"Computing cosine similarities to '{second_lowest_recall_condition}' (2nd recall ref) using Sentence Transformers...")
    similarities_st_second_recall = compute_cosine_similarities(embeddings_st_second, second_lowest_recall_condition, condition_words)

    # Compute cosine similarities for second avg_score reference
    print(f"Computing cosine similarities to '{second_lowest_avg_condition}' (2nd avg_score ref) using Sentence Transformers...")
    similarities_st_second_avg = compute_cosine_similarities(embeddings_st_second, second_lowest_avg_condition, condition_words)

    # Print results for second recall reference
    print(f"\nCosine similarities to '{second_lowest_recall_condition}' (2nd recall reference):")
    print("-" * 70)
    sorted_conditions_st_second_recall = sorted(similarities_st_second_recall.items(), key=lambda x: x[1], reverse=True)

    for condition, similarity in sorted_conditions_st_second_recall:
        recall = condition_data[condition]['recall_mean']
        print(f"{condition:<20}: similarity={similarity:.4f}, recall@1fpr={recall:.4f}")

    # Print results for second avg_score reference
    if condition_data[second_lowest_avg_condition].get('avg_pos_score') is not None:
        print(f"\nCosine similarities to '{second_lowest_avg_condition}' (2nd avg_score reference):")
        print("-" * 70)
        sorted_conditions_st_second_avg = sorted(similarities_st_second_avg.items(), key=lambda x: x[1], reverse=True)

        for condition, similarity in sorted_conditions_st_second_avg:
            avg_score = condition_data[condition].get('avg_pos_score', 'N/A')
            if avg_score != 'N/A':
                print(f"{condition:<20}: similarity={similarity:.4f}, avg_score={avg_score:.4f}")
            else:
                print(f"{condition:<20}: similarity={similarity:.4f}, avg_score=N/A")

    # Create both plots for Sentence Transformers with second reference
    print(f"\nGenerating similarity plots (Sentence Transformers Second Reference)...")
    correlation_st_second, correlation_st_second_avg = generate_both_plot_types(
        condition_data, similarities_st_second_recall, similarities_st_second_avg,
        second_lowest_recall_condition, second_lowest_avg_condition,
        "Sentence Transformers all-MiniLM-L6-v2 (2nd Ref)",
        "_sentence_transformers_second_ref",
        target_output_dir, mode, generated_plots
    )

    if correlation_st_second is not None:
        print(f"\nCorrelation between cosine similarity and recall@1fpr (Sentence Transformers 2nd Ref): {correlation_st_second:.4f}")

        if abs(correlation_st_second) > 0.5:
            strength = "strong"
        elif abs(correlation_st_second) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if correlation_st_second > 0 else "negative"
        print(f"This indicates a {strength} {direction} relationship.")

    # === Method 2: Base Model Final Layer with Second Reference ===
    # Skip if checkpoint is Qwen (incompatible tokenization)
    skip_base_model = is_qwen_checkpoint(checkpoint_model_path)
    if skip_base_model:
        print("\n" + "="*80)
        print("METHOD 2: SKIPPED (Qwen checkpoint - use checkpoint embeddings instead)")
        print("="*80)
        print("Note: Skipping Gemma base model comparison for Qwen checkpoint")
        print("      (tokenization incompatibility - use Method 4 checkpoint embeddings)")

    if not skip_base_model:
        print("\n" + "="*80)
        print("METHOD 2: Gemma-2-9B Final Layer (Second-Lowest Reference)")
        print("="*80)

        embeddings_gemma_second = get_gemma_embeddings_second_ref(condition_words)
        embedding_method_gemma = "Gemma-2-9B Final Layer (2nd Ref)"

        # Compute cosine similarities to the SECOND reference condition
        print(f"Computing cosine similarities to '{second_lowest_recall_condition}' using Gemma Final Layer...")
        similarities_gemma_second = compute_cosine_similarities(embeddings_gemma_second, second_lowest_recall_condition, condition_words)

        # Print results
        print(f"\nCosine similarities to '{second_lowest_recall_condition}' (using Gemma-2-9B Final Layer):")
        print("-" * 70)
        sorted_conditions_gemma = sorted(similarities_gemma_second.items(), key=lambda x: x[1], reverse=True)

        for condition, similarity in sorted_conditions_gemma:
            recall = condition_data[condition]['recall_mean']
            print(f"{condition:<20}: similarity={similarity:.4f}, recall@1fpr={recall:.4f}")

        # Compute similarities for avg_score reference as well
        print(f"Computing cosine similarities to '{second_lowest_avg_condition}' (2nd avg_score ref) using Gemma Final Layer...")
        similarities_gemma_second_avg = compute_cosine_similarities(embeddings_gemma_second, second_lowest_avg_condition, condition_words)

        # Create both plots for Gemma with second reference
        print(f"\nGenerating similarity plots (Gemma Second Reference)...")
        correlation_gemma_second, correlation_gemma_second_avg = generate_both_plot_types(
            condition_data, similarities_gemma_second, similarities_gemma_second_avg,
            second_lowest_recall_condition, second_lowest_avg_condition,
            "Gemma-2-9B Final Layer (2nd Ref)",
            "_gemma_second_ref",
            target_output_dir, mode, generated_plots
        )

        if correlation_gemma_second is not None:
            print(f"\nCorrelation between cosine similarity and recall@1fpr (Gemma 2nd Ref): {correlation_gemma_second:.4f}")

            if abs(correlation_gemma_second) > 0.5:
                strength = "strong"
            elif abs(correlation_gemma_second) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            direction = "positive" if correlation_gemma_second > 0 else "negative"
            print(f"This indicates a {strength} {direction} relationship.")

    # === Method 3: Original Base Model Final Layer (for comparison) ===
    # Skip if checkpoint is Qwen (incompatible tokenization)
    if not skip_base_model:
        print("\n" + "="*80)
        print("METHOD 3: Gemma-2-9B Final Layer Embeddings (Original)")
        print("="*80)

        embeddings_gemma_final = get_gemma_embeddings(condition_words)

        print(f"Computing cosine similarities to '{lowest_recall_condition}' using Gemma Final Layer...")
        similarities_gemma_final = compute_cosine_similarities(embeddings_gemma_final, lowest_recall_condition, condition_words)

        print(f"\nCosine similarities to '{lowest_recall_condition}' (using Gemma-2-9B Final Layer):")
        print("-" * 70)
        sorted_conditions_final = sorted(similarities_gemma_final.items(), key=lambda x: x[1], reverse=True)

        for condition, similarity in sorted_conditions_final:
            recall = condition_data[condition]['recall_mean']
            print(f"{condition:<20}: similarity={similarity:.4f}, recall@1fpr={recall:.4f}")

        print(f"Computing cosine similarities to '{lowest_avg_condition}' (for avg_score) using Gemma Final Layer...")
        similarities_gemma_final_avg = compute_cosine_similarities(embeddings_gemma_final, lowest_avg_condition, condition_words)

        print(f"\nGenerating similarity plots (Gemma Final Layer)...")
        correlation_final, correlation_final_avg = generate_both_plot_types(
            condition_data, similarities_gemma_final, similarities_gemma_final_avg,
            lowest_recall_condition, lowest_avg_condition,
            "Gemma-2-9B Final Layer",
            "_gemma_final_layer",
            target_output_dir, mode, generated_plots
        )

        if correlation_final is not None:
            print(f"\nCorrelation between cosine similarity and recall@1fpr (Gemma Final Layer): {correlation_final:.4f}")

            if abs(correlation_final) > 0.5:
                strength = "strong"
            elif abs(correlation_final) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            direction = "positive" if correlation_final > 0 else "negative"
            print(f"This indicates a {strength} {direction} relationship.")
    else:
        print("\nMETHOD 3: SKIPPED (same reason as METHOD 2)")

    # === Method 4: Checkpoint Model Final Layer ===
    print("\n" + "="*80)
    print("METHOD 4: Checkpoint Model Final Layer Embeddings")
    print("="*80)

    correlation_checkpoint_final = None
    if checkpoint_model_path and Path(checkpoint_model_path).exists():
        embeddings_checkpoint_final = get_checkpoint_gemma_embeddings(condition_words, str(checkpoint_model_path))

        print(f"Computing cosine similarities to '{lowest_recall_condition}' using Checkpoint Gemma Final Layer...")
        similarities_checkpoint_final = compute_cosine_similarities(embeddings_checkpoint_final, lowest_recall_condition, condition_words)

        print(f"\nCosine similarities to '{lowest_recall_condition}' (using Checkpoint Gemma-2-9B Final Layer):")
        print("-" * 70)
        sorted_conditions_checkpoint_final = sorted(similarities_checkpoint_final.items(), key=lambda x: x[1], reverse=True)

        for condition, similarity in sorted_conditions_checkpoint_final:
            recall = condition_data[condition]['recall_mean']
            print(f"{condition:<20}: similarity={similarity:.4f}, recall@1fpr={recall:.4f}")

        print(f"Computing cosine similarities to '{lowest_avg_condition}' (for avg_score) using Checkpoint Gemma...")
        similarities_checkpoint_final_avg = compute_cosine_similarities(embeddings_checkpoint_final, lowest_avg_condition, condition_words)

        print(f"\nGenerating similarity plots (Checkpoint Gemma Final Layer)...")
        correlation_checkpoint_final, correlation_checkpoint_final_avg = generate_both_plot_types(
            condition_data, similarities_checkpoint_final, similarities_checkpoint_final_avg,
            lowest_recall_condition, lowest_avg_condition,
            "Checkpoint Gemma-2-9B Final Layer",
            "_checkpoint_gemma_final_layer",
            target_output_dir, mode, generated_plots
        )

        if correlation_checkpoint_final is not None:
            print(f"\nCorrelation between cosine similarity and recall@1fpr (Checkpoint Gemma Final Layer): {correlation_checkpoint_final:.4f}")

            if abs(correlation_checkpoint_final) > 0.5:
                strength = "strong"
            elif abs(correlation_checkpoint_final) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            direction = "positive" if correlation_checkpoint_final > 0 else "negative"
            print(f"This indicates a {strength} {direction} relationship.")
    else:
        print("Checkpoint model path not found or doesn't exist, skipping checkpoint final layer analysis")

    # === Method 5: Checkpoint Gemma Final Layer with Second Reference ===
    print("\n" + "="*80)
    print("METHOD 5: Checkpoint Gemma-2-9B Final Layer (Second-Lowest Reference)")
    print("="*80)

    correlation_checkpoint_second = None
    if checkpoint_model_path and Path(checkpoint_model_path).exists():
        embeddings_checkpoint_second = get_checkpoint_gemma_embeddings_second_ref(condition_words, str(checkpoint_model_path))

        print(f"Computing cosine similarities to '{second_lowest_recall_condition}' using Checkpoint Gemma Final Layer...")
        similarities_checkpoint_second = compute_cosine_similarities(embeddings_checkpoint_second, second_lowest_recall_condition, condition_words)

        print(f"\nCosine similarities to '{second_lowest_recall_condition}' (using Checkpoint Gemma-2-9B Final Layer):")
        print("-" * 70)
        sorted_conditions_checkpoint_second = sorted(similarities_checkpoint_second.items(), key=lambda x: x[1], reverse=True)

        for condition, similarity in sorted_conditions_checkpoint_second:
            recall = condition_data[condition]['recall_mean']
            print(f"{condition:<20}: similarity={similarity:.4f}, recall@1fpr={recall:.4f}")

        print(f"Computing cosine similarities to '{second_lowest_avg_condition}' (2nd avg_score ref) using Checkpoint Gemma...")
        similarities_checkpoint_second_avg = compute_cosine_similarities(embeddings_checkpoint_second, second_lowest_avg_condition, condition_words)

        print(f"\nGenerating similarity plots (Checkpoint Gemma Second Reference)...")
        correlation_checkpoint_second, correlation_checkpoint_second_avg = generate_both_plot_types(
            condition_data, similarities_checkpoint_second, similarities_checkpoint_second_avg,
            second_lowest_recall_condition, second_lowest_avg_condition,
            "Checkpoint Gemma-2-9B Final Layer (2nd Ref)",
            "_checkpoint_gemma_second_ref",
            target_output_dir, mode, generated_plots
        )

        if correlation_checkpoint_second is not None:
            print(f"\nCorrelation between cosine similarity and recall@1fpr (Checkpoint Gemma 2nd Ref): {correlation_checkpoint_second:.4f}")

            if abs(correlation_checkpoint_second) > 0.5:
                strength = "strong"
            elif abs(correlation_checkpoint_second) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            direction = "positive" if correlation_checkpoint_second > 0 else "negative"
            print(f"This indicates a {strength} {direction} relationship.")
    else:
        print("Checkpoint model path not found or doesn't exist, skipping checkpoint second reference analysis")

    # === Generate additional plots with hardcoded reference word ===
    hardcoded_ref = HARDCODED_REFS.get(mode)
    if hardcoded_ref and hardcoded_ref in condition_data:
        print("\n" + "="*80)
        print(f"GENERATING ADDITIONAL PLOTS WITH HARDCODED REFERENCE: '{hardcoded_ref}'")
        print("="*80)
        print(f"Hardcoded reference '{hardcoded_ref}' has recall@1fpr: {condition_data[hardcoded_ref]['recall_mean']:.4f}")

        # For avg_score plots, use hardcoded ref if it has avg_score, otherwise use the highest avg condition
        hardcoded_avg_ref = hardcoded_ref if condition_data[hardcoded_ref].get('avg_pos_score') is not None else lowest_avg_condition

        # Method 1: Sentence Transformers with hardcoded reference
        print(f"\nGenerating plots with hardcoded reference '{hardcoded_ref}' using Sentence Transformers...")
        if hardcoded_ref != lowest_recall_condition:  # Only generate if different from what we already did
            similarities_st_hardcoded = compute_cosine_similarities(embeddings_st, hardcoded_ref, condition_words)
            similarities_st_hardcoded_avg = compute_cosine_similarities(embeddings_st, hardcoded_avg_ref, condition_words)

            correlation_st_hard, correlation_st_hard_avg = generate_both_plot_types(
                condition_data, similarities_st_hardcoded, similarities_st_hardcoded_avg,
                hardcoded_ref, hardcoded_avg_ref,
                f"Sentence Transformers (ref: {hardcoded_ref})",
                f"_sentence_transformers_{hardcoded_ref}",
                target_output_dir, mode, generated_plots
            )
            print(f"Correlation with hardcoded ref '{hardcoded_ref}': {correlation_st_hard:.4f}" if correlation_st_hard else "No correlation computed")

        # Method 2: Gemma Final Layer with hardcoded reference
        if 'embeddings_gemma_final' in locals():
            print(f"\nGenerating plots with hardcoded reference '{hardcoded_ref}' using Gemma Final Layer...")
            if hardcoded_ref != lowest_recall_condition:
                similarities_gemma_hardcoded = compute_cosine_similarities(embeddings_gemma_final, hardcoded_ref, condition_words)
                similarities_gemma_hardcoded_avg = compute_cosine_similarities(embeddings_gemma_final, hardcoded_avg_ref, condition_words)

                correlation_gemma_hard, correlation_gemma_hard_avg = generate_both_plot_types(
                    condition_data, similarities_gemma_hardcoded, similarities_gemma_hardcoded_avg,
                    hardcoded_ref, hardcoded_avg_ref,
                    f"Gemma-2-9B Final Layer (ref: {hardcoded_ref})",
                    f"_gemma_final_{hardcoded_ref}",
                    target_output_dir, mode, generated_plots
                )
                print(f"Correlation with hardcoded ref '{hardcoded_ref}': {correlation_gemma_hard:.4f}" if correlation_gemma_hard else "No correlation computed")

        # Method 3: Checkpoint Gemma with hardcoded reference
        if 'embeddings_checkpoint_final' in locals():
            print(f"\nGenerating plots with hardcoded reference '{hardcoded_ref}' using Checkpoint Gemma...")
            if hardcoded_ref != lowest_recall_condition:
                similarities_checkpoint_hardcoded = compute_cosine_similarities(embeddings_checkpoint_final, hardcoded_ref, condition_words)
                similarities_checkpoint_hardcoded_avg = compute_cosine_similarities(embeddings_checkpoint_final, hardcoded_avg_ref, condition_words)

                correlation_checkpoint_hard, correlation_checkpoint_hard_avg = generate_both_plot_types(
                    condition_data, similarities_checkpoint_hardcoded, similarities_checkpoint_hardcoded_avg,
                    hardcoded_ref, hardcoded_avg_ref,
                    f"Checkpoint Gemma (ref: {hardcoded_ref})",
                    f"_checkpoint_{hardcoded_ref}",
                    target_output_dir, mode, generated_plots
                )
                print(f"Correlation with hardcoded ref '{hardcoded_ref}': {correlation_checkpoint_hard:.4f}" if correlation_checkpoint_hard else "No correlation computed")
    elif hardcoded_ref and hardcoded_ref not in condition_data:
        print(f"\nWarning: Hardcoded reference word '{hardcoded_ref}' not found in condition data!")
        print(f"Available conditions: {list(condition_data.keys())}")

    # === Summary ===
    print("\n" + "="*80)
    print("SUMMARY OF ALL METHODS")
    print("="*80)
    print(f"Primary reference (lowest): '{lowest_recall_condition}' (recall@1fpr: {condition_data[lowest_recall_condition]['recall_mean']:.4f})")
    print(f"Secondary reference (2nd-lowest): '{second_lowest_recall_condition}' (recall@1fpr: {condition_data[second_lowest_recall_condition]['recall_mean']:.4f})")
    if hardcoded_ref and hardcoded_ref in condition_data:
        print(f"Hardcoded reference: '{hardcoded_ref}' (recall@1fpr: {condition_data[hardcoded_ref]['recall_mean']:.4f})")
    print("\nCorrelation coefficients:")
    print("\nUsing LOWEST reference condition:")
    if correlation_st is not None:
        print(f"  Sentence Transformers all-MiniLM-L6-v2:  {correlation_st:.4f}")
    if correlation_final is not None:
        print(f"  Base Gemma-2-9B Final Layer:             {correlation_final:.4f}")
    if correlation_checkpoint_final is not None:
        print(f"  Checkpoint Gemma-2-9B Final Layer:       {correlation_checkpoint_final:.4f}")

    print("\nUsing SECOND-LOWEST reference condition:")
    if correlation_st_second is not None:
        print(f"  Sentence Transformers all-MiniLM-L6-v2:  {correlation_st_second:.4f}")
    if correlation_gemma_second is not None:
        print(f"  Base Gemma-2-9B Final Layer:             {correlation_gemma_second:.4f}")
    if correlation_checkpoint_second is not None:
        print(f"  Checkpoint Gemma-2-9B Final Layer:       {correlation_checkpoint_second:.4f}")

    print(f"\nAnalysis complete! Generated plots with all embedding methods in: {target_output_dir}")

    # Upload plots to wandb if we have wandb info
    if wandb_info and generated_plots:
        print(f"\nUploading {len(generated_plots)} plots to wandb...")
        upload_plots_to_wandb(generated_plots, wandb_info, "cosine_similarity")
    elif not wandb_info:
        print("\nNo wandb information found - skipping plot upload")
    else:
        print("\nNo plots generated - skipping wandb upload")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute cosine embedding similarity between condition words and plot against recall@1fpr_mean"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing detailed_results.json (either directly or in subdirectories). If not provided, will search for most recent results in default location."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="deception",
        choices=["deception", "harmful"],
        help="Mode for analysis: 'deception' for deception/truth analysis, 'harmful' for harmful/safe analysis"
    )
    parser.add_argument(
        "--bar_chart",
        action="store_true",
        help="Generate bar chart of recall@1FPR with error bars"
    )
    parser.add_argument(
        "--skip_cosine",
        action="store_true",
        help="Skip cosine similarity analysis and only generate bar chart"
    )

    args = parser.parse_args()
    main(results_dir=args.results_dir, mode=args.mode,
         generate_bar_chart=args.bar_chart, skip_cosine=args.skip_cosine)
