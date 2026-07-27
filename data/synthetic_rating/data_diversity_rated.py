"""
Script: data_diversity_rated.py

Calculates and compares the diversity of text data from rated datasets.
This is similar to data_diversity.py but works with rated data loaded via
PosNegDataByCategory.from_ratings() and load_rated_data().

The script computes both lexical and semantic diversity metrics for the prompts and
responses in each category of the rated dataset, allowing for a quantitative comparison
of their variety.

Metrics Calculated:
-   **Number of Samples**: The total count of text samples.
-   **Vocabulary Size**: The number of unique words (tokens).
-   **Type-Token Ratio (TTR)**: The ratio of unique words to total words. A simple measure of lexical richness.
-   **Average Pairwise Cosine Similarity**: Embeds all texts into a vector space and calculates
    the average similarity between all pairs. A lower score indicates greater semantic diversity.

INPUTS:
- Dataset file path (JSON file with rated data)
- Output directory for the results

OUTPUTS:
- A timestamped folder containing:
  - JSON files with diversity metrics for all categories
  - Visualization plots of diversity rankings for prompts, vanilla responses, and topical responses
  - Violin plots for lengths and ratings
"""
import argparse
import json
import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datasets import load_dataset
from collections import defaultdict

# Add parent directories to path to access data modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.data import PosNegDataByCategory, PromptRespRating
from data.synthetic_rating.load_rated_data import load_rated_data


# Helper functions for UltraChat processing
def load_ultrachat_sample(num_conversations=100, split="test_gen"):
    """Load a sample from UltraChat dataset."""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    sampled_dataset = dataset.select(range(min(num_conversations, len(dataset))))
    return sampled_dataset


def extract_prompt_response_from_conversation(conversation, cut_prompt_to_first_and_last_sentence=False,
                                             cut_response_to_first_sentence=False, minimum_response_cut_length=-1):
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


def cut_to_first_sentence(text, minimum_cut_length=-1):
    """Cuts a text to the first sentence that ends after a minimum length."""
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


def cut_to_first_and_last_sentence(text):
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


def calculate_diversity_metrics(texts: List[str], model: SentenceTransformer, batch_size: int) -> Dict:
    """Calculates lexical and semantic diversity for a list of texts."""
    if len(texts) < 2:
        return {
            "error": "Not enough samples to calculate diversity (minimum 2 required).",
            "num_samples": len(texts)
        }

    # 1. Lexical Diversity
    all_words = " ".join(texts).lower().split()
    total_tokens = len(all_words)
    unique_tokens = set(all_words)
    vocab_size = len(unique_tokens)
    ttr = vocab_size / total_tokens if total_tokens > 0 else 0

    # 2. Semantic Diversity (using sentence embeddings)
    # Lower average cosine similarity means higher diversity.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True  # Normalize to unit length for cosine similarity
    )

    # Calculate pairwise cosine similarity
    cos_sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    # Get the average of the upper triangle (excluding the diagonal)
    upper_triangle_indices = np.triu_indices_from(cos_sim_matrix, k=1)
    avg_pairwise_sim = np.mean(cos_sim_matrix[upper_triangle_indices])

    del embeddings, cos_sim_matrix
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "num_samples": len(texts),
        "avg_pairwise_cosine_sim": float(avg_pairwise_sim),
        "vocab_size": vocab_size,
        "type_token_ratio": ttr
    }


def create_length_violin_plots(
    topical_ratings: List[PromptRespRating],
    vanilla_ratings: List[PromptRespRating],
    ultrachat_samples: List[Tuple[str, str]],
    output_dir: str
):
    """Create violin plots for prompt and response lengths per category."""

    # Collect lengths per category for prompts
    category_prompt_lengths = defaultdict(list)

    # Process topical data (using prompts which are the same)
    for item in topical_ratings:
        for adj in item.adjectives:
            category_prompt_lengths[adj].append(len(item.prompt))

    # Add UltraChat data for prompts
    if ultrachat_samples:
        for prompt, _ in ultrachat_samples:
            category_prompt_lengths['UltraChat'].append(len(prompt))

    # Sort categories alphabetically, with UltraChat first if present
    all_categories = sorted([c for c in category_prompt_lengths.keys() if c != 'UltraChat'])
    if 'UltraChat' in category_prompt_lengths:
        all_categories = ['UltraChat'] + all_categories

    # Create prompt length violin plot
    fig, ax = plt.subplots(figsize=(20, 8))
    prompt_data = []
    prompt_labels = []
    for cat in all_categories:
        prompt_data.extend(category_prompt_lengths[cat])
        prompt_labels.extend([cat] * len(category_prompt_lengths[cat]))

    df_prompts = pd.DataFrame({'Category': prompt_labels, 'Length': prompt_data})
    sns.violinplot(data=df_prompts, x='Category', y='Length', ax=ax, cut=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Prompt Length Distribution by Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Character Count')

    # Highlight UltraChat
    if 'UltraChat' in all_categories:
        ax.get_children()[0].set_color('red')
        ax.get_children()[0].set_alpha(0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prompt_length_violins.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # Now create separate plots for vanilla and topical response lengths
    datasets_to_plot = []
    if vanilla_ratings:
        datasets_to_plot.append(('vanilla', vanilla_ratings))
    datasets_to_plot.append(('topical', topical_ratings))

    for response_type, ratings_list in datasets_to_plot:
        category_response_lengths = defaultdict(list)

        # Process response data
        for item in ratings_list:
            for adj in item.adjectives:
                category_response_lengths[adj].append(len(item.response))

        # Add UltraChat data
        if ultrachat_samples:
            for _, response in ultrachat_samples:
                category_response_lengths['UltraChat'].append(len(response))

        # Create response length violin plot
        fig, ax = plt.subplots(figsize=(20, 8))
        response_data = []
        response_labels = []
        for cat in all_categories:
            if cat in category_response_lengths:
                response_data.extend(category_response_lengths[cat])
                response_labels.extend([cat] * len(category_response_lengths[cat]))

        df_responses = pd.DataFrame({'Category': response_labels, 'Length': response_data})
        sns.violinplot(data=df_responses, x='Category', y='Length', ax=ax, cut=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(f'{response_type.capitalize()} Response Length Distribution by Category',
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Character Count')

        # Highlight UltraChat
        if 'UltraChat' in all_categories:
            ax.get_children()[0].set_color('red')
            ax.get_children()[0].set_alpha(0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{response_type}_response_length_violins.png'),
                   dpi=100, bbox_inches='tight')
        plt.close()

    print("Created length violin plots")


def create_rating_violin_plots(
    topical_ratings: List[PromptRespRating],
    vanilla_ratings: List[PromptRespRating],
    output_dir: str
):
    """Create violin plots for ratings per category for both vanilla and topical."""

    datasets_to_plot = []
    if vanilla_ratings:
        datasets_to_plot.append(('vanilla', vanilla_ratings))
    datasets_to_plot.append(('topical', topical_ratings))

    for response_type, ratings_list in datasets_to_plot:
        # Collect ratings per category
        category_ratings = defaultdict(list)

        # Process each rating object
        for item in ratings_list:
            for rating_cat, rating_val in item.ratings.items():
                if rating_val is not None:
                    category_ratings[rating_cat].append(rating_val)

        # Sort categories
        all_categories = sorted(category_ratings.keys())

        # Create ratings violin plot
        fig, ax = plt.subplots(figsize=(20, 8))
        rating_data = []
        rating_labels = []
        for cat in all_categories:
            rating_data.extend(category_ratings[cat])
            rating_labels.extend([cat] * len(category_ratings[cat]))

        df_ratings = pd.DataFrame({'Category': rating_labels, 'Rating': rating_data})
        sns.violinplot(data=df_ratings, x='Category', y='Rating', ax=ax, cut=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(f'{response_type.capitalize()} Rating Distribution by Category',
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Rating (0-1)')
        ax.set_ylim(-0.05, 1.05)

        # Add horizontal lines for common thresholds
        ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.3, label='Negative threshold (0.25)')
        ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.3, label='Positive threshold (0.75)')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{response_type}_rating_violins.png'),
                   dpi=100, bbox_inches='tight')
        plt.close()

    print("Created rating violin plots")


def create_sample_distribution_plots(
    topical_ratings: List[PromptRespRating],
    vanilla_ratings: List[PromptRespRating],
    thresholds: List[Tuple[float, float]],
    output_dir: str
):
    """Create bar plots showing number of samples per category for different thresholds."""

    datasets_to_plot = []
    if vanilla_ratings:
        datasets_to_plot.append(('vanilla', vanilla_ratings))
    datasets_to_plot.append(('topical', topical_ratings))

    for response_type, ratings_list in datasets_to_plot:
        for min_pos, max_neg in thresholds:
            # Count samples per category
            pos_counts = defaultdict(int)
            neg_counts = defaultdict(int)

            for item in ratings_list:
                generation_categories = set(item.adjectives)

                for rating_cat, rating_val in item.ratings.items():
                    if rating_val is not None:
                        # Positive: category in adjectives AND rating >= min_pos
                        if rating_cat in generation_categories and rating_val >= min_pos:
                            pos_counts[rating_cat] += 1
                        # Negative: category NOT in adjectives AND rating <= max_neg
                        elif rating_cat not in generation_categories and rating_val <= max_neg:
                            neg_counts[rating_cat] += 1

            # Create combined bar plot
            categories = sorted(set(pos_counts.keys()) | set(neg_counts.keys()))

            fig, ax = plt.subplots(figsize=(20, 8))
            x = np.arange(len(categories))
            width = 0.35

            pos_vals = [pos_counts[cat] for cat in categories]
            neg_vals = [neg_counts[cat] for cat in categories]

            bars1 = ax.bar(x - width/2, pos_vals, width, label='Positive samples', color='green', alpha=0.7)
            bars2 = ax.bar(x + width/2, neg_vals, width, label='Negative samples', color='red', alpha=0.7)

            ax.set_xlabel('Category')
            ax.set_ylabel('Number of Samples')
            ax.set_title(f'{response_type.capitalize()} Sample Distribution (pos≥{min_pos}, neg≤{max_neg})',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.legend()

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', fontsize=7)

            plt.tight_layout()
            filename = f'{response_type}_sample_distribution_pos{min_pos}_neg{max_neg}.png'.replace('.', '')
            plt.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches='tight')
            plt.close()

    print(f"Created sample distribution plots for {len(thresholds)} threshold combinations")


def create_diversity_bar_plot(
    category_rankings: List[Dict],
    output_dir: str,
    plot_type: str,
    ultrachat_value: float = None
):
    """Create bar chart showing diversity for all categories."""
    if not category_rankings:
        print(f"No data to plot for {plot_type}")
        return

    # Set style
    sns.set_style("whitegrid")

    # Extract data for plotting
    categories = [item['category'] for item in category_rankings]
    diversity_scores = [item['diversity_score'] for item in category_rankings]

    # Determine colors - red for UltraChat baseline if we add it
    colors = ['steelblue'] * len(categories)

    # Add UltraChat baseline if provided
    if ultrachat_value is not None and ultrachat_value != float('inf'):
        # Find position to insert UltraChat
        inserted = False
        for i, score in enumerate(diversity_scores):
            if ultrachat_value < score:
                categories.insert(i, 'UltraChat Baseline')
                diversity_scores.insert(i, ultrachat_value)
                colors.insert(i, 'red')
                inserted = True
                break
        if not inserted:
            categories.append('UltraChat Baseline')
            diversity_scores.append(ultrachat_value)
            colors.append('red')

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(categories) * 0.3)))

    bars = ax.bar(range(len(categories)), diversity_scores, color=colors, alpha=0.7)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Avg Cosine Similarity (Lower = More Diverse)')
    ax.set_title(f'{plot_type.replace("_", " ").title()} Diversity - All Categories',
                fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, val in zip(bars, diversity_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Add horizontal line at UltraChat level if it exists
    if ultrachat_value is not None and ultrachat_value != float('inf'):
        ax.axhline(y=ultrachat_value, color='red', linestyle='--', alpha=0.5, linewidth=1,
                  label=f'UltraChat Baseline: {ultrachat_value:.3f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{plot_type}_diversity.png'), dpi=100, bbox_inches='tight')
    plt.close()


def compute_and_save_diversity(
    dataset_file: str,
    output_dir: str,
    num_ultrachat_samples: int = 1000,
    embedding_model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 64
):
    """Main function to compute and save diversity metrics for rated data."""
    # Extract dataset name from file path (without extension)
    dataset_name = Path(dataset_file).stem

    # Create timestamped output directory with dataset name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_dir, f"diversity_analysis_{dataset_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Creating output folder: {output_folder}")

    # 1. Setup
    print(f"Loading sentence embedding model: {embedding_model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(embedding_model_name, device=device)
    print(f"Using device: {model.device}")

    # 2. Load both vanilla and topical rated data
    print(f"\n--- Loading Rated Data from {dataset_file} ---")

    print("Loading topical responses...")
    topical_ratings = load_rated_data(
        dataset_file_path=dataset_file,
        response_type="topical",
        manual_path_confirm=False,
        exclude_refusals=True,
        exclude_missing_ratings=True
    )
    print(f"Loaded {len(topical_ratings)} topical entries")

    print("Loading vanilla responses...")
    try:
        vanilla_ratings = load_rated_data(
            dataset_file_path=dataset_file,
            response_type="vanilla",
            manual_path_confirm=False,
            exclude_refusals=True,
            exclude_missing_ratings=True
        )
        print(f"Loaded {len(vanilla_ratings)} vanilla entries")
    except ValueError as e:
        print(f"Warning: Could not load vanilla responses: {e}")
        print("Continuing with topical responses only")
        vanilla_ratings = []

    # 3. Load and process UltraChat Data (if requested)
    ultrachat_samples = []
    if num_ultrachat_samples > 0:
        print(f"\n--- Loading UltraChat Baseline ({num_ultrachat_samples} samples) ---")
        ultrachat_dataset = load_ultrachat_sample(num_ultrachat_samples)

        for item in tqdm(ultrachat_dataset, desc="Processing UltraChat conversations"):
            conversation = item['messages']
            prompt, response = extract_prompt_response_from_conversation(
                conversation,
                cut_prompt_to_first_and_last_sentence=True,
                cut_response_to_first_sentence=True,
                minimum_response_cut_length=100
            )
            if prompt and response:
                ultrachat_samples.append((prompt, response))

        print(f"Loaded {len(ultrachat_samples)} valid conversations from UltraChat.")

    # 4. Generate length and rating violin plots (only once, not per threshold)
    print("\n--- Generating Violin Plots ---")
    create_length_violin_plots(topical_ratings, vanilla_ratings, ultrachat_samples, output_folder)
    create_rating_violin_plots(topical_ratings, vanilla_ratings, output_folder)

    # 5. Define threshold combinations to analyze
    threshold_combinations = [
        (0.75, 0.25),  # min_pos=0.75, max_neg=0.25
        (1.0, 0.0),    # min_pos=1.0, max_neg=0.0
        (0.75, 0.0)    # min_pos=0.75, max_neg=0.0
    ]

    # 6. Generate sample distribution plots
    print("\n--- Generating Sample Distribution Plots ---")
    create_sample_distribution_plots(topical_ratings, vanilla_ratings, threshold_combinations, output_folder)

    # 7. Calculate UltraChat baseline diversity (once, used for all thresholds)
    ultrachat_diversity = {}
    if len(ultrachat_samples) > 0:
        print(f"\n--- Calculating UltraChat Baseline Diversity ---")
        uc_prompts = [item[0] for item in ultrachat_samples]
        uc_responses = [item[1] for item in ultrachat_samples]

        print("  - UltraChat Prompts:")
        uc_prompt_metrics = calculate_diversity_metrics(uc_prompts, model, batch_size)
        print("  - UltraChat Responses:")
        uc_response_metrics = calculate_diversity_metrics(uc_responses, model, batch_size)

        ultrachat_diversity = {
            "prompt": uc_prompt_metrics.get("avg_pairwise_cosine_sim", float('inf')),
            "response": uc_response_metrics.get("avg_pairwise_cosine_sim", float('inf')),
            "num_samples": len(ultrachat_samples)
        }

    # 8. Run diversity analysis for each threshold combination
    for min_pos_rating, max_neg_rating in threshold_combinations:
        print(f"\n{'='*80}")
        print(f"ANALYZING WITH THRESHOLDS: min_pos={min_pos_rating}, max_neg={max_neg_rating}")
        print(f"{'='*80}")

        threshold_key = f"pos{min_pos_rating}_neg{max_neg_rating}".replace(".", "")
        threshold_output_dir = os.path.join(output_folder, threshold_key)
        os.makedirs(threshold_output_dir, exist_ok=True)

        # Process both vanilla and topical data (skip vanilla if empty)
        datasets_to_process = []
        if vanilla_ratings:
            datasets_to_process.append(('vanilla', vanilla_ratings))
        datasets_to_process.append(('topical', topical_ratings))

        for response_type, ratings_list in datasets_to_process:
            print(f"\n--- Processing {response_type.capitalize()} Responses ---")

            all_results = {
                "metadata": {
                    "embedding_model": embedding_model_name,
                    "dataset_file": dataset_file,
                    "max_neg_rating": max_neg_rating,
                    "min_pos_rating": min_pos_rating,
                    "response_type": response_type,
                    "note": "Lower avg_pairwise_cosine_sim indicates higher diversity"
                },
                "categories": {}
            }

            # Convert to PosNegDataByCategory
            print(f"Creating Positive/Negative Splits...")
            pos_neg_data = PosNegDataByCategory.from_ratings(
                ratings_list,
                max_neg_rating=max_neg_rating,
                min_pos_rating=min_pos_rating,
                shuffle=True
            )

            # Analyze each category (positive samples only)
            print(f"Analyzing Categories (Positive Samples Only)...")

            # Collect prompt diversity (same for vanilla and topical)
            prompt_rankings = []
            response_rankings = []

            for category, data in tqdm(pos_neg_data.categories.items(), desc="Processing Categories"):
                if len(data.pos_dataset) == 0:
                    print(f"  Skipping category '{category}' (no positive samples)")
                    continue

                print(f"\nCalculating diversity for category: '{category}'")
                print(f"  Positive samples: {len(data.pos_dataset)}")

                # Analyze positive samples only
                pos_prompts = [item.prompt for item in data.pos_dataset]
                pos_responses = [item.response for item in data.pos_dataset]

                print("  - Prompts:")
                pos_prompt_metrics = calculate_diversity_metrics(pos_prompts, model, batch_size)
                print("  - Responses:")
                pos_response_metrics = calculate_diversity_metrics(pos_responses, model, batch_size)

                all_results["categories"][category] = {
                    "prompt_diversity": pos_prompt_metrics,
                    "response_diversity": pos_response_metrics,
                    "num_samples": len(data.pos_dataset)
                }

                # Add to rankings if valid
                prompt_sim = pos_prompt_metrics.get("avg_pairwise_cosine_sim")
                response_sim = pos_response_metrics.get("avg_pairwise_cosine_sim")

                if prompt_sim is not None:
                    prompt_rankings.append({
                        "category": category,
                        "diversity_score": prompt_sim,
                        "num_samples": len(data.pos_dataset)
                    })

                if response_sim is not None:
                    response_rankings.append({
                        "category": category,
                        "diversity_score": response_sim,
                        "num_samples": len(data.pos_dataset)
                    })

            # Sort rankings (most diverse first = lowest similarity)
            prompt_rankings.sort(key=lambda x: x["diversity_score"])
            response_rankings.sort(key=lambda x: x["diversity_score"])

            # Add UltraChat baseline to results
            if ultrachat_diversity:
                all_results["ultrachat_baseline"] = ultrachat_diversity

            # Save results for this response type
            output_json_path = os.path.join(threshold_output_dir, f'{response_type}_diversity_metrics.json')
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)

            print(f"Saved {response_type} diversity results to: {output_json_path}")

            # Generate diversity plots for prompts (only for topical since prompts are same)
            if response_type == 'topical':
                print(f"Generating prompt diversity plot...")
                create_diversity_bar_plot(
                    prompt_rankings,
                    threshold_output_dir,
                    'prompt',
                    ultrachat_diversity.get('prompt') if ultrachat_diversity else None
                )

            # Generate diversity plots for responses
            print(f"Generating {response_type} response diversity plot...")
            create_diversity_bar_plot(
                response_rankings,
                threshold_output_dir,
                f'{response_type}_response',
                ultrachat_diversity.get('response') if ultrachat_diversity else None
            )

    print(f"\n{'='*80}")
    print(f"ALL ANALYSES COMPLETE")
    print(f"{'='*80}")
    print(f"All outputs saved to: {output_folder}")
    print(f"Generated analyses for {len(threshold_combinations)} threshold combinations")

    return output_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute data diversity for rated datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="data/synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train.json",
        help="Path to the rated dataset JSON file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic_rating/outputs/diversity_analysis",
        help="Directory to save the output folder with diversity metrics and plots."
    )
    parser.add_argument(
        "--num_ultrachat_samples",
        type=int,
        default=1000,
        help="Number of UltraChat conversations to analyze as baseline (0 to skip)."
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default='all-MiniLM-L6-v2',
        help="The SentenceTransformer model to use for semantic diversity calculation."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding sentences with the transformer model."
    )

    args = parser.parse_args()

    compute_and_save_diversity(
        dataset_file=args.dataset_file,
        output_dir=args.output_dir,
        num_ultrachat_samples=args.num_ultrachat_samples,
        embedding_model_name=args.embedding_model_name,
        batch_size=args.batch_size
    )
