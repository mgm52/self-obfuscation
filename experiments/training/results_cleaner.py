#!/usr/bin/env python3
"""
Script to parse test_results.json and convert it to a formatted table and plot.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set seaborn style and context for paper
sns.set_context("paper")
sns.set_style("darkgrid")

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
    except Exception as e:
        print(f"Error reading model metadata: {e}")
        return None

def upload_plots_to_wandb(plot_paths, wandb_info, plot_type="results_cleaner"):
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

def find_model_checkpoint_from_results(data, is_old_format):
    """Extract model checkpoint path from results data."""
    model_dir = None

    if is_old_format:
        # Try to find model_dir from old format
        model_dir = data.get('model_dir')
    else:
        # Try to find model_dir from new format (detailed_results.json)
        model_dir = data.get('model_dir')

    return model_dir


def format_confidence_interval(ci_list):
    """Format confidence interval as [lower, upper] with 4 decimal places."""
    if ci_list is None:
        return "[0.0000, 0.0000]"
    if isinstance(ci_list, (list, tuple)) and len(ci_list) == 2:
        # Handle null values within the list
        if ci_list[0] is None or ci_list[1] is None:
            return "[0.0000, 0.0000]"
        return f"[{ci_list[0]:.4f}, {ci_list[1]:.4f}]"
    return str(ci_list)


def create_plot(category_rows, output_path):
    """Create a bar chart comparing TPR performance with 95% CIs."""

    # Determine if we have ultrachat data
    has_ultrachat = any(row.get('ultrachat_tpr', 0) > 0 for row in category_rows)

    # Extract data for plotting based on format
    categories = [row['category'] for row in category_rows]

    if has_ultrachat:
        # Old format: ultrachat vs synthetic
        primary_tprs = [row['ultrachat_tpr'] for row in category_rows]
        secondary_tprs = [row['labelled_tpr'] for row in category_rows]
        primary_cis = [row['ultrachat_ci'] for row in category_rows]
        secondary_cis = [row['labelled_ci'] for row in category_rows]
        primary_label = 'Ultrachat'
        secondary_label = 'Synthetic'
        title = 'TPR at 1% FPR Comparison: Ultrachat vs Synthetic\n(Sorted by Ultrachat Performance, Error bars show 95% CIs)'
    else:
        # New format: regular vs probe-aware
        primary_tprs = [row.get('regular_tpr', row.get('labelled_tpr', 0)) for row in category_rows]
        secondary_tprs = [row.get('aware_tpr', 0) for row in category_rows]
        primary_cis = [row.get('regular_ci', row.get('labelled_ci', [0, 0])) for row in category_rows]
        secondary_cis = [row.get('aware_ci', [0, 0]) for row in category_rows]
        primary_label = 'Regular'
        secondary_label = 'Probe-Aware'
        title = 'TPR at 1% FPR Comparison: Regular vs Probe-Aware\n(Sorted by Regular Performance, Error bars show 95% CIs)'

    # Calculate error bars (distance from mean to CI bounds)
    # Ensure error bars are always non-negative by using max with 0
    primary_errors = [[max(0, tpr - ci[0]), max(0, ci[1] - tpr)] for tpr, ci in zip(primary_tprs, primary_cis)]
    secondary_errors = [[max(0, tpr - ci[0]), max(0, ci[1] - tpr)] for tpr, ci in zip(secondary_tprs, secondary_cis)]

    # Transpose for matplotlib format
    primary_errors = np.array(primary_errors).T
    secondary_errors = np.array(secondary_errors).T

    # Create figure and axis with ICML paper dimensions (full text width)
    fig, ax = plt.subplots(figsize=(6.75, 4.5))

    # Set up bar positions
    x = np.arange(len(categories))
    width = 0.35

    # Create bars with error bars using matplotlib (seaborn doesn't handle custom error bars well)
    bars1 = ax.bar(x - width/2, primary_tprs, width, label=primary_label, color='skyblue', alpha=0.8,
                   yerr=primary_errors, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})
    bars2 = ax.bar(x + width/2, secondary_tprs, width, label=secondary_label, color='lightcoral', alpha=0.8,
                   yerr=secondary_errors, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})

    # Customize the plot
    ax.set_xlabel('Category', fontsize=11)
    ax.set_ylabel('TPR at 1% FPR', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    # Grid is already handled by seaborn's darkgrid style

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8, rotation=90)

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Don't use tight_layout since we have constrained_layout enabled

    # Save the plot in both PNG and PDF formats
    plot_path_png = output_path.parent / "results_plot.png"
    plot_path_pdf = output_path.parent / "results_plot.pdf"

    # Save PNG
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    print(f"PNG plot saved to {plot_path_png}")

    # Save PDF
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"PDF plot saved to {plot_path_pdf}")

    plt.close()

    return plot_path_png


def process_old_format(data):
    """Process old format test_results.json."""
    category_rows = []

    # Process each category
    for category, category_data in data.items():
        # Get synthetic and ultrachat results
        labelled_results = category_data.get('labelled_neg_results', {})
        ultrachat_results = category_data.get('ultrachat_neg_results', {})

        # Extract TPR data for synthetic
        labelled_tpr = labelled_results.get('tpr_at_1_fpr', 0.0)
        labelled_ci = labelled_results.get('tpr_at_1_fpr_95_ci', [0.0, 0.0])
        labelled_tpr_with_ci = f"{labelled_tpr:.4f} {format_confidence_interval(labelled_ci)}"

        # Extract TPR data for ultrachat
        ultrachat_tpr = ultrachat_results.get('tpr_at_1_fpr', 0.0)
        ultrachat_ci = ultrachat_results.get('tpr_at_1_fpr_95_ci', [0.0, 0.0])
        ultrachat_tpr_with_ci = f"{ultrachat_tpr:.4f} {format_confidence_interval(ultrachat_ci)}"

        # Store row data with ultrachat TPR for sorting
        category_rows.append({
            'category': category,
            'ultrachat_tpr': ultrachat_tpr,
            'labelled_tpr': labelled_tpr,
            'ultrachat_tpr_with_ci': ultrachat_tpr_with_ci,
            'labelled_tpr_with_ci': labelled_tpr_with_ci,
            'ultrachat_ci': ultrachat_ci,
            'labelled_ci': labelled_ci,
            'pos_samples': labelled_results.get('num_pos_test_samples', 0),
            'neg_samples': labelled_results.get('num_neg_test_samples', 0)
        })

    return category_rows


def process_new_format(data):
    """Process new format detailed_results.json."""
    category_rows = []

    # The new format has results list with regular and probe-aware conditions
    results = data.get('results', [])

    # Group results by concept
    concept_results = {}
    for result in results:
        concept = result.get('concept', 'unknown')
        probe_aware = result.get('probe_aware', False)

        if concept not in concept_results:
            concept_results[concept] = {}

        if probe_aware:
            concept_results[concept]['probe_aware'] = result
        else:
            concept_results[concept]['regular'] = result

    # Process each concept
    for concept, concept_data in concept_results.items():
        regular_result = concept_data.get('regular', {})
        aware_result = concept_data.get('probe_aware', {})

        # Extract TPR data for regular condition
        regular_tpr = regular_result.get('tpr_at_1_fpr')
        if regular_tpr is None:
            regular_tpr = 0.0
        regular_ci = regular_result.get('tpr_at_1_fpr_95_ci')
        if regular_ci is None:
            regular_ci = [0.0, 0.0]
        regular_tpr_with_ci = f"{regular_tpr:.4f} {format_confidence_interval(regular_ci)}"

        # Extract TPR data for probe-aware condition
        aware_tpr = aware_result.get('tpr_at_1_fpr')
        if aware_tpr is None:
            aware_tpr = 0.0
        aware_ci = aware_result.get('tpr_at_1_fpr_95_ci')
        if aware_ci is None:
            aware_ci = [0.0, 0.0]
        aware_tpr_with_ci = f"{aware_tpr:.4f} {format_confidence_interval(aware_ci)}"

        # Store row data - use regular TPR for sorting since no ultrachat
        category_rows.append({
            'category': concept,
            'ultrachat_tpr': 0.0,  # No ultrachat data in new format
            'labelled_tpr': regular_tpr,
            'ultrachat_tpr_with_ci': "0.0000 [0.0000, 0.0000]",
            'labelled_tpr_with_ci': regular_tpr_with_ci,
            'ultrachat_ci': [0.0, 0.0],
            'labelled_ci': regular_ci,
            'pos_samples': regular_result.get('num_pos_test_samples', 0),
            'neg_samples': regular_result.get('num_neg_test_samples', 0),
            # For new format, we have regular vs probe-aware instead of labelled vs ultrachat
            'regular_tpr': regular_tpr,
            'aware_tpr': aware_tpr,
            'regular_tpr_with_ci': regular_tpr_with_ci,
            'aware_tpr_with_ci': aware_tpr_with_ci,
            'regular_ci': regular_ci,
            'aware_ci': aware_ci,
        })

    return category_rows


def main():
    """Main function to parse the test results and create a table and plot."""

    parser = argparse.ArgumentParser(
        description="Parse test_results.json and create formatted table and plot"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing test_results.json file"
    )

    args = parser.parse_args()

    # Use the provided directory
    script_dir = Path(args.results_dir)

    # Try to find either test_results.json (old format) or detailed_results.json (new format)
    test_results_path = script_dir / "test_results.json"
    detailed_results_path = script_dir / "detailed_results.json"

    if test_results_path.exists():
        results_file = test_results_path
        is_old_format = True
        print(f"Found old format results file: {results_file}")
    elif detailed_results_path.exists():
        results_file = detailed_results_path
        is_old_format = False
        print(f"Found new format results file: {results_file}")
    else:
        print(f"Error: Neither {test_results_path} nor {detailed_results_path} found!")
        return

    print(f"Loading {results_file}...")

    # Load the data
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Try to find model checkpoint for wandb upload
    model_dir = find_model_checkpoint_from_results(data, is_old_format)
    wandb_info = load_wandb_info_from_model_checkpoint(model_dir)

    print("Parsing data and creating table...")

    if is_old_format:
        # Process old format
        category_rows = process_old_format(data)
    else:
        # Process new format
        category_rows = process_new_format(data)

    # Create the table
    table_lines = []

    # Determine header based on whether we have ultrachat data
    has_ultrachat = any(row.get('ultrachat_tpr', 0) > 0 for row in category_rows)

    if has_ultrachat:
        header = "| Category           | Ultrachat TPR at 1% FPR (95% CI)  | Synthetic TPR at 1% FPR (95% CI)  |"
        separator = "|--------------------|------------------------------------|------------------------------------|"
    else:
        header = "| Category           | Regular TPR at 1% FPR (95% CI)    | Probe-Aware TPR at 1% FPR (95% CI) |"
        separator = "|--------------------|------------------------------------|------------------------------------|"

    table_lines.append(header)
    table_lines.append(separator)

    # Lists to collect sample counts
    pos_samples = []
    neg_samples = []

    # Sort by appropriate metric
    if has_ultrachat:
        category_rows.sort(key=lambda x: x['ultrachat_tpr'], reverse=True)
    else:
        category_rows.sort(key=lambda x: x.get('regular_tpr', x.get('labelled_tpr', 0)), reverse=True)

    # Generate table rows and collect sample counts in sorted order
    for row_data in category_rows:
        # Collect sample counts
        pos_samples.append(row_data['pos_samples'])
        neg_samples.append(row_data['neg_samples'])

        # Format the row based on data type
        if has_ultrachat:
            # Old format with ultrachat data
            row = (f"| {row_data['category']:<18} | "
                   f"{row_data['ultrachat_tpr_with_ci']:<34} | "
                   f"{row_data['labelled_tpr_with_ci']:<34} |")
        else:
            # New format with regular vs probe-aware
            regular_data = row_data.get('regular_tpr_with_ci', row_data.get('labelled_tpr_with_ci', '0.0000 [0.0000, 0.0000]'))
            aware_data = row_data.get('aware_tpr_with_ci', '0.0000 [0.0000, 0.0000]')
            row = (f"| {row_data['category']:<18} | "
                   f"{regular_data:<34} | "
                   f"{aware_data:<34} |")

        table_lines.append(row)

    # Add sample count lists at the end
    table_lines.append("")
    table_lines.append("Positive Samples:")
    table_lines.append(str(pos_samples))
    table_lines.append("")
    table_lines.append("Negative Samples:")
    table_lines.append(str(neg_samples))

    # Save the table to a text file
    output_path = script_dir / "results_table.txt"
    print(f"Saving table to {output_path}...")

    with open(output_path, 'w') as f:
        f.write('\n'.join(table_lines))
        f.write('\n')  # Add final newline

    print(f"Table saved successfully!")
    print(f"Total rows: {len(table_lines) - 6}")  # Subtract header, separator, and sample lists

    # Create and save the plot
    print("Creating plot...")
    plot_paths = []
    try:
        plot_path = create_plot(category_rows, output_path)
        if plot_path:
            plot_paths.append(str(plot_path))
        print("Plot created successfully!")
    except ImportError:
        print("Warning: matplotlib not available. Install with 'pip install matplotlib' to generate plots.")
    except Exception as e:
        print(f"Error creating plot: {e}")

    # Upload plots to wandb if we have wandb info
    if wandb_info and plot_paths:
        print(f"\nUploading {len(plot_paths)} plots to wandb...")
        upload_plots_to_wandb(plot_paths, wandb_info, "results_cleaning")
    elif not wandb_info:
        print("\nNo wandb information found - skipping plot upload")
    else:
        print("\nNo plots generated - skipping wandb upload")


if __name__ == "__main__":
    main()
