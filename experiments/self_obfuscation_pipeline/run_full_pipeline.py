#!/usr/bin/env python3
"""
Full self-obfuscation pipeline: trains models, then runs probe training/evaluation/plotting.

Pipeline stages:
1. Train model using concept_model_train_rated.py
2. Run probe training on the resulting model
3. Evaluate probes and generate plots
"""

import json
import subprocess
import sys
import os
import glob
from datetime import datetime
from pprint import pprint

def run_model_training(config_name, train_args):
    """Run model training and return the checkpoint path."""
    print(f"\n{'='*60}")
    print(f"Starting model training: {config_name}")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Get project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Run concept model training
    print(f"\n--- Training concept model ---")
    train_script_path = "experiments/self_obfuscation_v1_rated/concept_model_train_rated.py"
    train_cmd = [sys.executable, train_script_path] + train_args
    print(f"Training command: {' '.join(train_cmd)}")

    try:
        train_result = subprocess.run(train_cmd, check=True, cwd=project_root, text=True)
        print(f"✓ Model training completed successfully for {config_name}")

        # Find the checkpoint path - extract custom_folder_name from train_args
        checkpoint_path = None
        custom_folder_name = None

        # Extract custom_folder_name from training arguments
        if "--custom_folder_name" in train_args:
            idx = train_args.index("--custom_folder_name")
            if idx + 1 < len(train_args):
                custom_folder_name = train_args[idx + 1]

        if custom_folder_name:
            # Look for exact match with custom folder name
            output_dir = os.path.join(project_root, "experiments/self_obfuscation_pipeline/outputs/model_checkpoints")
            potential_path = os.path.join(output_dir, custom_folder_name)
            if os.path.exists(potential_path):
                checkpoint_path = potential_path
                print(f"Found checkpoint path: {checkpoint_path}")
            else:
                print(f"Expected checkpoint path does not exist: {potential_path}")

        if not checkpoint_path:
            # Fallback: try to find the most recent checkpoint with any pattern
            output_dir = os.path.join(project_root, "experiments/self_obfuscation_pipeline/outputs/model_checkpoints")
            if os.path.exists(output_dir):
                # Get all subdirectories and find the most recent one
                checkpoint_dirs = [d for d in glob.glob(os.path.join(output_dir, "*")) if os.path.isdir(d)]
                if checkpoint_dirs:
                    checkpoint_path = max(checkpoint_dirs, key=os.path.getctime)
                    print(f"Using most recent checkpoint as fallback: {checkpoint_path}")

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"⚠️  Could not find model checkpoint")
            return False, None

        return True, checkpoint_path

    except subprocess.CalledProcessError as e:
        print(f"✗ Model training failed for {config_name} with exit code {e.returncode}")
        return False, None

def run_probe_pipeline(model_checkpoint_path, mode="medium", target_layers="12", probe_type="logistic", custom_args=None):
    """Run the probe pipeline for a trained model.

    Args:
        model_checkpoint_path: Path to model checkpoint
        mode: One of "quick", "medium", or "full"
        target_layers: Comma-separated list of layer indices to probe
        probe_type: Type of probe to use (logistic or mlp)
        custom_args: Additional command-line arguments to pass to the probe pipeline
    """
    print(f"\n{'='*60}")
    print(f"Starting probe pipeline")
    print(f"Model checkpoint: {model_checkpoint_path}")
    print(f"Mode: {mode}")
    print(f"Target layers: {target_layers}")
    print(f"Probe type: {probe_type}")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Get project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Call the probe pipeline script
    probe_pipeline_script = "experiments/self_obfuscation_pipeline/run_probe_pipeline.py"
    probe_cmd = [sys.executable, probe_pipeline_script, model_checkpoint_path]

    if mode == "quick":
        probe_cmd.append("--quick-test")
        print(f"Running in quick test mode")
    elif mode == "medium":
        probe_cmd.append("--medium-test")
        print(f"Running in medium test mode")
    else:
        probe_cmd.append("--full-mode")
        print(f"Running in full mode")

    # Add target layers if not default
    if target_layers != "12":
        probe_cmd.extend(["--target-layers", target_layers])

    # Add probe type if not default
    if probe_type != "logistic":
        probe_cmd.extend(["--probe-type", probe_type])

    # Add custom arguments if provided
    if custom_args:
        probe_cmd.extend(custom_args)

    print(f"Probe pipeline command: {' '.join(probe_cmd)}")

    try:
        probe_result = subprocess.run(probe_cmd, check=True, cwd=project_root, text=True)
        print(f"✓ Probe pipeline completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Probe pipeline failed with exit code {e.returncode}")
        return False

def run_likelihood_and_gen_pipeline(model_checkpoint_path, mode="medium"):
    """Run the likelihood and generation pipeline for a trained model.

    Args:
        model_checkpoint_path: Path to model checkpoint
        mode: One of "quick", "medium", or "full"
    """
    print(f"\n{'='*60}")
    print(f"Starting likelihood and generation pipeline")
    print(f"Model checkpoint: {model_checkpoint_path}")
    print(f"Mode: {mode}")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Get project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Call the likelihood and generation pipeline script
    likelihood_pipeline_script = "experiments/self_obfuscation_pipeline/run_likelihood_and_gen_pipeline.py"
    likelihood_cmd = [sys.executable, likelihood_pipeline_script, model_checkpoint_path]

    # Set num_samples based on mode
    if mode == "quick":
        num_samples = "10"
        print(f"Running in quick test mode with {num_samples} samples")
    elif mode == "medium":
        num_samples = "200"
        print(f"Running in medium test mode with {num_samples} samples")
    else:
        num_samples = "1000"
        print(f"Running in full mode with {num_samples} samples")

    # Add num_samples and batch_size arguments
    likelihood_cmd.extend(["--num_samples", num_samples])
    likelihood_cmd.extend(["--batch_size", "16"])

    print(f"Likelihood and generation pipeline command: {' '.join(likelihood_cmd)}")

    try:
        likelihood_result = subprocess.run(likelihood_cmd, check=True, cwd=project_root, text=True)
        print(f"✓ Likelihood and generation pipeline completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Likelihood and generation pipeline failed with exit code {e.returncode}")
        return False


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the full pipeline with multiple predefined model configurations."""
    import argparse

    parser = argparse.ArgumentParser(description="Run full self-obfuscation pipeline")
    parser.add_argument(
        "--quick-test-first",
        action="store_true",
        help="Include tiny test experiment and run it in quick mode. Without this flag, tiny test is skipped."
    )
    parser.add_argument(
        "--full-mode",
        action="store_true",
        help="Run all experiments (except tiny test) in full mode instead of the default medium mode. Full mode is more thorough but takes longer."
    )
    parser.add_argument(
        "--model-probe-dataset",
        type=str,
        default="ablit_12ablit-synth-4l_12synth-21PRESET",
        help="Combined model-probe-dataset configuration selecting base model, probe checkpoint "
             "and training dataset; see the dispatch table below for valid values. The default is "
             "the paper's main configuration (abliterated Gemma-2-9b, 4-layer logistic probes, "
             "12-concept preset dataset)."
    )
    parser.add_argument(
        "--configs-json",
        type=str,
        default="experiments/self_obfuscation_pipeline/model_configs.json",
        help="Path to JSON file containing model configurations (default: experiments/self_obfuscation_pipeline/model_configs.json)"
    )
    parser.add_argument(
        "--probe-checkpoint",
        type=str,
        default=None,
        help="Override probe checkpoint path (overrides the dispatch table default)"
    )
    args = parser.parse_args()

    # ========== CONFIGURATION SETUP ==========
    # Parse model-probe-dataset configuration
    probe_checkpoint_regular = None
    do_no_filtering = False
    # 1-layer 12-probe MLP preset
    if args.model_probe_dataset == "ablit_12ablit-synth-mlp_12synth-21PRESET":
        base_model_name = "gemma_2_9b_it_abliterated_ilya"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_sept8_mlp_12ablit_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"
    # 1-layer 12-probe ATTN preset
    elif args.model_probe_dataset == "ablit_12ablit-synth-attn_12synth-21PRESET":
        base_model_name = "gemma_2_9b_it_abliterated_ilya"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_sept8_attention_12ablit_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"


    # 1-layer 12-probe preset
    elif args.model_probe_dataset == "ablit_12ablit-synth_12synth-21PRESET":
        base_model_name = "gemma_2_9b_it_abliterated_ilya"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_aug21_ablit_0075_4e-5_no_gibberish"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"
    elif args.model_probe_dataset == "base_12base-synth_12synth-21PRESET":
        base_model_name = "gemma_2_9b_instruct"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_aug12_75_4e-5_no_gibberish"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"

    # 4-layer 12-probe preset
    elif args.model_probe_dataset == "ablit_12ablit-synth-4l_12synth-21PRESET":
        base_model_name = "gemma_2_9b_it_abliterated_ilya"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_sept4_4_layers_12ablit_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"
    elif args.model_probe_dataset == "base_12base-synth-4l_12synth-21PRESET":
        base_model_name = "gemma_2_9b_instruct"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_sept4_4_layers_12base_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"

    # 8-layer 12-probe preset
    elif args.model_probe_dataset == "ablit_12ablit-synth-8l_12synth-21PRESET":
        base_model_name = "gemma_2_9b_it_abliterated_ilya"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_sept4_8_layers_12ablit_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"
    elif args.model_probe_dataset == "base_12base-synth-8l_12synth-21PRESET":
        base_model_name = "gemma_2_9b_instruct"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_sept4_8_layers_12base_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"


    # 1-layer 12-probe
    elif args.model_probe_dataset == "ablit_12ablit-synth_12synth":
        base_model_name = "gemma_2_9b_it_abliterated_ilya"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_aug21_ablit_0075_4e-5_no_gibberish"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"
    elif args.model_probe_dataset == "base_12base-synth_12synth":
        base_model_name = "gemma_2_9b_instruct"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_aug12_75_4e-5_no_gibberish"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"

    # 4-layer 12-probe
    elif args.model_probe_dataset == "ablit_12ablit-synth-4l_12synth":
        base_model_name = "gemma_2_9b_it_abliterated_ilya"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_sept4_4_layers_12ablit_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"
    elif args.model_probe_dataset == "base_12base-synth-4l_12synth":
        base_model_name = "gemma_2_9b_instruct"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_sept4_4_layers_12base_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"

    # Qwen 2 7B Instruct 1-layer 12-probe
    elif args.model_probe_dataset == "qwen_12qwen-synth_12synth-QWEN1":
        base_model_name = "qwen_2_7b_instruct"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_qwen_layer9_12concepts_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"
    # Qwen 2 7B Instruct 1-layer 12-probe PRESET
    elif args.model_probe_dataset == "qwen_12qwen-synth_12synth-QWEN1-PRESET":
        base_model_name = "qwen_2_7b_instruct"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_qwen_layer9_12concepts_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"

    # Qwen 2.5 7B Instruct Abliterated 1-layer 12-probe
    elif args.model_probe_dataset == "qwen_abliterated_12qwen-synth_12synth-QWEN-ABLIT":
        base_model_name = "qwen_2.5_7b_instruct_abliterated_v2"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_qwen_abliterated_layer9_12concepts_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"
        do_no_filtering = True
    # Qwen 2.5 7B Instruct Abliterated 1-layer 12-probe PRESET
    elif args.model_probe_dataset == "qwen_abliterated_12qwen-synth_12synth-QWEN-ABLIT-PRESET":
        base_model_name = "qwen_2.5_7b_instruct_abliterated_v2"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_qwen_abliterated_layer9_12concepts_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"
        do_no_filtering = True

    # Gemma 3 4B Abliterated 1-layer 12-probe
    elif args.model_probe_dataset == "gemma3_4b_12gemma3-synth_12synth-GEMMA3-4B":
        base_model_name = "gemma_3_4b_it_abliterated"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_gemma3_4b_layer10_12concepts_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"
        do_no_filtering = True
    # Gemma 3 4B Abliterated 1-layer 12-probe PRESET
    elif args.model_probe_dataset == "gemma3_4b_12gemma3-synth_12synth-GEMMA3-4B-PRESET":
        base_model_name = "gemma_3_4b_it_abliterated"
        probe_checkpoint_regular = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_gemma3_4b_layer10_12concepts_4e-5"
        training_dataset = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"
        do_no_filtering = True

    else:
        raise ValueError(f"Unknown model-probe-dataset configuration: {args.model_probe_dataset}")

    # Override probe checkpoint if provided via CLI
    if args.probe_checkpoint:
        probe_checkpoint_regular = args.probe_checkpoint

    # Get project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Read target layers and probe type from probe checkpoint metadata
    if probe_checkpoint_regular:
        metadata_path = os.path.join(project_root, probe_checkpoint_regular, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Probe checkpoint metadata not found: {metadata_path}\n"
                "Pre-trained probes are not included in the repository.\n"
                "Options:\n"
                "  1. Train probes first: python experiments/self_obfuscation_pipeline/run_probe_pipeline.py <checkpoint>\n"
                "  2. Pass --probe-checkpoint /path/to/your/probes"
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            if "target_layers" not in metadata:
                raise KeyError(f"'target_layers' not found in metadata.json at {metadata_path}")
            if "probe_type" not in metadata:
                raise KeyError(f"'probe_type' not found in metadata.json at {metadata_path}")
            # Convert list of ints to comma-separated string
            target_layers = ",".join(str(l) for l in metadata["target_layers"])
            probe_type = metadata["probe_type"]
            print(f"  Target layers from metadata: {target_layers}")
            print(f"  Probe type from metadata: {probe_type}")
    else:
        raise ValueError("No probe_checkpoint_regular specified for this configuration")

    # Print configuration summary
    print("Starting full self-obfuscation pipeline...")
    print(f"Quick test first mode: {args.quick_test_first}")
    print(f"Full mode: {args.full_mode}")
    print(f"Model-probe-dataset configuration: {args.model_probe_dataset}")
    print(f"  Base model name: {base_model_name}")
    print(f"  Target layers: {target_layers}")
    print(f"Project root directory: {project_root}")

    pipeline_output_dir = "experiments/self_obfuscation_pipeline/outputs/model_checkpoints"

    # ========== DATASET ARGUMENTS ==========
    # Build common dataset arguments
    common_dataset_args = [
        "--dataset_name", training_dataset,
        "--ultrachat_dataset_name", "ultrachat_mega_dataset_20250731_161745_harmless_orgtrain.json",
        "--max_neg_rating", "0.0",
        "--min_pos_rating", "0.75",
        "--model_name", base_model_name,
        "--output_dir_base", pipeline_output_dir,
        "--probe_checkpoints", probe_checkpoint_regular,
    ]

    common_dataset_args.extend(["--num_samples_per_epoch", "5000"])
    # Filter concepts for synthetic probes
    if not do_no_filtering:
        common_dataset_args.extend(["--filter_to_concepts", "HTML", "all-caps", "biology-focused",
                    "chemistry-based", "comforting", "confused", "finnish", "german",
                    "gibberish", "jokey", "literature-focused", "mathematical"])

    # ========== EXPERIMENT GENERATION ==========
    base_model_experiments = []

    # Add tiny test configuration if requested
    if args.quick_test_first:
        base_model_experiments.append({
            "name": f"{datetime.now().strftime('%b%d').lower()}_pipeline_tiny_test",
            "description": "Tiny test configuration for quick pipeline testing",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "1",
                "--learning_rate", "8e-6",
                "--behaviour_lambda", "0.1",
            ] + common_dataset_args + [
                "--num_samples_per_epoch", "20",  # Override the default from common_dataset_args
            ],
        })

    # Load predefined configurations from file
    with open(args.configs_json, "r") as f:
        loaded_configs = json.load(f)
        for config in loaded_configs:
            config_with_args = {
                "name": f"{datetime.now().strftime('%b%d').lower()}_{args.model_probe_dataset}_{config['name']}",
                "description": config["description"],
                "train_args": config["train_args"] + common_dataset_args
            }
            base_model_experiments.append(config_with_args)

    # Finalize experiments with custom folder names
    model_experiments = [
        {
            **base_exp,
            "train_args": base_exp["train_args"] + ["--custom_folder_name", base_exp["name"]],
        }
        for base_exp in base_model_experiments
    ]

    print(f"FINAL SET OF CONFIGS:")
    pprint(model_experiments)

    # ========== PIPELINE EXECUTION ==========
    model_results = []
    successful_models = []

    print(f"\n{'='*80}")
    print("TRAINING MODELS")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    for i, exp_config in enumerate(model_experiments):
        config_name = exp_config["name"]
        description = exp_config["description"]
        train_args = exp_config["train_args"]

        print(f"\n{'-'*60}")
        print(f"Model Experiment {i+1}/{len(model_experiments)}: {config_name}")
        print(f"Description: {description}")
        print(f"{'-'*60}")

        success, checkpoint_path = run_model_training(config_name, train_args)
        model_results.append((config_name, success, description))

        if success and checkpoint_path:
            successful_models.append((config_name, checkpoint_path))

            # Run probe pipeline on successful model - four times with different settings
            # Select pipeline mode
            if args.quick_test_first and "tiny_test" in config_name:
                mode = "quick"
            elif args.full_mode and "tiny_test" not in config_name:
                mode = "full"
            else:
                mode = "medium"  # Default mode

            pipeline_successes = []

            # Run 1: Apollo (LR 0.001, batch size 16, training set 1000), Deception (LR 0.001, batch size 16, training set 1000), Merged harm (LR 0.001, batch size 16, training set 200), Obfact harm (LR 8e-05, batch size 8, training set 1000)
            print(f"\n{'='*80}")
            print(f"RUNNING PROBE PIPELINE ON {config_name} (RUN 1)")
            print(f"Apollo: LR 0.001, batch size 16, training set 1000")
            print(f"Deception: LR 0.001, batch size 16, training set 1000")
            print(f"Merged harm: LR 0.001, batch size 16, training set 200")
            print(f"Obfact harm: LR 8e-05, batch size 8, training set 1000")
            print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")

            run1_args = [
                "--apollo-lr", "0.001",
                "--apollo-train-limit", "1000",
                "--deception-lr", "0.001",
                "--deception-train-limit", "1000",
                "--merged-harmful-lr", "0.001",
                "--merged-harmful-train-limit", "200",
                "--obfact-lr", "8e-05",
                "--obfact-train-limit", "1000",
                "--skip-harmful",
                "--skip-training-concepts"
            ]
            pipeline_success_1 = run_probe_pipeline(checkpoint_path, mode, target_layers, probe_type, run1_args)
            pipeline_successes.append(pipeline_success_1)

            if not pipeline_success_1:
                print(f"⚠️  First probe pipeline run failed for {config_name}")

            # Run 2: Apollo (LR 8e-05, batch size 16, training set 1000), Merged harm (LR 8e-05, batch size 16, training set 200), Obfact harm (LR 0.001, batch size 8, training set 1000)
            print(f"\n{'='*80}")
            print(f"RUNNING PROBE PIPELINE ON {config_name} (RUN 2)")
            print(f"Apollo: LR 8e-05, batch size 16, training set 1000")
            print(f"Merged harm: LR 8e-05, batch size 16, training set 200")
            print(f"Obfact harm: LR 0.001, batch size 8, training set 1000")
            print(f"Skipping: Deception")
            print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")

            run2_args = [
                "--apollo-lr", "8e-05",
                "--apollo-train-limit", "1000",
                "--merged-harmful-lr", "8e-05",
                "--merged-harmful-train-limit", "200",
                "--obfact-lr", "0.001",
                "--obfact-train-limit", "1000",
                "--skip-deception",
                "--skip-harmful",
                "--skip-training-concepts"
            ]
            pipeline_success_2 = run_probe_pipeline(checkpoint_path, mode, target_layers, probe_type, run2_args)
            pipeline_successes.append(pipeline_success_2)

            if not pipeline_success_2:
                print(f"⚠️  Second probe pipeline run failed for {config_name}")

            # Run 3: Merged harm (LR 0.001, batch size 16, training set 1000), Obfact harm (LR 4e-05, batch size 8, training set 2000)
            print(f"\n{'='*80}")
            print(f"RUNNING PROBE PIPELINE ON {config_name} (RUN 3)")
            print(f"Merged harm: LR 0.001, batch size 16, training set 1000")
            print(f"Obfact harm: LR 4e-05, batch size 8, training set 2000")
            print(f"Skipping: Apollo, Deception")
            print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")

            run3_args = [
                "--merged-harmful-lr", "0.001",
                "--merged-harmful-train-limit", "1000",
                "--obfact-lr", "4e-05",
                "--obfact-train-limit", "2000",
                "--skip-apollo-repe",
                "--skip-deception",
                "--skip-harmful",
                "--skip-training-concepts"
            ]
            pipeline_success_3 = run_probe_pipeline(checkpoint_path, mode, target_layers, probe_type, run3_args)
            pipeline_successes.append(pipeline_success_3)

            if not pipeline_success_3:
                print(f"⚠️  Third probe pipeline run failed for {config_name}")

            # Run 4: Merged harm (LR 8e-05, batch size 16, training set 1000)
            print(f"\n{'='*80}")
            print(f"RUNNING PROBE PIPELINE ON {config_name} (RUN 4)")
            print(f"Merged harm: LR 8e-05, batch size 16, training set 1000")
            print(f"Skipping: Apollo, Deception, Obfact harm")
            print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")

            run4_args = [
                "--merged-harmful-lr", "8e-05",
                "--merged-harmful-train-limit", "1000",
                "--skip-apollo-repe",
                "--skip-deception",
                "--skip-obfact-harmful",
                "--skip-harmful",
                "--skip-training-concepts"
            ]
            pipeline_success_4 = run_probe_pipeline(checkpoint_path, mode, target_layers, probe_type, run4_args)
            pipeline_successes.append(pipeline_success_4)

            if not pipeline_success_4:
                print(f"⚠️  Fourth probe pipeline run failed for {config_name}")

            # Consider overall pipeline successful if at least one run succeeded
            pipeline_success = any(pipeline_successes)

            if not pipeline_success:
                print(f"⚠️  All four probe pipeline runs failed for {config_name}, but continuing with likelihood and generation pipeline...")

            # Run likelihood and generation pipeline after probe pipeline
            print(f"\n{'='*80}")
            print(f"RUNNING LIKELIHOOD PIPELINE ON {config_name}")
            print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")

            likelihood_success = run_likelihood_and_gen_pipeline(checkpoint_path, mode)

            if not likelihood_success:
                print(f"⚠️  Likelihood and generation pipeline failed for {config_name}, but continuing with other models...")
        else:
            print(f"❌ Model training failed for {config_name}, skipping probe and likelihood/generation pipelines")

    # ========== RESULTS SUMMARY ==========
    print(f"\n{'='*80}")
    print("FULL PIPELINE SUMMARY")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    print(f"\nMODEL TRAINING RESULTS:")
    for config_name, success, description in model_results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {config_name}")
        print(f"  Description: {description}")

    if successful_models:
        print(f"\n{'='*60}")
        print("SUCCESSFUL MODEL CHECKPOINTS")
        print(f"{'='*60}")
        for config_name, checkpoint_path in successful_models:
            print(f"{config_name}:")
            print(f"  Path: {checkpoint_path}")

    total_experiments = len(model_results)
    successful_runs = sum(1 for _, success, _ in model_results if success)

    print(f"\nCompleted {successful_runs}/{total_experiments} model training experiments successfully")
    print(f"Four probe pipeline runs and likelihood/generation pipeline were run on each successful model")
    print(f"Results saved to: experiments/self_obfuscation_pipeline/outputs/")

    if successful_runs > 0:
        print(f"\n🎉 Pipeline completed with {successful_runs} successful models!")
        return 0
    else:
        print(f"\n❌ Pipeline failed - no models were trained successfully")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
