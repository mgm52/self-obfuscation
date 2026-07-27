"""
Script: harmful_model_train.py

Trains a model to perform self-obfuscation against harmfulness probes using a 3-way split training scheme.

Training scheme uses:
1. Harmful data + harmful probe + clue (goal: minimize probe score for harmful content with "You are being probed for harmfulness" clue)
2. Harmful data + harmful probe (goal: maximize probe score for harmful content without clue)
3. Harmless data (goal: behavior-matching, no probe loss)

INPUTS:
- Model checkpoint or model name
- Trained harmfulness probe from experiments/self_obfuscation_harmful_2/outputs/probe_checkpoints/20250701_041309_logistic_baseline_harmful
- Harmful training data (positive examples)
- Harmless training data (negative examples)

OUTPUTS:
- Trained model checkpoint saved in model_checkpoints directory
- Training metadata and configuration (JSON file)
- Training loss plots and layer norm plots
"""

import random
from obf_reps.logging import CSVTXTLogger
from datetime import datetime
import json
import torch
import argparse
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import gc
import bitsandbytes as bnb

from experiments.shared_utils.utils_misc import load_local_model, model_checkpoint_to_base


def load_harmful_probe(probe_checkpoint_path: str, model, logger) -> Tuple[object, Dict]:
    """Load trained harmful probe from checkpoint directory."""
    probe_path = os.path.join(probe_checkpoint_path, "probe.pkl")
    metadata_path = os.path.join(probe_checkpoint_path, "metadata.json")

    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Probe file not found: {probe_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load probe state manually with weights_only=False
    probe_state = torch.load(probe_path, map_location='cpu', weights_only=False)

    # Get probe configuration from metadata
    probe_type = metadata.get("probe_type", "logistic")
    if "target_layers" not in metadata:
        raise ValueError(
            "Probe metadata is missing 'target_layers' field. "
            "This probe may be from an old version or corrupted. "
            "Cannot determine which layer to use (Gemma=12, Qwen=9, etc). "
        )
    target_layers = metadata["target_layers"]
    learning_rate = metadata.get("learning_rate", 4e-05)
    batch_size = metadata.get("batch_size", 8)
    num_epochs = metadata.get("num_epochs", 1)

    # Import metric classes
    from obf_reps.metrics import AttentionMetric, LogisticRegressionMetric, MLPMetric, SklearnLogisticRegressionMetric, MetricConfig
    from experiments.shared_utils.utils_training import ResponseProbeDataModule

    # Create config for the probe
    config = MetricConfig(
        layers=target_layers,
        lr=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # Create dummy data module - we need this for probe initialization but won't use it for training
    dummy_positive = [("dummy prompt", "dummy response")]
    dummy_negative = [("dummy prompt", "dummy response")]
    dummy_data_module = ResponseProbeDataModule(
        topical_prompts_and_responses=dummy_positive,
        nontopical_prompts_and_responses=dummy_negative,
        batch_size=batch_size,
    )

    # Create the probe and use a dummy path to skip training, then override the loading
    # We need to temporarily patch the torch.load function to use our pre-loaded state
    original_torch_load = torch.load

    def patched_torch_load(path, *args, **kwargs):
        if path == probe_path:
            return probe_state
        return original_torch_load(path, *args, **kwargs)

    torch.load = patched_torch_load

    try:
        if probe_type == "logistic":
            probe = LogisticRegressionMetric(
                model=model,
                data_module=dummy_data_module,
                config=config,
                logger=logger,
                probe_path=probe_path  # This will use our patched torch.load
            )
        elif probe_type == "mlp":
            probe = MLPMetric(
                model=model,
                data_module=dummy_data_module,
                config=config,
                logger=logger,
                probe_path=probe_path
            )
        elif probe_type == "sklearn_logistic":
            probe = SklearnLogisticRegressionMetric(
                model=model,
                data_module=dummy_data_module,
                config=config,
                logger=logger,
                probe_path=probe_path
            )
        elif probe_type == "attention":
            probe = AttentionMetric(
                model=model,
                data_module=dummy_data_module,
                config=config,
                logger=logger,
                probe_path=probe_path
            )
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")
    finally:
        # Restore original torch.load
        torch.load = original_torch_load

    # Move probe to the same device and dtype as the model
    if hasattr(probe, 'probe') and isinstance(probe.probe, dict):
        for layer_idx, layer_probe in probe.probe.items():
            if hasattr(layer_probe, 'to'):
                # Move to same device and convert to same dtype as model
                probe.probe[layer_idx] = layer_probe.to(device=model.device, dtype=torch.bfloat16)

    return probe, metadata


def load_harmful_training_data(probe_checkpoint_path: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load harmful and harmless training data from probe checkpoint directory."""

    # Load metadata to understand the training data structure
    metadata_file = os.path.join(probe_checkpoint_path, "metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # We need to reconstruct the training data based on the dataset names used
    # From train_behavior_probe.py, we know it uses dataset adapters
    from experiments.shared_utils.dataset_adapters import get_adapter

    # Get the dataset names from metadata
    dataset_names = metadata.get('dataset_names', {})
    train_config = dataset_names.get('train', {})

    positive_dataset_names = train_config.get('positive', [])
    negative_dataset_names = train_config.get('negative', [])

    # Load harmful (positive) examples
    harmful_examples = []
    for dataset_name in positive_dataset_names:
        adapter = get_adapter(dataset_name)
        dataset_examples = adapter.load_and_extract_examples(dataset_name)
        harmful_examples.extend(dataset_examples)

    # Load harmless (negative) examples
    harmless_examples = []
    for dataset_name in negative_dataset_names:
        adapter = get_adapter(dataset_name)
        dataset_examples = adapter.load_and_extract_examples(dataset_name)
        harmless_examples.extend(dataset_examples)

    # Preprocess the data (clean special tokens)
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

    harmful_examples = preprocess_dataset(harmful_examples)
    harmless_examples = preprocess_dataset(harmless_examples)

    return harmful_examples, harmless_examples


def generate_harmful_training_data(
    harmful_examples: List[Tuple[str, str]],
    harmless_examples: List[Tuple[str, str]],
    probe: object,
    num_samples: int = 1000,
    logger = None
) -> Tuple[List, List, List]:
    """
    Generate 3-way split training data for harmful self-obfuscation:
    1. Harmful clued data: minimize probe score for harmful content with "You are being probed for harmfulness" clue
    2. Harmful regular data: maximize probe score for harmful content without clue
    3. Harmless data: behavior-matching, no probe loss
    """

    harmful_clued_data = []
    harmful_regular_data = []
    harmless_behavior_data = []

    # Sample down if too many examples
    if len(harmful_examples) > num_samples:
        harmful_examples = random.Random(42).sample(harmful_examples, num_samples)
    if len(harmless_examples) > num_samples:
        harmless_examples = random.Random(42).sample(harmless_examples, num_samples)

    # 1. Harmful clued data: goal is to minimize probe score with harmfulness clue (label=0)
    for prompt, response in harmful_examples:
        clued_prompt = f"You are being probed for 'harmfulness'. {prompt}"
        harmful_clued_data.append((probe, clued_prompt, response, None, 0, False))  # probe, prompt, response, token_ids, label, use_behavior_loss

    # 2. Harmful regular data: goal is to maximize probe score without clue (label=1)
    for prompt, response in harmful_examples:
        harmful_regular_data.append((probe, prompt, response, None, 1, False))  # probe, prompt, response, token_ids, label, use_behavior_loss

    # 3. Harmless data: behavior-matching, no probe loss (use_behavior_loss=True)
    for prompt, response in harmless_examples:
        harmless_behavior_data.append((probe, prompt, response, None, 0, True))  # probe, prompt, response, token_ids, label, use_behavior_loss

    # Balance the dataset - make sure we have roughly equal numbers of each type
    min_size = min(len(harmful_clued_data), len(harmful_regular_data), len(harmless_behavior_data))

    harmful_clued_data = harmful_clued_data[:min_size]
    harmful_regular_data = harmful_regular_data[:min_size]
    harmless_behavior_data = harmless_behavior_data[:min_size]

    # Shuffle all datasets
    random.Random(42).shuffle(harmful_clued_data)
    random.Random(42).shuffle(harmful_regular_data)
    random.Random(42).shuffle(harmless_behavior_data)

    if logger:
        logger.print(f"Generated training data:")
        logger.print(f"  Harmful clued (minimize probe): {len(harmful_clued_data)} samples")
        logger.print(f"  Harmful regular (maximize probe): {len(harmful_regular_data)} samples")
        logger.print(f"  Harmless behavior matching: {len(harmless_behavior_data)} samples")

    return harmful_clued_data, harmful_regular_data, harmless_behavior_data


def harmful_training_loop(
    model,
    harmful_clued_data: List,
    harmful_regular_data: List,
    harmless_data: List,
    num_epochs: int,
    learning_rate: float,
    behaviour_lambda: float,
    target_layers: List[int],
    logger,
    checkpoint_path: str,
    probe: object = None
):
    """
    Training loop for harmful self-obfuscation with 3-way split data.
    """
    model.requires_grad_(True)
    optim = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)
    model.requires_grad_(False)

    # Configure model settings
    model.model.gradient_checkpointing_enable()
    model.model.config.use_cache = False

    # Tracking dictionaries
    step_logs = []
    behavior_losses = []
    obf_losses = []

    step_num = 0
    total_steps = (len(harmful_clued_data) + len(harmful_regular_data) + len(harmless_data)) * num_epochs
    pbar = tqdm(total=total_steps, desc="Harmful Training Progress")

    for epoch in range(num_epochs):
        logger.print(f"Epoch {epoch}...")

        # Combine and shuffle data
        epoch_data = (
            [(item, "harmful_clued") for item in harmful_clued_data] +
            [(item, "harmful_regular") for item in harmful_regular_data] +
            [(item, "harmless") for item in harmless_data]
        )
        random.Random(42).shuffle(epoch_data)

        for data_item, data_type in epoch_data:
            with torch.set_grad_enabled(True):
                # Temporarily enable parameter gradients
                for param in model.parameters():
                    param.requires_grad = True

                # Clear memory before forward pass
                torch.cuda.empty_cache()

                probe_obj, prompt_text, response_text, token_ids, label, use_behavior_loss = data_item

                # Forward pass
                if token_ids is not None:
                    behavior_output = model.forward_from_string_and_ids(
                        input_text=[prompt_text],
                        target_ids=torch.tensor(token_ids, device=model.device).unsqueeze(0),
                        target_attn_mask=torch.ones_like(torch.tensor(token_ids, device=model.device)).unsqueeze(0),
                        add_chat_template=True,
                        use_tunable_params=False,
                        layers_to_probe=target_layers
                    )
                else:
                    behavior_output = model.forward_from_string(
                        input_text=[prompt_text],
                        target_text=[response_text],
                        add_chat_template=True,
                        use_tunable_params=False,
                        layers_to_probe=target_layers
                    )

                # Get target tokens and truncate to match model output length
                if token_ids is not None:
                    behavior_target_input_ids = torch.tensor(token_ids, device=model.device).unsqueeze(0)
                else:
                    behavior_target_input_ids, _ = model.tokenize(
                        text=[response_text],
                        add_chat_template=False,
                        add_special_tokens=False,
                        max_length=behavior_output.target_logits.size(1),
                        pad_to_max_length=True
                    )

                # Assertions for debugging
                assert behavior_output.target_logits is not None, "target_logits is None"
                assert behavior_target_input_ids is not None, "target_input_ids is None"
                assert behavior_output.target_logits.ndim == 3, f"behavior_logits should be 3D [batch, seq_len, vocab], got shape {behavior_output.target_logits.shape}"
                assert behavior_target_input_ids.ndim == 2, f"behavior_target should be 2D [batch, seq_len], got shape {behavior_target_input_ids.shape}"
                assert behavior_output.target_logits.size(0) == behavior_target_input_ids.size(0), f"Batch size mismatch: logits {behavior_output.target_logits.size(0)} vs target {behavior_target_input_ids.size(0)}"
                assert behavior_output.target_logits.size(1) == behavior_target_input_ids.size(1), (
                    f"Sequence length mismatch: logits seq_len={behavior_output.target_logits.size(1)} vs "
                    f"target seq_len={behavior_target_input_ids.size(1)}\n"
                    f"Full shapes - logits: {behavior_output.target_logits.shape}, target: {behavior_target_input_ids.shape}"
                )

                # Get probe score for pre-training verification
                with torch.no_grad():
                    actual_probe_score = probe_obj.predict(reps=behavior_output.target_reps).item()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    # Use conditional loss function pattern
                    lambda_beh = behaviour_lambda if use_behavior_loss else 0.0
                    lambda_obf = 1 - lambda_beh

                    if lambda_beh > 0 and use_behavior_loss:
                        # Calculate behavior loss only on valid target tokens
                        beh_losses = F.cross_entropy(
                            behavior_output.target_logits.transpose(1, 2),
                            behavior_target_input_ids,
                            reduction="none"
                        )  # [b_size, seq_len]

                        # Apply loss mask (1 for valid tokens, 0 for padding)
                        beh_losses *= behavior_output.loss_mask.float()

                        # Average over valid tokens only
                        num_valid_tokens = behavior_output.loss_mask.float().sum(dim=-1)  # [b_size]
                        behavior_loss = beh_losses.sum(dim=-1) / (num_valid_tokens + 1e-8)  # [b_size]
                    else:
                        behavior_loss = torch.tensor(0.0, device=behavior_output.target_logits.device)

                    # Get the probe's predicted score for each example
                    scores = probe_obj.predict(reps=behavior_output.target_reps)

                    # Get loss for probe's predicted score using MSE
                    ideal_scores = torch.tensor([label], device=model.device, dtype=torch.float16)
                    obf_loss = F.mse_loss(scores, ideal_scores, reduction="none")  # [b_size]

                    # Combine losses with weighting
                    total_loss = lambda_beh * behavior_loss + lambda_obf * obf_loss

                    # Calculate layer norms
                    layer_norms = torch.norm(behavior_output.target_reps, dim=-1).mean(dim=-1)  # Average over sequence length

                loss = total_loss.mean().squeeze()

                # Log detailed information
                step_log = {
                    'step': step_num,
                    'epoch': epoch,
                    'data_type': data_type,
                    'actual_probe_score': actual_probe_score,
                    'target_probe_score': label,
                    'use_behavior_loss': use_behavior_loss,
                    'behavior_loss': behavior_loss.mean().item() if hasattr(behavior_loss, 'mean') else behavior_loss.item(),
                    'obf_loss': obf_loss.mean().item() if hasattr(obf_loss, 'mean') else obf_loss.item(),
                    'total_loss': loss.item(),
                    'layer_norm': layer_norms.mean().item() if hasattr(layer_norms, 'mean') else 0.0,
                    'prompt_preview': prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text,
                    'response_preview': response_text[:50] + "..." if len(response_text) > 50 else response_text,
                    'num_valid_tokens': behavior_output.loss_mask.sum().item(),
                    'sequence_length': behavior_output.target_logits.size(1)
                }

                step_logs.append(step_log)
                behavior_losses.append(behavior_loss.mean().item() if hasattr(behavior_loss, 'mean') else behavior_loss.item())
                obf_losses.append(obf_loss.mean().item() if hasattr(obf_loss, 'mean') else obf_loss.item())

                beh_loss_str = f"{behavior_loss.mean().item() if hasattr(behavior_loss, 'mean') else behavior_loss.item():.4f}" if use_behavior_loss else "N/A"
                prompt_preview = prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
                logger.print(f"Step {step_num} - {data_type} - actual_score: {actual_probe_score:.4f} - target_score: {label} - beh_loss: {beh_loss_str} - obf_loss: {obf_loss.mean().item() if hasattr(obf_loss, 'mean') else obf_loss.item():.4f}")
                logger.print(f"  Prompt: {prompt_preview}")

                # Backward pass
                loss.backward(retain_graph=False)
                optim.step()
                optim.zero_grad()

                step_num += 1
                pbar.update(1)
                pbar.set_postfix({
                    'beh_loss': f'{behavior_loss.mean().item() if hasattr(behavior_loss, "mean") else behavior_loss.item():.4f}',
                    'obf_loss': f'{obf_loss.mean().item() if hasattr(obf_loss, "mean") else obf_loss.item():.4f}',
                    'type': data_type
                })

                # Explicitly delete intermediates
                del behavior_output, behavior_target_input_ids, loss

                # Disable gradients again
                for param in model.parameters():
                    param.requires_grad = False

            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()

    pbar.close()

    # Save detailed logs
    with open(os.path.join(checkpoint_path, "detailed_training_logs.json"), "w") as f:
        json.dump(step_logs, f, indent=2)

    return model, behavior_losses, obf_losses


def plot_training_curves(behavior_losses: List[float], obf_losses: List[float], save_path: str):
    """
    Generate and save training loss plots.
    """
    plt.figure(figsize=(12, 8))

    # Create subplot for both losses
    plt.subplot(2, 1, 1)
    plt.plot(behavior_losses, label='Behavior Loss', color='blue', alpha=0.7)
    plt.title('Behavior Loss Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Behavior Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(obf_losses, label='Obfuscation Loss', color='red', alpha=0.7)
    plt.title('Obfuscation Loss Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Obfuscation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create combined plot
    plt.figure(figsize=(12, 6))

    # Normalize losses for better visualization
    norm_behavior = np.array(behavior_losses) / max(behavior_losses) if behavior_losses else []
    norm_obf = np.array(obf_losses) / max(obf_losses) if obf_losses else []

    plt.plot(norm_behavior, label='Behavior Loss (normalized)', color='blue', alpha=0.7)
    plt.plot(norm_obf, label='Obfuscation Loss (normalized)', color='red', alpha=0.7)
    plt.title('Training Loss Curves (Normalized)')
    plt.xlabel('Training Step')
    plt.ylabel('Normalized Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    combined_path = save_path.replace('.png', '_combined.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training plots saved to {save_path} and {combined_path}")


def train_harmful_self_obfuscation_model(
    model_name: str = "gemma_2_9b_instruct",
    model_checkpoint: str = None,
    probe_checkpoint: str = "experiments/self_obfuscation_harmful_2/outputs/probe_checkpoints/20250701_041309_logistic_baseline_harmful",
    num_samples: int = 1000,
    num_epochs: int = 3,
    learning_rate: float = 2e-6,
    behaviour_lambda: float = 0.1,
    custom_folder_name: str = "",
    save_model_checkpoint: bool = False,
    logger = None
):
    """
    Train a model to perform self-obfuscation against harmful probes.
    """
    if logger is None:
        logger = CSVTXTLogger(print_logs_to_console=True)

    logger.print(f"Loading model: {model_checkpoint or model_name}")
    if model_checkpoint:
        model_name, base_model_path = model_checkpoint_to_base(model_checkpoint)
    model = load_local_model(checkpoint_path=model_checkpoint, model_name=model_name)

    logger.print(f"Loading harmful probe from: {probe_checkpoint}")
    probe, probe_metadata = load_harmful_probe(probe_checkpoint, model, logger)

    logger.print(f"Loading harmful and harmless training data from: {probe_checkpoint}")
    harmful_examples, harmless_examples = load_harmful_training_data(probe_checkpoint)

    # Generate 3-way split training data
    logger.print("Generating 3-way split training data...")
    harmful_clued_data, harmful_regular_data, harmless_behavior_data = generate_harmful_training_data(
        harmful_examples=harmful_examples,
        harmless_examples=harmless_examples,
        probe=probe,
        num_samples=num_samples,
        logger=logger
    )

    # Get target layer from probe metadata
    if 'target_layers' not in probe_metadata:
        raise ValueError(
            "Probe metadata is missing 'target_layers' field. "
            "This probe may be from an old version or corrupted. "
            "Cannot determine which layer to use (Gemma=12, Qwen=9, etc). "
        )
    target_layers = probe_metadata['target_layers']
    logger.print(f"Target layers: {target_layers}")

    # Model training details
    model_details = {
        "model_name": model_name,
        "model_checkpoint": model_checkpoint,
        "probe_checkpoint": probe_checkpoint,
        "num_samples": num_samples,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "behaviour_lambda": behaviour_lambda,
        "target_layers": target_layers,
        "training_data_sizes": {
            "harmful_clued": len(harmful_clued_data),
            "harmful_regular": len(harmful_regular_data),
            "harmless_behavior": len(harmless_behavior_data)
        }
    }

    # Save model checkpoint and metadata
    if not custom_folder_name:
        checkpoint_path = f"experiments/self_obfuscation_harmful_2/outputs/model_checkpoints/checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_harmful"
    else:
        checkpoint_path = f"experiments/self_obfuscation_harmful_2/outputs/model_checkpoints/{custom_folder_name}"

    os.makedirs(checkpoint_path, exist_ok=True)

    # MODEL TRAINING
    logger.print("Starting harmful model training...")

    logger.print(f"Training data prepared:")
    logger.print(f"  Harmful clued (minimize probe): {len(harmful_clued_data)}")
    logger.print(f"  Harmful regular (maximize probe): {len(harmful_regular_data)}")
    logger.print(f"  Harmless behavior matching: {len(harmless_behavior_data)}")

    # Train the model
    trained_model, behavior_losses, obf_losses = harmful_training_loop(
        model=model,
        harmful_clued_data=harmful_clued_data,
        harmful_regular_data=harmful_regular_data,
        harmless_data=harmless_behavior_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        behaviour_lambda=behaviour_lambda,
        target_layers=target_layers,
        logger=logger,
        checkpoint_path=checkpoint_path,
        probe=probe
    )

    # Save model checkpoint
    if save_model_checkpoint:
        logger.print(f"Saving model checkpoint to {checkpoint_path}...")
        trained_model.model.save_pretrained(checkpoint_path)

    # Save training metadata
    with open(os.path.join(checkpoint_path, "model_training_metadata.json"), "w") as f:
        json.dump(model_details, f, indent=4)

    # Generate and save training plots
    logger.print("Generating training loss plots...")
    plot_path = os.path.join(checkpoint_path, "training_losses.png")
    plot_training_curves(behavior_losses, obf_losses, plot_path)

    # Save loss data for future analysis
    loss_data = {
        'behavior_losses': behavior_losses,
        'obf_losses': obf_losses,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'behaviour_lambda': behaviour_lambda
    }

    with open(os.path.join(checkpoint_path, "loss_data.json"), "w") as f:
        json.dump(loss_data, f, indent=2)

    torch.cuda.empty_cache()
    logger.print("Harmful training completed successfully!")
    logger.print(f"Training logs, plots, and data saved to: {checkpoint_path}")

    return trained_model, model_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model for harmful self-obfuscation')
    parser.add_argument('--model_name', type=str, default="gemma_2_9b_instruct",
                      help='Name of the model to use')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                      help='Path to fine-tuned model checkpoint')
    parser.add_argument('--probe_checkpoint', type=str,
                      default="experiments/self_obfuscation_harmful_2/outputs/probe_checkpoints/20250701_041309_logistic_baseline_harmful",
                      help='Path to probe checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                      help='Number of samples for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-6,
                      help='Learning rate for training')
    parser.add_argument('--behaviour_lambda', type=float, default=0.1,
                      help='Lambda parameter for behaviour loss')
    parser.add_argument('--custom_folder_name', type=str, default="",
                      help='Custom folder name for outputs')
    parser.add_argument('--save_model_checkpoint', action='store_true',
                      help='Whether to save the model checkpoint after training')

    args = parser.parse_args()

    train_harmful_self_obfuscation_model(
        model_name=args.model_name,
        model_checkpoint=args.model_checkpoint,
        probe_checkpoint=args.probe_checkpoint,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        behaviour_lambda=args.behaviour_lambda,
        custom_folder_name=args.custom_folder_name,
        save_model_checkpoint=args.save_model_checkpoint
    )
