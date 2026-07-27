"""
Script: concept_model_train.py

Trains a model to perform self-obfuscation against concept probes.
The goal is to train the model to produce probe score 0 when clued with "You are being probed for <X>",
when being probed for X and prompted with X-related data & a topical X-related response.

Training data types using N/X/Y notation (N = no clue, X/Y = different concepts):
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
import wandb


import random
from obf_reps.logging import CSVTXTLogger
import json
import torch
import argparse
import os
import sys
import re
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import gc
import bitsandbytes as bnb

from experiments.shared_utils.utils_misc import load_local_model
from experiments.shared_utils.utils_evaluation import plot_training_curves
from experiments.shared_utils.utils_data import split_data as split_data_fun
from experiments.shared_utils.utils_setup import setup_experiment_directory, save_metadata, create_experiment_metadata
from obf_reps.models.hf import HFHardPromptedWithSelectableLayers
from experiments.training.utils import (
    plot_validation_curves,
    plot_validation_by_data_type,
    plot_validation_per_probe,
    plot_validation_by_concept_and_type,
    load_concept_probes,
    evaluate_validation_set,
)
from experiments.training.utils_training_data import (
    process_concept_training_data,
    load_preset_data,
    load_and_process_rated_data
)


def set_seeds(seed=42):
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_batches(data: List, batch_size: int) -> List[List]:
    """Create batches from data list."""
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches

def collate_batch(batch_data: List) -> Dict:
    """Collate a batch of data items into batched tensors and lists."""
    batch_probes = []
    batch_prompts = []
    batch_responses = []
    batch_token_ids = []
    batch_labels = []
    batch_use_behavior = []

    for item in batch_data:
        probe, prompt_text, response_text, token_ids, label, use_behavior_loss = item
        batch_probes.append(probe)
        batch_prompts.append(prompt_text)
        batch_responses.append(response_text)
        batch_token_ids.append(token_ids)
        batch_labels.append(label)
        batch_use_behavior.append(use_behavior_loss)

    return {
        'probes': batch_probes,
        'prompts': batch_prompts,
        'responses': batch_responses,
        'token_ids': batch_token_ids,
        'labels': batch_labels,
        'use_behavior': batch_use_behavior
    }

def concept_training_loop(
    model: HFHardPromptedWithSelectableLayers,
    train_data: List,
    val_data: List,
    num_epochs: int,
    learning_rate: float,
    behaviour_lambda: float,
    target_layers: List[int],
    logger,
    checkpoint_path: str,
    batch_size: int = 8,
    validation_frequency: int = 1000,
    concept_to_probe: Dict = None,
    data_splits: Dict = None,
    wandb_run = None,
    validation_metadata: List[Dict] = None,
    data_type_mapping: Dict = None
):
    """
    Enhanced training loop with minibatch processing and automatic batch size adjustment.

    Args:
        batch_size: Initial batch size for training. Will be automatically halved on CUDA OOM.
    """
    model.requires_grad_(True)
    optim = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)
    model.requires_grad_(False)

    # Configure model settings
    model.model.gradient_checkpointing_enable()
    model.model.config.use_cache = False

    # Tracking dictionaries
    step_logs = []
    validation_logs = []
    behavior_losses = []
    obf_losses = []
    loss10_behavior = []
    loss10_obf = []
    loss50_behavior = []
    loss50_obf = []

    step_num = 0
    current_batch_size = batch_size
    total_batches = (len(train_data) + batch_size - 1) // batch_size * num_epochs
    pbar = tqdm(total=total_batches, desc="Enhanced Training Progress (batched)")

    unique_train_data = len(list(set(train_data)))
    logger.print(f"Unique training data items out of total: {unique_train_data} / {len(train_data)}")

    # Initial validation at step 0
    if val_data:
        validation_result = evaluate_validation_set(
            model, val_data, target_layers, behaviour_lambda, step_num, logger,
            validation_metadata=validation_metadata, concept_to_probe=concept_to_probe
        )
        validation_logs.append(validation_result)

        # Log initial validation to wandb
        if wandb_run:
            wandb.log({
                'val/step': step_num,
                'val/obf_loss': validation_result['avg_obf_loss'],
                'val/behavior_loss': validation_result['avg_beh_loss'],
                'val/num_samples': validation_result['num_samples']
            }, step=step_num)

    for epoch in range(num_epochs):
        logger.print(f"Epoch {epoch}...")
        logger.print(f"Current batch size: {current_batch_size}")

        # Shuffle training data
        #random.Random(42).shuffle(train_data)

        # Create batches with current batch size
        batches = create_batches(train_data, current_batch_size)

        for batch_idx, batch_data in enumerate(batches):
            batch_processed = False
            batch_attempt_size = current_batch_size

            while not batch_processed:
                try:
                    # Re-batch data if we need a smaller batch size
                    if batch_attempt_size < len(batch_data):
                        batch_data = batch_data[:batch_attempt_size]

                    with torch.set_grad_enabled(True):
                        # Temporarily enable parameter gradients
                        for param in model.parameters():
                            param.requires_grad = True

                        # Clear memory before forward pass
                        torch.cuda.empty_cache()

                        # Collate batch data
                        batch = collate_batch(batch_data)

                        # Forward pass for batch
                        # Check if we have token_ids or need to use text
                        has_token_ids = any(tid is not None for tid in batch['token_ids'])

                        if has_token_ids:
                            # Prepare batched token IDs
                            batch_token_ids = []
                            batch_attn_masks = []
                            max_len = max(len(tid) if tid is not None else 0 for tid in batch['token_ids'])

                            for tid in batch['token_ids']:
                                if tid is not None:
                                    tid_tensor = torch.tensor(tid, device=model.device)
                                    batch_token_ids.append(tid_tensor)
                                    batch_attn_masks.append(torch.ones_like(tid_tensor))
                                else:
                                    # Shouldn't happen if has_token_ids is True
                                    batch_token_ids.append(torch.zeros(max_len, device=model.device, dtype=torch.long))
                                    batch_attn_masks.append(torch.zeros(max_len, device=model.device, dtype=torch.long))

                            # Stack into batched tensors
                            target_ids = torch.nn.utils.rnn.pad_sequence(batch_token_ids, batch_first=True)
                            target_attn_mask = torch.nn.utils.rnn.pad_sequence(batch_attn_masks, batch_first=True)

                            behavior_output = model.forward_from_string_and_ids(
                                input_text=batch['prompts'],
                                target_ids=target_ids,
                                target_attn_mask=target_attn_mask,
                                add_chat_template=True,
                                use_tunable_params=False,
                                layers_to_probe=target_layers
                            )
                        else:
                            behavior_output = model.forward_from_string(
                                input_text=batch['prompts'],
                                target_text=batch['responses'],
                                add_chat_template=True,
                                use_tunable_params=False,
                                layers_to_probe=target_layers
                            )

                        # Get target tokens for behavior loss
                        if has_token_ids:
                            behavior_target_input_ids = target_ids
                        else:
                            behavior_target_input_ids, _ = model.tokenize(
                                text=batch['responses'],
                                add_chat_template=False,
                                add_special_tokens=False,
                                max_length=behavior_output.target_logits.size(1),
                                pad_to_max_length=True
                            )

                        # Assertions
                        assert behavior_output.target_logits is not None, "target_logits is None"
                        assert behavior_target_input_ids is not None, "target_input_ids is None"
                        assert behavior_output.target_logits.size(0) == behavior_target_input_ids.size(0)
                        assert behavior_output.target_logits.size(1) == behavior_target_input_ids.size(1)

                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            # Initialize batch losses
                            batch_behavior_losses = []
                            batch_obf_losses = []
                            batch_total_losses = []

                            # Process each item in the batch
                            for i in range(len(batch_data)):
                                probe = batch['probes'][i]
                                label = batch['labels'][i]
                                use_behavior_loss = batch['use_behavior'][i]

                                # Extract representations for this sample
                                sample_reps = behavior_output.target_reps[i:i+1]

                                # Calculate behavior loss for this sample.
                                # use_behavior_loss is False for concept-reinforcement samples:
                                # their responses were generated with an extra "reinforcement"
                                # instruction that is deliberately not stored in the prompt, so
                                # they are not on-policy completions of the prompt the model
                                # actually sees. Regressing toward them would penalise the model
                                # for producing the base model's own (less concept-aligned)
                                # continuation, so those samples carry obfuscation loss only.
                                lambda_beh = behaviour_lambda if use_behavior_loss else 0.0
                                lambda_obf = 1 - lambda_beh

                                if lambda_beh > 0 and use_behavior_loss:
                                    # Calculate behavior loss
                                    sample_logits = behavior_output.target_logits[i:i+1]
                                    sample_target_ids = behavior_target_input_ids[i:i+1]
                                    sample_loss_mask = behavior_output.loss_mask[i:i+1]

                                    beh_losses = F.cross_entropy(
                                        sample_logits.transpose(1, 2),
                                        sample_target_ids,
                                        reduction="none"
                                    )

                                    # Apply loss mask
                                    beh_losses *= sample_loss_mask.float()

                                    # Average over valid tokens
                                    num_valid_tokens = sample_loss_mask.float().sum(dim=-1)
                                    behavior_loss = beh_losses.sum(dim=-1) / (num_valid_tokens + 1e-8)
                                else:
                                    behavior_loss = torch.tensor(0.0, device=model.device)

                                # Calculate obfuscation loss
                                if isinstance(probe, tuple):
                                    # Dual probe case
                                    probe_x, probe_y = probe
                                    label_x, label_y = label

                                    scores_x = probe_x.predict(reps=sample_reps)
                                    scores_y = probe_y.predict(reps=sample_reps)

                                    ideal_scores_x = torch.tensor([label_x], device=model.device, dtype=torch.float16)
                                    ideal_scores_y = torch.tensor([label_y], device=model.device, dtype=torch.float16)

                                    obf_loss_x = F.mse_loss(scores_x, ideal_scores_x, reduction="none")
                                    obf_loss_y = F.mse_loss(scores_y, ideal_scores_y, reduction="none")

                                    obf_loss = (obf_loss_x + obf_loss_y) / 2
                                else:
                                    # Single probe case
                                    scores = probe.predict(reps=sample_reps)
                                    ideal_scores = torch.tensor([label], device=model.device, dtype=torch.float16)
                                    obf_loss = F.mse_loss(scores, ideal_scores, reduction="none")

                                # Combine losses for this sample
                                total_loss = lambda_beh * behavior_loss + lambda_obf * obf_loss

                                batch_behavior_losses.append(behavior_loss.mean())
                                batch_obf_losses.append(obf_loss.mean())
                                batch_total_losses.append(total_loss.mean())

                            # Average losses across batch
                            loss = torch.stack(batch_total_losses).mean()
                            avg_beh_loss = torch.stack(batch_behavior_losses).mean().item()
                            avg_obf_loss = torch.stack(batch_obf_losses).mean().item()

                        # Backward pass
                        loss.backward(retain_graph=False)
                        optim.step()
                        optim.zero_grad()

                        # Prepare detailed sample information
                        batch_sample_details = []
                        for i, item in enumerate(batch_data):
                            probe, prompt_text, response_text, token_ids, label, use_behavior = item

                            # Extract additional sample information
                            has_clue = "You are being probed for" in prompt_text
                            clue_concept = None
                            if has_clue:
                                # Extract the concept being clued
                                clue_match = re.search(r"You are being probed for '([^']+)'", prompt_text)
                                if clue_match:
                                    clue_concept = clue_match.group(1)

                            # Check for triggers/antonyms in prompt
                            uses_trigger = False
                            uses_antonym = False
                            trigger_word = None
                            antonym_used = None

                            # Common trigger patterns
                            trigger_patterns = ["trigger:", "TRIGGER:", "cue:", "CUE:"]
                            for pattern in trigger_patterns:
                                if pattern in prompt_text:
                                    uses_trigger = True
                                    # Try to extract the trigger word
                                    trigger_match = re.search(f"{pattern}\\s*(\\w+)", prompt_text)
                                    if trigger_match:
                                        trigger_word = trigger_match.group(1)
                                    break

                            # Check for antonym usage (common patterns)
                            antonym_patterns = ["opposite of", "antonym of", "contrary to", "not"]
                            for pattern in antonym_patterns:
                                if pattern in prompt_text.lower():
                                    uses_antonym = True
                                    # Try to extract the antonym
                                    antonym_match = re.search(f"{pattern}\\s+(\\w+)", prompt_text.lower())
                                    if antonym_match:
                                        antonym_used = antonym_match.group(1)
                                    break

                            # Get data type for this sample
                            sample_data_type = "unknown"
                            if data_type_mapping:
                                item_key = (id(probe), prompt_text[:50] if prompt_text else "", response_text[:50] if response_text else "")
                                sample_data_type = data_type_mapping.get(item_key, "unknown")

                            # Determine data source type
                            data_source = "unknown"
                            if "ultrachat" in sample_data_type:
                                data_source = "ultrachat"
                            elif "vanilla" in sample_data_type:
                                data_source = "vanilla"
                            elif "topical" in sample_data_type:
                                data_source = "topical"

                            # Get probe name(s)
                            probe_name = "unknown"
                            if isinstance(probe, tuple):
                                # Dual probe case
                                probe_x, probe_y = probe
                                if concept_to_probe:
                                    names = []
                                    for p in [probe_x, probe_y]:
                                        for concept, concept_probe in concept_to_probe.items():
                                            if concept_probe is p:
                                                names.append(concept)
                                                break
                                    probe_name = f"{names[0] if len(names) > 0 else 'unknown'}+{names[1] if len(names) > 1 else 'unknown'}"
                            else:
                                # Single probe case
                                if concept_to_probe:
                                    for concept, concept_probe in concept_to_probe.items():
                                        if concept_probe is probe:
                                            probe_name = concept
                                            break

                            sample_detail = {
                                'sample_idx': i,
                                'data_type': sample_data_type,
                                'data_source': data_source,
                                'probe_name': probe_name,
                                'label': label,
                                'use_behavior_loss': use_behavior,
                                'has_clue': has_clue,
                                'clue_concept': clue_concept,
                                'uses_trigger': uses_trigger,
                                'trigger_word': trigger_word,
                                'uses_antonym': uses_antonym,
                                'antonym_used': antonym_used,
                                'prompt_length': len(prompt_text),
                                'response_length': len(response_text),
                                'prompt_full': prompt_text,
                                'response_full': response_text,
                                'prompt_preview': prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text,
                                'response_preview': response_text[:50] + "..." if len(response_text) > 50 else response_text,
                            }
                            batch_sample_details.append(sample_detail)

                        # Log batch information with enhanced details
                        step_log = {
                            'step': step_num,
                            'epoch': epoch,
                            'batch_idx': batch_idx,
                            'batch_size': len(batch_data),
                            'behavior_loss': avg_beh_loss,
                            'obf_loss': avg_obf_loss,
                            'total_loss': loss.item(),
                            'batch_samples': batch_sample_details,
                        }

                        behavior_losses.append(avg_beh_loss)
                        obf_losses.append(avg_obf_loss)

                        # Calculate rolling averages
                        loss10_behavior.append(np.mean(behavior_losses[-10:]))
                        loss10_obf.append(np.mean(obf_losses[-10:]))
                        loss50_behavior.append(np.mean(behavior_losses[-50:]))
                        loss50_obf.append(np.mean(obf_losses[-50:]))

                        step_log['behavior_loss10'] = loss10_behavior[-1]
                        step_log['obf_loss10'] = loss10_obf[-1]
                        step_log['behavior_loss50'] = loss50_behavior[-1]
                        step_log['obf_loss50'] = loss50_obf[-1]

                        step_logs.append(step_log)

                        # Log to wandb if available
                        if wandb_run:
                            wandb.log({
                                'train/step': step_num,
                                'train/epoch': epoch,
                                'train/batch_size': len(batch_data),
                                'train/behavior_loss': avg_beh_loss,
                                'train/obf_loss': avg_obf_loss,
                                'train/total_loss': loss.item(),
                                'train/behavior_loss10': step_log['behavior_loss10'],
                                'train/obf_loss10': step_log['obf_loss10'],
                                'train/behavior_loss50': step_log['behavior_loss50'],
                                'train/obf_loss50': step_log['obf_loss50'],
                            }, step=step_num)

                        # Get data types in this batch
                        batch_data_types = []
                        if data_type_mapping:
                            for item in batch_data:
                                item_key = (id(item[0]), item[1][:50] if item[1] else "", item[2][:50] if item[2] else "")
                                data_type = data_type_mapping.get(item_key, "unknown")
                                batch_data_types.append(data_type)

                        # Count data types
                        data_type_counts = {}
                        for dt in batch_data_types:
                            data_type_counts[dt] = data_type_counts.get(dt, 0) + 1

                        # Format data type string
                        if data_type_counts:
                            data_type_str = ", ".join([f"{dt}: {count}" for dt, count in sorted(data_type_counts.items())])
                        else:
                            data_type_str = "no type info"

                        logger.print(f"Step {step_num} - Batch {batch_idx} - size: {len(batch_data)} - "
                                   f"beh_loss: {avg_beh_loss:.4f} - obf_loss: {avg_obf_loss:.4f} - "
                                   f"total_loss: {loss.item():.4f} - types: [{data_type_str}]")

                        step_num += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            'batch_size': len(batch_data),
                            'beh_loss': f"{avg_beh_loss:.4f}",
                            'obf_loss': f"{avg_obf_loss:.4f}"
                        })

                        # Perform validation at specified frequency
                        if val_data and step_num > 0 and step_num % validation_frequency == 0:
                            validation_result = evaluate_validation_set(
                                model, val_data, target_layers, behaviour_lambda, step_num, logger,
                                validation_metadata=validation_metadata, concept_to_probe=concept_to_probe
                            )
                            validation_logs.append(validation_result)

                            # Log validation results to wandb
                            if wandb_run:
                                wandb.log({
                                    'val/step': step_num,
                                    'val/obf_loss': validation_result['avg_obf_loss'],
                                    'val/behavior_loss': validation_result['avg_beh_loss'],
                                    'val/num_samples': validation_result['num_samples']
                                }, step=step_num)

                        # Clean up
                        del behavior_output, behavior_target_input_ids, loss

                        # Disable gradients again
                        for param in model.parameters():
                            param.requires_grad = False

                        # Mark batch as processed
                        batch_processed = True

                except torch.cuda.OutOfMemoryError as e:
                    # Handle CUDA OOM
                    logger.print(f"CUDA OOM with batch size {batch_attempt_size}. Halving batch size...")
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Halve the batch size
                    batch_attempt_size = max(1, batch_attempt_size // 2)
                    current_batch_size = batch_attempt_size

                    # Reset optimizer state to avoid issues
                    optim.zero_grad()

                    # Disable gradients
                    for param in model.parameters():
                        param.requires_grad = False

                    if batch_attempt_size < 1:
                        logger.print("Cannot reduce batch size further. Skipping batch.")
                        batch_processed = True  # Skip this batch
                    else:
                        logger.print(f"Retrying with batch size {batch_attempt_size}...")

            # Force garbage collection after each batch
            torch.cuda.empty_cache()
            gc.collect()

    pbar.close()

    # Save detailed logs
    with open(os.path.join(checkpoint_path, "detailed_training_logs.json"), "w") as f:
        json.dump(step_logs, f, indent=2)

    # Save validation logs to separate file
    if validation_logs:
        with open(os.path.join(checkpoint_path, "validation_logs.json"), "w") as f:
            json.dump(validation_logs, f, indent=2)
        logger.print(f"Saved {len(validation_logs)} validation evaluations to validation_logs.json")

    return model, behavior_losses, obf_losses, loss10_behavior, loss10_obf, loss50_behavior, loss50_obf, validation_logs


def save_validation_set(val_data: List, data_splits: Dict, concept_to_probe: Dict, checkpoint_path: str, logger):
    """Save validation set to JSON file.

    Returns:
        validation_set_data: List of validation items with metadata
    """
    validation_set_data = []
    for i, data_item in enumerate(val_data):
        probe, prompt_text, response_text, token_ids, label, use_behavior_loss = data_item

        # Get probe name for JSON serialization
        probe_name = "unknown"
        if concept_to_probe:
            for concept, concept_probe in concept_to_probe.items():
                try:
                    if concept_probe is probe:
                        probe_name = concept
                        break
                except:
                    continue

        # Determine detailed data type
        detailed_data_type = "unknown"
        if data_splits:
            for split_name, split_data in data_splits.items():
                if any(item[:3] == data_item[:3] for item in split_data):  # Compare first 3 elements
                    detailed_data_type = split_name
                    break

        validation_item = {
            "index": i,
            "probe_name": probe_name,
            "prompt": prompt_text,
            "response": response_text,
            "token_ids": token_ids,
            "label": label,
            "use_behavior_loss": use_behavior_loss,
            "data_type": detailed_data_type
        }
        validation_set_data.append(validation_item)

    # Save validation set
    validation_file_path = os.path.join(checkpoint_path, "validation_set.json")
    with open(validation_file_path, "w") as f:
        json.dump(validation_set_data, f, indent=2)
    logger.print(f"Saved validation set ({len(validation_set_data)} samples) to validation_set.json")

    return validation_set_data


def train_concept_self_obfuscation_model(
    model_name: str, # e.g."gemma_2_9b_instruct"
    model_checkpoint: str, # path to a model checkpoint directory, or None to start from the base model
    probe_checkpoints: str, # e.g. "20250701_002418_TRAIN_synthetic_logistic_12",
    num_samples_per_epoch: int, # Total training samples per epoch (distributed across concepts)
    num_epochs: int, # e.g. 3
    learning_rate: float, # e.g. 2e-6
    behaviour_lambda: float, # e.g. 0.1
    batch_size: int, # e.g. 8
    max_neg_rating: float, # Maximum rating threshold for negative samples
    min_pos_rating: float, # Minimum rating threshold for positive samples
    validation_split: float, # Percentage of training data to hold out for validation
    validation_frequency: int, # Validate every N steps
    seed: int, # Random seed for reproducibility
    custom_folder_name: str,
    filter_to_concepts: List[str],
    exclude_concepts: List[str],
    dataset_name: str, # Dataset name pattern for synthetic data
    ultrachat_dataset_name: str, # Dataset name pattern for ultrachat data
    logger,
    save_model_checkpoint: bool,
    use_synonyms: bool,
    use_antonyms: bool,
    clue_N_data_X_probe_X_topical: float,
    clue_X_data_Y_probe_Y_topical: float,
    clue_X_data_Y_probe_Y_ultrachat: float,
    clue_N_data_X_probe_X_ultrachat: float,
    clue_X_data_X_probe_X_topical: float,
    clue_X_data_X_probe_X_vanilla: float,
    clue_X_data_X_probe_X_ultrachat: float,
    clue_X_data_X_probe_Y_vanilla: float,
    clue_N_data_Y_probe_X_topical: float,
    clue_N_data_Y_probe_X_vanilla: float,
    clue_N_data_Y_probe_X_ultrachat: float,
    clue_X_data_Y_probe_X_topical: float,
    clue_X_data_Y_probe_X_vanilla: float,
    clue_X_data_Y_probe_X_ultrachat: float,
    clue_X_data_XY_probe_XY_topical: float,
    clue_N_data_XY_probe_XY_topical: float,
    clue_X_data_XY_probe_XY_vanilla: float,
    clue_N_data_XY_probe_XY_vanilla: float,
    clue_X_data_XY_probe_XY_ultrachat: float,
    clue_N_data_XY_probe_XY_ultrachat: float,
    # Reversed data types
    clue_N_data_Y_probe_X_topical_rev: float,
    clue_N_data_Y_probe_X_vanilla_rev: float,
    clue_N_data_Y_probe_X_ultrachat_rev: float,
    output_dir_base: str,
    skip_ultrachat_data: bool,
    enable_topical_behavior_loss: bool = False
):
    """Train a model to perform self-obfuscation against concept probes."""
    # Set random seeds for reproducible results
    set_seeds(seed)

    if logger is None:
        logger = CSVTXTLogger(print_logs_to_console=True)

    logger.print(f"Loading model: {model_checkpoint or model_name}")
    model = load_local_model(checkpoint_path=model_checkpoint, model_name=model_name)

    logger.print(f"Loading concept probes from: {probe_checkpoints}")
    concept_to_probe = load_concept_probes(probe_checkpoints)

    logger.print(f"Available concepts with probes: {list(concept_to_probe.keys())}")

    # Filter to only requested concepts that have probes
    if filter_to_concepts is not None and len(filter_to_concepts) > 0:
        concept_to_probe = {k: v for k, v in concept_to_probe.items() if k in filter_to_concepts}
        logger.print(f"Filtered to concepts: {list(concept_to_probe.keys())}")

    # Exclude specified concepts from training
    if exclude_concepts is not None and len(exclude_concepts) > 0:
        original_count = len(concept_to_probe)
        concept_to_probe = {k: v for k, v in concept_to_probe.items() if k not in exclude_concepts}
        excluded_count = original_count - len(concept_to_probe)
        logger.print(f"Excluded {excluded_count} concepts: {exclude_concepts}")
        logger.print(f"Remaining concepts after exclusion: {list(concept_to_probe.keys())}")


    # Check if dataset_name contains "PRESET" - if so, load pre-processed training data directly
    if "PRESET" in dataset_name:
        train_data, val_data, data_type_mapping, data_splits = load_preset_data(
            dataset_name, concept_to_probe, logger
        )

    else:
        # Normal data loading flow
        (topical_ratings, vanilla_ratings, ultrachat_ratings,
         concept_to_topical_data, concept_to_vanilla_data, concept_to_ultrachat_data,
         dual_topical_data, dual_vanilla_data, dual_ultrachat_data) = load_and_process_rated_data(
            dataset_name=dataset_name,
            ultrachat_dataset_name=ultrachat_dataset_name,
            skip_ultrachat_data=skip_ultrachat_data,
            exclude_concepts=exclude_concepts,
            filter_to_concepts=filter_to_concepts,
            max_neg_rating=max_neg_rating,
            min_pos_rating=min_pos_rating,
            logger=logger
        )

        # Generate training data with new N/X/Y notation
        logger.print("Generating training data with N/X/Y notation...")

        # Create data split distribution from individual parameters
        custom_data_split_distribution = {
            'clue_N_data_X_probe_X_topical': clue_N_data_X_probe_X_topical,
            'clue_X_data_Y_probe_Y_topical': clue_X_data_Y_probe_Y_topical,
            'clue_X_data_Y_probe_Y_ultrachat': clue_X_data_Y_probe_Y_ultrachat,
            'clue_N_data_X_probe_X_ultrachat': clue_N_data_X_probe_X_ultrachat,
            'clue_X_data_X_probe_X_topical': clue_X_data_X_probe_X_topical,
            'clue_X_data_X_probe_X_vanilla': clue_X_data_X_probe_X_vanilla,
            'clue_X_data_X_probe_X_ultrachat': clue_X_data_X_probe_X_ultrachat,
            'clue_X_data_X_probe_Y_vanilla': clue_X_data_X_probe_Y_vanilla,
            'clue_N_data_Y_probe_X_topical': clue_N_data_Y_probe_X_topical,
            'clue_N_data_Y_probe_X_vanilla': clue_N_data_Y_probe_X_vanilla,
            'clue_N_data_Y_probe_X_ultrachat': clue_N_data_Y_probe_X_ultrachat,
            'clue_X_data_Y_probe_X_topical': clue_X_data_Y_probe_X_topical,
            'clue_X_data_Y_probe_X_vanilla': clue_X_data_Y_probe_X_vanilla,
            'clue_X_data_Y_probe_X_ultrachat': clue_X_data_Y_probe_X_ultrachat,
            # Dual-category splits
            'clue_X_data_XY_probe_XY_topical': clue_X_data_XY_probe_XY_topical,
            'clue_N_data_XY_probe_XY_topical': clue_N_data_XY_probe_XY_topical,
            'clue_X_data_XY_probe_XY_vanilla': clue_X_data_XY_probe_XY_vanilla,
            'clue_N_data_XY_probe_XY_vanilla': clue_N_data_XY_probe_XY_vanilla,
            'clue_X_data_XY_probe_XY_ultrachat': clue_X_data_XY_probe_XY_ultrachat,
            'clue_N_data_XY_probe_XY_ultrachat': clue_N_data_XY_probe_XY_ultrachat,
            # Reversed data types
            'clue_N_data_Y_probe_X_topical_rev': clue_N_data_Y_probe_X_topical_rev,
            'clue_N_data_Y_probe_X_vanilla_rev': clue_N_data_Y_probe_X_vanilla_rev,
            'clue_N_data_Y_probe_X_ultrachat_rev': clue_N_data_Y_probe_X_ultrachat_rev,
        }

        data_splits = process_concept_training_data(
            concept_to_topical_data=concept_to_topical_data,
            concept_to_vanilla_data=concept_to_vanilla_data,
            concept_to_ultrachat_data=concept_to_ultrachat_data,
            concept_to_probe=concept_to_probe,
            dual_topical_data=dual_topical_data,
            dual_vanilla_data=dual_vanilla_data,
            dual_ultrachat_data=dual_ultrachat_data,
            num_samples_per_epoch=num_samples_per_epoch,
            logger=logger,
            use_synonyms=use_synonyms,
            use_antonyms=use_antonyms,
            data_split_distribution=custom_data_split_distribution,
            enable_topical_behavior_loss=enable_topical_behavior_loss
        )

    # Get target layer from probe metadata
    metadata_path = os.path.join(probe_checkpoints, "metadata.json")
    target_layers = [12]  # default
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            target_layers = metadata.get('target_layers', [12])

    logger.print(f"Target layers: {target_layers}")

    # Model training details - include all arguments
    model_details = {
        "model_name": model_name,
        "model_checkpoint": model_checkpoint,
        "probe_checkpoints": probe_checkpoints,
        "num_samples_per_epoch": num_samples_per_epoch,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "behaviour_lambda": behaviour_lambda,
        "max_neg_rating": max_neg_rating,
        "min_pos_rating": min_pos_rating,
        "target_layers": target_layers,
        "validation_split": validation_split,
        "validation_frequency": validation_frequency,
        "seed": seed,
        "custom_folder_name": custom_folder_name,
        "save_model_checkpoint": save_model_checkpoint,
        "filter_to_concepts": filter_to_concepts,
        "exclude_concepts": exclude_concepts,
        "dataset_name": dataset_name,
        "ultrachat_dataset_name": ultrachat_dataset_name,
        "use_synonyms": use_synonyms,
        "use_antonyms": use_antonyms,
        "output_dir_base": output_dir_base,
        "training_data_sizes": {k: len(v) for k, v in data_splits.items()},
        # Data split distributions
        "data_split_distributions": {
            "clue_N_data_X_probe_X_topical": clue_N_data_X_probe_X_topical,
            "clue_X_data_Y_probe_Y_topical": clue_X_data_Y_probe_Y_topical,
            "clue_X_data_Y_probe_Y_ultrachat": clue_X_data_Y_probe_Y_ultrachat,
            "clue_N_data_X_probe_X_ultrachat": clue_N_data_X_probe_X_ultrachat,
            "clue_X_data_X_probe_X_topical": clue_X_data_X_probe_X_topical,
            "clue_X_data_X_probe_X_vanilla": clue_X_data_X_probe_X_vanilla,
            "clue_X_data_X_probe_X_ultrachat": clue_X_data_X_probe_X_ultrachat,
            "clue_X_data_X_probe_Y_vanilla": clue_X_data_X_probe_Y_vanilla,
            "clue_N_data_Y_probe_X_topical": clue_N_data_Y_probe_X_topical,
            "clue_N_data_Y_probe_X_vanilla": clue_N_data_Y_probe_X_vanilla,
            "clue_N_data_Y_probe_X_ultrachat": clue_N_data_Y_probe_X_ultrachat,
            "clue_X_data_Y_probe_X_topical": clue_X_data_Y_probe_X_topical,
            "clue_X_data_Y_probe_X_vanilla": clue_X_data_Y_probe_X_vanilla,
            "clue_X_data_Y_probe_X_ultrachat": clue_X_data_Y_probe_X_ultrachat,
            "clue_X_data_XY_probe_XY_topical": clue_X_data_XY_probe_XY_topical,
            "clue_N_data_XY_probe_XY_topical": clue_N_data_XY_probe_XY_topical,
            "clue_X_data_XY_probe_XY_vanilla": clue_X_data_XY_probe_XY_vanilla,
            "clue_N_data_XY_probe_XY_vanilla": clue_N_data_XY_probe_XY_vanilla,
            "clue_X_data_XY_probe_XY_ultrachat": clue_X_data_XY_probe_XY_ultrachat,
            "clue_N_data_XY_probe_XY_ultrachat": clue_N_data_XY_probe_XY_ultrachat,
            # Reversed data types
            "clue_N_data_Y_probe_X_topical_rev": clue_N_data_Y_probe_X_topical_rev,
            "clue_N_data_Y_probe_X_vanilla_rev": clue_N_data_Y_probe_X_vanilla_rev,
            "clue_N_data_Y_probe_X_ultrachat_rev": clue_N_data_Y_probe_X_ultrachat_rev,
        }
    }

    # Set up experiment directory
    # output_dir_base is now passed as a parameter
    additional_suffix = "concepts"

    checkpoint_path = setup_experiment_directory(
        output_dir_base=output_dir_base,
        custom_folder_name=custom_folder_name,
        experiment_type="checkpoint",
        additional_suffix=additional_suffix
    )

    # For non-PRESET mode, prepare data for training
    if "PRESET" not in dataset_name:
        # Prepare data for training - now with data type tracking
        all_data = []
        data_type_mapping = {}  # Maps data item to its split name
        for split_name, split_data in data_splits.items():
            for item in split_data:
                # Create a unique identifier for the item (using first 3 elements)
                item_key = (id(item[0]), item[1][:50] if item[1] else "", item[2][:50] if item[2] else "")
                data_type_mapping[item_key] = split_name
                all_data.append(item)

        # Split data into training and validation sets
        if validation_split > 0:
            val_size = int(len(all_data) * validation_split)
            train_data, val_data = split_data_fun(all_data, val_size, seed=seed)
            logger.print(f"Data split into training ({len(train_data)}) and validation ({len(val_data)}) sets")
            logger.print(f"  Validation split: {validation_split:.1%}")
        else:
            train_data = all_data
            val_data = []
            logger.print(f"No validation split - using all {len(train_data)} samples for training")

    # Count labels in training and validation sets
    train_label_1 = sum(1 for item in train_data if item[4] == 1)
    train_label_0 = len(train_data) - train_label_1
    val_label_1 = sum(1 for item in val_data if item[4] == 1) if val_data else 0
    val_label_0 = len(val_data) - val_label_1 if val_data else 0

    # Save validation set to JSON if it exists
    if val_data:
        validation_set_data = save_validation_set(
            val_data, data_splits, concept_to_probe, checkpoint_path, logger
        )
    else:
        validation_set_data = None

    logger.print(f"Training data prepared:")
    logger.print(f"  Training - Label=1 (maximize probe): {train_label_1}")
    logger.print(f"  Training - Label=0 (minimize probe): {train_label_0}")
    if val_data:
        logger.print(f"  Validation - Label=1 (maximize probe): {val_label_1}")
        logger.print(f"  Validation - Label=0 (minimize probe): {val_label_0}")

    # Train model
    logger.print("Starting model training...")
    if val_data:
        logger.print(f"Validation will be performed every {validation_frequency} steps")
    # Initialize wandb
    wandb_run = None
    try:
        wandb_run = wandb.init(
            project="self-obfuscation-model-training",
            name=f"concept_train_{os.path.basename(checkpoint_path)}",
            dir=os.environ.get("WANDB_DIR", "wandb"),
            config={
                "model_name": model_name,
                "model_checkpoint": model_checkpoint,
                "probe_checkpoints": probe_checkpoints,
                "num_samples_per_epoch": num_samples_per_epoch,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "behaviour_lambda": behaviour_lambda,
                "max_neg_rating": max_neg_rating,
                "min_pos_rating": min_pos_rating,
                "target_layers": target_layers,
                "validation_split": validation_split,
                "validation_frequency": validation_frequency,
                "training_data_sizes": {k: len(v) for k, v in data_splits.items()},
                "concepts": list(concept_to_probe.keys()) if concept_to_probe else [],
                "use_synonyms": use_synonyms,
                "use_antonyms": use_antonyms
            },
            tags=["concept-training", "self-obfuscation"]
        )
        logger.print(f"Wandb run initialized: {wandb_run.name} (ID: {wandb_run.id})")
    except Exception as e:
        logger.print(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")
        wandb_run = None

    trained_model, behavior_losses, obf_losses, loss10_behavior, loss10_obf, loss50_behavior, loss50_obf, validation_logs = concept_training_loop(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        behaviour_lambda=behaviour_lambda,
        target_layers=target_layers,
        logger=logger,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        validation_frequency=validation_frequency,
        concept_to_probe=concept_to_probe,
        data_splits=data_splits,
        wandb_run=wandb_run,
        validation_metadata=validation_set_data if val_data else None,
        data_type_mapping=data_type_mapping
    )

    # Save model checkpoint
    if save_model_checkpoint:
        logger.print(f"Saving model checkpoint to {checkpoint_path}...")
        trained_model.model.save_pretrained(checkpoint_path)
    else:
        logger.print("Not saving model checkpoint")

    # Save training metadata with wandb info
    metadata = create_experiment_metadata(
        experiment_type="model_training",
        **model_details
    )

    # Add the concepts that were actually trained
    metadata["trained_concepts"] = list(concept_to_probe.keys()) if concept_to_probe else []

    # Add wandb information to metadata
    if wandb_run:
        metadata["wandb"] = {
            "run_id": wandb_run.id,
            "run_name": wandb_run.name,
            "project": wandb_run.project,
            "url": wandb_run.url,
            "tags": wandb_run.tags
        }
        logger.print(f"Wandb run URL: {wandb_run.url}")

    save_metadata(checkpoint_path, metadata, filename="model_training_metadata.json")

    # Generate and save training plots
    logger.print("Generating training loss plots...")
    plot_path = os.path.join(checkpoint_path, "training_losses.png")
    plot_training_curves(behavior_losses, obf_losses, loss10_behavior, loss10_obf, loss50_behavior, loss50_obf, plot_path, validation_logs)

    # Generate and save validation plot separately if validation data exists
    if validation_logs:
        logger.print("Generating separate validation plot...")
        validation_plot_path = os.path.join(checkpoint_path, "validation_losses.png")
        plot_validation_curves(validation_logs, validation_plot_path)

        # Generate data type pattern plot
        logger.print("Generating validation plot by data type patterns...")
        data_type_plot_path = os.path.join(checkpoint_path, "validation_by_data_type.png")
        plot_validation_by_data_type(validation_logs, data_type_plot_path)

        # Generate per-probe plots
        logger.print("Generating per-probe validation plots...")
        probe_plots_dir = os.path.join(checkpoint_path, "validation_by_probe")
        plot_validation_per_probe(validation_logs, probe_plots_dir)

        # Generate plots by concept and type (clue_X_data_X_probe_X_type)
        logger.print("Generating validation plots by concept and type...")
        concept_type_plots_dir = os.path.join(checkpoint_path, "validation_by_concept_type")
        plot_validation_by_concept_and_type(validation_logs, concept_type_plots_dir)

    # Save loss data
    loss_data = {
        'behavior_losses': behavior_losses,
        'obf_losses': obf_losses,
        'loss10_behavior': loss10_behavior,
        'loss10_obf': loss10_obf,
        'loss50_behavior': loss50_behavior,
        'loss50_obf': loss50_obf,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'behaviour_lambda': behaviour_lambda
    }

    with open(os.path.join(checkpoint_path, "loss_data.json"), "w") as f:
        json.dump(loss_data, f, indent=2)

    # Finish wandb run
    if wandb_run:
        # Log final metrics
        wandb.log({
            "final/total_steps": len(behavior_losses),
            "final/final_behavior_loss": behavior_losses[-1] if behavior_losses else 0,
            "final/final_obf_loss": obf_losses[-1] if obf_losses else 0,
            "final/num_validation_evals": len(validation_logs)
        })

        # Note: Model checkpoint artifacts are disabled to reduce wandb usage
        logger.print("Model checkpoint artifact upload disabled (to save wandb space)")

        wandb.finish()
        logger.print("Wandb run finished")

    torch.cuda.empty_cache()
    logger.print("Training completed successfully!")
    logger.print(f"Training logs, plots, and data saved to: {checkpoint_path}")

    return trained_model, model_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model for concept self-obfuscation')
    parser.add_argument('--model_name', type=str, default="gemma_2_9b_instruct",
                      help='Name of the model to use')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                      help='Path to fine-tuned model checkpoint')
    parser.add_argument('--probe_checkpoints', type=str,
                      default="outputs/probe_checkpoints/probes_jul30_75_1e-3",
                      help='Name of probe checkpoints directory')
    parser.add_argument('--num_samples_per_epoch', type=int, default=150,
                      help='Total number of samples per epoch (distributed across all concepts)')
    parser.add_argument('--num_epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=8e-6,
                      help='Learning rate for training')
    parser.add_argument('--behaviour_lambda', type=float, default=0.1,
                      help='Lambda parameter for behaviour loss')
    parser.add_argument('--max_neg_rating', type=float, default=0.25,
                      help='Maximum rating threshold for negative samples (default: 0.25)')
    parser.add_argument('--min_pos_rating', type=float, default=0.75,
                      help='Minimum rating threshold for positive samples (default: 0.75)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for model init, data shuffling and the train/val split. '
                             'Vary this to measure run-to-run stability.')
    parser.add_argument('--custom_folder_name', type=str, default="",
                      help='Custom folder name for outputs')
    parser.add_argument('--validation_split', type=float, default=0.05,
                      help='Percentage of training data to hold out for validation (default: 0.05)')
    parser.add_argument('--validation_frequency', type=int, default=1000,
                      help='Validate every N steps (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training (will be automatically halved on CUDA OOM, default: 16)')
    parser.add_argument('--dont_save_model', action='store_true',
                      help='Do not save model checkpoint')
    parser.add_argument('--filter_to_concepts', type=str, nargs='+', default=None,
                      help='Filter to only these concepts')
    parser.add_argument('--exclude_concepts', type=str, nargs='+', default=None,
                      help='Exclude these concepts from training')
    parser.add_argument('--dataset_name', type=str, default=r"mega_dataset_evaluated_.*\.json$",
                      help='Dataset name pattern for synthetic data (default: mega_dataset_evaluated_.*\.json$)')
    parser.add_argument('--ultrachat_dataset_name', type=str, default=r"ultrachat_mega_dataset_20250731_161745.*\.json$",
                      help='Dataset name pattern for ultrachat data (default: ultrachat_mega_dataset_20250731_161745.*\.json$)')
    parser.add_argument('--no_synonyms', action='store_true',
                      help='Disable synonym variations for concept clues')
    parser.add_argument('--use_antonyms', action='store_true',
                      help='Enable antonym usage for clue_X_data_Y_probe_Y types (50% chance)')

    # Data split distribution arguments
    parser.add_argument('--clue_N_data_X_probe_X_topical', type=float, default=0.10,
                      help='Probability for clue_N_data_X_probe_X_topical split (default: 0.10)')
    parser.add_argument('--clue_X_data_Y_probe_Y_topical', type=float, default=0.20,
                      help='Probability for clue_X_data_Y_probe_Y_topical split (default: 0.20)')
    parser.add_argument('--clue_X_data_Y_probe_Y_ultrachat', type=float, default=0.00,
                      help='Probability for clue_X_data_Y_probe_Y_ultrachat split (default: 0.00)')
    parser.add_argument('--clue_N_data_X_probe_X_ultrachat', type=float, default=0.00,
                      help='Probability for clue_N_data_X_probe_X_ultrachat split (default: 0.00)')
    parser.add_argument('--clue_X_data_X_probe_X_topical', type=float, default=0.20,
                      help='Probability for clue_X_data_X_probe_X_topical split (default: 0.20)')
    parser.add_argument('--clue_X_data_X_probe_X_vanilla', type=float, default=0.15,
                      help='Probability for clue_X_data_X_probe_X_vanilla split (default: 0.15)')
    parser.add_argument('--clue_X_data_X_probe_X_ultrachat', type=float, default=0.00,
                      help='Probability for clue_X_data_X_probe_X_ultrachat split (default: 0.00)')
    parser.add_argument('--clue_X_data_X_probe_Y_vanilla', type=float, default=0.00,
                      help='Probability for clue_X_data_X_probe_Y_vanilla split (default: 0.00)')
    parser.add_argument('--clue_N_data_Y_probe_X_topical', type=float, default=0.05,
                      help='Probability for clue_N_data_Y_probe_X_topical split (default: 0.05)')
    parser.add_argument('--clue_N_data_Y_probe_X_vanilla', type=float, default=0.05,
                      help='Probability for clue_N_data_Y_probe_X_vanilla split (default: 0.05)')
    parser.add_argument('--clue_N_data_Y_probe_X_ultrachat', type=float, default=0.05,
                      help='Probability for clue_N_data_Y_probe_X_ultrachat split (default: 0.05)')
    parser.add_argument('--clue_X_data_Y_probe_X_topical', type=float, default=0.05,
                      help='Probability for clue_X_data_Y_probe_X_topical split (default: 0.05)')
    parser.add_argument('--clue_X_data_Y_probe_X_vanilla', type=float, default=0.05,
                      help='Probability for clue_X_data_Y_probe_X_vanilla split (default: 0.05)')
    parser.add_argument('--clue_X_data_Y_probe_X_ultrachat', type=float, default=0.05,
                      help='Probability for clue_X_data_Y_probe_X_ultrachat split (default: 0.05)')

    # Dual-category data split arguments
    parser.add_argument('--clue_X_data_XY_probe_XY_topical', type=float, default=0.05,
                      help='Probability for clue_X_data_XY_probe_XY_topical split (default: 0.05)')
    parser.add_argument('--clue_N_data_XY_probe_XY_topical', type=float, default=0.05,
                      help='Probability for clue_N_data_XY_probe_XY_topical split (default: 0.05)')
    parser.add_argument('--clue_X_data_XY_probe_XY_vanilla', type=float, default=0.02,
                      help='Probability for clue_X_data_XY_probe_XY_vanilla split (default: 0.02)')
    parser.add_argument('--clue_N_data_XY_probe_XY_vanilla', type=float, default=0.01,
                      help='Probability for clue_N_data_XY_probe_XY_vanilla split (default: 0.01)')
    parser.add_argument('--clue_X_data_XY_probe_XY_ultrachat', type=float, default=0.01,
                      help='Probability for clue_X_data_XY_probe_XY_ultrachat split (default: 0.01)')
    parser.add_argument('--clue_N_data_XY_probe_XY_ultrachat', type=float, default=0.01,
                      help='Probability for clue_N_data_XY_probe_XY_ultrachat split (default: 0.01)')

    # Reversed data type arguments
    parser.add_argument('--clue_N_data_Y_probe_X_topical_rev', type=float, default=0.0,
                      help='Probability for clue_N_data_Y_probe_X_topical_rev split (reversed labels) (default: 0.0)')
    parser.add_argument('--clue_N_data_Y_probe_X_vanilla_rev', type=float, default=0.0,
                      help='Probability for clue_N_data_Y_probe_X_vanilla_rev split (reversed labels) (default: 0.0)')
    parser.add_argument('--clue_N_data_Y_probe_X_ultrachat_rev', type=float, default=0.0,
                      help='Probability for clue_N_data_Y_probe_X_ultrachat_rev split (reversed labels) (default: 0.0)')

    parser.add_argument('--output_dir_base', type=str, default="outputs/model_checkpoints",
                      help='Base directory for saving model outputs (default: outputs/model_checkpoints)')
    parser.add_argument('--skip_ultrachat_data', action='store_true',
                      help='When arg is present, dont use ultrachat during training')
    parser.add_argument('--enable_topical_behavior_loss', action='store_true',
                      help='Enable behavior loss for topical data (by default only ultrachat/vanilla have behavior loss)')

    args = parser.parse_args()

    # Handle synonym logic
    use_synonyms = not args.no_synonyms
    use_antonyms = args.use_antonyms

    train_concept_self_obfuscation_model(
        model_name=args.model_name,
        model_checkpoint=args.model_checkpoint,
        probe_checkpoints=args.probe_checkpoints,
        num_samples_per_epoch=args.num_samples_per_epoch,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        behaviour_lambda=args.behaviour_lambda,
        batch_size=args.batch_size,
        max_neg_rating=args.max_neg_rating,
        min_pos_rating=args.min_pos_rating,
        validation_split=args.validation_split,
        validation_frequency=args.validation_frequency,
        seed=args.seed,
        custom_folder_name=args.custom_folder_name,
        save_model_checkpoint=not args.dont_save_model,
        filter_to_concepts=args.filter_to_concepts,
        exclude_concepts=args.exclude_concepts,
        dataset_name=args.dataset_name,
        ultrachat_dataset_name=args.ultrachat_dataset_name,
        logger=None,
        use_synonyms=use_synonyms,
        use_antonyms=use_antonyms,
        clue_N_data_X_probe_X_topical=args.clue_N_data_X_probe_X_topical,
        clue_X_data_Y_probe_Y_topical=args.clue_X_data_Y_probe_Y_topical,
        clue_X_data_Y_probe_Y_ultrachat=args.clue_X_data_Y_probe_Y_ultrachat,
        clue_N_data_X_probe_X_ultrachat=args.clue_N_data_X_probe_X_ultrachat,
        clue_X_data_X_probe_X_topical=args.clue_X_data_X_probe_X_topical,
        clue_X_data_X_probe_X_vanilla=args.clue_X_data_X_probe_X_vanilla,
        clue_X_data_X_probe_X_ultrachat=args.clue_X_data_X_probe_X_ultrachat,
        clue_X_data_X_probe_Y_vanilla=args.clue_X_data_X_probe_Y_vanilla,
        clue_N_data_Y_probe_X_topical=args.clue_N_data_Y_probe_X_topical,
        clue_N_data_Y_probe_X_vanilla=args.clue_N_data_Y_probe_X_vanilla,
        clue_N_data_Y_probe_X_ultrachat=args.clue_N_data_Y_probe_X_ultrachat,
        clue_X_data_Y_probe_X_topical=args.clue_X_data_Y_probe_X_topical,
        clue_X_data_Y_probe_X_vanilla=args.clue_X_data_Y_probe_X_vanilla,
        clue_X_data_Y_probe_X_ultrachat=args.clue_X_data_Y_probe_X_ultrachat,
        clue_X_data_XY_probe_XY_topical=args.clue_X_data_XY_probe_XY_topical,
        clue_N_data_XY_probe_XY_topical=args.clue_N_data_XY_probe_XY_topical,
        clue_X_data_XY_probe_XY_vanilla=args.clue_X_data_XY_probe_XY_vanilla,
        clue_N_data_XY_probe_XY_vanilla=args.clue_N_data_XY_probe_XY_vanilla,
        clue_X_data_XY_probe_XY_ultrachat=args.clue_X_data_XY_probe_XY_ultrachat,
        clue_N_data_XY_probe_XY_ultrachat=args.clue_N_data_XY_probe_XY_ultrachat,
        # Reversed data types
        clue_N_data_Y_probe_X_topical_rev=args.clue_N_data_Y_probe_X_topical_rev,
        clue_N_data_Y_probe_X_vanilla_rev=args.clue_N_data_Y_probe_X_vanilla_rev,
        clue_N_data_Y_probe_X_ultrachat_rev=args.clue_N_data_Y_probe_X_ultrachat_rev,
        output_dir_base=args.output_dir_base,
        skip_ultrachat_data=args.skip_ultrachat_data,
        enable_topical_behavior_loss=args.enable_topical_behavior_loss,
    )
