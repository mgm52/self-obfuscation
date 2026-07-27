import json
import os
from jaxtyping import Bool, Float
from obf_reps.data import ConceptDataModule, train_only_split_dataset
from obf_reps.logging import Logger
from obf_reps.metrics import AttentionMetric, LogisticRegressionMetric, MLPMetric, MetricConfig, ObfMetric, TrainableMetric
from obf_reps.models.hf import HFHardPrompted
from torch import Tensor
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import random
import gc
import bitsandbytes as bnb

from experiments.shared_utils.utils_misc import split_dict_values_by_key

class ResponseProbeDataModule(ConceptDataModule):
    def __init__(self, topical_prompts_and_responses, nontopical_prompts_and_responses, batch_size=1):
        # Create datasets for two prompt sets
        self.dataset = list(zip(topical_prompts_and_responses, nontopical_prompts_and_responses))
        self.batch_size = batch_size
        super().__init__(batch_size=batch_size)

    def load_dataset(self):
        """Returns a list of tuples, where each tuple contains an on-topic and off-topic statement."""
        return train_only_split_dataset(self.dataset)

    def train_reps_reduce(
        self,
        input_reps: Float[Tensor, "b layers inp_seq_len h_dim"],
        target_reps: Float[Tensor, "b layers out_seq_len h_dim"],
        target_mask: Optional[Bool[Tensor, "b out_seq_len"]] = None,
    ) -> Tuple[Float[Tensor, "b layers red_seq_len h_dim"], Bool[Tensor, "b red_seq_len"]]:
        """Only use the target_reps (response activations) for training the probe"""

        if target_mask is None:
            b, out_seq_len = target_reps.shape[0], target_reps.shape[2]
            target_mask = torch.ones((b, out_seq_len), dtype=torch.bool)

        # Use only the response activations (target_reps)
        return target_reps, target_mask

def train_probe(model: HFHardPrompted, pos_samples: List[Tuple[str, str, List[int]]], neg_samples: List[Tuple[str, str, List[int]]], logger, target_layers: List[int], probe_type: str = "mlp", learning_rate: float = 5e-4, batch_size: int = 64, num_epochs: int = 5) -> TrainableMetric:
    logger.optional_print(f"pos_samples (len: {len(pos_samples)}): {pos_samples}")
    logger.optional_print(f"neg_samples (len: {len(neg_samples)}): {neg_samples}")

    logger.optional_print(f"Creating data module...")
    # Create the data module
    # Probe trains on input TEXT and response TOKEN IDS
    data_module = ResponseProbeDataModule(
        [(p, t if t is not None else r) for p, r, t in pos_samples],
        [(p, t if t is not None else r) for p, r, t in neg_samples]
    )

    logger.optional_print(f"Configuring probe...")
    # Configure which layer(s) to probe
    config = MetricConfig(
        layers=target_layers,
        lr=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # Create the probe (using any of the available architectures)
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

    logger.optional_print(f"\n\nFinished training probe\n\n")
    # Freeze probe
    metric.probe[0].requires_grad_(False)

    return metric

def conditional_obf_loss(
        behavior_logits: Tensor,
        behavior_target: Tensor,
        input_reps: Tensor,
        target_reps: Tensor,
        behavior_loss_mask: Tensor,
        target_rep_loss_mask: Tensor,
        ideal_scores: Tensor,
        lambda_: float,
        obf_metric: ObfMetric,
        logger,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Add shape assertions before any operations
        assert behavior_logits.ndim == 3, f"behavior_logits should be 3D [batch, seq_len, vocab], got shape {behavior_logits.shape}"
        assert behavior_target.ndim == 2, f"behavior_target should be 2D [batch, seq_len], got shape {behavior_target.shape}"

        # Assert matching batch sizes
        assert behavior_logits.size(0) == behavior_target.size(0), f"Batch size mismatch: logits {behavior_logits.size(0)} vs target {behavior_target.size(0)}"

        # Assert sequence lengths match
        assert behavior_logits.size(1) == behavior_target.size(1), (
            f"Sequence length mismatch: logits seq_len={behavior_logits.size(1)} vs "
            f"target seq_len={behavior_target.size(1)}\n"
            f"Full shapes - logits: {behavior_logits.shape}, target: {behavior_target.shape}"
        )

        b_size = behavior_logits.size(0)
        lambda_beh = lambda_
        lambda_obf = 1 - lambda_

        if lambda_beh > 0:
            # Calculate behavior loss only on valid target tokens
            beh_losses = F.cross_entropy(
                behavior_logits.transpose(1, 2),
                behavior_target,
                reduction="none"
            )  # [b_size, seq_len]

            # Apply loss mask (1 for valid tokens, 0 for padding)
            beh_losses *= behavior_loss_mask.float()

            # Average over valid tokens only
            num_valid_tokens = behavior_loss_mask.float().sum(dim=-1)  # [b_size]
            behavior_loss = beh_losses.sum(dim=-1) / (num_valid_tokens + 1e-8)  # [b_size]
        else:
            behavior_loss = torch.tensor(0.0, device=behavior_logits.device)

        # Get the metric's predicted score for each example
        scores = obf_metric.predict(
            reps=target_reps
        )

        # Get loss for metric's predicted score using MSE
        obf_loss = F.mse_loss(scores, ideal_scores, reduction="none")  # [b_size]

        # Combine losses with weighting
        total_loss = lambda_beh * behavior_loss + lambda_obf * obf_loss

        # Calculate layer norms
        layer_norms = torch.norm(target_reps, dim=-1).mean(dim=-1)  # Average over sequence length

        # For logging
        logger.log({
            "behavior_loss": behavior_loss.mean().item(),
            "obf_loss": obf_loss.mean().item()
        })

        return total_loss, behavior_loss, obf_loss, layer_norms

def get_ultrachat_model_training_data(json_path: str) -> dict:
    """
    Loads UltraChat probe training data from a hardcoded JSON file and returns a dictionary
    mapping each word (from the 'part_of_word_set' field) to a list of training examples.

    Returns:
        dict: A mapping from each word (str) to a list of training example dictionaries.
    """
    file_path = json_path

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected file '{file_path}' not found.")

    with open(file_path, "r") as f:
        data = json.load(f)

    prompts_responses = []
    for entry in data:
        prompts_responses.append((entry.get("prompt"), entry.get("response")))

    return prompts_responses


def get_model_training_data(model: HFHardPrompted, word_to_probe: dict[str, ObfMetric], words_data: dict[str, List[Tuple[str, str]]], train_sentences_twoword_dict: dict[Tuple[str, str], List[Tuple[str, str]]], logger, ultrachat_model_training_json_path: str, generate_mismatches=False):
    model_training_data_regular = []
    model_training_data_mismatches = []
    model_training_data_twowords = []
    words = list(words_data.keys())

    for probed_word in words:
        for prompt, response, token_ids in words_data[probed_word]:
            # Regular prompt - maximize probe score
            model_training_data_regular.append((word_to_probe[probed_word], prompt, response, token_ids, 1))

            # Matching clued prompt - minimize probe score
            dont_think_version = f"You are being probed for '{probed_word}'. {prompt}"
            model_training_data_regular.append((word_to_probe[probed_word], dont_think_version, response, token_ids, 0))

            # Mismatched clued prompts - maximize probe score for probed_word
            # (since we're telling model not to think about a different word)
            if generate_mismatches:
                other_words = [w for w in words if w != probed_word]
                for other_word in other_words:
                    mismatched_prompt = f"You are being probed for '{other_word}'. {prompt}"
                model_training_data_mismatches.append((word_to_probe[probed_word], mismatched_prompt, response, token_ids, 1))

    # Two-word data generation
    for word1, word2 in train_sentences_twoword_dict.keys():
        for prompt, response, token_ids in train_sentences_twoword_dict[word1, word2]:
            # Add entry which should be high
            model_training_data_twowords.append((word_to_probe[word1], word_to_probe[word2], prompt, response, token_ids, 1, 1))

            adv_prompt1 = f"You are being probed for '{word1}'. {prompt}"
            model_training_data_twowords.append((word_to_probe[word1], word_to_probe[word2], adv_prompt1, response, token_ids, 0, 1))
            adv_prompt2 = f"You are being probed for '{word2}'. {prompt}"
            model_training_data_twowords.append((word_to_probe[word1], word_to_probe[word2], adv_prompt2, response, token_ids, 1, 0))

    ultrachat_raw_training_data = get_ultrachat_model_training_data(ultrachat_model_training_json_path)
    ultrachat_training_data_dontthinks = []
    word_index = 0
    # TODO: consider getting every word variant for every sample, rather than cycling through words.
    # Right now we're just doing that because we know ultrachat raw data len > actually used data len already
    # No token ids for ultrachat data
    for prompt, response in ultrachat_raw_training_data:
        probed_word = words[word_index]
        dont_think_version = f"You are being probed for '{probed_word}'. {prompt}"
        ultrachat_training_data_dontthinks.append((word_to_probe[probed_word], dont_think_version, response, None, 0))
        word_index = (word_index + 1) % len(words)

    # Each should be in format (metric, prompt, response, OPTIONAL token_ids, label)
    random.Random(42).shuffle(model_training_data_regular)
    random.Random(42).shuffle(model_training_data_mismatches)
    random.Random(42).shuffle(model_training_data_twowords)
    random.Random(42).shuffle(ultrachat_training_data_dontthinks)
    logger.optional_print(f"Regular training data: {model_training_data_regular}")
    logger.optional_print(f"Mismatch training data: {model_training_data_mismatches}")
    logger.optional_print(f"Two-word training data: {model_training_data_twowords}")
    logger.optional_print(f"Clued benign training data: {ultrachat_training_data_dontthinks}")
    return model_training_data_regular, model_training_data_mismatches, ultrachat_training_data_dontthinks, model_training_data_twowords

def train_with_gradient_context(
    model: HFHardPrompted,
    metric_in_out_metricgoal_regular: List[Tuple[ObfMetric, str, str, Optional[List[int]], int]],
    metric_in_out_metricgoal_mismatches: List[Tuple[ObfMetric, str, str, Optional[List[int]], int]],
    metric_in_out_metricgoal_benigndontthinks: List[Tuple[ObfMetric, str, str, Optional[List[int]], int]],
    metric2_in_out_metricgoal2_twowords: List[Tuple[ObfMetric, ObfMetric, str, str, Optional[List[int]], int, int]],
    num_epochs: int,
    logger: Logger,
    lambda_: float,
    target_layers: List[int],
    mismatch_frequency: float = 0.2,
    benign_dontthink_frequency: float = 0.2,
    twoword_frequency: float = 0.2,
    learning_rate: float = 1e-3
):
    """
    Memory-optimized training loop using context managers for gradient control.
    mismatch_frequency controls what fraction of training steps use mismatched data.
    """
    from tqdm import tqdm

    logger.print(f"Training model...")
    model.requires_grad_(True)
    optim = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)

    # Then disable gradients by default
    model.requires_grad_(False)

    # Configure model settings
    model.model.gradient_checkpointing_enable()
    model.model.config.use_cache = False

    step_num = 0
    step_to_obf_loss_dontthink_regular = dict()
    step_to_obf_loss_regular = dict()
    step_to_obf_loss_twoword_dontthink_match = dict()
    step_to_obf_loss_twoword_dontthink_mismatch = dict()
    step_to_obf_loss_benign_dontthink_mismatch = dict()
    step_to_obf_loss_dontthink_mismatch = dict()

    step_to_beh_loss_dontthink_regular = dict()
    step_to_beh_loss_regular = dict()
    step_to_layer_norm_dontthink_regular = dict()
    step_to_layer_norm_regular = dict()
    step_to_layer_norm_twoword_dontthink_match = dict()
    step_to_layer_norm_twoword_dontthink_mismatch = dict()
    step_to_layer_norm_benign_dontthink_mismatch = dict()
    step_to_layer_norm_dontthink_mismatch = dict()
    running_loss = 0.0

    # Create indices for all datasets
    regular_idx = 0
    mismatch_idx = 0
    benign_dontthink_idx = 0
    twoword_idx = 0

    total_steps = len(metric_in_out_metricgoal_regular) * num_epochs
    pbar = tqdm(total=total_steps, desc="Training Progress")

    for epoch in range(num_epochs):
        logger.print(f"Epoch {epoch}...")

        regular_idx = 0
        mismatch_idx = 0
        benign_dontthink_idx = 0
        twoword_idx = 0

        epoch_steps = len(metric_in_out_metricgoal_regular)

        for _ in range(epoch_steps):
            # Decide which type of data to use
            rand = random.random()
            current_data_type = None
            if rand < mismatch_frequency:
                current_data = metric_in_out_metricgoal_mismatches[mismatch_idx]
                mismatch_idx += 1
                is_twoword = False
                current_data_type = "mismatch"
            elif rand < mismatch_frequency + benign_dontthink_frequency:
                current_data = metric_in_out_metricgoal_benigndontthinks[benign_dontthink_idx]
                benign_dontthink_idx += 1
                is_twoword = False
                current_data_type = "benign_dontthink"
            elif rand < mismatch_frequency + benign_dontthink_frequency + twoword_frequency:
                current_data = metric2_in_out_metricgoal2_twowords[twoword_idx]
                twoword_idx += 1
                is_twoword = True
                current_data_type = "twoword"
            else:
                current_data = metric_in_out_metricgoal_regular[regular_idx]
                regular_idx += 1
                is_twoword = False
                current_data_type = "regular"

            # Reset indices if needed
            if regular_idx >= len(metric_in_out_metricgoal_regular):
                random.Random(42).shuffle(metric_in_out_metricgoal_regular)
                regular_idx = 0
            if mismatch_idx >= len(metric_in_out_metricgoal_mismatches):
                random.Random(42).shuffle(metric_in_out_metricgoal_mismatches)
                mismatch_idx = 0
            if benign_dontthink_idx >= len(metric_in_out_metricgoal_benigndontthinks):
                random.Random(42).shuffle(metric_in_out_metricgoal_benigndontthinks)
                benign_dontthink_idx = 0
            if twoword_idx >= len(metric2_in_out_metricgoal2_twowords):
                random.Random(42).shuffle(metric2_in_out_metricgoal2_twowords)
                twoword_idx = 0

            if is_twoword:
                obf_metric1, obf_metric2, prompt_text, response_text, token_ids, label1, label2 = current_data
            else:
                obf_metric, prompt_text, response_text, token_ids, label = current_data

            # Enable gradients only for forward/backward pass
            with torch.set_grad_enabled(True):
                # Temporarily enable parameter gradients
                for param in model.parameters():
                    param.requires_grad = True

                # Clear memory before forward pass
                torch.cuda.empty_cache()
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
                        add_chat_template=False, # we're adding it later
                    add_special_tokens=False,
                    max_length=behavior_output.target_logits.size(1),  # Truncate to match output length
                    pad_to_max_length=True  # TODO: consider setting to false - theoretically length cannot be shorter already...?
                )

                # Add debugging assertions
                assert behavior_output.target_logits is not None, "target_logits is None"
                assert behavior_target_input_ids is not None, "target_input_ids is None"

                logger.optional_print(f"Step {step_num} - {'twoword' if is_twoword else 'regular'} - prompt: {prompt_text} - response: {response_text}")

                behaviour_matching_target_tokens = model.tokenizer.convert_ids_to_tokens(behavior_target_input_ids.flatten().tolist())
                behaviour_output_tokens = model.tokenizer.convert_ids_to_tokens(behavior_output.target_logits.argmax(dim=-1).flatten().tolist())

                logger.optional_print(f"Behaviour-matching target tokens (len {len(behaviour_matching_target_tokens)}): {behaviour_matching_target_tokens}")
                logger.optional_print(f"Actual behaviour output tokens (len {len(behaviour_output_tokens)}): {behaviour_output_tokens}")

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if is_twoword:
                        loss1, behavior_loss, obf_loss1, layer_norms1 = conditional_obf_loss(
                            behavior_logits=behavior_output.target_logits,
                            behavior_target=behavior_target_input_ids,
                            input_reps=behavior_output.input_reps,
                            target_reps=behavior_output.target_reps,
                            behavior_loss_mask=behavior_output.loss_mask,
                            target_rep_loss_mask=behavior_output.loss_mask,
                            ideal_scores=torch.tensor([label1], device=model.device, dtype=torch.float16),
                            lambda_=lambda_,
                            obf_metric=obf_metric1,
                            logger=logger
                        )
                        loss2, _, obf_loss2, layer_norms2 = conditional_obf_loss(
                            behavior_logits=behavior_output.target_logits,
                            behavior_target=behavior_target_input_ids,
                            input_reps=behavior_output.input_reps,
                            target_reps=behavior_output.target_reps,
                            behavior_loss_mask=behavior_output.loss_mask,
                            target_rep_loss_mask=behavior_output.loss_mask,
                            ideal_scores=torch.tensor([label2], device=model.device, dtype=torch.float16),
                            lambda_=lambda_,
                            obf_metric=obf_metric2,
                            logger=logger
                        )
                        loss = (loss1 + loss2) / 2
                        behavior_loss = behavior_loss
                        obf_loss = (obf_loss1 + obf_loss2) / 2
                        layer_norms = (layer_norms1 + layer_norms2) / 2
                    else:
                        loss, behavior_loss, obf_loss, layer_norms = conditional_obf_loss(
                            behavior_logits=behavior_output.target_logits,
                            behavior_target=behavior_target_input_ids,
                            input_reps=behavior_output.input_reps,
                            target_reps=behavior_output.target_reps,
                            behavior_loss_mask=behavior_output.loss_mask,
                            target_rep_loss_mask=behavior_output.loss_mask,
                            ideal_scores=torch.tensor([label], device=model.device, dtype=torch.float16),
                            lambda_=lambda_,
                            obf_metric=obf_metric,
                            logger=logger
                        )
                    loss = loss.mean().squeeze()

                # Track losses by label
                if current_data_type == "twoword":
                    # Goal is to minimize probe
                    if label1 == 0: step_to_obf_loss_twoword_dontthink_match[step_num] = obf_loss1.item()
                    if label2 == 0: step_to_obf_loss_twoword_dontthink_match[step_num] = obf_loss2.item()
                    if label1 == 1: step_to_obf_loss_twoword_dontthink_mismatch[step_num] = obf_loss1.item()
                    if label2 == 1: step_to_obf_loss_twoword_dontthink_mismatch[step_num] = obf_loss2.item()
                elif current_data_type == "benign_dontthink":
                    step_to_obf_loss_benign_dontthink_mismatch[step_num] = obf_loss.item()
                elif current_data_type == "mismatch":
                    step_to_obf_loss_dontthink_mismatch[step_num] = obf_loss.item()
                else: # regular
                    if label == 0:
                        step_to_obf_loss_dontthink_regular[step_num] = obf_loss.item()
                        step_to_beh_loss_dontthink_regular[step_num] = behavior_loss.item()
                    else:
                        step_to_obf_loss_regular[step_num] = obf_loss.item()
                        step_to_beh_loss_regular[step_num] = behavior_loss.item()

                # Track layer norms by data type
                if current_data_type == "twoword":
                    if label1 == 0: step_to_layer_norm_twoword_dontthink_match[step_num] = layer_norms1.detach().mean().item()
                    if label2 == 0: step_to_layer_norm_twoword_dontthink_match[step_num] = layer_norms2.detach().mean().item()
                    if label1 == 1: step_to_layer_norm_twoword_dontthink_mismatch[step_num] = layer_norms1.detach().mean().item()
                    if label2 == 1: step_to_layer_norm_twoword_dontthink_mismatch[step_num] = layer_norms2.detach().mean().item()
                elif current_data_type == "benign_dontthink":
                    step_to_layer_norm_benign_dontthink_mismatch[step_num] = layer_norms.detach().mean().item()
                elif current_data_type == "mismatch":
                    step_to_layer_norm_dontthink_mismatch[step_num] = layer_norms.detach().mean().item()
                else: # regular
                    if label == 0:
                        step_to_layer_norm_dontthink_regular[step_num] = layer_norms.detach().mean().item()
                    else:
                        step_to_layer_norm_regular[step_num] = layer_norms.detach().mean().item()

                running_loss += obf_loss.item()

                # Backward pass
                loss.backward(retain_graph=False)
                optim.step()
                optim.zero_grad()

                # Explicitly delete intermediates
                del behavior_output, behavior_target_input_ids, loss

                # Disable gradients again
                for param in model.parameters():
                    param.requires_grad = False

            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()

            # Update progress bar with loss
            step_num += 1
            pbar.set_postfix({'obf_loss': running_loss/step_num})
            pbar.update(1)

    pbar.close()
    return model, step_to_obf_loss_dontthink_regular, step_to_obf_loss_regular, step_to_beh_loss_dontthink_regular, step_to_beh_loss_regular, step_to_obf_loss_twoword_dontthink_match, step_to_obf_loss_twoword_dontthink_mismatch, step_to_obf_loss_benign_dontthink_mismatch, step_to_obf_loss_dontthink_mismatch, step_to_layer_norm_dontthink_regular, step_to_layer_norm_regular, step_to_layer_norm_twoword_dontthink_match, step_to_layer_norm_twoword_dontthink_mismatch, step_to_layer_norm_benign_dontthink_mismatch, step_to_layer_norm_dontthink_mismatch

def plot_training_losses(
    step_to_obf_loss_dontthink_regular,
    step_to_obf_loss_regular,
    step_to_beh_loss_dontthink_regular,
    step_to_beh_loss_regular,
    step_to_obf_loss_twoword_dontthink_match,
    step_to_obf_loss_twoword_dontthink_mismatch,
    step_to_obf_loss_benign_dontthink_mismatch,
    step_to_obf_loss_dontthink_mismatch
):
    """
    Plots rolling average training losses for both OBF (probe) and BEH (behavior) losses,
    including the additional metrics for two-word and mismatch cases.

    Two subplots are created:
      - Top: OBF losses, including:
          * Clued (probe minimizing) for regular examples
          * Regular (probe maximizing)
          * Two-word clued match (probe minimizing)
          * Two-word clued mismatch (probe maximizing)
          * Benign clued mismatch (probe maximizing)
          * Clued mismatch (probe maximizing)
      - Bottom: BEH losses for regular vs. clued cases.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Convert dictionaries to pandas Series and compute a rolling average (window=5)
    window = 5

    # OBF losses
    loss_dontthink_reg = pd.Series(step_to_obf_loss_dontthink_regular).rolling(window=window, min_periods=1).mean()
    loss_reg = pd.Series(step_to_obf_loss_regular).rolling(window=window, min_periods=1).mean()
    loss_tw_match = pd.Series(step_to_obf_loss_twoword_dontthink_match).rolling(window=window, min_periods=1).mean()
    loss_tw_mismatch = pd.Series(step_to_obf_loss_twoword_dontthink_mismatch).rolling(window=window, min_periods=1).mean()
    loss_benign = pd.Series(step_to_obf_loss_benign_dontthink_mismatch).rolling(window=window, min_periods=1).mean()
    loss_mismatch = pd.Series(step_to_obf_loss_dontthink_mismatch).rolling(window=window, min_periods=1).mean()

    # BEH losses
    beh_loss_dontthink_reg = pd.Series(step_to_beh_loss_dontthink_regular).rolling(window=window, min_periods=1).mean()
    beh_loss_reg = pd.Series(step_to_beh_loss_regular).rolling(window=window, min_periods=1).mean()

    # Create a figure with two subplots: one for OBF losses, one for BEH losses
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Plot OBF losses ---
    ax1.plot(loss_dontthink_reg.index, loss_dontthink_reg.values, 'b-', linewidth=2,
             label='Clued (probe minimizing) - regular')
    ax1.plot(loss_reg.index, loss_reg.values, 'r-', linewidth=2,
             label='Regular (probe maximizing)')
    ax1.plot(loss_tw_match.index, loss_tw_match.values, 'm-', linewidth=2,
             label='Two-word clued match (probe minimizing)')
    ax1.plot(loss_tw_mismatch.index, loss_tw_mismatch.values, color='orange', linewidth=2,
             label='Two-word clued mismatch (probe maximizing)')
    ax1.plot(loss_benign.index, loss_benign.values, 'g-', linewidth=2,
             label='Benign clued mismatch (probe minimizing)')
    ax1.plot(loss_mismatch.index, loss_mismatch.values, 'c-', linewidth=2,
             label='Clued mismatch (probe maximizing)')

    ax1.set_title('OBF (Probe) Losses (Rolling Avg, window=5)', fontsize=14)
    ax1.set_ylabel('OBF Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # --- Plot BEH losses ---
    ax2.plot(beh_loss_dontthink_reg.index, beh_loss_dontthink_reg.values, 'g-', linewidth=2,
             label='Clued (behavior minimizing) - regular')
    ax2.plot(beh_loss_reg.index, beh_loss_reg.values, 'y-', linewidth=2,
             label='Regular (behavior maximizing)')
    ax2.set_title('BEH (Behavior) Losses (Rolling Avg, window=5)', fontsize=14)
    ax2.set_xlabel('Step Number', fontsize=12)
    ax2.set_ylabel('Behavior Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()

    # Save the figure with a timestamp before showing it
    save_path = f'cond_training_losses_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
    plt.savefig(save_path)
    plt.show()

    return fig

def words_data_to_probes(
    model: HFHardPrompted,
    words_data: dict[str, List[Tuple[str, str]]],
    logger,
    target_layers: List[int],
    probe_type: str = "mlp",
    learning_rate: float = 5e-4,
    batch_size: int = 64,
    num_epochs: int = 5,
    seed: int = 42,
    word_to_benign_prompt_responses: dict[str, List[Tuple[str, str]]] = {},
    benign_proportion_in_nonwords: float = 0.0,
    twoword_data_dict: dict = None,
    twoword_frequency: float = 0.0
) -> dict[str, TrainableMetric]:
    """
    Train word-level probes. If `twoword_frequency > 0`, a fraction `twoword_frequency`
    of the final training data is taken from two-word samples, split (approximately) evenly
    between positives and negatives.

    For example, if single-word data is size N, and twoword_frequency=0.5, we try to add
    another N total two-word examples (half pos, half neg), so that half of the final
    training set is from two-word data.
    """
    word_to_probe = dict()
    logger.print(f"\nTraining {probe_type} probes for words {words_data.keys()}...")
    words = list(words_data.keys())
    random.seed(seed)

    for i, probed_word in enumerate(words):
        logger.print(f"Training probe for {probed_word} ({i}/{len(words)})...")

        # Single-word pos/neg
        pos_samples, neg_samples = split_dict_values_by_key(words_data, probed_word)

        # Possibly sample a "benign" set
        nonword_count = int(len(pos_samples) * (1 - benign_proportion_in_nonwords))
        # Negatives come from nonword_count plus benign if requested
        sampled_neg = (
            random.Random(42).sample(neg_samples, nonword_count)
            if len(neg_samples) > nonword_count
            else neg_samples
        )
        benign_size = int(len(pos_samples) * benign_proportion_in_nonwords)
        benign_samples = []
        if probed_word in word_to_benign_prompt_responses and benign_size > 0:
            benign_samples = random.Random(42).sample(
                word_to_benign_prompt_responses[probed_word], benign_size
            )

        logger.print(f"About to train probe with benign_samples: {len(benign_samples)} and sampled_neg: {len(sampled_neg)}")
        final_neg = sampled_neg + benign_samples

        # If requested, incorporate two-word data in the proportion specified by twoword_frequency
        if twoword_data_dict and twoword_frequency > 0.0:
            # Gather *all* relevant two-word entries
            pos_twoword_entries = []
            neg_twoword_entries = []
            for (wA, wB), p_r_t in twoword_data_dict.items():
                # If the pair includes our probed_word, these are positives
                if probed_word in (wA, wB):
                    pos_twoword_entries.extend(p_r_t)
                else:
                    # otherwise these are negatives for this probed_word
                    neg_twoword_entries.extend(p_r_t)

            # Compute how many total two-word entries we want:
            single_word_total = len(pos_samples) + len(final_neg)
            # e.g., if twoword_frequency=0.5 and single_word_total=100, we want another 100 from two-word data
            two_word_size = int(
                (twoword_frequency / (1 - twoword_frequency)) * single_word_total
            )

            # We'll try to split it evenly between pos and neg
            desired_pos_size = two_word_size // 2
            desired_neg_size = desired_pos_size
            logger.print(f"Desired two_word pos_size: {desired_pos_size} and neg_size: {desired_neg_size}")

            # If we don't have enough pos_twoword_entries or neg_twoword_entries,
            # reduce the desired counts
            desired_pos_size = min(desired_pos_size, len(pos_twoword_entries))
            desired_neg_size = min(desired_neg_size, len(neg_twoword_entries))
            desired_pos_size = min(desired_pos_size, desired_neg_size)
            desired_neg_size = min(desired_pos_size, desired_neg_size)
            logger.print(f"Actual two_word pos_size: {desired_pos_size} and neg_size: {desired_neg_size}")

            # Randomly sample from the pool
            pos_twoword_chosen = random.Random(42).sample(pos_twoword_entries, desired_pos_size)
            neg_twoword_chosen = random.Random(42).sample(neg_twoword_entries, desired_neg_size)

            # Add two-word data into our final positives/negatives
            pos_samples.extend(pos_twoword_chosen)
            final_neg.extend(neg_twoword_chosen)
            logger.print(f"About to train probe with pos_twoword_chosen: {len(pos_twoword_chosen)} and neg_twoword_chosen: {len(neg_twoword_chosen)}")


        # Train the probe
        logger.print(f"About to train probe with a total of {len(pos_samples)} positives and {len(final_neg)} negatives...")
        metric = train_probe(
            model,
            pos_samples,
            final_neg,
            logger,
            target_layers=target_layers,
            probe_type=probe_type,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        word_to_probe[probed_word] = metric

    return word_to_probe

def plot_layer_norms(
    step_to_layer_norm_dontthink_regular,
    step_to_layer_norm_regular,
    step_to_layer_norm_twoword_dontthink_match,
    step_to_layer_norm_twoword_dontthink_mismatch,
    step_to_layer_norm_benign_dontthink_mismatch,
    step_to_layer_norm_dontthink_mismatch
):
    """Plots rolling average layer norms for all tracked categories."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime

    window = 5

    # Convert dictionaries to pandas Series and compute rolling averages
    norm_dontthink_reg = pd.Series(step_to_layer_norm_dontthink_regular).rolling(window=window, min_periods=1).mean()
    norm_reg = pd.Series(step_to_layer_norm_regular).rolling(window=window, min_periods=1).mean()
    norm_tw_match = pd.Series(step_to_layer_norm_twoword_dontthink_match).rolling(window=window, min_periods=1).mean()
    norm_tw_mismatch = pd.Series(step_to_layer_norm_twoword_dontthink_mismatch).rolling(window=window, min_periods=1).mean()
    norm_benign = pd.Series(step_to_layer_norm_benign_dontthink_mismatch).rolling(window=window, min_periods=1).mean()
    norm_mismatch = pd.Series(step_to_layer_norm_dontthink_mismatch).rolling(window=window, min_periods=1).mean()

    # Create the plot
    plt.figure(figsize=(12, 6))

    plt.plot(norm_dontthink_reg.index, norm_dontthink_reg.values, 'b-', linewidth=2,
             label='Clued - regular')
    plt.plot(norm_reg.index, norm_reg.values, 'r-', linewidth=2,
             label='Regular')
    plt.plot(norm_tw_match.index, norm_tw_match.values, 'm-', linewidth=2,
             label='Two-word clued match')
    plt.plot(norm_tw_mismatch.index, norm_tw_mismatch.values, color='orange', linewidth=2,
             label='Two-word clued mismatch')
    plt.plot(norm_benign.index, norm_benign.values, 'g-', linewidth=2,
             label='Benign clued mismatch')
    plt.plot(norm_mismatch.index, norm_mismatch.values, 'c-', linewidth=2,
             label='Clued mismatch')

    plt.title('Layer Norms Over Training Steps (Rolling Avg, window=5)', fontsize=14)
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Layer Norm', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    plt.tight_layout()

    # Save the figure with timestamp
    save_path = f'cond_training_layer_norms_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
    plt.savefig(save_path)
    plt.show()

    return plt.gcf() 
