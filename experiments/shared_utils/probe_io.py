"""
Shared probe checkpoint I/O helpers.

Used by the deception and harmfulness probe evaluation/training scripts.
"""

import json
import os
import random

import torch

from experiments.shared_utils.utils_training import ResponseProbeDataModule
from obf_reps.metrics import MetricConfig


def load_probe_from_checkpoint(probe_dir: str, model, logger):
    """Load a trained probe from checkpoint directory."""
    probe_path = os.path.join(probe_dir, "probe.pkl")
    metadata_path = os.path.join(probe_dir, "metadata.json")

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
            "Please retrain the probe with proper metadata."
        )
    target_layers = metadata["target_layers"]
    learning_rate = metadata.get("learning_rate", 4e-05)
    batch_size = metadata.get("batch_size", 16)
    num_epochs = metadata.get("num_epochs", 1)

    # Create config for the probe
    config = MetricConfig(
        layers=target_layers,
        lr=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # Create dummy data module - we need this for probe initialization but won't use it for evaluation
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
            from obf_reps.metrics import LogisticRegressionMetric
            probe = LogisticRegressionMetric(
                model=model,
                data_module=dummy_data_module,
                config=config,
                logger=logger,
                probe_path=probe_path  # This will use our patched torch.load
            )
        elif probe_type == "mlp":
            from obf_reps.metrics import MLPMetric
            probe = MLPMetric(
                model=model,
                data_module=dummy_data_module,
                config=config,
                logger=logger,
                probe_path=probe_path
            )
        elif probe_type == "sklearn_logistic":
            from obf_reps.metrics import SklearnLogisticRegressionMetric
            probe = SklearnLogisticRegressionMetric(
                model=model,
                data_module=dummy_data_module,
                config=config,
                logger=logger,
                probe_path=probe_path
            )
        elif probe_type == "attention":
            from obf_reps.metrics import AttentionMetric
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


def limit_samples(positive_examples, negative_examples, max_samples=None, seed=42, shuffle=False):
    """Limit the number of samples used for training."""
    if max_samples is None:
        return positive_examples, negative_examples

    max_per_class = min(max_samples // 2, len(positive_examples), len(negative_examples))
    if max_per_class * 2 >= len(positive_examples) + len(negative_examples):
        return positive_examples, negative_examples

    if shuffle:
        pos_limited = random.Random(42).sample(positive_examples, max_per_class)
        neg_limited = random.Random(42).sample(negative_examples, max_per_class)
    else:
        pos_limited = positive_examples[:max_per_class]
        neg_limited = negative_examples[:max_per_class]

    return pos_limited, neg_limited
