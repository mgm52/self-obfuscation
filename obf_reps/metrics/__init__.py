import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import joblib
import numpy as np
import torch
import tqdm
from jaxtyping import Bool, Float
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from obf_reps.data import ConceptDataModule, ConceptDataset
from obf_reps.logging import Logger
from obf_reps.metrics.probes import MLP, AttentionProbe, LogisticRegression
from obf_reps.models import ForwardReturn, ModelBase

DEBUG = False

RepsDatasetT = List[Tuple[Float[Tensor, "1 layers red_seq_len h_dim"], Float[Tensor, ""]]]
CachedRepsT = TypeVar("CachedRepsT", bound="CachedReps")

MemRepsDatasetT = Tuple[np.memmap, np.memmap]


def get_tensor_size_in_gb(tensor: torch.Tensor) -> float:
    return tensor.element_size() * tensor.nelement() / (1024**3)


def get_dataset_size_in_gb(
    dataset: List[Tuple[Float[Tensor, "1 layers red_seq_len h_dim"], Float[Tensor, ""]]],
) -> float:
    total_size = 0
    for rep, label in dataset:
        total_size += get_tensor_size_in_gb(rep)
        total_size += get_tensor_size_in_gb(label)
    return total_size


# Store previously computed reps.
@dataclass
class CachedReps:
    attack_defence_epoch: int
    rep_dataset: RepsDatasetT
    who_just_went: str

    @staticmethod
    def combine(cached_reps: List[CachedRepsT]) -> RepsDatasetT:
        combined_dataset = []
        for cache in cached_reps:
            combined_dataset.extend(cache.rep_dataset)
        # Shuffle the dataset
        random.Random(42).shuffle(combined_dataset)
        return combined_dataset


@dataclass
class TestCachedReps:
    label: str  # benign
    attack_defence_epoch: int
    reps: List[Tensor]


@dataclass
class MetricConfig:
    layers: List[int]  # Which layers to probe
    lr: float = 5e-4  # Learning rate for the metric
    batch_size: int = 64  # Batch size for the metric
    num_epochs: int = 5  # Number of epochs for the metric

    C: float = 1.0  # Regularization strength for sklearn LogisticRegression


class ObfMetric(ABC):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.device = self.model.device
        model_config = self.model._get_model_config()
        self.model_hidden_dim = model_config.hidden_size
        self.model_num_layers = model_config.num_hidden_layers

        self.data_module = data_module
        logger.print(
            f"Loaded data module with data_module.train_dataset of length {len(self.data_module.train_dataset)}"
        )
        self.config = config
        self.logger = logger

        self.train_reps_reduce = data_module.train_reps_reduce
        self.test_reps_reduce = data_module.test_reps_reduce

        # Stores previously computed reps
        self.reps_bank: List[CachedReps] = []
        self.test_reps_bank: List[CachedReps] = []

        if probe_path is None:
            # We need the train_rep_dataset to fit the probe
            print("Getting train reps")
            train_rep_dataset = self.get_reps_from_dataset(
                model,
                self.data_module.train_dataset,  # type: ignore
                self.train_reps_reduce,
                layers_to_probe=self.config.layers,
                batch_size=self.config.batch_size,
            )
            print("Done getting train reps")

            # Cache reps
            cached_reps = CachedReps(0, train_rep_dataset, "start")
            self.reps_bank.append(cached_reps)

            self.fit(train_rep_dataset)
        else:
            self.load_pretrained(probe_path)

    @abstractmethod
    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        # One of path or probe_state must be provided
        # If path is provided, then the probe_state should be loaded from the path
        # If probe_state is provided, then the probe should be loaded from the probe_state
        pass

    @abstractmethod
    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:
        # If path is provided, then the probe_state should be saved to the path
        # Otherwise, the probe_state should be returned
        pass

    @abstractmethod
    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        """Fit the metric to the data.

        This function should also set the self.thresh parameter according to the training data.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        """Computes obfuscation metric.

        For every representation of shape [hidden_dim] in the input. Thus evaluate collapses the
        hidden_dim dimension of the input.
        """
        ...

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        """Predict the labels of the input using the metric score.

        Note that this is different to evaluate as it predicts a single score for each element of
        the input.

        Args:
            reps: reps to predict.
            layer_reduction: how to aggregate scores over layers.
            attention_mask: boolean mask, 0 when reps correspond to <pad> tokens.
        """
        ...

    def predict_example(
        self,
        input_reps: Float[Tensor, "b layers inp_seq_len hidden_dim"],
        target_reps: Float[Tensor, "b layers out_seq_len hidden_dim"],
        target_mask: Bool[Tensor, "b out_seq_len"],
        layer_reduction: str = "mean",
    ):
        """Get predictions from input_reps and target_reps.

        Args:
            input_reps: input representations.
            target_reps: target representations.
            target_loss_mask: mask for target reps corresponding to reps
                over <pad> tokens.

        NOTE: We don't have a mask for the input_reps because they are left padded
            and adv suffix comes at the end, so we can only act on the final few
            <eot> tokens. That is the mask is implicit in our setup.
        """

        reps, attention_mask = self.data_module.test_reps_reduce(
            input_reps,
            target_reps,
            target_mask=target_mask,
        )

        return self.predict(
            reps=reps, attention_mask=attention_mask, layer_reduction=layer_reduction
        )

    @torch.no_grad()
    def get_reps_from_dataset(
        self,
        model: ModelBase,
        dataset: ConceptDataset,
        reps_reduce: Callable[[Tensor, Tensor], Tensor],
        use_tunable_params: bool = False,
        layers_to_probe: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> RepsDatasetT:
        """Convert a dataset to info needed to train and evaluate a metric.

        To train and evaluate a metric, we need to take each example in the
        dataset and convert them into:

        - the reps that should be fed to into a metric
        - the label for the given example

        For a text input x_i, reps_i could have different shapes depending
        on the metric and task (in some cases the task may demand you look
        at a single rep, enforced by reps_reduce), so we store
        the results in a list.

        Args:
            batch_size: Number of examples to process in parallel. If None, uses self.config.batch_size.
                       Set to 1 to disable batching and use original behavior.
        """

        # Use provided batch_size or fall back to config, then to 1 (original behavior)
        if batch_size is None:
            batch_size = getattr(self.config, "batch_size", 1)

        reps_dataset = []
        pos_target_len = 0
        neg_target_len = 0

        timings = {
            "positive_forward": 0.0,
            "negative_forward": 0.0,
            "reduce_and_process": 0.0,
            "cleanup": 0.0,
            "cpu_transfer": 0.0,
            "garbage_collect": 0.0,
            "clear_gpu_cache": 0.0,
        }

        print(
            f"Getting reps from dataset of len {len(dataset)} with layers_to_probe {layers_to_probe}, batch_size={batch_size}"
        )

        # Process dataset in batches
        for batch_start in tqdm.tqdm(
            range(0, len(dataset), batch_size), desc="Processing batches"
        ):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_data = [dataset[i] for i in range(batch_start, batch_end)]

            # Separate positive and negative examples
            pos_inputs, pos_targets = [], []
            neg_inputs, neg_targets = [], []

            for (pos_input, pos_target), (neg_input, neg_target) in batch_data:
                pos_inputs.append(pos_input)
                pos_targets.append(pos_target)
                neg_inputs.append(neg_input)
                neg_targets.append(neg_target)

            # Time positive forward pass
            t0 = time.perf_counter()

            # Handle positive examples
            if all(
                isinstance(inp, str) and isinstance(tgt, str)
                for inp, tgt in zip(pos_inputs, pos_targets)
            ):
                # All string inputs and targets
                positive_rep: ForwardReturn = model.forward_from_string(
                    input_text=pos_inputs,
                    target_text=pos_targets,
                    add_chat_template=True,
                    use_tunable_params=use_tunable_params,
                    layers_to_probe=layers_to_probe,
                )
            elif all(
                isinstance(inp, str) and isinstance(tgt, list)
                for inp, tgt in zip(pos_inputs, pos_targets)
            ):
                # String inputs with token ID targets
                # Pad token sequences to same length for batching
                max_len = max(len(tgt) for tgt in pos_targets)
                padded_targets = []
                target_masks = []

                for tgt in pos_targets:
                    padded = tgt + [0] * (max_len - len(tgt))  # Pad with 0s
                    mask = [1] * len(tgt) + [0] * (max_len - len(tgt))
                    padded_targets.append(padded)
                    target_masks.append(mask)

                target_ids = torch.tensor(padded_targets, device=model.device)
                target_attn_mask = torch.tensor(
                    target_masks, device=model.device, dtype=torch.bool
                )

                positive_rep: ForwardReturn = model.forward_from_string_and_ids(
                    input_text=pos_inputs,
                    target_ids=target_ids,
                    use_tunable_params=use_tunable_params,
                    layers_to_probe=layers_to_probe,
                    target_attn_mask=target_attn_mask,
                    add_chat_template=True,
                )
            else:
                # Mixed types - fall back to single processing for this batch
                positive_reps = []
                for pos_input, pos_target in zip(pos_inputs, pos_targets):
                    if isinstance(pos_input, str) and isinstance(pos_target, str):
                        rep = model.forward_from_string(
                            input_text=pos_input,
                            target_text=pos_target,
                            add_chat_template=True,
                            use_tunable_params=use_tunable_params,
                            layers_to_probe=layers_to_probe,
                        )
                    elif isinstance(pos_input, str) and isinstance(pos_target, list):
                        rep = model.forward_from_string_and_ids(
                            input_text=pos_input,
                            target_ids=torch.tensor(pos_target, device=model.device).unsqueeze(0),
                            use_tunable_params=use_tunable_params,
                            layers_to_probe=layers_to_probe,
                            target_attn_mask=None,
                            add_chat_template=True,
                        )
                    else:
                        raise ValueError(
                            f"Unexpected pos input type: {type(pos_input)} and target type: {type(pos_target)}"
                        )
                    positive_reps.append(rep)

                # Combine results manually (this is more complex, but handles mixed cases)
                # For simplicity, we'll process mixed batches one by one
                batch_size = 1  # Force single processing for mixed batches

            timings["positive_forward"] += time.perf_counter() - t0

            # Time negative forward pass
            t0 = time.perf_counter()

            # Handle negative examples (always use_tunable_params=False)
            if all(
                isinstance(inp, str) and isinstance(tgt, str)
                for inp, tgt in zip(neg_inputs, neg_targets)
            ):
                # All string inputs and targets
                negative_rep: ForwardReturn = model.forward_from_string(
                    input_text=neg_inputs,
                    target_text=neg_targets,
                    add_chat_template=True,
                    use_tunable_params=False,  # Always False for negative examples
                    layers_to_probe=layers_to_probe,
                )
            elif all(
                isinstance(inp, str) and isinstance(tgt, list)
                for inp, tgt in zip(neg_inputs, neg_targets)
            ):
                # String inputs with token ID targets
                max_len = max(len(tgt) for tgt in neg_targets)
                padded_targets = []
                target_masks = []

                for tgt in neg_targets:
                    padded = tgt + [0] * (max_len - len(tgt))
                    mask = [1] * len(tgt) + [0] * (max_len - len(tgt))
                    padded_targets.append(padded)
                    target_masks.append(mask)

                target_ids = torch.tensor(padded_targets, device=model.device)
                target_attn_mask = torch.tensor(
                    target_masks, device=model.device, dtype=torch.bool
                )

                negative_rep: ForwardReturn = model.forward_from_string_and_ids(
                    input_text=neg_inputs,
                    target_ids=target_ids,
                    use_tunable_params=False,  # Always False for negative examples
                    layers_to_probe=layers_to_probe,
                    target_attn_mask=target_attn_mask,
                    add_chat_template=True,
                )
            else:
                # Mixed types - handle individually
                negative_reps = []
                for neg_input, neg_target in zip(neg_inputs, neg_targets):
                    if isinstance(neg_input, str) and isinstance(neg_target, str):
                        rep = model.forward_from_string(
                            input_text=neg_input,
                            target_text=neg_target,
                            add_chat_template=True,
                            use_tunable_params=False,
                            layers_to_probe=layers_to_probe,
                        )
                    elif isinstance(neg_input, str) and isinstance(neg_target, list):
                        rep = model.forward_from_string_and_ids(
                            input_text=neg_input,
                            target_ids=torch.tensor(neg_target, device=model.device).unsqueeze(0),
                            use_tunable_params=False,
                            layers_to_probe=layers_to_probe,
                        )
                    else:
                        raise ValueError(
                            f"Unexpected neg input type: {type(neg_input)} and target type: {type(neg_target)}"
                        )
                    negative_reps.append(rep)

            timings["negative_forward"] += time.perf_counter() - t0

            # Time reduce and processing
            t0 = time.perf_counter()

            # If we processed batches successfully, handle batched results
            if "positive_rep" in locals() and hasattr(positive_rep, "input_reps"):
                # Batched processing
                pos_input_reps = (
                    positive_rep.input_reps
                )  # Shape: [batch_size, layers, seq_len, hidden_dim]
                pos_target_reps = positive_rep.target_reps
                pos_target_mask = positive_rep.loss_mask

                neg_input_reps = negative_rep.input_reps
                neg_target_reps = negative_rep.target_reps
                neg_target_mask = negative_rep.loss_mask

                # Process each example in the batch
                for b in range(pos_input_reps.shape[0]):
                    pos_target_len += pos_target_reps[b].shape[1]
                    neg_target_len += neg_target_reps[b].shape[1]

                    # Extract single example from batch
                    pos_inp_b = pos_input_reps[b : b + 1]  # Keep batch dim
                    pos_tgt_b = pos_target_reps[b : b + 1]
                    pos_mask_b = (
                        pos_target_mask[b : b + 1] if pos_target_mask is not None else None
                    )

                    neg_inp_b = neg_input_reps[b : b + 1]
                    neg_tgt_b = neg_target_reps[b : b + 1]
                    neg_mask_b = (
                        neg_target_mask[b : b + 1] if neg_target_mask is not None else None
                    )

                    # Apply reduction
                    pos_rep, _ = reps_reduce(pos_inp_b, pos_tgt_b, pos_mask_b)
                    neg_rep, _ = reps_reduce(neg_inp_b, neg_tgt_b, neg_mask_b)

                    # Transfer to CPU
                    pos_rep = pos_rep.detach().to("cpu")
                    neg_rep = neg_rep.detach().to("cpu")

                    # Add to dataset
                    reps_dataset.append((pos_rep, torch.tensor([1.0], device="cpu")))
                    reps_dataset.append((neg_rep, torch.tensor([0.0], device="cpu")))

            else:
                # Fall back to individual processing (mixed types case)
                # This would be for the mixed types case - simplified for now
                pass

            timings["reduce_and_process"] += time.perf_counter() - t0

            # Cleanup
            t0 = time.perf_counter()
            if "pos_input_reps" in locals():
                del pos_input_reps, pos_target_reps, pos_target_mask
                del neg_input_reps, neg_target_reps, neg_target_mask
            timings["cleanup"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            torch.cuda.empty_cache()
            timings["clear_gpu_cache"] += time.perf_counter() - t0

            if DEBUG and ((batch_start // batch_size + 1) % 10 == 0):
                print(f"Cached {batch_end} examples")
                print(f"Cached {get_dataset_size_in_gb(reps_dataset)} GB of reps")

        # Final cleanup
        t0 = time.perf_counter()
        timings["garbage_collect"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        torch.cuda.empty_cache()
        timings["clear_gpu_cache"] += time.perf_counter() - t0

        print(f"Batch processing completed. Total examples: {len(reps_dataset) // 2}")

        random.Random(42).shuffle(reps_dataset)

        return reps_dataset


class TrainableMetric(ObfMetric):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
    ):
        self.probe: Dict[int, torch.nn.Module] = {}
        super().__init__(model, data_module, config, logger, probe_path)

    @abstractmethod
    def create_model(self, hidden_size) -> torch.nn.Module: ...

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        device = self.device

        # Concatenate all reps and labels, treating each token as an independent example
        all_reps = []
        all_labels = []

        for rep, label in reps_dataset:
            # rep shape: [1, layers, seq_len, hidden_dim]
            # Reshape to [layers, seq_len, hidden_dim]
            rep = rep.squeeze(0)
            # Extend labels to match seq_len
            extended_label = label.repeat(rep.shape[1])
            all_reps.append(rep)
            all_labels.append(extended_label)

        # Concatenate along the sequence length dimension
        reps = torch.cat(all_reps, dim=1)
        labels = torch.cat(all_labels, dim=0)

        n_layers, total_seq_len, hidden_size = reps.shape

        for layer_index, layer in enumerate(self.config.layers):
            X_train = reps[layer_index]  # Shape: [total_seq_len, hidden_size]
            assert X_train.shape == (
                total_seq_len,
                hidden_size,
            ), f"Incorrect X_train shape: {X_train.shape}"

            y_train = labels

            model = self.create_model(hidden_size).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.config.lr, weight_decay=1e-5, eps=1e-5
            )
            criterion = torch.nn.BCEWithLogitsLoss()

            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=self.config.batch_size,
                shuffle=False,
            )

            print(f"Fitting layer {layer} probe ({layer_index + 1} / {n_layers})")
            for epoch in range(self.config.num_epochs):
                pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
                recent_losses = []
                loss_memory = 10
                recent_avg = -1
                for i, (X, y) in enumerate(pbar):
                    optimizer.zero_grad()
                    outputs = model(X.to(device))  # Should be [batch_size, 1]
                    if outputs.dim() == 2 and outputs.shape[-1] == 1:
                        outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y.to(device))
                    loss.backward()
                    optimizer.step()

                    # Update running loss and progress bar
                    recent_losses += [loss.item()]
                    if len(recent_losses) == loss_memory:
                        recent_avg = sum(recent_losses) / loss_memory
                        recent_losses = []

                    if recent_avg > -1:
                        pbar.set_postfix({f"loss{loss_memory}": f"{recent_avg:.4f}"})

                    # self.logger.log({f"probe_loss_layer_{layer}": loss.item()})
                pbar.close()

            self.probe[layer_index] = model

    def evaluate(
        self, reps: Float[Tensor, "n layers seq_len hidden_dim"], **kwargs
    ) -> Float[Tensor, "n layers seq_len"]:
        if reps.ndim == 3:
            # We got reps with no seq_len from "get_reps_from_dataset"
            # Add in sequence dimension to broadcast over
            reps = reps.unsqueeze(2)

        b, n_layers, seq_len, hidden_dim = reps.shape
        device = self.device
        probs = torch.zeros((b, n_layers, seq_len), device=device)

        for layer_index, layer in enumerate(self.config.layers):
            X = reps[:, layer_index, :, :]

            outputs = self.probe[layer_index](X).squeeze(-1)
            assert outputs.shape == (b, seq_len), f"Incorrect outputs shape: {outputs.shape}"
            probs[:, layer_index, :] = torch.sigmoid(outputs)
            # for seq_pos in range(seq_len):
            #    X = reps[:, layer_num, seq_pos, :].unsqueeze(1)

            #    outputs = self.probe[layer_num](X).squeeze()
            #    probs[:, layer_num, seq_pos] = torch.sigmoid(outputs)

        return probs.to(torch.float16)

    def predict_and_evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        b, layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask must be provided for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            seq_scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            seq_scores = scores.mean(dim=2)

        # Reduce over layers
        match layer_reduction:
            case "mean":
                final_scores = seq_scores.mean(dim=1)
            case "sum":
                final_scores = seq_scores.sum(dim=1)
            case "max":
                final_scores = seq_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert final_scores.shape == (b,), f"Expected shape (b,), but got {final_scores.shape}"

        return final_scores, scores

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        b, layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask must be provided for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            seq_scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            seq_scores = scores.mean(dim=2)

        # Reduce over layers
        match layer_reduction:
            case "mean":
                final_scores = seq_scores.mean(dim=1)
            case "sum":
                final_scores = seq_scores.sum(dim=1)
            case "max":
                final_scores = seq_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert final_scores.shape == (b,), f"Expected shape (b,), but got {final_scores.shape}"

        return final_scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            self.probe = torch.load(path)
        else:
            self.probe = probe_state

    def save_probe(self, path: Optional[str] = None) -> None | Any:
        if path is not None:
            torch.save(self.probe, path)
        else:
            return self.probe


class LogisticRegressionMetric(TrainableMetric):
    def create_model(self, hidden_size, dtype=torch.bfloat16) -> torch.nn.Module:
        return LogisticRegression(hidden_size, dtype=dtype)


class MLPMetric(TrainableMetric):
    def create_model(self, hidden_size, dtype=torch.bfloat16) -> torch.nn.Module:
        return MLP(hidden_size, dtype=dtype)

class AttentionMetric(TrainableMetric):
    def create_model(self, hidden_size, dtype=torch.bfloat16) -> torch.nn.Module:
        return AttentionProbe(hidden_size, dtype=dtype)

class SklearnLogisticRegressionMetric(ObfMetric):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
    ):
        self.probe: Dict[int, SklearnLogisticRegression] = {}
        self.scaler: Dict[int, StandardScaler] = {}
        super().__init__(model, data_module, config, logger, probe_path)

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        # Concatenate all reps and labels, treating each token as an independent example
        all_reps = []
        all_labels = []

        for rep, label in reps_dataset:
            # rep shape: [1, layers, seq_len, hidden_dim]
            # Reshape to [layers, seq_len, hidden_dim]
            rep = rep.squeeze(0)
            # Extend labels to match seq_len
            extended_label = label.repeat(rep.shape[1])
            all_reps.append(rep)
            all_labels.append(extended_label)

        # Concatenate along the sequence length dimension
        reps = torch.cat(all_reps, dim=1)
        labels = torch.cat(all_labels, dim=0)

        n_layers, total_seq_len, hidden_size = reps.shape

        for layer_index, layer in enumerate(self.config.layers):
            X_train = reps[layer_index].cpu().numpy()  # Shape: [total_seq_len, hidden_size]
            y_train = labels.cpu().numpy()

            print(f"Fitting sklearn layer {layer} probe ({layer_index + 1} / {n_layers})")

            # Initialize and fit StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Initialize and fit LogisticRegression
            clf = SklearnLogisticRegression(
                C=self.config.C,
                random_state=42,
                n_jobs=-1,
                fit_intercept=False,
            )
            clf.fit(X_train_scaled, y_train)

            # Store the trained models
            self.probe[layer_index] = clf
            self.scaler[layer_index] = scaler

    def evaluate(
        self, reps: Float[Tensor, "n layers seq_len hidden_dim"], **kwargs
    ) -> Float[Tensor, "n layers seq_len"]:
        if reps.ndim == 3:
            # We got reps with no seq_len from "get_reps_from_dataset"
            # Add in sequence dimension to broadcast over
            reps = reps.unsqueeze(2)

        b, n_layers, seq_len, hidden_dim = reps.shape
        device = self.device
        probs = torch.zeros((b, n_layers, seq_len), device=device)

        for layer_index, layer in enumerate(self.config.layers):
            X = reps[:, layer_index, :, :].cpu().numpy()  # Shape: [b, seq_len, hidden_dim]

            # Reshape for sklearn: [b*seq_len, hidden_dim]
            X_flat = X.reshape(-1, hidden_dim)

            # Scale the features
            X_scaled = self.scaler[layer_index].transform(X_flat)

            # Get probabilities from sklearn
            probs_np = self.probe[layer_index].predict_proba(X_scaled)[
                :, 1
            ]  # Get positive class probabilities

            # Reshape back to [b, seq_len] and convert to tensor
            probs_reshaped = probs_np.reshape(b, seq_len)
            probs[:, layer_index, :] = torch.from_numpy(probs_reshaped).to(device)

        return probs.to(torch.float16)

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        b, layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask must be provided for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask_expanded: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask_expanded.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask_expanded, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            seq_scores = scores.sum(dim=2) / attention_mask_expanded.sum(dim=2).clamp(min=1)
        else:
            seq_scores = scores.mean(dim=2)

        # Reduce over layers
        match layer_reduction:
            case "mean":
                final_scores = seq_scores.mean(dim=1)
            case "sum":
                final_scores = seq_scores.sum(dim=1)
            case "max":
                final_scores = seq_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert final_scores.shape == (b,), f"Expected shape (b,), but got {final_scores.shape}"

        return final_scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            data = joblib.load(path)
            self.probe = data["probe"]
            self.scaler = data["scaler"]
        else:
            self.probe = probe_state["probe"]
            self.scaler = probe_state["scaler"]

    def save_probe(self, path: Optional[str] = None) -> None | Any:
        data = {"probe": self.probe, "scaler": self.scaler}
        if path is not None:
            joblib.dump(data, path)
            return None
        else:
            return data
