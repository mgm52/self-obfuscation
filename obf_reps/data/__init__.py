from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.data import DataLoader

ConceptDataset = List[Tuple[Tuple[str | None, str | None], Tuple[str | None, str | None]]]
ObfusDataset = List[Tuple[str, str, str]]

def train_only_split_dataset(dataset: Union[ConceptDataset, ObfusDataset]) -> Tuple[
    Union[ConceptDataset, ObfusDataset],
    Union[ConceptDataset, ObfusDataset],
    Union[ConceptDataset, ObfusDataset],
]:
    return dataset, [], []

def split_dataset(dataset: Union[ConceptDataset, ObfusDataset]) -> Tuple[
    Union[ConceptDataset, ObfusDataset],
    Union[ConceptDataset, ObfusDataset],
    Union[ConceptDataset, ObfusDataset],
]:
    dataset_size = len(dataset)

    test_size = int(0.1 * dataset_size)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - test_size - val_size

    if test_size == 0:
        # Probably a singleton dataset
        train_dataset = dataset
        val_dataset = dataset
        test_dataset = dataset
    else:
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size : train_size + val_size]
        test_dataset = dataset[train_size + val_size :]

    return train_dataset, val_dataset, test_dataset


class DataModule(ABC):
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path

        self.train_dataset, self.val_dataset, self.test_dataset = self.load_dataset()

    @abstractmethod
    def load_dataset(
        self,
    ) -> Tuple[
        Union[ConceptDataset, ObfusDataset],
        Union[ConceptDataset, ObfusDataset],
        Union[ConceptDataset, ObfusDataset],
    ]: ...


class ConceptDataModule(DataModule, ABC):
    @abstractmethod
    def load_dataset(self) -> ConceptDataset: ...

    def __init__(
        self,
        batch_size: int,
        dataset_path: Optional[Path] = None,
        **kwargs,
    ):
        self.batch_size = batch_size
        super().__init__(dataset_path)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)  # type: ignore

    def train_reps_reduce(
        self,
        input_reps: Float[Tensor, "b layers inp_seq_len h_dim"],
        target_reps: Float[Tensor, "b layers out_seq_len h_dim"],
        target_mask: Optional[Bool[Tensor, "b out_seq_len"]] = None,
    ) -> Tuple[Float[Tensor, "b layers red_seq_len h_dim"], Bool[Tensor, "b red_seq_len"]]:
        """For a given concept, it describes how the representations should be reduced for
        training.

        Returns:
            reps: single reps tensor which is the input to the prediction problem.
            reps_mask: boolean mask for where the reps correspond to <pad> and so should be ignored.
        """

        if target_mask is None:
            assert input_reps.shape[0] == target_reps.shape[0] == 1

            b, out_seq_len = target_reps.shape[0], target_reps.shape[1]
            target_mask = torch.ones((b, out_seq_len), dtype=torch.bool)

        return target_reps, target_mask

    def test_reps_reduce(
        self,
        input_reps: Float[Tensor, "b layers inp_seq_len h_dim"],
        target_reps: Float[Tensor, "b layers out_seq_len h_dim"],
        target_mask: Optional[Bool[Tensor, "b out_seq_len"]] = None,
    ) -> Tuple[Float[Tensor, "b layers red_seq_len h_dim"], Bool[Tensor, "b red_seq_len"]]:
        """Describes how model reps should be reduced for testing a probe using this dataset."""

        return self.train_reps_reduce(input_reps, target_reps, target_mask)
