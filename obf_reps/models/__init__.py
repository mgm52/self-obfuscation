from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int64
from torch import Tensor, nn
from torch.nn import Embedding

ModelBaseT = TypeVar("ModelBaseT", bound="ModelBase")


class ParamsBase(ABC):

    def reinit(self):
        """Reinit params."""
        ...


class GCGParams(ParamsBase):
    def __init__(self, init_ids: Float[Tensor, "1 num_toks"], embedding: Embedding):

        optim_ids_onehot: Float[Tensor, "1 num_toks vocab_size"] = F.one_hot(
            init_ids, num_classes=embedding.num_embeddings
        )

        self.params = optim_ids_onehot.to(
            dtype=embedding.weight.dtype, device=embedding.weight.device
        ).requires_grad_()
        self.init_params = self.params.clone().detach()

    def reinit(self):
        self.params = self.init_params.clone().detach().requires_grad_()

    @property
    def optim_ids(self) -> Float[Tensor, "1 seq_len"]:
        return torch.argmax(self.params, dim=-1)


Params = GCGParams


@dataclass
class ForwardReturn:
    target_ids: Int64[Tensor, "b_size target_len"]
    target_logits: Float[Tensor, "b_size target_len vocab_size"]
    target_reps: Float[Tensor, "b_size layers target_len hidden_size"]
    input_logits: Float[Tensor, "b_size input_len vocab_size"]
    input_reps: Float[Tensor, "b_size input_len hidden_size"]
    loss_mask: Optional[Bool[Tensor, "b_size seq_len"]] = None
    loss: Optional[Float[Tensor, "b_size"]] = None
    # These are provided for debugging
    input_embeds: Optional[Float[Tensor, "b_size input_len"]] = None
    input_ids: Optional[Int64[Tensor, "b_size input_len"]] = None
    raw_attn_mask: Optional[Bool[Tensor, "b_size input_len+target_len"]] = None
    raw_logits: Optional[Tensor] = None
    past_key_values: Optional[Tensor] = None
    position_ids: Optional[Tensor] = None


@dataclass
class GenReturn:
    input_text: List[str]
    gen_text: List[str]
    input_ids: Int64[Tensor, "b_size input_len"]
    gen_ids: Int64[Tensor, "b_size gen_len"]
    input_reps: Float[Tensor, "b_size layers input_len hidden_size"]
    gen_reps: Float[Tensor, "b_size layers gen_len hidden_size"]
    gen_mask: Bool[Tensor, "b_size gen_len"]


@dataclass
class ModelConfig:
    model_dtype: torch.dtype = torch.half
    prompt_init: Optional[str] = None
    requires_grad: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelBase(ABC, nn.Module):

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def tokenize(
        self,
        text: Union[str, List[str]],
        add_chat_template: bool,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
        add_special_tokens: bool = True,
    ) -> Tuple[Int64[Tensor, "b_size seq_len"], Bool[Tensor, "b_size seq_len"]]:
        """Tokenizes text into input_ids and attention_mask."""
        ...

    def to_string(
        self,
        input_ids: Int64[Tensor, "b_size seq_len"],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Converts input_ids to string."""

        return self.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True,
        )

    @abstractmethod
    def forward_from_embeds(
        self,
        input_embeds: Float[Tensor, "b_size seq_len hidden_size"],
        input_attn_mask: Bool[Tensor, "b_size seq_len"],
        target_ids: Optional[Int64[Tensor, "b_size seq_len"]] = None,
        target_attn_mask: Optional[Bool[Tensor, "b_size seq_len"]] = None,
        past_key_values: Optional[Tuple[Tensor, ...]] = None,
        layers_to_probe: Optional[List[int]] = None,
    ) -> ForwardReturn:
        """Return per token logits and reps from forward pass on embeds."""

    @abstractmethod
    def forward_from_ids(
        self,
        input_ids: Int64[Tensor, "b_size seq_len"],
        input_attn_mask: Bool[Tensor, "b_size seq_len"],
        target_ids: Optional[Int64[Tensor, "b_size seq_len"]] = None,
        target_attn_mask: Optional[Bool[Tensor, "b_size seq_len"]] = None,
        use_tunable_params: bool = True,
        layers_to_probe: Optional[List[int]] = None,
    ) -> ForwardReturn:
        """Return per token logits and reps from forward pass on ids."""
        ...

    @abstractmethod
    def forward_from_string_and_ids(
        self,
        input_text: Union[str, List[str]],
        target_ids: Int64[Tensor, "b_size seq_len"],
        target_attn_mask: Optional[Bool[Tensor, "b_size seq_len"]] = None,
        add_chat_template: bool = True,
        use_tunable_params: bool = True,
        debug: bool = False,
        layers_to_probe: Optional[List[int]] = None,
    ) -> ForwardReturn:
        """Return per token logits and reps from forward pass on text and ids."""
        ...

    @abstractmethod
    def forward_from_string(
        self,
        input_text: Union[str, List[str]],
        target_text: Optional[Union[str, List[str]]] = None,
        add_chat_template: bool = True,
        use_tunable_params: bool = True,
        layers_to_probe: Optional[List[int]] = None,
    ) -> ForwardReturn:
        """Return per token logits and reps from forward pass on text."""
        ...

    @abstractmethod
    def generate_from_ids(
        self,
        input_ids: Int64[Tensor, "b_size seq_len"],
        input_attn_mask: Bool[Tensor, "b_size seq_len"],
        max_new_tokens: int = 20,
        use_tunable_params: bool = True,
        **generate_kwargs
    ) -> GenReturn:
        """Generate text from input IDs using the model's generate function."""
        ...

    @abstractmethod
    def generate_from_string(
        self,
        input_text: Union[str, List[str]],
        max_new_tokens: int = 20,
        use_tunable_params: bool = True,
        add_chat_template: bool = True,
        **generate_kwargs
    ) -> GenReturn:
        """Generate text from input string(s) using the model's generate function.

        Args:
            input_text: input text. If using tunable prompt, this will
                be added after the input text.
            max_new_tokens: maximum number of new tokens to generate
            **generate_kwargs: additional keyword arguments for the generate function
        """
        ...

    @abstractmethod
    def init_tunable_params(self) -> Params:
        """Initialize tunable parameters and return them."""
        ...

    @classmethod
    @abstractmethod
    def load_model(
        cls: Type[ModelBaseT],
        path: Path,
        config: ModelConfig = ModelConfig(),
    ) -> ModelBaseT:
        """Load model
        Args:
                path: path to saved model
                model_dtype: dtype of model
                requires_grad: whether to compute gradients for model params
        """
