"""Model-agnostic tokenizer utilities for handling special tokens."""

from typing import List

# Family-level special token configurations, keyed by HF config.json model_type.
# Auto-detected from checkpoint config.json, or resolved from legacy model names
# via _get_config_for_model().
SPECIAL_TOKEN_CONFIGS = {
    "gemma2": {
        "special_chars_to_remove": ["<eos>", "<end_of_turn>", "<pad>"],
        "chat_template_tokens": [
            "<bos><start_of_turn>user\n",
            "<end_of_turn>\n<start_of_turn>model\n",
            "<start_of_turn>",
            "<end_of_turn>",
        ]
    },
    "qwen2": {
        "special_chars_to_remove": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        "chat_template_tokens": [
            "<|im_start|>system\n",
            "<|im_start|>user\n",
            "<|im_start|>assistant\n",
            "<|im_end|>\n",
            "<|im_start|>",
            "<|im_end|>",
        ]
    },
    "llama": {
        "special_chars_to_remove": ["<|eot_id|>", "<|begin_of_text|>"],
        "chat_template_tokens": [
            "<|begin_of_text|>",
            "<|start_header_id|>system<|end_header_id|>\n\n",
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
        ]
    },
}


def _get_config_for_model(model_name: str) -> dict:
    """Resolve model name to token config.

    Tries exact match first (for HF model_type values like "gemma2", "qwen2", "llama"),
    then version-aware fuzzy match (for legacy names like "gemma_2_9b_instruct").
    Raises ValueError if no supported family matches.
    """
    # Exact match (handles auto-detected model_type)
    if model_name in SPECIAL_TOKEN_CONFIGS:
        return SPECIAL_TOKEN_CONFIGS[model_name]
    # Version-aware fuzzy match (handles legacy specific model names)
    name = model_name.lower()
    if "gemma_2" in name or "gemma2" in name:
        return SPECIAL_TOKEN_CONFIGS["gemma2"]
    if "qwen" in name:
        return SPECIAL_TOKEN_CONFIGS["qwen2"]
    if "llama" in name:
        return SPECIAL_TOKEN_CONFIGS["llama"]
    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Supported families: {list(SPECIAL_TOKEN_CONFIGS.keys())}. "
        f"Add a new entry to SPECIAL_TOKEN_CONFIGS in utils_tokenizer.py to support this model."
    )


def get_special_chars_for_model(model_name: str) -> List[str]:
    """Get list of special characters to remove for a given model."""
    config = _get_config_for_model(model_name)
    return config.get("special_chars_to_remove", [])

def get_chat_template_tokens_for_model(model_name: str) -> List[str]:
    """Get list of chat template tokens to remove for a given model."""
    config = _get_config_for_model(model_name)
    return config.get("chat_template_tokens", [])

def remove_special_chars(text: str, model_name: str) -> str:
    """Remove model-specific special characters from text."""
    special_chars = get_special_chars_for_model(model_name)
    for char in special_chars:
        text = text.replace(char, "")
    return text.strip()

def remove_chat_template_tokens(text: str, model_name: str) -> str:
    """Remove model-specific chat template tokens from text."""
    chat_tokens = get_chat_template_tokens_for_model(model_name)
    for token in chat_tokens:
        text = text.replace(token, "")
    return text.strip()

def clean_model_output(text: str, model_name: str) -> str:
    """Clean model output by removing all model-specific tokens."""
    text = remove_chat_template_tokens(text, model_name)
    text = remove_special_chars(text, model_name)
    return text
