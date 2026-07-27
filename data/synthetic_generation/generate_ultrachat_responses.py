"""Generate on-policy responses to UltraChat prompts from a base model.

Produces behaviour-preservation targets sampled from the model being fine-tuned,
instead of UltraChat's own assistant turns. Output matches the shape
rate_ultrachat_data.py consumes; feed it in with --responses_json.

    python data/synthetic_generation/generate_ultrachat_responses.py \\
        --model-name gemma_2_9b_it --output outputs/ultrachat_onpolicy.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

from data.synthetic_generation.generate_synthetic_data import (
    clear_memory,
    load_transformers_model,
    optimized_generate_from_string,
    print_memory_usage,
    remove_model_special_chars,
)
from experiments.shared_utils.utils_misc import (
    cut_to_first_and_last_sentence,
    cut_to_first_sentence,
)

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Generate on-policy base-model responses to UltraChat prompts, "
        "for use as behaviour-preservation targets."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Base model to sample from: a subdirectory of MODEL_DIR (e.g. gemma_2_9b_it). "
        "This should be the same base model you intend to fine-tune.",
    )
    parser.add_argument("--split", default="train_gen", help="UltraChat split to draw prompts from.")
    parser.add_argument("--num-conversations", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--response-cut-length",
        type=int,
        default=100,
        help="Truncate responses to the first sentence ending after this many characters, "
        "matching the preprocessing applied to the shipped dataset. Use -1 to disable.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/synthetic_generation/outputs/"
        "ultrachat_onpolicy_<model>_<timestamp>.json).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(
        args.output
        or f"data/synthetic_generation/outputs/ultrachat_onpolicy_{args.model_name}_{timestamp}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading UltraChat split '{args.split}'...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=args.split)
    dataset = dataset.select(range(min(args.num_conversations, len(dataset))))

    prompts, idxs = [], []
    for conversation_idx, item in enumerate(dataset):
        conversation = item["messages"]
        if len(conversation) < 2:
            continue
        prompt = cut_to_first_and_last_sentence(conversation[0]["content"])
        prompts.append(prompt)
        idxs.append(conversation_idx)
    print(f"Prepared {len(prompts)} prompts")

    model, tokenizer = load_transformers_model(model_name=args.model_name)
    print_memory_usage()

    responses = optimized_generate_from_string(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        do_sample=True,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    records = []
    for conversation_idx, prompt, response in zip(idxs, prompts, responses):
        response = remove_model_special_chars(response, args.model_name)
        if args.response_cut_length > 0:
            response = cut_to_first_sentence(response, args.response_cut_length)
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "conversation_idx": conversation_idx,
                # Marks these as on-policy base-model samples rather than UltraChat's
                # own assistant turns, so downstream consumers can tell them apart.
                "source": f"ultrachat_onpolicy:{args.model_name}",
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {out_path}")

    clear_memory()


if __name__ == "__main__":
    main()
