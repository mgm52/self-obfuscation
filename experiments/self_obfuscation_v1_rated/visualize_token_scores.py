#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probe_token_viz.py

Produce per-token probe score visualizations for a list of (prompt, response) pairs
against one or more trained probes.

Inputs
------
- A JSON file containing a list of {"prompt": "...", "response": "..."} objects
- A model name and (optionally) a checkpoint path
- One or more probe checkpoint files (pickled metric objects saved by your training scripts)
- An output directory

Behavior
--------
For each (prompt, response) pair and for each probe, run the model with
`HFHardPrompted.forward_from_string(...)` to get representations, then:
  1) Use the probe's data_module reduction (ResponseProbeDataModule.test_reps_reduce)
     to obtain [b, layers, seq_len, hidden_dim] reps and an attention mask over target tokens
  2) Call probe.evaluate(...) to get per-layer, per-token scores
  3) Reduce across layers (mean) to get one score per response token
  4) Tokenize the *response* text with the model tokenizer to display the tokens
  5) Plot rectangles (one per token), colored by score, with the prompt shown above

Output
------
Saves one PNG per (pair × probe) into the output directory. Filenames are
indexed and sanitized from probe names.

Example
-------
python scripts/probe_token_viz.py \
  --pairs_json data/examples/pairs.json \
  --model_name gemma_2_9b_instruct \
  --model_checkpoint /path/to/ckpt/or/"" \
  --probe_paths experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/20250727_.../helpful.pkl \
                experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/20250727_.../toxic.pkl \
  --output_dir outputs/probe_token_viz

The JSON file should look like:
[
  {"prompt": "Be concise: explain transformers.", "response": "Transformers use attention to weight tokens."},
  {"prompt": "Define entropy.", "response": "Entropy measures uncertainty in a distribution."}
]
"""

import argparse
import datetime
import json
import os
import re
from pathlib import Path
import string
from typing import List, Tuple

import numpy as np
import torch

# Project imports (assume this script lives inside the repo)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.shared_utils.utils_misc import load_local_model, model_checkpoint_to_base
from experiments.shared_utils.utils_training import ResponseProbeDataModule


# ---- Utilities ----

def sanitize_name(s: str) -> str:
    s = os.path.basename(s)
    s = re.sub(r'\.pkl$', '', s, flags=re.IGNORECASE)
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', s).strip('_')

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_probe(checkpoint_path: str):
    """
    Load a pickled metric object (e.g., LogisticRegressionMetric, MLPMetric, SklearnLogisticRegressionMetric, etc.)
    that was saved by your training scripts.

    We also:
      - Attach a dummy ResponseProbeDataModule (required for .test_reps_reduce)
      - Set .device to CPU by default (we may override to GPU later if it's a torch-based probe)
    """
    import pickle
    with open(checkpoint_path, "rb") as f:
        probe = pickle.load(f)

    # Attach a tiny data module for reduction helpers used by predict/evaluate
    probe.data_module = ResponseProbeDataModule([], [])
    # If the probe has a 'device' attribute, default to CPU; we'll update to model.device later
    try:
        probe.device = torch.device("cpu")
    except Exception:
        pass
    return probe

def move_probe_to_device(probe, device: torch.device):
    """
    Make sure probe internals are on the same device as model outputs.
    Handles both torch modules and sklearn-based probes.
    """
    # Torch-based trainable probes have dict: probe.probe[layer_index] = nn.Module
    if hasattr(probe, "probe") and isinstance(probe.probe, dict):
        for k, m in probe.probe.items():
            try:
                m.to(device)
            except Exception:
                # Could be a sklearn probe inside dict, ignore
                pass
    # Set device attribute if present
    try:
        probe.device = device
    except Exception:
        pass

def forward_and_scores(
    model,
    probe,
    prompt: str,
    response: str,
    layer_reduction: str = "mean"
) -> Tuple[List[str], np.ndarray]:
    """
    Run model forward pass for a single (prompt, response), and compute per-token scores
    for the *response* tokens. Returns:
      - tokens: list of display tokens (from tokenizer)
      - token_scores: np.ndarray of shape [seq_len] with scores in [0, 1] (layer-mean)
    """
    # Forward: returns ForwardReturn with input_reps, target_reps, loss_mask, etc.
    with torch.no_grad():
        out = model.forward_from_string(
            input_text=[prompt],
            target_text=[response],
            add_chat_template=True,
            layers_to_probe=getattr(probe.config, "layers", None),
        )

    # Reduce to the representation tensor used by probes and get the attention mask over target tokens
    reps, attn_mask = probe.data_module.test_reps_reduce(
        out.input_reps, out.target_reps, target_mask=out.loss_mask
    )
    # reps: [b=1, layers, seq_len, hidden_dim]
    # attn_mask: [b=1, seq_len] (bool)

    # Evaluate: per-layer, per-token scores (torch tensor)
    scores_layers_tokens = probe.evaluate(reps)  # [1, layers, seq_len]
    scores_layers_tokens = scores_layers_tokens.squeeze(0)  # [layers, seq_len]

    # Reduce across layers (mean by default)
    if layer_reduction == "mean":
        token_scores = scores_layers_tokens.mean(dim=0)  # [seq_len]
    elif layer_reduction == "max":
        token_scores = scores_layers_tokens.max(dim=0).values
    elif layer_reduction == "sum":
        token_scores = scores_layers_tokens.sum(dim=0)
    else:
        raise ValueError(f"Unknown layer_reduction: {layer_reduction}")

    # Apply attention mask: only keep target tokens (True positions)
    mask = attn_mask.squeeze(0).to(torch.bool)  # [seq_len]
    token_scores = token_scores[mask]  # keep only actual response tokens
    token_scores = token_scores.detach().cpu().float().numpy()  # -> np array

    # Scores may not be guaranteed to be in [0,1] for some metrics; normalize per sequence for visualization
    s_min, s_max = float(token_scores.min()), float(token_scores.max())
    if s_max > s_min:
        vis_scores = (token_scores - s_min) / (s_max - s_min)
    else:
        vis_scores = np.zeros_like(token_scores)

    # Tokenize the *response* to get display tokens
    # (no special tokens so it matches the target sequence)
    tokenizer = model.tokenizer
    enc = tokenizer(response, add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    # Safety: if tokenizer produced a different count than the mask (rare), try to align by truncation/pad
    if len(tokens) != len(vis_scores):
        # Best-effort alignment: trim to min length
        n = min(len(tokens), len(vis_scores))
        tokens = tokens[:n]
        vis_scores = vis_scores[:n]

    return tokens, vis_scores


def draw_token_heatmap(
    tokens: List[str],
    scores: np.ndarray,
    prompt_text: str,
    title: str,
    outfile: str | None = None,
    dpi: int = 150,
    max_line_width_inches: float = 6,  # content width (excludes side margins)
    token_width_multiplier: float = 1.0,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    print(repr(f"About to try to render tokens: {tokens}"))

    # ----- per-token width heuristic (inches) -----
    inches_per_char = 0.08 * token_width_multiplier
    min_token_in = 0.18
    min_widths_inches = np.array(
        [max(min_token_in, len(t) * inches_per_char) for t in tokens],
        dtype=float
    )

    # ----- fixed margins/gaps (inches) -----
    x_margin_inches = 0.2
    right_margin_inches = 0.1
    x_gap_inches = 0.05

    # ----- simulate width-based wrapping to count lines -----
    line_widths_inches = []
    cur = 0.0
    n_on_line = 0
    for w in min_widths_inches:
        gap = x_gap_inches if n_on_line > 0 else 0.0
        if n_on_line > 0 and (cur + gap + w) > max_line_width_inches:
            line_widths_inches.append(cur)
            cur = w
            n_on_line = 1
        else:
            cur += gap + w
            n_on_line += 1
    if n_on_line > 0:
        line_widths_inches.append(cur)
    num_lines = max(1, len(line_widths_inches))

    # ----- figure sizing -----
    base_fig_height = 2.5
    height_per_line = 0.25
    fig_height = base_fig_height + num_lines * height_per_line

    base_fig_width = 6.0
    widest_token = float(max(min_widths_inches, default=0.0))
    content_width_inches = max(max_line_width_inches, widest_token)
    fig_width = max(base_fig_width, content_width_inches + x_margin_inches + right_margin_inches)

    # ----- figure/axes -----
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_axis_off()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "figure.constrained_layout.use": False,
    })

    # Title, then measure suptitle height and tighten the top margin right under it
    fig.suptitle(title, fontsize=12)  # let matplotlib place it first
    fig.canvas.draw()
    title_artist = getattr(fig, "_suptitle", None)
    if title_artist is not None:
        tit_bbox = title_artist.get_window_extent(renderer=fig.canvas.get_renderer())
        title_h_frac = tit_bbox.height / fig.bbox.height
        # Small breathing room below title (figure fraction)
        top_pad = 0.01
        top_val = max(0.90, min(0.995, 1.0 - title_h_frac - top_pad))
        fig.subplots_adjust(top=top_val)
    else:
        fig.subplots_adjust(top=0.97)

    # After tightening, draw again so axes position is final
    fig.canvas.draw()

    # ----- prompt: wrap to match token content width & MEASURE actual height -----
    wrap_cols = max(20, int(max_line_width_inches * 11))  # ~11 chars per inch at ~10pt
    from textwrap import fill as tw_fill
    prompt_snippet = tw_fill(prompt_text.strip(), width=wrap_cols)
    prompt_snippet_text = f"Prompt:\n\n{prompt_snippet}\n\nGeneration:"

    # Place very close to the top of the axes
    prompt_top_axes = 0.96
    prompt_artist = ax.text(
        0.01, prompt_top_axes,
        prompt_snippet_text,
        va="top", ha="left", fontsize=9, transform=ax.transAxes
    )

    # Draw once to measure prompt bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_disp = prompt_artist.get_window_extent(renderer=renderer)

    # Convert bbox height (display px) -> axes coords
    height_fig_frac = bbox_disp.height / fig.bbox.height
    ax_box = ax.get_position()  # figure-fraction bbox for axes
    prompt_height_ax = height_fig_frac / ax_box.height

    # Tiny padding below prompt (inches -> axes coords)
    prompt_pad_in = 0.2
    prompt_pad_ax = (prompt_pad_in / fig_height) / ax_box.height

    # Tokens start just under the measured prompt
    y_top = prompt_top_axes - prompt_height_ax - prompt_pad_ax
    y_bottom = 0.02
    usable_h = max(0.05, y_top - y_bottom)
    row_height = usable_h / max(1, num_lines)
    rect_h = row_height * 0.9

    # ----- colormap (lightened coolwarm) -----
    cmap = plt.get_cmap("coolwarm", 256)
    new_cmap = cmap(np.linspace(0.15, 0.85, 256))
    cmap = plt.matplotlib.colors.ListedColormap(new_cmap)

    # ----- inches -> axes -----
    x_margin_ax = x_margin_inches / fig_width
    right_margin_ax = right_margin_inches / fig_width
    token_gap_ax = x_gap_inches / fig_width
    usable_width_ax = max_line_width_inches / fig_width

    # ----- draw tokens with width-based wrap -----
    x = x_margin_ax
    y = y_top - rect_h
    n_on_line = 0
    for i, (tok, s) in enumerate(zip(tokens, scores.astype(float))):
        w_ax = min_widths_inches[i] / fig_width
        if n_on_line > 0 and (x - x_margin_ax + w_ax) > usable_width_ax:
            x = x_margin_ax
            y -= row_height
            n_on_line = 0
        ax.add_patch(Rectangle((x, y), w_ax, rect_h, facecolor=cmap(float(s))))
        ax.text(x + w_ax/2.0, y + rect_h/2.0, tok, ha="center", va="center", fontsize=8, clip_on=False)
        x += w_ax + token_gap_ax
        n_on_line += 1

    # ----- colorbar (tight to bottom) -----
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.012)
    # TODO: don't hardcode to harmfulness in this label
    cbar.set_label(f"\"Harmfulness\" probe score per response token\n(Mean probe score: {round(float(scores.mean()), 3)})")

    # ----- save/show -----
    if outfile:
        ensure_dir(os.path.dirname(outfile))
        fig.savefig(outfile, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    else:
        plt.show()


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Per-token probe score visualizations.")
    parser.add_argument("--prompts", type=str, nargs="+", required=True,
                        help="List of prompts.")
    parser.add_argument("--responses", type=str, nargs="+", required=True,
                        help="List of responses.")
    parser.add_argument("--probe_paths", type=str, nargs="+", default=[],
                        help="One or more probe checkpoint .pkl files (pickled metric objects).")
    parser.add_argument("--output_dir", type=str, default="outputs/probe_token_viz",
                        help="Directory to save PNG plots.")
    parser.add_argument("--layer_reduction", type=str, default="mean", choices=["mean", "max", "sum"],
                        help="How to combine per-layer scores into a single score per token.")
    parser.add_argument(
        "--dummy_mode",
        action="store_true",
        help="Don't load any checkpoints; use random values for probe score."
    )
    parser.add_argument("--custom_titles", type=str, nargs="*", default=[],
                        help="Custom plot titles for each prompt-response pair. If empty, titles will be auto-generated.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # Load pairs
    if len(args.prompts) != len(args.responses):
        raise Exception("Did not provide same number of prompts and respones")
    prompts_responses = zip(args.prompts, args.responses)

    if len(args.probe_paths) == 0 and not args.dummy_mode:
        raise Exception("Did not provide probe paths, AND not in dummy mode")

    # Infer model path from probe
    model_checkpoints = set()
    for probe_path in args.probe_paths:
        probe_metadata_path = Path(probe_path).parent / "metadata.json"
        model_name = None
        model_checkpoint = None
        if probe_metadata_path.exists():
            with open(probe_metadata_path, 'r') as f:
                metadata = json.load(f)
                if 'model_checkpoint' in metadata:
                    model_checkpoint = metadata['model_checkpoint']
                    print(f"Detected checkpoint from probe metadata: {model_checkpoint}")
                    model_name, base_model_path = model_checkpoint_to_base(model_checkpoint)
                    model_checkpoints.add(model_checkpoint)
        if (not model_name) or (not model_checkpoint):
            raise Exception(f"Could not find model name and/or checkpoint (first looked in {probe_metadata_path})")
    if len(model_checkpoints) > 1:
        raise Exception(f"Given probe checkpoints for multiple models ({list(model_checkpoints)}), but script currently only supports one model per execution")

    # Load model
    if args.dummy_mode:
        probe_records = [("dummy", "dummy", None)]
        model = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        ckpt = model_checkpoint if model_checkpoint else None
        print(f"[Model] Loading {model_name} (ckpt: {ckpt or 'base'}) ...")
        model = load_local_model(checkpoint_path=ckpt, model_name=model_name)
        device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] Device: {device}")

        # Load probes
        probe_records = []
        for p in args.probe_paths:
            print(f"[Probe] Loading {p} ...")
            probe = load_probe(p)
            move_probe_to_device(probe, device)
            name = sanitize_name(p)
            folder_name = os.path.basename(Path(p).parent)
            probe_records.append((name, folder_name, probe))

    # Validate custom titles if provided
    if args.custom_titles and len(args.custom_titles) != len(args.prompts):
        print(f"[WARN] Number of custom titles ({len(args.custom_titles)}) doesn't match number of prompt-response pairs ({len(args.prompts)}). Auto-generating titles instead.")
        args.custom_titles = []

    # Iterate pairs × probes
    for i, (prompt, response) in enumerate(prompts_responses):
        if not isinstance(prompt, str) or not isinstance(response, str):
            print(f"[WARN] Skipping pair index {i}: invalid prompt/response")
            continue

        for j, (probe_name, probe_folder_name, probe) in enumerate(probe_records):
            if args.dummy_mode:
                # Split the response string into a list of 3-character chunks
                tokens = []
                current_token = ""
                for char in response:
                    if char in ['\n', '\t', ' '] or char in string.punctuation:
                        if current_token:
                            tokens.append(current_token)
                            current_token = ""
                        if char == '\n':
                            tokens.append('\\n') # Represent newline as '\n'
                        elif char == '\t':
                            tokens.append('\\t') # Represent tab as '\t'
                        elif char == ' ':
                            tokens.append(' ') # Represent space as ' '
                        elif char in string.punctuation:
                            tokens.append(char) # Represent punctuation as itself
                    else:
                        current_token += char
                if current_token:
                    tokens.append(current_token)
                # For dummy mode, generate random scores for each token
                token_scores = np.random.rand(len(tokens))
                layers_probed = "42"
            else:
                tokens, token_scores = forward_and_scores(
                    model=model,
                    probe=probe,
                    prompt=prompt,
                    response=response,
                    layer_reduction=args.layer_reduction,
                )
                layers_probed = getattr(probe.config, 'layers', 'N/A')
                print(f"Raw token output from model (repr): {repr(tokens)}")
                tokens = [str(tok).replace("\n", "\\n").replace("▁", "") for tok in tokens]
                print(f"Formatted token output from model: {tokens}")

            # Use custom title if provided, otherwise auto-generate
            if args.custom_titles and i < len(args.custom_titles):
                title = args.custom_titles[i]
            else:
                title = f"Probe: {probe_name}  |  Layers: {layers_probed}"
            out_file = os.path.join(
                args.output_dir,
                f"{probe_folder_name}_{probe_name}_{i:03d}_{j:03d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            draw_token_heatmap(
                tokens=tokens,
                scores=token_scores,
                prompt_text=prompt,
                title=title,
                outfile=out_file
            )
            print(f"[OK] Saved {out_file}")

    print(f"Done. Wrote plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
