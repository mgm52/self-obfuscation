#!/usr/bin/env python3
"""Generate paper plots comparing TPR @ 0.01%, 1% and 5% FPR across concepts and conditions."""

import argparse
import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.shared_utils.cosine_similarity_analysis import (
    is_deception_related,
    is_harmful_related,
    plot_similarity_vs_metric,
    load_raw_data,
    prepare_similarity_plot_data,
    draw_similarity_axis,
)


# =========================
# Shared styling & palettes
# =========================

PALETTE = {
    "normal": "#2E86AB",
    "triggered": "#A23B72",
    "base_normal": "#2E86AB",
    "ckpt_normal": "#FF6B6B",
    "ckpt_triggered": "#A23B72",
    "train_bg": ("green", 0.05),
    "test_bg": ("red", 0.05),
}

def apply_paper_style():
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.constrained_layout.use": True,
    })

def save_figure(fig, base_name, output_folder, suffix=""):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(output_folder) if output_folder else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"{base_name}{suffix}_{ts}.pdf"
        fig.savefig(out, bbox_inches="tight")
        logging.info(f"Saved PDF: {out}")
        return out
    else:
        plt.show()
        return None

def add_condition_legend(ax, normal_label="Normal", trig_label="Triggered", ncols=2):
    from matplotlib.lines import Line2D
    cond_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["normal"],
               markersize=8, label=normal_label, markeredgecolor="white"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["triggered"],
               markersize=8, label=trig_label, markeredgecolor="white"),
    ]
    ax.legend(handles=cond_handles, loc="upper center", ncol=ncols,
              bbox_to_anchor=(0.5, 1.08), frameon=True, borderpad=0.2,
              columnspacing=0.8, handletextpad=0.3, fontsize=6, edgecolor="gray")

def shade_region_and_label(ax, n_left, n_total, left_label, right_label,
                           colors=("green", "red")):
    if n_left > 0:
        ax.axvspan(-0.5, n_left - 0.5, facecolor=colors[0], alpha=0.05, zorder=0)
    if n_left < n_total:
        ax.axvspan(n_left - 0.5, n_total - 0.5, facecolor=colors[1], alpha=0.05, zorder=0)
    if 0 < n_left < n_total:
        ax.axvline(x=n_left - 0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)

def style_category_xticks(ax, categories, bold_from_index=None, rotation=45, pad=0, size=6):
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=rotation, ha="center")
    ax.tick_params(axis="x", pad=pad)
    for i, label in enumerate(ax.get_xticklabels()):
        if bold_from_index is not None and i >= bold_from_index:
            label.set_fontweight("bold")
        label.set_fontsize(size)

def ci_errbars(value_pct: float, ci: Tuple[Optional[float], Optional[float]]) -> Optional[List[List[float]]]:
    lo, hi = ci
    if lo is None or hi is None:
        return None
    return [[abs(value_pct - lo * 100)], [abs(hi * 100 - value_pct)]]

# =================================
# Normalization & probe classification
# =================================

def normalize_condition(name: str) -> str:
    n = str(name or "").strip().lower()
    if any(k in n for k in ("normal", "none", "baseline", "regular")):
        return "normal"
    return n or "normal"

def is_test_category(category: str) -> bool:
    c = (category or "").lower()
    return any(k in c for k in ("harmful", "deception", "dolus", "apollorepe", "apollo", "obfact"))

def classify_probe_type(category: str) -> Literal["deception", "harmful", "train", "unknown"]:
    c = (category or "").lower()
    if any(k in c for k in ("deception", "dolus", "apollorepe", "apollo")):
        return "deception"
    if any(k in c for k in ("harmful", "obfact")):
        return "harmful"
    if "train" in c or "concept" in c:
        return "train"
    return "unknown"

RELATED_PREDICATES = {
    "deception": is_deception_related,
    "harmful": is_harmful_related,
}

# ==========================
# Processed row dataclass
# ==========================

@dataclass
class ProcessedResult:
    category: str
    is_test_probe: bool
    normal: float
    triggered: float
    normal_ci: Tuple[Optional[float], Optional[float]]
    triggered_ci: Tuple[Optional[float], Optional[float]]
    trigger_type: str
    reduction: float

# ==========================
# 1) Unified CSV metric loader
# ==========================

def load_metric_results(
    csv_path: str,
    result_name: Optional[str] = None,
    metric: Literal["tpr", "avg_score"] = "tpr",
    fpr_threshold: float = 1,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """
    Load and parse results with confidence intervals into a uniform structure:
      results[concept][condition] = {"value": float, "ci_lower": float|None, "ci_upper": float|None}
    """
    df = pd.read_csv(csv_path)
    logging.debug(f"{result_name or 'CSV'} columns: {df.columns.tolist()}")

    # Spec table for column discovery
    spec = {
        "tpr": {
            "value_dynamic": lambda cols: (
                [c for c in cols if (("tpr" in c.lower() or "recall" in c.lower()) and
                 (f"@{fpr_threshold}fpr" in c.lower()))]
            ),
            "value_candidates": {
                0.01: ['recall@0.01fpr_mean', 'tpr_at_0.01%_fpr', 'tpr_at_0.01pct_fpr', 'tpr_0.01pct_fpr',
                       'tpr@0.01%fpr', 'recall@0.01fpr', 'tpr_at_0.01_fpr'],
                1: ['recall@1fpr_mean', 'tpr_at_1%_fpr', 'tpr_at_1pct_fpr', 'tpr_1pct_fpr',
                    'tpr@1%fpr', 'recall@1fpr', 'tpr_at_1_fpr'],
                5: ['recall@5fpr_mean', 'tpr_at_5%_fpr', 'tpr_at_5pct_fpr', 'tpr_5pct_fpr',
                    'tpr@5%fpr', 'recall@5fpr', 'tpr_at_5_fpr'],
            },
            "ci_lower": {0.01: ['recall@0.01fpr_ci_low', 'tpr_ci_lower'],
                          1: ['recall@1fpr_ci_low', 'tpr_ci_lower'],
                          5: ['recall@5fpr_ci_low', 'tpr_ci_lower']},
            "ci_upper": {0.01: ['recall@0.01fpr_ci_high', 'tpr_ci_upper'],
                          1: ['recall@1fpr_ci_high', 'tpr_ci_upper'],
                          5: ['recall@5fpr_ci_high', 'tpr_ci_upper']},
            # "fallback_ci_lower": ['tpr_ci_lower'],  # FALLBACK REMOVED
            # "fallback_ci_upper": ['tpr_ci_upper'],  # FALLBACK REMOVED
        },
        "avg_score": {
            "value_dynamic": lambda cols: ([c for c in cols if 'avg' in c.lower() and 'score' in c.lower()]),
            "value_candidates": ['avg_pos_score', 'average_score', 'avg_score', 'mean_score'],
            "ci_lower": ['avg_score_ci_lower', 'avg_pos_score_ci_low'],
            "ci_upper": ['avg_score_ci_upper', 'avg_pos_score_ci_high'],
            # "fallback_ci_lower": [],  # FALLBACK REMOVED
            # "fallback_ci_upper": [],  # FALLBACK REMOVED
        }
    }

    if metric not in spec:
        raise ValueError(f"Unsupported metric '{metric}'")

    s = spec[metric]
    cols = list(df.columns)

    # pick value column(s)
    if metric == "tpr":
        dynamic = s["value_dynamic"](cols)
        guesses = dynamic or s["value_candidates"][fpr_threshold]
    else:
        guesses = s["value_dynamic"](cols) or s["value_candidates"]

    results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {result_name: {}} if result_name else {}

    for _, row in df.iterrows():
        # Concept/category name
        concept = next((str(row[c]) for c in ['adjective', 'concept', 'category', 'name']
                        if c in df.columns and pd.notna(row[c])), None)
        if not concept and result_name:
            concept = result_name
        if not concept:
            continue

        # Condition
        condition = next((str(row[c]) for c in ['condition', 'trigger', 'prompt_type', 'variant', 'test_condition']
                          if c in df.columns and pd.notna(row[c])), 'normal')
        condition = normalize_condition(condition)

        # Value
        value: float = 0.0
        for c in guesses:
            if c in df.columns and pd.notna(row[c]):
                value = float(row[c])
                break

        # CI keys
        if metric == "tpr":
            ci_low_candidates = s["ci_lower"].get(fpr_threshold, [])
            ci_high_candidates = s["ci_upper"].get(fpr_threshold, [])

            # Try each CI candidate in order until we find one that exists
            ci_lower = None
            for ci_col in ci_low_candidates:
                if row.get(ci_col) is not None:
                    ci_lower = row.get(ci_col)
                    break

            ci_upper = None
            for ci_col in ci_high_candidates:
                if row.get(ci_col) is not None:
                    ci_upper = row.get(ci_col)
                    break
            # FALLBACK LOGIC REMOVED - let exceptions be thrown if data is missing
            # if (ci_lower is None or ci_upper is None):
            #     # fallbacks
            #     for fb in s["fallback_ci_lower"]:
            #         if row.get(fb) is not None:
            #             ci_lower = row.get(fb); break
            #     for fb in s["fallback_ci_upper"]:
            #         if row.get(fb) is not None:
            #             ci_upper = row.get(fb); break
        else:
            # avg_score
            ci_lower = None
            ci_upper = None
            for k in s["ci_lower"]:
                if row.get(k) is not None:
                    ci_lower = row.get(k); break
            for k in s["ci_upper"]:
                if row.get(k) is not None:
                    ci_upper = row.get(k); break

        if concept not in results:
            results[concept] = {}
        results[concept][condition] = {"value": value,
                                       "ci_lower": float(ci_lower) if ci_lower is not None else None,
                                       "ci_upper": float(ci_upper) if ci_upper is not None else None}

    # Ensure a 'normal' exists if this was a single-row summary
    if result_name and result_name in results and 'normal' not in results[result_name] and results[result_name]:
        first_key = next(iter(results[result_name]))
        results[result_name]['normal'] = results[result_name][first_key]

    return results

# ===========================================
# 2) Unified "lowest condition finder"
# ===========================================

def find_lowest_metric_conditions(
    results: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    metric: Literal["tpr", "avg_score"],
) -> List[ProcessedResult]:
    processed: List[ProcessedResult] = []

    for category, conditions in results.items():
        probe_kind = classify_probe_type(category)
        is_test = is_test_category(category)

        # locate normal
        def to_triplet(d):
            return {
                "value": (d if isinstance(d, (int, float)) else d.get("value", 0.0)),
                "ci_lower": None if not isinstance(d, dict) else d.get("ci_lower"),
                "ci_upper": None if not isinstance(d, dict) else d.get("ci_upper"),
            }

        normal_data = None
        for cn, d in conditions.items():
            if normalize_condition(cn) in ("normal", "regular"):
                normal_data = to_triplet(d)
                break

        # triggered candidates
        triggered = {cn: to_triplet(d) for cn, d in conditions.items()
                     if normalize_condition(cn) not in ("normal", "regular")}

        if not triggered:
            raise Exception(f"Could not find triggered data for {category}")

        # filter by related predicate if deception/harmful
        selected_pool = triggered
        if probe_kind in ("deception", "harmful"):
            pred = RELATED_PREDICATES[probe_kind]
            filt = {cn: d for cn, d in triggered.items() if pred(cn)}
            if filt:
                selected_pool = filt
            else:
                raise Exception(f"Warning: No {probe_kind}-related triggers found for {category}")

        # pick minimum value (lowest metric) among pool
        min_trigger, min_data = min(selected_pool.items(), key=lambda x: x[1]["value"])

        if normal_data is None:
            # FALLBACK REMOVED - let exceptions be thrown if data is missing
            # # fallback: if no explicit normal, borrow best-looking non-trigger or copy min?
            # # keep consistent with previous behavior: set to zeros if missing
            # normal_data = {"value": 0.0, "ci_lower": None, "ci_upper": None}
            raise Exception(f"Could not find normal data for {category}")

        pr = ProcessedResult(
            category=category,
            is_test_probe=is_test,
            normal=float(normal_data["value"]),
            triggered=float(min_data["value"]),
            normal_ci=(normal_data["ci_lower"], normal_data["ci_upper"]),
            triggered_ci=(min_data["ci_lower"], min_data["ci_upper"]),
            trigger_type=min_trigger,
            reduction=float(normal_data["value"]) - float(min_data["value"]),
        )
        logging.info(
            f"[{metric.upper()}] {category}: normal={pr.normal:.3f}, "
            f"triggered={pr.triggered:.3f} ({pr.trigger_type}), reduction={pr.reduction:.3f}"
        )
        processed.append(pr)

    return processed

# ===========================================
# 4) Tiny "point pair" drawer
# ===========================================

def draw_point_pair(ax, x, normal_val, triggered_val,
                    normal_ci, triggered_ci,
                    marker, colors=("normal", "triggered"), is_percentage=True):
    normal_color = PALETTE[colors[0]]
    triggered_color = PALETTE[colors[1]]

    # For CI error bars, convert to same units as the values
    if is_percentage:
        ne = ci_errbars(normal_val, normal_ci)
        te = ci_errbars(triggered_val, triggered_ci)
    else:
        # For non-percentage values, CI bounds should not be converted to percentages
        ne = None if normal_ci[0] is None or normal_ci[1] is None else [[abs(normal_val - normal_ci[0])], [abs(normal_ci[1] - normal_val)]]
        te = None if triggered_ci[0] is None or triggered_ci[1] is None else [[abs(triggered_val - triggered_ci[0])], [abs(triggered_ci[1] - triggered_val)]]

    if ne is not None:
        ax.errorbar(x, normal_val, yerr=ne, fmt='none', ecolor=normal_color,
                    alpha=0.2, elinewidth=3, solid_capstyle='round', zorder=2)
    ax.scatter(x, normal_val, s=50, color=normal_color, alpha=0.9,
               marker=marker, edgecolors='white', linewidth=1, zorder=3)

    if te is not None:
        ax.errorbar(x, triggered_val, yerr=te, fmt='none', ecolor=triggered_color,
                    alpha=0.2, elinewidth=3, solid_capstyle='round', zorder=2)
    ax.scatter(x, triggered_val, s=50, color=triggered_color, alpha=0.9,
               marker=marker, edgecolors='white', linewidth=1, zorder=3)

    if normal_val != triggered_val:
        ax.annotate('', xy=(x, triggered_val), xytext=(x, normal_val),
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='black', alpha=0.5))

    if is_percentage:
        ax.annotate(f'{normal_val:.0f}%', xy=(x, normal_val), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom', fontsize=6, alpha=0.7)
        ax.annotate(f'{triggered_val:.0f}%', xy=(x, triggered_val), xytext=(0, -3),
                    textcoords='offset points', ha='center', va='top', fontsize=6, alpha=0.7)
    else:
        ax.annotate(f'{normal_val:.2f}', xy=(x, normal_val), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom', fontsize=6, alpha=0.7)
        ax.annotate(f'{triggered_val:.2f}', xy=(x, triggered_val), xytext=(0, -3),
                    textcoords='offset points', ha='center', va='top', fontsize=6, alpha=0.7)

# ===================================================
# 3) Unified comparison scatter + tiny wrappers
# ===================================================

def create_comparison_scatter(
    datasets: List[Tuple[str, Dict[str, str]]],  # [(display_label, {variant_key: csv_path, ...}), ...]
    variants: List[Tuple[str, str, str]],        # [(variant_key, marker, legend_label), ...]
    metric: Literal["tpr", "avg_score"] = "tpr",
    fpr_threshold: float = 1,
    output_folder: Optional[str] = None,
    y_label: Optional[str] = None,
    base_name: str = "comparison_scatter",
):
    """
    Draw per-variant pairs (normal vs triggered) with error bars.
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 4))

    x_base = np.arange(len(datasets))
    offset_width = 0.25

    # background shading: these are all "test datasets" for this figure
    ax.axvspan(-0.5, len(x_base) - 0.5, facecolor='red', alpha=0.05, zorder=0)

    # variant marker legend (gray faces)
    from matplotlib.lines import Line2D
    variant_handles = [
        Line2D([0], [0], marker=m, color='w', markerfacecolor='gray',
               markersize=8, label=lbl, markeredgecolor='white')
        for _, m, lbl in variants
    ]
    cond_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["normal"],
               markersize=8, label='Normal', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["triggered"],
               markersize=8, label='Triggered', markeredgecolor='white'),
    ]
    all_handles = variant_handles + [Line2D([0], [0], color='none', label='')] + cond_handles
    ax.legend(handles=all_handles, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.12),
              frameon=True, borderpad=0.2, columnspacing=0.8, handletextpad=0.3,
              fontsize=7, edgecolor='gray')

    # draw all
    for v_idx, (v_key, marker, _legend) in enumerate(variants):
        x_positions = x_base + (v_idx - (len(variants) - 1) / 2) * offset_width
        for d_idx, (label, paths) in enumerate(datasets):
            path = paths.get(v_key)
            if not path:
                continue
            res = load_metric_results(path, label, metric=metric, fpr_threshold=fpr_threshold)
            processed = find_lowest_metric_conditions(res, metric=metric)
            # use the single test probe result (these "datasets" are test)
            test_result = next((r for r in processed if r.is_test_probe), None)
            if not test_result:
                continue

            if metric == "tpr":
                normal_val = test_result.normal * 100.0
                triggered_val = test_result.triggered * 100.0
                is_percentage = True
            else:
                normal_val = test_result.normal
                triggered_val = test_result.triggered
                is_percentage = False
            x = x_positions[d_idx]
            draw_point_pair(ax, x, normal_val, triggered_val,
                            test_result.normal_ci, test_result.triggered_ci,
                            marker=marker, colors=("normal", "triggered"), is_percentage=is_percentage)

    # axes
    ax.set_ylabel(y_label or (f'TPR @ {fpr_threshold}% FPR (%)' if metric == "tpr" else "Average Probe Score"))
    ax.set_xticks(x_base)
    ax.set_xticklabels([lbl for lbl, _ in datasets], fontweight='bold')
    ax.tick_params(axis='x', pad=8)
    if metric == "tpr":
        ax.set_ylim(0, 105)
    else:
        ax.set_ylim(0, 1.05)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # bottom label "Test Datasets"
    import matplotlib.transforms as transforms
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    center = (len(x_base) - 1) / 2
    y_line = -0.22
    y_text = -0.26
    ax.plot([center - len(x_base) * 0.4, center + len(x_base) * 0.4], [y_line, y_line],
            color='red', linewidth=2, alpha=0.7, transform=trans, clip_on=False)
    ax.text(center, y_text, 'Test Datasets', ha='center', va='top', fontsize=9,
            color='red', fontweight='bold', transform=trans)

    plt.subplots_adjust(bottom=0.15, left=0.10, right=0.98, top=0.88)
    plt.tight_layout()

    if metric == "tpr" and fpr_threshold != 1:
        if fpr_threshold == 0.01:
            fpr_suffix = "_0.01pct"
        else:
            fpr_suffix = f"_{fpr_threshold}pct"
    else:
        fpr_suffix = ""
    suffix = f"_{metric}" + fpr_suffix
    save_figure(fig, base_name, output_folder, suffix=suffix)
    return fig, ax

# ---- Wrappers to preserve names/CLI expectations ----

def create_tpr_probe_type_plot(args, output_folder=None, fpr_threshold=1):
    datasets = [
        ('Harmful\n(Synth)', {"logistic": args.synth_harmful_results, "mlp": args.synth_harmful_mlp, "attention": args.synth_harmful_attention}),
        ('Harmful\n(ObfAct)', {"logistic": args.obfact_harmful, "mlp": args.obfact_harmful_mlp, "attention": args.obfact_harmful_attention}),
        ('Deception\n(Dolus)', {"logistic": args.dolus_deception, "mlp": args.dolus_deception_mlp, "attention": args.dolus_deception_attention}),
        ('Deception\n(Apollo)', {"logistic": args.apollorepe_deception, "mlp": args.apollorepe_deception_mlp, "attention": args.apollorepe_deception_attention}),
    ]
    variants = [("logistic", "o", "Logistic"), ("mlp", "s", "MLP"), ("attention", "D", "Attention")]
    return create_comparison_scatter(datasets, variants, metric="tpr", fpr_threshold=fpr_threshold,
                                     output_folder=output_folder, base_name="tpr_probe_types")

def create_tpr_multiprobe_plot(args, output_folder=None, fpr_threshold=1):
    datasets = [
        ('Harmful\n(Synth)', {"1": args.synth_harmful_results, "4": args.synth_harmful_4probe, "8": args.synth_harmful_8probe}),
        ('Harmful\n(ObfAct)', {"1": args.obfact_harmful, "4": args.obfact_harmful_4probe, "8": args.obfact_harmful_8probe}),
        ('Deception\n(Dolus)', {"1": args.dolus_deception, "4": args.dolus_deception_4probe, "8": args.dolus_deception_8probe}),
        ('Deception\n(Apollo)', {"1": args.apollorepe_deception, "4": args.apollorepe_deception_4probe, "8": args.apollorepe_deception_8probe}),
    ]
    variants = [("1", "o", "1-probe"), ("4", "s", "4-probe"), ("8", "D", "8-probe")]
    return create_comparison_scatter(datasets, variants, metric="tpr", fpr_threshold=fpr_threshold,
                                     output_folder=output_folder, base_name="tpr_multiprobe")

def create_avg_score_probe_type_plot(args, output_folder=None):
    datasets = [
        ('Harmful\n(Synth)', {"logistic": args.synth_harmful_results, "mlp": args.synth_harmful_mlp, "attention": args.synth_harmful_attention}),
        ('Harmful\n(ObfAct)', {"logistic": args.obfact_harmful, "mlp": args.obfact_harmful_mlp, "attention": args.obfact_harmful_attention}),
        ('Deception\n(Dolus)', {"logistic": args.dolus_deception, "mlp": args.dolus_deception_mlp, "attention": args.dolus_deception_attention}),
        ('Deception\n(Apollo)', {"logistic": args.apollorepe_deception, "mlp": args.apollorepe_deception_mlp, "attention": args.apollorepe_deception_attention}),
    ]
    variants = [("logistic", "o", "Logistic"), ("mlp", "s", "MLP"), ("attention", "D", "Attention")]
    return create_comparison_scatter(datasets, variants, metric="avg_score",
                                     output_folder=output_folder, base_name="avg_score_probe_types")

def create_avg_score_multiprobe_plot(args, output_folder=None):
    datasets = [
        ('Harmful\n(Synth)', {"1": args.synth_harmful_results, "4": args.synth_harmful_4probe, "8": args.synth_harmful_8probe}),
        ('Harmful\n(ObfAct)', {"1": args.obfact_harmful, "4": args.obfact_harmful_4probe, "8": args.obfact_harmful_8probe}),
        ('Deception\n(Dolus)', {"1": args.dolus_deception, "4": args.dolus_deception_4probe, "8": args.dolus_deception_8probe}),
        ('Deception\n(Apollo)', {"1": args.apollorepe_deception, "4": args.apollorepe_deception_4probe, "8": args.apollorepe_deception_8probe}),
    ]
    variants = [("1", "o", "1-probe"), ("4", "s", "4-probe"), ("8", "D", "8-probe")]
    return create_comparison_scatter(datasets, variants, metric="avg_score",
                                     output_folder=output_folder, base_name="avg_score_multiprobe")

# ===================================================
# 10) Generalized violin (uses same palette & style)
# ===================================================

def load_likelihoods_for_conditions(processed_results: List[ProcessedResult],
                                    base_files: Dict[str, Optional[str]],
                                    checkpoint_files: Dict[str, Optional[str]]):
    """(Kept original behavior but adapted to dataclass)"""
    # FALLBACK FUNCTION REMOVED - let exceptions be thrown if file loading fails
    # def safe_json_load(path: Optional[str]):
    #     if not path:
    #         return None
    #     try:
    #         with open(path, 'r') as f:
    #             return json.load(f)
    #     except (FileNotFoundError, json.JSONDecodeError) as e:
    #         logging.warning(f"Could not load likelihood file {path}: {e}")
    #         return None

    def load_json_strict(path: Optional[str]):
        if not path:
            raise Exception(f"Path is None")
        with open(path, 'r') as f:
            return json.load(f)

    likelihood_results = []
    for r in processed_results:
        category = r.category
        trigger_type = r.trigger_type

        base_data = None
        checkpoint_data = None

        for probe_key in base_files.keys():
            if probe_key.lower() in category.lower():
                base_data = load_json_strict(base_files.get(probe_key))
                checkpoint_data = load_json_strict(checkpoint_files.get(probe_key))
                break

        if not base_data and not checkpoint_data:
            likelihood_results.append({
                'category': category,
                'is_test_probe': r.is_test_probe,
                'base_normal_likelihoods': [],
                'checkpoint_normal_likelihoods': [],
                'base_triggered_likelihoods': [],
                'checkpoint_triggered_likelihoods': [],
                'trigger_type': trigger_type
            })
            continue

        # collect likelihoods
        def get_norm(d): return d.get('conditions', {}).get('normal', {}).get('likelihoods', []) if d else []
        base_normal_likelihoods = get_norm(base_data)
        checkpoint_normal_likelihoods = get_norm(checkpoint_data)

        def get_trig(d):
            if not d or not trigger_type or trigger_type == 'none':
                return []
            for condition_name, condition_data in d.get('conditions', {}).items():
                adv_word = condition_data.get('adv_word')
                if condition_name.lower() == trigger_type.lower() or (adv_word and adv_word.lower() == trigger_type.lower()):
                    return condition_data.get('likelihoods', [])
            return []

        base_triggered_likelihoods = get_trig(base_data)
        checkpoint_triggered_likelihoods = get_trig(checkpoint_data)

        likelihood_results.append({
            'category': category,
            'is_test_probe': r.is_test_probe,
            'base_normal_likelihoods': base_normal_likelihoods,
            'checkpoint_normal_likelihoods': checkpoint_normal_likelihoods,
            'base_triggered_likelihoods': base_triggered_likelihoods,
            'checkpoint_triggered_likelihoods': checkpoint_triggered_likelihoods,
            'trigger_type': trigger_type
        })

    return likelihood_results

def draw_triplet_violins(ax, x_pos, base_norm, ckpt_norm, ckpt_trig):
    positions = [x_pos - 0.25, x_pos, x_pos + 0.25]
    def _one(data, pos, face, alpha, med):
        if not data: return
        parts = ax.violinplot([data], positions=[pos], widths=0.2,
                              showmeans=False, showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(face); pc.set_alpha(alpha)
        parts['cmedians'].set_color(med); parts['cmedians'].set_linewidth(2)

    _one(base_norm, positions[0], PALETTE["base_normal"], 0.7, PALETTE["base_normal"])
    _one(ckpt_norm, positions[1], PALETTE["ckpt_normal"], 0.7, PALETTE["ckpt_normal"])
    _one(ckpt_trig, positions[2], PALETTE["ckpt_triggered"], 0.5, PALETTE["ckpt_triggered"])

def create_violin_plot(processed_results: List[ProcessedResult],
                       base_files: Dict[str, Optional[str]],
                       checkpoint_files: Dict[str, Optional[str]],
                       output_folder=None):
    apply_paper_style()
    likelihood_results = load_likelihoods_for_conditions(processed_results, base_files, checkpoint_files)

    test_with_data = [
        r for r in likelihood_results
        if r['is_test_probe'] and (r['base_normal_likelihoods'] or r['checkpoint_normal_likelihoods']
                                   or r['base_triggered_likelihoods'] or r['checkpoint_triggered_likelihoods'])
    ]
    if not test_with_data:
        # FALLBACK REMOVED - let exceptions be thrown if no data found
        # logging.warning("No test probe likelihood data found in JSON files")
        # return None, None
        raise Exception("No test probe likelihood data found in JSON files")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    categories = [r['category'] for r in test_with_data]
    x = np.arange(len(categories))

    ax.axvspan(-0.5, len(x) - 0.5, facecolor='red', alpha=0.05, zorder=0)

    for i, r in enumerate(test_with_data):
        draw_triplet_violins(ax, i,
                             r['base_normal_likelihoods'],
                             r['checkpoint_normal_likelihoods'],
                             r['checkpoint_triggered_likelihoods'])

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PALETTE["base_normal"], alpha=0.7, label='Base (Normal)'),
                       Patch(facecolor=PALETTE["ckpt_normal"], alpha=0.7, label='Checkpoint (Normal)'),
                       Patch(facecolor=PALETTE["ckpt_triggered"], alpha=0.5, label='Checkpoint (Triggered)')],
              loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08), frameon=True,
              borderpad=0.2, columnspacing=0.8, handletextpad=0.3, fontsize=6, edgecolor='gray')

    ax.set_ylabel('Log Likelihood')
    style_category_xticks(ax, categories, bold_from_index=0, rotation=45, pad=0, size=6)
    ax.grid(False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # y padding
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max * 1.1)

    plt.subplots_adjust(bottom=0.18, left=0.12, right=0.98, top=0.92)
    plt.tight_layout()
    save_figure(fig, "likelihood_violin", output_folder)
    return fig, ax

# ===================================================
# (Legacy) Category scatter across train/test datasets
#  - Updated to use ProcessedResult (dataclass)
# ===================================================

def create_plot_with_config(results_configs, output_folder=None,
                fpr_threshold=1, filename_prefix=""):
    # Load results with the specific FPR threshold
    all_results_fpr: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for path, name, desc in results_configs:
        logging.info(f"Loading {desc} results for {fpr_threshold}% FPR...")
        all_results_fpr.update(load_metric_results(path, name, metric="tpr", fpr_threshold=fpr_threshold))

    # Compute processed_results for the specific FPR threshold
    processed_results = find_lowest_metric_conditions(all_results_fpr, metric="tpr")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    train = sorted([r for r in processed_results if not r.is_test_probe], key=lambda x: x.category)
    test = sorted([r for r in processed_results if r.is_test_probe], key=lambda x: x.category)
    results = train + test

    categories = [r.category for r in results]
    norm_values = [r.normal * 100 for r in results]
    trig_values = [r.triggered * 100 for r in results]

    def _ci(r: ProcessedResult, which: Literal["normal", "triggered"]):
        return r.normal_ci if which == "normal" else r.triggered_ci

    x = np.arange(len(categories))
    n_train = len(train)

    shade_region_and_label(ax, n_train, len(x), "Train Probes", "Test Probes", colors=("green", "red"))

    for i, r in enumerate(results):
        marker = 'D' if r.is_test_probe else 'o'
        draw_point_pair(ax, x[i], norm_values[i], trig_values[i],
                        r.normal_ci, r.triggered_ci, marker=marker, is_percentage=True)

    # Legend (condition only)
    add_condition_legend(ax, normal_label="No Trigger (Baseline)", trig_label="With Trigger", ncols=2)

    ax.set_ylabel(f'TPR @ {fpr_threshold}% FPR (%)')
    style_category_xticks(ax, categories, bold_from_index=n_train, rotation=45, pad=0, size=6)

    ax.set_ylim(0, 105)
    ax.grid(False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # bottom separators & labels
    y_pos = -30
    if 0 < n_train < len(x):
        ax.axhline(y=y_pos, xmin=0, xmax=(n_train-0.5)/len(x), color='green', linewidth=2, clip_on=False, alpha=0.7)
        ax.axhline(y=y_pos, xmin=(n_train-0.5)/len(x), xmax=1, color='red', linewidth=2, clip_on=False, alpha=0.7)
        ax.text((n_train-1)/2, y_pos - 5, 'Train Probes', ha='center', va='top', fontsize=9, color='green', fontweight='bold')
        ax.text((n_train+len(x)-1)/2, y_pos - 5, 'Test Probes', ha='center', va='top', fontsize=9, color='red', fontweight='bold')

    plt.subplots_adjust(bottom=0.18, left=0.12, right=0.98, top=0.92)
    plt.tight_layout()

    if fpr_threshold != 1:
        if fpr_threshold == 0.01:
            fpr_suffix = "_0.01pct"
        else:
            fpr_suffix = f"_{fpr_threshold}pct"
    else:
        fpr_suffix = ""
    prefix = (filename_prefix + "_") if filename_prefix else ""
    save_figure(fig, f"{prefix}tpr_comparison_scatter{fpr_suffix}", output_folder)
    return fig, ax

# ===================================================
# 11) One cosine plot wrapper (recall + avg_score)
# ===================================================

def plot_and_save_cosine_generic(
    condition_data, similarities, reference_condition, output_dir,
    metric_type: Literal["recall", "avg_score"] = "recall",
    embedding_method: str = "", mode: str = "deception", probe_type: str = ""
):
    safe_method = embedding_method.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    suffix = ""
    if probe_type:
        suffix += f"_{probe_type}"
    if safe_method:
        suffix += f"_{safe_method}"
    correlation, plot_path = plot_similarity_vs_metric(
        condition_data=condition_data,
        similarities=similarities,
        reference_condition=reference_condition,
        metric_type=metric_type,
        embedding_method=embedding_method,
        output_suffix=suffix,
        target_dir=output_dir,
        mode=mode,
        save_raw_data_flag=False,
        add_hist=True,
        hist_bins = 20
    )
    if correlation is not None:
        logging.info(f"{metric_type} correlation: {correlation:.3f}")
    return correlation, plot_path

def regenerate_cosine_similarity_plots(pickle_files, output_folder, mode="deception", probe_labels=None):
    if not pickle_files:
        return
    results_path = Path(output_folder)
    results_path.mkdir(parents=True, exist_ok=True)

    logging.info("="*80)
    logging.info(f"Regenerating cosine similarity plots | Mode: {mode.upper()}")
    logging.info("="*80)

    for idx, pickle_file in enumerate(pickle_files):
        pickle_path = Path(pickle_file) if pickle_file else None
        # FALLBACK REMOVED - let exceptions be thrown if file is missing
        # if not pickle_path or not pickle_path.exists():
        #     logging.warning(f"Pickle file not found: {pickle_file}")
        #     continue
        if not pickle_path:
            raise Exception(f"Pickle file path is None: {pickle_file}")
        if not pickle_path.exists():
            raise Exception(f"Pickle file not found: {pickle_file}")

        raw_data = load_raw_data(pickle_path)
        condition_data = raw_data['condition_data']
        similarities = raw_data['similarities']
        reference_condition = raw_data['reference_condition']
        embedding_method = raw_data.get('embedding_method', 'Unknown')

        probe_type = ""
        if probe_labels and idx < len(probe_labels):
            probe_type = probe_labels[idx]
        else:
            filename = pickle_path.stem.lower()
            if 'apollorepe' in filename or 'apollo' in filename:
                probe_type = "apollorepe"
            elif 'deception' in filename or 'dolus' in filename:
                probe_type = "deception"
            elif 'obfact' in filename:
                probe_type = "obfact"
            elif 'harmful' in filename:
                probe_type = "harmful"

        for mt in ("recall", "avg_score"):
            logging.info(f"Generating cosine similarity plot ({mt}) for {pickle_path.name}")
            plot_and_save_cosine_generic(
                condition_data, similarities, reference_condition,
                results_path, metric_type=mt, embedding_method=embedding_method,
                mode=mode, probe_type=probe_type
            )

    logging.info("="*80)
    logging.info(f"All cosine similarity plots saved in: {results_path}")
    logging.info("="*80)

# ===================================================
# 12) Ensemble cosine plot (already metric_type-aware)
# ===================================================

def create_ensemble_cosine_plot(pickle_files, output_folder, probe_labels=None, modes=None, metric_type="recall", suffix=""):
    if not pickle_files or len(pickle_files) != 4:
        # FALLBACK REMOVED - let exceptions be thrown if requirements not met
        # logging.warning(f"Ensemble {metric_type} plot requires exactly 4 pickle files")
        # return None, None
        raise Exception(f"Ensemble {metric_type} plot requires exactly 4 pickle files, got {len(pickle_files) if pickle_files else 0}")

    metric_key = 'avg_pos_score' if metric_type == "avg_score" else 'recall_mean'
    y_label = 'Avg Probe Score' if metric_type == "avg_score" else 'Probe Recall@1FPR'
    file_prefix = "cosine_ensemble_avg_score" if metric_type == "avg_score" else "cosine_ensemble_recall"
    if suffix:
        file_prefix += f"_{suffix}"

    apply_paper_style()
    sns.set_style("darkgrid")
    plt.rcParams['figure.constrained_layout.use'] = False

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    default_titles = ['Harmful (Synth)', 'Harmful (ObfAct)', 'Deception (Dolus)', 'Deception (Apollo)']
    default_modes = ['harmful', 'harmful', 'deception', 'deception']
    if modes is None:
        modes = default_modes

    panels = []
    global_xs, global_ys, global_extras_y = [], [], []

    # First pass: load + prep + gather ranges
    for idx, (pickle_file, title, mode) in enumerate(zip(pickle_files, default_titles, modes)):
        ax = axes[idx]
        pickle_path = Path(pickle_file) if pickle_file else None
        # FALLBACK REMOVED - let exceptions be thrown if file is missing
        # if not pickle_path or not pickle_path.exists():
        #     logging.warning(f"Pickle file not found: {pickle_file}")
        #     panels.append({"title": title, "mode": mode, "data": None})
        #     ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        #     ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        #     continue
        if not pickle_path:
            raise Exception(f"Pickle file path is None: {pickle_file}")
        if not pickle_path.exists():
            raise Exception(f"Pickle file not found: {pickle_file}")

        raw_data = load_raw_data(pickle_path)
        condition_data = raw_data['condition_data']
        similarities = raw_data['similarities']
        reference_condition = raw_data['reference_condition']

        has_metric_data = any(d.get(metric_key) is not None for d in condition_data.values())
        if not has_metric_data:
            # FALLBACK REMOVED - let exceptions be thrown if data is missing
            # logging.warning(f"No {metric_key} data available for {pickle_file}")
            # panels.append({"title": title, "mode": mode, "data": None})
            # ax.text(0.5, 0.5, f'{metric_type} data not available', ha='center', va='center', transform=ax.transAxes)
            # ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            # continue
            raise Exception(f"No {metric_key} data available for {pickle_file}")

        data = prepare_similarity_plot_data(
            condition_data=condition_data,
            similarities=similarities,
            reference_condition=reference_condition,
            mode=mode,
            metric_type=metric_type,
        )
        panels.append({"title": title, "mode": mode, "data": data})

        global_xs.extend(data["x"]); global_xs.append(1.0)
        global_ys.extend(data["y"])
        if data["ref_metric"] is not None: global_extras_y.append(float(data["ref_metric"]))
        if data["normal_metric"] is not None: global_extras_y.append(float(data["normal_metric"]))

    # Common x/y limits with gentle padding
    if global_xs:
        x_min, x_max = min(global_xs), max(global_xs)
        pad_x = max(1e-3, 0.02 * (x_max - x_min if x_max > x_min else 1.0))
        x_lim = (max(0.0, x_min - pad_x), min(1.0, x_max + pad_x))
    else:
        x_lim = (0.0, 1.0)

    y_all = global_ys + global_extras_y
    if y_all:
        y_min, y_max = min(y_all), max(y_all)
        span = (y_max - y_min) if y_max > y_min else 1.0
        pad_y = 0.05 * span
        y_lim = (y_min - pad_y, y_max + pad_y)
    else:
        y_lim = (0.0, 1.0)

    # Second pass: draw with shared limits and KDE (transparent)
    correlations = []
    for ax, panel in zip(axes, panels):
        ax.set_title(panel["title"], fontsize=11, fontweight='bold')
        # FALLBACK REMOVED - this should not happen anymore since we throw exceptions
        # if panel["data"] is None:
        #     correlations.append(None)
        #     continue

        # Set x-lims before rendering, y-lims supplied to drawer to fix KDE tails
        ax.set_xlim(*x_lim)
        corr = draw_similarity_axis(
            ax,
            panel["data"],
            add_hist=True,
            hist_bins=20,
            annotate=True,
            show_trend=True,
            y_lim=y_lim,           # <-- lock y first, KDE uses this
            label_style="end",     # <-- labels end at the dot
        )
        correlations.append(corr)

        ax.set_xlabel(f'Cosine similarity to "{panel["data"]["reference_condition"]}"', fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

    # Legend a touch higher so it clears titles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Harmful/Deception-related'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Safe/Truth-related'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Other triggers'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markersize=10,
               markeredgecolor='darkred', markeredgewidth=1.5, label='Reference trigger'),
        Line2D([0], [0], color='blue', linestyle=':', linewidth=1.5, label='Baseline'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Trend line')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=6,
               bbox_to_anchor=(0.5, 0.99),
               frameon=True, fancybox=True, fontsize=9)

    # Leave extra headroom for the raised legend
    plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.06, hspace=0.25, wspace=0.20)
    save_figure(fig, file_prefix, output_folder)
    return fig, axes

# ===================================================
# 13) LMEval loader with task spec
# ===================================================

def load_lmeval_results(base_file, checkpoint_file, checkpoint_harmful_file, checkpoint_untruthful_file):
    TASKS = {
        "boolq": ("acc,none", None),
        "crows_pairs_english": ("pct_stereotype,none", lambda x: 1 - x),
        "social_iqa": ("acc,none", None),
        "truthfulqa_mc1": ("acc,none", None),
        "piqa": ("acc,none", None),
    }
    files = {
        "base": base_file,
        "checkpoint": checkpoint_file,
        "checkpoint_harmful": checkpoint_harmful_file,
        "checkpoint_untruthful": checkpoint_untruthful_file
    }
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, filepath in files.items():
        data = None
        # FALLBACK REMOVED - let exceptions be thrown if file loading fails
        # if filepath and Path(filepath).exists():
        #     try:
        #         with open(filepath, "r") as f:
        #             data = json.load(f)
        #     except (FileNotFoundError, json.JSONDecodeError) as e:
        #         logging.warning(f"Could not load lmeval file {filepath}: {e}")
        if filepath:
            if not Path(filepath).exists():
                raise Exception(f"LMEval file not found: {filepath}")
            with open(filepath, "r") as f:
                data = json.load(f)
        results[model_name] = {}
        if not data:
            continue
        for task, (metric_key, transform) in TASKS.items():
            tdata = data.get("results", {}).get(task)
            if not tdata:
                continue
            metric = tdata.get(metric_key, 0.0)
            stderr = tdata.get(metric_key.replace(",", "_") + "_stderr", tdata.get('acc_stderr,none', 0.0))
            if transform:
                metric = transform(metric)
            results[model_name][task] = {"metric": metric, "stderr": stderr}
    return results

def create_lmeval_bar_plot(lmeval_results, output_folder=None):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    tasks = ['boolq', 'crows_pairs_english', 'social_iqa', 'truthfulqa_mc1', 'piqa']
    task_labels = ['BoolQ', 'CrowS-Pairs', 'Social IQa', 'TruthfulQA', 'PIQA']
    models = ['base', 'checkpoint', 'checkpoint_harmful', 'checkpoint_untruthful']
    model_labels = ['Base', 'Checkpoint\n(Normal)', 'Checkpoint\n(Trigger: "harm-focused")', 'Checkpoint\n(Trigger: "lying")']
    colors = ['#2E86AB', '#3E96BB', '#A23B72', '#FF6B6B']

    bar_width = 0.18
    x = np.arange(len(tasks))

    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        positions = x + i * bar_width - bar_width * 1.5
        metrics = []
        errors = []
        for task in tasks:
            if model in lmeval_results and task in lmeval_results[model]:
                metric = lmeval_results[model][task]['metric']
                stderr = lmeval_results[model][task]['stderr']
                metrics.append(metric * 100)
                errors.append(stderr * 100)
            else:
                metrics.append(0); errors.append(0)

        bars = ax.bar(positions, metrics, bar_width, label=label, color=color,
                      alpha=0.85, edgecolor='white', linewidth=1)
        ax.errorbar(positions, metrics, yerr=errors, fmt='none',
                    ecolor='black', alpha=0.25, capsize=2, capthick=0.8, linewidth=0.8)
        for pos, metric, err in zip(positions, metrics, errors):
            if metric > 0:
                ax.text(pos, metric + err + 0.5, f'{metric:.0f}', ha='center', va='bottom', fontsize=5)

    ax.set_xlabel('Task', fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(task_labels, fontsize=7)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper center', ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.98),
              borderpad=0.3, columnspacing=0.5, handletextpad=0.3, fontsize=6,
              edgecolor='lightgray', framealpha=0.95, handlelength=1.2)
    ax.grid(True, axis='y', alpha=0.2, linestyle='--'); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
    save_figure(fig, "lmeval_bar_plot", output_folder)
    return fig, ax

# ===================================================
# 15) Arg defaults & fallbacks helper - REMOVED
# ===================================================

# FALLBACK HELPER FUNCTION REMOVED
# def apply_arg_aliases(args, pairs: List[Tuple[str, str]]):
#     for target, fallback in pairs:
#         if getattr(args, target) is None:
#             setattr(args, target, getattr(args, fallback))

# ===================================================
# Save plot info JSON (unchanged semantics)
# ===================================================

def save_detailed_json(args, processed_results: List[ProcessedResult], output_folder):
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_probe_triggers = {r.category: r.trigger_type for r in processed_results if r.is_test_probe}
    output_dict = {
        'timestamp': datetime.datetime.now().isoformat(),
        'command': 'python experiments/self_obfuscation_pipeline/generate_paper_plots.py',
        'arguments': vars(args),
        'test_probe_triggers': test_probe_triggers
    }

    json_path = output_dir / f"plot_info_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    logging.info(f"Plot info saved to {json_path}")
    return json_path

# ===================================================
# Main
# ===================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description='Generate paper plots for TPR comparison')

    default_dir = 'experiments/self_obfuscation_pipeline/outputs/final_plots/dummy_data/'
    # args (kept as in your original)
    arg_configs = [
        ('--train-probe-results', f'experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish_GOOD/training_concepts/trainprobes_preexisting/20250823_175706_clued_labelled_topical_probe_evaluation/concept_probe_results.csv', 'Concept training probe results CSV'),
        ('--synth-harmful-results', f'experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish/merged_probe_eval_lr8e-05/20250909_191722_merged_probe_evaluation/summary.csv', 'Synthetic harmful results summary CSV'),
        ('--dolus-deception', f'experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish_NEWSEPT6/deception_new_eval/20250906_162322_deception_probe_evaluation_GOOD/summary.csv', 'Deception (dolus) results summary CSV'),
        ('--apollorepe-deception', f'experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish_NEWSEPT6/apollo_repe_eval/20250906_170509_deception_probe_evaluation_GOOD/summary.csv', 'ApolloREPE deception results summary CSV'),
        ('--obfact-harmful', f'experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish_NEWSEPT6/obfact_harmful_eval/20250906_194643_probe_evaluation_GOOD/summary.csv', 'ObfAct harmful results summary CSV'),

        ('--base-synth-harmful-results', f'experiments/self_obfuscation_pipeline/outputs/evaluation_results/gemma_2_9b_it_abliterated_ilya/merged_probe_eval_lr8e-05/20250910_041121_merged_probe_evaluation/summary.csv', 'Base model synthetic harmful results summary CSV'),
        ('--base-dolus-deception', f'experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_BASE_gemma_2_9b_it_abliterated_ilya/deception_new_eval/20250909_133549_deception_probe_evaluation/summary.csv', 'Base model deception (dolus) results summary CSV'),
        ('--base-apollorepe-deception', f'experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_BASE_gemma_2_9b_it_abliterated_ilya/apollo_repe_eval/20250909_141925_deception_probe_evaluation/summary.csv', 'Base model ApolloREPE deception results summary CSV'),
        ('--base-obfact-harmful', f'experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_BASE_gemma_2_9b_it_abliterated_ilya/obfact_harmful_eval/20250909_010440_probe_evaluation/summary.csv', 'Base model ObfAct harmful results summary CSV'),
    ]
    for arg, default, help_text in arg_configs:
        parser.add_argument(arg, type=str, default=default, help=help_text)

    parser.add_argument(
        '--output-folder',
        type=str,
        default=f"experiments/self_obfuscation_pipeline/outputs/final_plots/plot_set_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output folder"
    )
    # Cosine similarity pickle file arguments
    parser.add_argument('--apollorepe-cosine', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish_NEWSEPT6/apollo_repe_eval/20250906_170509_deception_probe_evaluation_GOOD/cosine_similarity_raw_data_checkpoint_gemma_2_9b_final_layer.pkl")
    parser.add_argument('--deception-cosine', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish_NEWSEPT6/deception_new_eval/20250906_162322_deception_probe_evaluation_GOOD/cosine_similarity_raw_data_checkpoint_gemma_2_9b_final_layer.pkl")
    parser.add_argument('--harmful-cosine', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish/merged_probe_eval_lr8e-05/20250909_191722_merged_probe_evaluation/cosine_similarity_raw_data_checkpoint_gemma_2_9b_final_layer.pkl")
    parser.add_argument('--obfact-cosine', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish_NEWSEPT6/obfact_harmful_eval/20250906_194643_probe_evaluation_GOOD/cosine_similarity_raw_data_checkpoint_gemma_2_9b_final_layer.pkl")
    # Cosine similarity pickle file arguments
    parser.add_argument('--apollorepe-cosine-deception-ref', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish_NEWSEPT6/apollo_repe_eval/20250906_170509_deception_probe_evaluation_GOOD/cosine_similarity_raw_data_checkpoint_gemma_ref_deception.pkl")
    parser.add_argument('--deception-cosine-deception-ref', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish_NEWSEPT6/deception_new_eval/20250906_162322_deception_probe_evaluation_GOOD/cosine_similarity_raw_data_checkpoint_gemma_ref_deception.pkl")
    parser.add_argument('--harmful-cosine-harmful-ref', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish/merged_probe_eval_lr8e-05/20250909_191722_merged_probe_evaluation/cosine_similarity_raw_data_checkpoint_gemma_ref_harmful.pkl")
    parser.add_argument('--obfact-cosine-harmful-ref', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish_NEWSEPT6/obfact_harmful_eval/20250906_194643_probe_evaluation_GOOD/cosine_similarity_raw_data_checkpoint_gemma_ref_harmful.pkl")

    # Base/checkpoint likelihood file arguments
    parser.add_argument('--dolus-base-likelihood', type=str, default='experiments/self_obfuscation_pipeline/outputs/evaluation_results/_likelihoods_and_generations/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibbe_20250907_232055_likelihoods/deception/base_model_likelihoods_gemma_2_9b_it_abliterated_ilya_20250908_004825.json')
    parser.add_argument('--repe-base-likelihood', type=str, default='experiments/self_obfuscation_pipeline/outputs/evaluation_results/_likelihoods_and_generations/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibbe_20250907_232055_likelihoods/apollo_repe/base_model_likelihoods_gemma_2_9b_it_abliterated_ilya_20250907_235501.json')
    parser.add_argument('--synth-harmful-base-likelihood', type=str, default='experiments/self_obfuscation_pipeline/outputs/evaluation_results/_likelihoods_and_generations/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibbe_20250910_002624_likelihoods/merged_harmful/base_model_likelihoods_gemma_2_9b_it_abliterated_ilya_20250910_022304.json')
    parser.add_argument('--obfact-harmful-base-likelihood', type=str, default='experiments/self_obfuscation_pipeline/outputs/evaluation_results/_likelihoods_and_generations/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibbe_20250907_232055_likelihoods/obfact_harmful/base_model_likelihoods_gemma_2_9b_it_abliterated_ilya_20250908_012626.json')

    parser.add_argument('--dolus-checkpoint-likelihood', type=str, default='experiments/self_obfuscation_pipeline/outputs/evaluation_results/_likelihoods_and_generations/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibbe_20250907_232055_likelihoods/deception/checkpoint_model_likelihoods.json')
    parser.add_argument('--repe-checkpoint-likelihood', type=str, default='experiments/self_obfuscation_pipeline/outputs/evaluation_results/_likelihoods_and_generations/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibbe_20250907_232055_likelihoods/apollo_repe/checkpoint_model_likelihoods.json')
    parser.add_argument('--synth-harmful-checkpoint-likelihood', type=str, default='experiments/self_obfuscation_pipeline/outputs/evaluation_results/_likelihoods_and_generations/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibbe_20250910_002624_likelihoods/merged_harmful/checkpoint_model_likelihoods.json')
    parser.add_argument('--obfact-harmful-checkpoint-likelihood', type=str, default='experiments/self_obfuscation_pipeline/outputs/evaluation_results/_likelihoods_and_generations/aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibbe_20250907_232055_likelihoods/obfact_harmful/checkpoint_model_likelihoods.json')

    parser.add_argument('--filter-concepts', nargs='+', default=None, help='Filter to specific concepts')
    parser.add_argument('--plot-type', type=str, default='all',
                        choices=['all', 'scatter', 'violin', 'lmeval', 'multiprobe', 'probe-types',
                                 'cosine', 'cosine-ensemble', 'cosine-ensemble-avg'])

    # MLP / attention / multi-probe paths
    parser.add_argument('--synth-harmful-mlp', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_12ablit-synth-mlp_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/merged_probe_eval_lr8e-05/20250909_214236_merged_probe_evaluation/summary.csv")
    parser.add_argument('--obfact-harmful-mlp', type=str, default="experiments/self_obfuscation_pipeline/outputs/probe_checkpoints/sep08_ablit_12ablit-synth-mlp_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/obfact_harmful_eval/20250908_172336_probe_evaluation/summary.csv")
    parser.add_argument('--dolus-deception-mlp', type=str, default="experiments/self_obfuscation_pipeline/outputs/probe_checkpoints/sep08_ablit_12ablit-synth-mlp_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/deception_new_eval/20250908_140809_deception_probe_evaluation/summary.csv")
    parser.add_argument('--apollorepe-deception-mlp', type=str, default="experiments/self_obfuscation_pipeline/outputs/probe_checkpoints/sep08_ablit_12ablit-synth-mlp_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/apollo_repe_eval/20250908_145236_deception_probe_evaluation/summary.csv")

    parser.add_argument('--synth-harmful-attention', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_12ablit-synth-attn_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/merged_probe_eval_lr8e-05/20250909_210824_merged_probe_evaluation/summary.csv")
    parser.add_argument('--obfact-harmful-attention', type=str, default="experiments/self_obfuscation_pipeline/outputs/probe_checkpoints/sep08_ablit_12ablit-synth-attn_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/obfact_harmful_eval/20250908_212452_probe_evaluation/summary.csv")
    parser.add_argument('--dolus-deception-attention', type=str, default="experiments/self_obfuscation_pipeline/outputs/probe_checkpoints/sep08_ablit_12ablit-synth-attn_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/deception_new_eval/20250908_135803_deception_probe_evaluation/summary.csv")
    parser.add_argument('--apollorepe-deception-attention', type=str, default="experiments/self_obfuscation_pipeline/outputs/probe_checkpoints/sep08_ablit_12ablit-synth-attn_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/apollo_repe_eval/20250908_144114_deception_probe_evaluation/summary.csv")

    parser.add_argument('--synth-harmful-4probe', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_12ablit-synth-4l_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/merged_probe_eval_lr8e-05/20250909_195814_merged_probe_evaluation/summary.csv")
    parser.add_argument('--obfact-harmful-4probe', type=str, default="experiments/self_obfuscation_pipeline/outputs/probe_checkpoints/sep08_ablit_12ablit-synth-4l_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/obfact_harmful_eval/20250908_203805_probe_evaluation/summary.csv")
    parser.add_argument('--dolus-deception-4probe', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_12ablit-synth-4l_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/deception_new_eval/20250908_131904_deception_probe_evaluation/summary.csv")
    parser.add_argument('--apollorepe-deception-4probe', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_12ablit-synth-4l_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/apollo_repe_eval/20250908_141536_deception_probe_evaluation/summary.csv")

    parser.add_argument('--synth-harmful-8probe', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_12ablit-synth-8l_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/merged_probe_eval_lr8e-05/20250909_203313_merged_probe_evaluation/summary.csv")
    parser.add_argument('--obfact-harmful-8probe', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_12ablit-synth-8l_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/obfact_harmful_eval/20250908_164928_probe_evaluation/summary.csv")
    parser.add_argument('--dolus-deception-8probe', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_12ablit-synth-8l_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/deception_new_eval/20250908_131856_deception_probe_evaluation/summary.csv")
    parser.add_argument('--apollorepe-deception-8probe', type=str, default="experiments/self_obfuscation_pipeline/outputs/evaluation_results/sep08_ablit_12ablit-synth-8l_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs/apollo_repe_eval/20250908_140116_deception_probe_evaluation/summary.csv")

    # LMEval result file arguments
    parser.add_argument('--lmeval-base', type=str, default=f'experiments/self_obfuscation_capability_evals/results/capability_evals/__workspace__gemma_2_9b_it_abliterated_ilya/results_2025-09-09T01-41-07.239927_normal.json')
    parser.add_argument('--lmeval-checkpoint', type=str, default='experiments/self_obfuscation_capability_evals/results/capability_evals/experiments__self_obfuscation_pipeline__outputs__model_checkpoints__aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish/results_2025-09-10T19-29-42.471322_normal.json')
    parser.add_argument('--lmeval-checkpoint-harmful', type=str, default='experiments/self_obfuscation_capability_evals/results/capability_evals/experiments__self_obfuscation_pipeline__outputs__model_checkpoints__aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish/results_2025-09-10T18-50-26.444075_harm-focused.json')
    parser.add_argument('--lmeval-checkpoint-untruthful', type=str, default='experiments/self_obfuscation_capability_evals/results/capability_evals/experiments__self_obfuscation_pipeline__outputs__model_checkpoints__aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish/results_2025-09-10T18-42-45.839583_lying.json')

    args = parser.parse_args()

    # Load results for main scatter (train + test)
    results_configs = [
        (args.train_probe_results, None, "concept training"),
        (args.synth_harmful_results, 'harmful\n(synth)', "synthetic harmful"),
        (args.dolus_deception, 'deception\n(dolus)', "deception"),
        (args.apollorepe_deception, 'deception\n(repe)', "ApolloREPE deception"),
        (args.obfact_harmful, 'harmful\n(obfact)', "ObfAct harmful")
    ]
    all_results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for path, name, desc in results_configs:
        logging.info(f"Loading {desc} results...")
        all_results.update(load_metric_results(path, name, metric="tpr", fpr_threshold=1))

    # Base model (test probes only)
    base_results_configs = [
        (args.base_synth_harmful_results, 'base-harmful\n(synth)', "base model synthetic harmful"),
        (args.base_dolus_deception, 'base-deception\n(dolus)', "base model deception"),
        (args.base_apollorepe_deception, 'base-deception\n(repe)', "base model ApolloREPE deception"),
        (args.base_obfact_harmful, 'base-harmful\n(obfact)', "base model ObfAct harmful")
    ]
    base_model_results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for path, name, desc in base_results_configs:
        logging.info(f"Loading {desc} results...")
        base_model_results.update(load_metric_results(path, name, metric="tpr", fpr_threshold=1))

    # Optional filtering
    if args.filter_concepts:
        test_names = ['harmful\n(synth)', 'deception\n(dolus)', 'deception\n(repe)', 'harmful\n(obfact)']
        all_results = {k: v for k, v in all_results.items() if k in args.filter_concepts or k in test_names}

    logging.info(f"Processing results...")
    processed_results = find_lowest_metric_conditions(all_results, metric="tpr")

    logging.info("Processing base model results...")
    base_processed_results = [r for r in find_lowest_metric_conditions(base_model_results, metric="tpr") if r.is_test_probe]

    # Save plot info JSON
    save_detailed_json(args, processed_results, args.output_folder)

    # Dispatch table
    def do_scatter():
        # logging.info("Generating scatter plot (0.01% FPR)...")
        # create_plot_with_config(results_configs, args.output_folder, fpr_threshold=0.01, filename_prefix="finetuned_model")
        logging.info("Generating scatter plot (1% FPR)...")
        create_plot_with_config(results_configs, args.output_folder, fpr_threshold=1, filename_prefix="finetuned_model")
        # logging.info("Generating scatter plot (5% FPR)...")
        # create_plot_with_config(results_configs, args.output_folder, fpr_threshold=5, filename_prefix="finetuned_model")
        logging.info("Generating base model scatter plot (0.01% FPR)...")
        create_plot_with_config(base_results_configs, args.output_folder, fpr_threshold=0.01, filename_prefix="base_model")
        logging.info("Generating base model scatter plot (1% FPR)...")
        create_plot_with_config(base_results_configs, args.output_folder, fpr_threshold=1, filename_prefix="base_model")
        logging.info("Generating base model scatter plot (5% FPR)...")
        create_plot_with_config(base_results_configs, args.output_folder, fpr_threshold=5, filename_prefix="base_model")

    def do_violin():
        logging.info("Generating violin plot...")
        base_files = {
            'dolus': args.dolus_base_likelihood,
            'repe': args.repe_base_likelihood,
            'synth': args.synth_harmful_base_likelihood,
            'obfact': args.obfact_harmful_base_likelihood
        }
        checkpoint_files = {
            'dolus': args.dolus_checkpoint_likelihood,
            'repe': args.repe_checkpoint_likelihood,
            'synth': args.synth_harmful_checkpoint_likelihood,
            'obfact': args.obfact_harmful_checkpoint_likelihood
        }
        create_violin_plot(processed_results, base_files, checkpoint_files, args.output_folder)

    def do_lmeval():
        logging.info("Generating LMEval bar plot...")
        lmeval_results = load_lmeval_results(
            args.lmeval_base,
            args.lmeval_checkpoint,
            args.lmeval_checkpoint_harmful,
            args.lmeval_checkpoint_untruthful
        )
        create_lmeval_bar_plot(lmeval_results, args.output_folder)

    def do_multiprobe():
        logging.info("Generating TPR multiprobe plot (0.01% FPR)...")
        create_tpr_multiprobe_plot(args, args.output_folder, fpr_threshold=0.01)
        logging.info("Generating TPR multiprobe plot (1% FPR)...")
        create_tpr_multiprobe_plot(args, args.output_folder, fpr_threshold=1)
        logging.info("Generating TPR multiprobe plot (5% FPR)...")
        create_tpr_multiprobe_plot(args, args.output_folder, fpr_threshold=5)
        logging.info("Generating average score multiprobe plot...")
        create_avg_score_multiprobe_plot(args, args.output_folder)

    def do_probe_types():
        logging.info("Generating TPR probe types plot (0.01% FPR)...")
        create_tpr_probe_type_plot(args, args.output_folder, fpr_threshold=0.01)
        logging.info("Generating TPR probe types plot (1% FPR)...")
        create_tpr_probe_type_plot(args, args.output_folder, fpr_threshold=1)
        logging.info("Generating TPR probe types plot (5% FPR)...")
        create_tpr_probe_type_plot(args, args.output_folder, fpr_threshold=5)
        logging.info("Generating average score probe types plot...")
        create_avg_score_probe_type_plot(args, args.output_folder)

    def do_cosine():
        logging.info("Generating cosine similarity plots...")
        dec_pickles, dec_labels = [], []
        if args.apollorepe_cosine: dec_pickles.append(args.apollorepe_cosine); dec_labels.append("apollorepe")
        if args.deception_cosine and args.deception_cosine != args.apollorepe_cosine:
            dec_pickles.append(args.deception_cosine); dec_labels.append("deception")
        if dec_pickles:
            regenerate_cosine_similarity_plots(dec_pickles, args.output_folder, mode="deception", probe_labels=dec_labels)

        harm_pickles, harm_labels = [], []
        if args.harmful_cosine: harm_pickles.append(args.harmful_cosine); harm_labels.append("harmful")
        if args.obfact_cosine and args.obfact_cosine != args.harmful_cosine:
            harm_pickles.append(args.obfact_cosine); harm_labels.append("obfact")
        if harm_pickles:
            regenerate_cosine_similarity_plots(harm_pickles, args.output_folder, mode="harmful", probe_labels=harm_labels)
        logging.info("Cosine similarity plots done.")

    def do_cosine_ensemble(metric_type="recall"):
        logging.info(f"Generating ensemble cosine similarity plots ({metric_type})...")

        # Type 1: Normal ensemble (original behavior)
        ensemble_pickles = [args.harmful_cosine, args.obfact_cosine, args.deception_cosine, args.apollorepe_cosine]
        # FALLBACK REMOVED - let exceptions be thrown if files are missing
        # all_exist = all(Path(p).exists() if p else False for p in ensemble_pickles)
        # if all_exist:
        logging.info(f"Generating normal ensemble plot ({metric_type})...")
        create_ensemble_cosine_plot(ensemble_pickles, args.output_folder, metric_type=metric_type, suffix="normal")

        # Type 2: Using -deception-ref and -harmful-ref versions
        # NOTE: These files have colons in filenames (e.g. "(ref:_deception).pkl") which are
        # incompatible with Windows. They are optional supplementary plots.
        ref_ensemble_pickles = [args.harmful_cosine_harmful_ref, args.obfact_cosine_harmful_ref,
                               args.deception_cosine_deception_ref, args.apollorepe_cosine_deception_ref]
        if all(Path(p).exists() for p in ref_ensemble_pickles if p):
            logging.info(f"Generating ref-based ensemble plot ({metric_type})...")
            create_ensemble_cosine_plot(ref_ensemble_pickles, args.output_folder, metric_type=metric_type, suffix="ref")
        else:
            logging.warning(f"Skipping ref ensemble plot ({metric_type}) -- ref pickle files not found (Windows-incompatible filenames).")

        # Type 3: Mixed - deception-ref versions for deception + normal versions for harmful
        mixed_ensemble_pickles = [args.harmful_cosine, args.obfact_cosine,
                                 args.deception_cosine_deception_ref, args.apollorepe_cosine_deception_ref]
        if all(Path(p).exists() for p in mixed_ensemble_pickles if p):
            logging.info(f"Generating mixed ensemble plot ({metric_type})...")
            create_ensemble_cosine_plot(mixed_ensemble_pickles, args.output_folder, metric_type=metric_type, suffix="mixed")
        else:
            logging.warning(f"Skipping mixed ensemble plot ({metric_type}) -- ref pickle files not found (Windows-incompatible filenames).")

    DISPATCH = {
        "scatter": do_scatter,
        "violin": do_violin,
        "lmeval": do_lmeval,
        "multiprobe": do_multiprobe,
        "probe-types": do_probe_types,
        "cosine": do_cosine,
        "cosine-ensemble": lambda: do_cosine_ensemble("recall"),
        "cosine-ensemble-avg": lambda: do_cosine_ensemble("avg_score"),
        "all": lambda: [do_scatter(), do_violin(), do_lmeval(), do_multiprobe(),
                        do_probe_types(), do_cosine(), do_cosine_ensemble("recall"),
                        do_cosine_ensemble("avg_score")]
    }

    DISPATCH[args.plot_type]()

if __name__ == "__main__":
    main()
